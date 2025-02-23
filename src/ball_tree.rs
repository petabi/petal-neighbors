use std::cmp;
use std::collections::BinaryHeap;
use std::num::NonZeroUsize;
use std::ops::{AddAssign, DivAssign, Range};

use ndarray::{Array1, ArrayBase, ArrayView1, CowArray, Data, Ix1, Ix2};
use num_traits::{Float, FromPrimitive, Zero};
use ordered_float::{FloatCore, OrderedFloat};

use crate::distance::{self, Euclidean, Metric};
use crate::ArrayError;

/// A data structure for nearest neighbor search in a multi-dimensional space,
/// which is partitioned into a nested set of hyperspheres, or "balls".
pub struct BallTree<'a, A, M>
where
    A: FloatCore,
    M: Metric<A>,
{
    pub points: CowArray<'a, A, Ix2>,
    pub idx: Vec<usize>,
    pub nodes: Vec<Node<A>>,
    pub metric: M,
}

impl<'a, A, M> BallTree<'a, A, M>
where
    A: FloatCore + Zero + AddAssign + DivAssign + FromPrimitive,
    M: Metric<A>,
{
    /// Builds a ball tree using the given distance metric.
    ///
    /// # Errors
    ///
    /// * `ArrayError::Empty` if `points` is an empty array.
    /// * `ArrayError::NotContiguous` if any row in `points` is not
    ///   contiguous in memory.
    pub fn new<T>(points: T, metric: M) -> Result<Self, ArrayError>
    where
        T: Into<CowArray<'a, A, Ix2>>,
    {
        let points = points.into();
        let n_points: usize = points.nrows();
        if n_points == 0 {
            return Err(ArrayError::Empty);
        }
        if !points.row(0).is_standard_layout() {
            return Err(ArrayError::NotContiguous);
        }

        let height = usize::BITS - n_points.leading_zeros();
        let size = 1_usize.wrapping_shl(height) - 1;

        let mut idx: Vec<usize> = (0..n_points).collect();
        let mut nodes = vec![Node::default(); size];
        build_subtree(&mut nodes, &mut idx, &points, 0, 0..n_points, &metric);
        Ok(BallTree {
            points,
            idx,
            nodes,
            metric,
        })
    }

    /// Finds the nearest neighbor and its distance in the tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::{array, aview1};
    /// use petal_neighbors::BallTree;
    ///
    /// let points = array![[1., 1.], [1., 2.], [9., 9.]];
    /// let tree = BallTree::euclidean(points).expect("valid array");
    /// let (index, distance) = tree.query_nearest(&aview1(&[8., 8.]));
    /// assert_eq!(index, 2);  // points[2] is the nearest.
    /// assert!((2_f64.sqrt() - distance).abs() < 1e-8);
    /// ```
    #[allow(clippy::missing_panics_doc)] // never panics
    pub fn query_nearest<S>(&self, point: &ArrayBase<S, Ix1>) -> (usize, A)
    where
        S: Data<Elem = A>,
    {
        self.nearest_neighbor_in_subtree(&point.view(), 0, A::infinity())
            .expect("0 is a valid index")
    }

    /// Finds the nearest `k` neighbors and their distances in the tree. The
    /// return values are sorted in the ascending order in distance.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::{array, aview1};
    /// use petal_neighbors::BallTree;
    ///
    /// let points = array![[1., 1.], [1., 2.], [9., 9.]];
    /// let tree = BallTree::euclidean(points).expect("non-empty input");
    /// let (indices, distances) = tree.query(&aview1(&[3., 3.]), 2);
    /// assert_eq!(indices, &[1, 0]);  // points[1] is the nearest, followed by points[0].
    /// ```
    pub fn query<S>(&self, point: &ArrayBase<S, Ix1>, k: usize) -> (Vec<usize>, Vec<A>)
    where
        S: Data<Elem = A>,
    {
        let Some(k) = NonZeroUsize::new(k) else {
            return (Vec::new(), Vec::new());
        };
        let mut neighbors = BinaryHeap::with_capacity(k.get());
        self.nearest_k_neighbors_in_subtree(&point.view(), 0, A::infinity(), k, &mut neighbors);
        let sorted = neighbors.into_sorted_vec();
        let indices = sorted.iter().map(|v| v.idx).collect();
        let distances = sorted.iter().map(|v| v.distance.into_inner()).collect();
        (indices, distances)
    }

    /// Finds all neighbors whose distances from `point` are less than or equal
    /// to `distance`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::{array, aview1};
    /// use petal_neighbors::BallTree;
    ///
    /// let points = array![[1., 0.], [2., 0.], [9., 0.]];
    /// let tree = BallTree::euclidean(points).expect("non-empty input");
    /// let indices = tree.query_radius(&aview1(&[3., 0.]), 1.5);
    /// assert_eq!(indices, &[1]);  // The distance to points[1] is less than 1.5.
    /// ```
    pub fn query_radius<S>(&self, point: &ArrayBase<S, Ix1>, distance: A) -> Vec<usize>
    where
        S: Data<Elem = A>,
    {
        self.neighbors_within_radius_in_subtree(&point.view(), distance, 0)
    }

    /// Finds the nearest neighbor and its distance in the subtree rooted at `root`.
    ///
    /// # Panics
    ///
    /// Panics if `root` is out of bound.
    fn nearest_neighbor_in_subtree(
        &self,
        point: &ArrayView1<A>,
        root: usize,
        radius: A,
    ) -> Option<(usize, A)> {
        let root_node = &self.nodes[root];
        let lower_bound = self.nodes[root].distance_lower_bound(point, &self.metric);
        if lower_bound > radius {
            return None;
        }

        if root_node.is_leaf {
            let (min_i, min_dist) = self.idx[root_node.range.clone()].iter().fold(
                (0, A::infinity()),
                |(min_i, min_dist), &i| {
                    let dist = self.metric.distance(point, &self.points.row(i));

                    if dist < min_dist {
                        (i, dist)
                    } else {
                        (min_i, min_dist)
                    }
                },
            );
            if min_dist <= radius {
                Some((min_i, min_dist))
            } else {
                None
            }
        } else {
            let child1 = root * 2 + 1;
            let child2 = child1 + 1;
            let lb1 = self.nodes[child1].distance_lower_bound(point, &self.metric);
            let lb2 = self.nodes[child2].distance_lower_bound(point, &self.metric);
            let (child1, child2) = if lb1 < lb2 {
                (child1, child2)
            } else {
                (child2, child1)
            };
            match self.nearest_neighbor_in_subtree(point, child1, radius) {
                Some(neighbor) => self
                    .nearest_neighbor_in_subtree(point, child2, neighbor.1)
                    .map_or(Some(neighbor), Some),
                None => self.nearest_neighbor_in_subtree(point, child2, radius),
            }
        }
    }

    /// Finds the nearest k neighbors within the radius in the subtree rooted at `root`.
    ///
    /// # Panics
    ///
    /// Panics if `root` is out of bound.
    fn nearest_k_neighbors_in_subtree(
        &self,
        point: &ArrayView1<A>,
        root: usize,
        radius: A,
        k: NonZeroUsize,
        neighbors: &mut BinaryHeap<Neighbor<A>>,
    ) {
        let root_node = &self.nodes[root];
        if root_node.distance_lower_bound(point, &self.metric) > radius {
            return;
        }

        if root_node.is_leaf {
            self.idx[root_node.range.clone()]
                .iter()
                .filter_map(|&i| {
                    let dist = self.metric.distance(point, &self.points.row(i));

                    if dist < radius {
                        Some(Neighbor::new(i, dist))
                    } else {
                        None
                    }
                })
                .fold(neighbors, |neighbors, n| {
                    if neighbors.len() < k.get() {
                        neighbors.push(n);
                    } else if n < *neighbors.peek().expect("not empty") {
                        neighbors.pop();
                        neighbors.push(n);
                    }
                    neighbors
                });
        } else {
            let child1 = root * 2 + 1;
            let child2 = child1 + 1;
            let lb1 = self.nodes[child1].distance_lower_bound(point, &self.metric);
            let lb2 = self.nodes[child2].distance_lower_bound(point, &self.metric);
            let (child1, child2) = if lb1 < lb2 {
                (child1, child2)
            } else {
                (child2, child1)
            };
            self.nearest_k_neighbors_in_subtree(point, child1, radius, k, neighbors);
            self.nearest_k_neighbors_in_subtree(point, child2, radius, k, neighbors);
        }
    }

    /// Finds the neighbors within the radius in the subtree rooted at `root`.
    ///
    /// # Panics
    ///
    /// Panics if `root` is out of bound.
    fn neighbors_within_radius_in_subtree(
        &self,
        point: &ArrayView1<A>,
        radius: A,
        root: usize,
    ) -> Vec<usize> {
        let mut neighbors = Vec::new();
        let mut subtrees_to_visit = vec![root];

        loop {
            let subroot = subtrees_to_visit.pop().expect("should not be empty");
            let root_node = &self.nodes[subroot];
            let (lb, ub) = root_node.distance_bounds(point, &self.metric);

            if lb > radius {
                if subtrees_to_visit.is_empty() {
                    break;
                }
                continue;
            }

            if ub <= radius {
                neighbors.reserve(root_node.range.end - root_node.range.start);
                neighbors.extend(self.idx[root_node.range.clone()].iter().copied());
            } else if root_node.is_leaf {
                neighbors.extend(self.idx[root_node.range.clone()].iter().filter_map(|&i| {
                    let dist = self.metric.distance(point, &self.points.row(i));
                    if dist < radius {
                        Some(i)
                    } else {
                        None
                    }
                }));
            } else {
                subtrees_to_visit.push(subroot * 2 + 1);
                subtrees_to_visit.push(subroot * 2 + 2);
            }

            if subtrees_to_visit.is_empty() {
                break;
            }
        }

        neighbors
    }

    // calculate minimum distance between the nodes:
    //   max(||n1_centroid - n2_centroid|| - R_n1 - R_n2, 0)
    ///
    /// # Panics
    ///
    /// Panics if `n1 >= self.nodes.len()` or `n2 >= self.nodes.len()`
    #[inline]
    pub fn node_distance_lower_bound(&self, n1: usize, n2: usize) -> A {
        assert!(n1 < self.nodes.len() && n2 < self.nodes.len());
        let n1 = &self.nodes[n1];
        let n2 = &self.nodes[n2];
        let lb = self
            .metric
            .distance(&n1.centroid.view(), &n2.centroid.view())
            - n1.radius
            - n2.radius;
        if lb < A::zero() {
            A::zero()
        } else {
            lb
        }
    }

    #[inline]
    pub fn children_of(&self, n: usize) -> Option<(usize, usize)> {
        if self.nodes[n].is_leaf {
            None
        } else {
            let left = 2 * n + 1;
            let right = left + 1;
            Some((left, right))
        }
    }

    #[inline]
    pub fn points_of(&self, n: usize) -> &[usize] {
        &self.idx[self.nodes[n].range.clone()]
    }

    #[inline]
    pub fn radius_of(&self, n: usize) -> A {
        self.nodes[n].radius
    }

    #[inline]
    pub fn compare_nodes(&self, x: usize, y: usize) -> Option<std::cmp::Ordering> {
        self.nodes[x].radius.partial_cmp(&self.nodes[y].radius)
    }

    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    pub fn num_points(&self) -> usize {
        self.points.nrows()
    }
}

impl<'a, A> BallTree<'a, A, Euclidean>
where
    A: FloatCore + Float + Zero + AddAssign + DivAssign + FromPrimitive,
{
    /// Builds a ball tree with a euclidean distance metric.
    ///
    /// # Errors
    ///
    /// * `ArrayError::Empty` if `points` is an empty array.
    /// * `ArrayError::NotContiguous` if any row in `points` is not
    ///   contiguous in memory.
    pub fn euclidean<T>(points: T) -> Result<BallTree<'a, A, Euclidean>, ArrayError>
    where
        A: Float + Zero + AddAssign + DivAssign + FromPrimitive,
        T: Into<CowArray<'a, A, Ix2>>,
    {
        BallTree::<'a, A, Euclidean>::new(points, distance::Euclidean::default())
    }
}

/// An error returned when an array is not suitable to build a `BallTree`.
#[derive(Clone, Debug)]
struct Neighbor<A> {
    pub idx: usize,
    pub distance: OrderedFloat<A>,
}

impl<A> Neighbor<A>
where
    A: FloatCore,
{
    #[must_use]
    pub fn new(idx: usize, distance: A) -> Self {
        Self {
            idx,
            distance: distance.into(),
        }
    }
}

impl<A> Ord for Neighbor<A>
where
    A: FloatCore,
{
    #[must_use]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.distance.cmp(&other.distance)
    }
}

impl<A> PartialOrd for Neighbor<A>
where
    A: FloatCore,
{
    #[must_use]
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<A> PartialEq for Neighbor<A>
where
    A: FloatCore,
{
    #[must_use]
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl<A> Eq for Neighbor<A> where A: FloatCore {}

/// A node containing a range of points in a ball tree.
#[derive(Clone, Debug)]
pub struct Node<A> {
    range: Range<usize>,
    centroid: Array1<A>,
    radius: A,
    is_leaf: bool,
}

impl<A> Node<A>
where
    A: FloatCore + Zero + AddAssign + DivAssign + FromPrimitive,
{
    /// Computes the centroid of the node.
    ///
    /// # Panics
    ///
    /// Panics if any row in `points` is not contiguous in memory.
    #[allow(clippy::cast_precision_loss)] // The precision provided by 54-bit-wide mantissa is
                                          // good enough in computing mean.
    fn init(&mut self, points: &CowArray<A, Ix2>, idx: &[usize], metric: &dyn Metric<A>) {
        let mut sum = idx
            .iter()
            .fold(Array1::<A>::zeros(points.ncols()), |mut sum, &i| {
                for (s, v) in sum.iter_mut().zip(points.row(i)) {
                    *s += *v;
                }
                sum
            });
        let len = A::from_usize(idx.len()).expect("approximation");
        sum.iter_mut().for_each(|v| *v /= len);
        self.centroid = sum;

        self.radius = idx.iter().fold(A::zero(), |max, &i| {
            A::max(metric.distance(&self.centroid.view(), &points.row(i)), max)
        });
    }

    fn distance_bounds(&self, point: &ArrayView1<A>, metric: &dyn Metric<A>) -> (A, A) {
        let centroid_dist = metric.distance(point, &self.centroid.view());
        let mut lb = centroid_dist - self.radius;
        if lb < A::zero() {
            lb = A::zero();
        }
        let ub = centroid_dist + self.radius;
        (lb, ub)
    }

    fn distance_lower_bound(&self, point: &ArrayView1<A>, metric: &dyn Metric<A>) -> A {
        let centroid_dist = metric.distance(point, &self.centroid.view());
        let lb = centroid_dist - self.radius;
        if lb < A::zero() {
            A::zero()
        } else {
            lb
        }
    }
}

impl<A> Default for Node<A>
where
    A: FloatCore + Zero,
{
    #[allow(clippy::reversed_empty_ranges)] // An empty range is valid because `centroid` is empty.
    fn default() -> Self {
        Self {
            range: (0..0),
            centroid: Array1::<A>::zeros(0),
            radius: A::zero(),
            is_leaf: false,
        }
    }
}

/// Builds a subtree recursively.
///
/// # Panics
///
/// Panics if `root` or `range` is out of range.
fn build_subtree<A, M>(
    nodes: &mut [Node<A>],
    idx: &mut [usize],
    points: &CowArray<A, Ix2>,
    root: usize,
    range: Range<usize>,
    metric: &M,
) where
    A: FloatCore + AddAssign + DivAssign + FromPrimitive,
    M: Metric<A>,
{
    let n_nodes = nodes.len();
    let root_node = nodes.get_mut(root).expect("root node index out of range");
    root_node.init(
        points,
        idx.get(range.clone()).expect("invalid subtree range"),
        metric,
    );
    root_node.range = range.clone();
    let left = root * 2 + 1;
    if left >= n_nodes {
        root_node.is_leaf = true;
        return;
    }

    #[allow(clippy::deref_addrof)]
    let col_idx = max_spread_column(points, &idx[range.clone()]);
    debug_assert!(col_idx < points.ncols());
    let col = points.column(col_idx);
    halve_node_indices(&mut idx[range.clone()], &col);

    let mid = (range.start + range.end) / 2;
    build_subtree(nodes, idx, points, left, range.start..mid, metric);
    build_subtree(nodes, idx, points, left + 1, mid..range.end, metric);
}

/// Divides the node index array into two equal-sized parts.
///
/// # Panics
///
/// Panics if `col` is empty.
fn halve_node_indices<A>(idx: &mut [usize], col: &ArrayView1<A>)
where
    A: FloatCore,
{
    let (mut first, mut last) = (0, idx.len() - 1);
    let mid = idx.len() / 2;
    loop {
        let mut cur = first;
        for i in first..last {
            if col[idx[i]] < col[idx[last]] {
                idx.swap(i, cur);
                cur += 1;
            }
        }
        idx.swap(cur, last);
        if cur == mid {
            break;
        }
        if cur < mid {
            first = cur + 1;
        } else {
            last = cur - 1;
        }
    }
}

/// Finds the column with the maximum spread.
///
/// # Panics
///
/// Panics if either `matrix` or `idx` is empty, or any element of `idx` is
/// greater than or equal to the number of rows in `matrix`.
fn max_spread_column<A, S>(matrix: &ArrayBase<S, Ix2>, idx: &[usize]) -> usize
where
    A: FloatCore,
    S: Data<Elem = A>,
{
    let mut spread_iter = matrix
        .columns()
        .into_iter()
        .map(|col| {
            let (min, max) = idx
                .iter()
                .skip(1)
                .fold((col[idx[0]], col[idx[0]]), |(min, max), &i| {
                    (A::min(min, col[i]), A::max(max, col[i]))
                });
            max - min
        })
        .enumerate();
    let (_, max_spread) = spread_iter.next().expect("empty matrix");
    let (max_spread_col, _) = spread_iter.fold(
        (0, max_spread),
        |(max_spread_col, max_spread), (i, spread)| {
            if spread.partial_cmp(&max_spread) == Some(cmp::Ordering::Greater) {
                (i, spread)
            } else {
                (max_spread_col, max_spread)
            }
        },
    );
    max_spread_col
}

#[cfg(test)]
mod test {
    use ndarray::{arr1, array, aview1, aview2, Array, Axis};
    use ordered_float::FloatCore;

    use super::*;
    use crate::distance;

    #[test]
    #[should_panic]
    fn ball_tree_empty() {
        let data: [[f64; 0]; 0] = [];
        let tree = BallTree::euclidean(aview2(&data)).expect("`data` should not be empty");
        let point = aview1(&[0., 0.]);
        tree.query_nearest(&point);
    }

    #[test]
    #[should_panic]
    fn ball_tree_column_base() {
        let array = array![[1., 1.], [1., 1.1], [9., 9.]];
        let fortran = array.reversed_axes();
        let _ = BallTree::euclidean(fortran).expect("`array` should not be empty");
    }

    #[test]
    fn ball_tree_metric() {
        let array = array![[1., 1.], [1., 1.1], [9., 9.]];
        let tree = BallTree::new(array.clone(), Euclidean::default())
            .expect("`array` should not be empty");
        let tree1 = BallTree::euclidean(array).expect("`array` should not be empty");
        assert_eq!(tree.metric, tree1.metric);
    }

    #[test]
    fn ball_tree_3() {
        let array = array![[1., 1.], [1., 1.1], [9., 9.]];
        let tree = BallTree::euclidean(array).expect("`array` should not be empty");

        let point = aview1(&[0., 0.]);
        let neighbor = tree.query_nearest(&point);
        assert_eq!(neighbor.0, 0);
        assert!(approx::abs_diff_eq!(neighbor.1, 2_f64.sqrt()));
        let (indices, distances) = tree.query(&point, 0);
        assert!(indices.is_empty());
        assert!(distances.is_empty());
        let (indices, distances) = tree.query(&point, 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(distances.len(), 1);
        assert_eq!(indices[0], neighbor.0);
        assert!(approx::abs_diff_eq!(distances[0], neighbor.1));
        let mut neighbors = tree.query_radius(&point, 2.);
        neighbors.sort_unstable();
        assert_eq!(neighbors, &[0, 1]);

        let neighbors = tree.nearest_neighbor_in_subtree(&aview1(&[20., 20.]), 0, 1.);
        assert_eq!(neighbors, None);

        let neighbors = tree.query_radius(&aview1(&[20., 20.]), 1.);
        assert_eq!(neighbors, &[]);

        let point = aview1(&[1.1, 1.2]);
        let neighbor = tree.query_nearest(&point);
        assert_eq!(neighbor.0, 1);
        assert!(approx::abs_diff_eq!(
            neighbor.1,
            (2f64 * 0.1_f64 * 0.1_f64).sqrt()
        ));
        let (indices, distances) = tree.query(&point, 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(distances.len(), 1);
        assert_eq!(indices[0], neighbor.0);
        assert!(approx::abs_diff_eq!(distances[0], neighbor.1));

        let point = aview1(&[7., 7.]);
        let neighbor = tree.query_nearest(&point);
        assert_eq!(neighbor.0, 2);
        assert!(approx::abs_diff_eq!(neighbor.1, 8_f64.sqrt()));
        let (indices, distances) = tree.query(&point, 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(distances.len(), 1);
        assert_eq!(indices[0], neighbor.0);
        assert!(approx::abs_diff_eq!(distances[0], neighbor.1));
    }

    #[test]
    fn ball_tree_6() {
        let array = array![
            [1.0, 2.0],
            [1.1, 2.2],
            [0.9, 1.9],
            [1.0, 2.1],
            [-2.0, 3.0],
            [-2.2, 3.1],
        ];
        let tree = BallTree::euclidean(array).expect("`array` should not be empty");

        let point = aview1(&[1., 2.]);
        let neighbor = tree.query_nearest(&point);
        assert_eq!(neighbor.0, 0);
        assert!(approx::abs_diff_eq!(neighbor.1, 0_f64.sqrt()));
    }

    #[test]
    fn ball_tree_identical_points() {
        let array = array![
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ];
        let tree = BallTree::new(array, distance::Euclidean::default())
            .expect("`array` should not be empty");

        let point = aview1(&[1., 2.]);
        let neighbor = tree.query_nearest(&point);
        assert!(approx::abs_diff_eq!(neighbor.1, 1_f64.sqrt()));

        let point = aview1(&[1., 1.]);
        let neighbor = tree.query_nearest(&point);
        assert!(approx::abs_diff_eq!(neighbor.1, 0_f64.sqrt()));
    }

    #[test]
    fn ball_tree_query() {
        use ndarray_rand::rand_distr::Uniform;
        use ndarray_rand::RandomExt;
        const DIMENSION: usize = 3;

        let array = Array::random((40, DIMENSION), Uniform::new(0., 1.));
        let bt = BallTree::euclidean(array.view()).expect("`array` should not be empty");
        let euclidean = distance::Euclidean::default();
        for _ in 0..10 {
            let query = Array::random(DIMENSION, Uniform::new(0., 1.));
            let (_, bt_distances) = bt.query(&query, 5);
            let naive_neighbors = naive_k_nearest_neighbors(&array, &query.view(), 5, &euclidean);
            for (bt_dist, naive_neighbor) in bt_distances.iter().zip(naive_neighbors.iter()) {
                assert!(approx::abs_diff_eq!(
                    *bt_dist,
                    naive_neighbor.distance.into_inner()
                ));
            }
        }
    }

    #[test]
    fn ball_tree_query_radius() {
        let array = array![[0.], [2.], [3.], [4.], [6.], [8.], [10.]];
        let bt = BallTree::new(array, distance::Euclidean::default())
            .expect("`array` should not be empty");

        let neighbors = bt.query_radius(&aview1(&[0.1]), 1.);
        assert_eq!(neighbors, &[0]);

        let mut neighbors = bt.query_radius(&aview1(&[3.2]), 1.);
        neighbors.sort_unstable();
        assert_eq!(neighbors, &[2, 3]);

        let neighbors = bt.query_radius(&aview1(&[9.]), 0.9);
        assert!(neighbors.is_empty());
    }

    #[test]
    fn node_init() {
        let array = array![[0., 1.], [0., 9.], [0., 2.]];
        let idx: [usize; 3] = [0, 1, 2];
        let mut node = Node::default();
        let metric = distance::Euclidean::default();
        node.init(&array.view().into(), &idx, &metric);
        assert_eq!(node.centroid, arr1(&[0., 4.]));
        assert_eq!(node.radius, 5.);

        let idx: [usize; 2] = [0, 2];
        node.init(&array.into(), &idx, &metric);
        assert_eq!(node.centroid, arr1(&[0., 1.5]));
    }

    #[test]
    #[should_panic]
    fn halve_node_indices_empty() {
        let col: [f64; 0] = [];
        let mut idx: [usize; 0] = [];
        halve_node_indices(&mut idx, &aview1(&col));
    }

    #[test]
    fn halve_node_indices_one() {
        let col = [1.];
        let mut idx = [0];
        halve_node_indices(&mut idx, &aview1(&col));
        assert_eq!(idx, [0]);
    }

    #[test]
    fn halve_node_indices_odd() {
        let col = [1., 2., 3., 4., 5.];
        let mut idx = [0, 1, 4, 3, 2];
        halve_node_indices(&mut idx, &aview1(&col));
        assert!(idx[0] < idx[2]);
        assert!(idx[1] < idx[2]);
        assert!(idx[2] <= idx[3]);
        assert!(idx[2] <= idx[4]);
    }

    #[test]
    fn halve_node_indices_even() {
        let col = [1., 2., 3., 4.];
        let mut idx = [3, 2, 1, 0];
        halve_node_indices(&mut idx, &aview1(&col));
        assert!(idx[0] < idx[2]);
        assert!(idx[1] < idx[2]);
        assert!(idx[2] <= idx[3]);
    }

    #[test]
    #[should_panic]
    fn max_spread_column_empty_idx() {
        let data = [[0., 1.], [0., 9.], [0., 2.]];
        let idx: [usize; 0] = [];
        super::max_spread_column(&aview2(&data), &idx);
    }

    #[test]
    #[should_panic]
    fn max_spread_column_idx_out_of_bound() {
        let data = [[0., 1.], [0., 9.], [0., 2.]];
        let idx: [usize; 3] = [0, 4, 2];
        super::max_spread_column(&aview2(&data), &idx);
    }

    #[test]
    #[should_panic]
    fn max_spread_column_empty_matrix() {
        let data: [[f64; 0]; 0] = [];
        let idx: [usize; 3] = [0, 1, 2];
        super::max_spread_column(&aview2(&data), &idx);
    }

    #[test]
    fn max_spread_column() {
        let data = [[0., 1.], [0., 9.], [0., 2.]];
        let idx: [usize; 3] = [0, 1, 2];
        assert_eq!(super::max_spread_column(&aview2(&data), &idx), 1);
    }

    /// Finds k nearest neighbors by compuing distance to every point.
    ///
    /// # Panics
    ///
    /// Panics if any row in `neighbors` is not contiguous in memory.
    fn naive_k_nearest_neighbors<A, S, M>(
        neighbors: &ArrayBase<S, Ix2>,
        point: &ArrayView1<A>,
        k: usize,
        metric: &M,
    ) -> Vec<Neighbor<A>>
    where
        A: FloatCore,
        S: Data<Elem = A>,
        M: Metric<A>,
    {
        let mut knn = neighbors
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(i, n)| Neighbor {
                idx: i,
                distance: metric.distance(&n, point).into(),
            })
            .collect::<Vec<Neighbor<A>>>();
        knn.sort();
        knn[0..k].to_vec()
    }
}
