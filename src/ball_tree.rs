use crate::distance::{self, Distance};
use crate::ArrayError;
use ndarray::{ArrayBase, ArrayView1, CowArray, Data, Ix2};
use num_traits::{Float, FromPrimitive, Zero};
use std::cmp;
use std::collections::BinaryHeap;
use std::convert::TryFrom;
use std::mem::size_of;
use std::ops::{AddAssign, DivAssign, Range};

/// A data structure for nearest neighbor search in a multi-dimensional space,
/// which is partitioned into a nested set of hyperspheres, or "balls".
pub struct BallTree<'a, A>
where
    A: Float,
{
    points: CowArray<'a, A, Ix2>,
    idx: Vec<usize>,
    nodes: Vec<Node<A>>,
    distance: Distance<A>,
    _reduced_distance: Distance<A>,
}

impl<'a, A> BallTree<'a, A>
where
    A: Float + Zero + AddAssign + DivAssign + FromPrimitive,
{
    /// Builds a ball tree using the given distance metric.
    ///
    /// # Errors
    ///
    /// * `ArrayError::Empty` if `points` is an empty array.
    /// * `ArrayError::NotContiguous` if any row in `points` is not
    ///   contiguous in memory.
    pub fn new<T>(
        points: T,
        distance: Distance<A>,
        reduced_distance: Option<Distance<A>>,
    ) -> Result<Self, ArrayError>
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

        let height = u32::try_from(size_of::<usize>() * 8).expect("smaller than u32::max_value()")
            - n_points.leading_zeros();
        let size = 1_usize.wrapping_shl(height) - 1;

        let mut idx: Vec<usize> = (0..n_points).collect();
        let mut nodes = vec![Node::default(); size];
        let reduced_distance = reduced_distance.unwrap_or(distance);
        build_subtree(
            &mut nodes,
            &mut idx,
            &points,
            0,
            0..n_points,
            distance,
            reduced_distance,
        );
        Ok(BallTree {
            points,
            idx,
            nodes,
            distance,
            _reduced_distance: reduced_distance,
        })
    }

    /// Builds a ball tree with a euclidean distance metric.
    ///
    /// # Errors
    ///
    /// * `ArrayError::Empty` if `points` is an empty array.
    /// * `ArrayError::NotContiguous` if any row in `points` is not
    ///   contiguous in memory.
    pub fn euclidean<T>(points: T) -> Result<Self, ArrayError>
    where
        T: Into<CowArray<'a, A, Ix2>>,
    {
        Self::new(
            points,
            distance::euclidean,
            Some(distance::euclidean_reduced),
        )
    }

    /// Finds the nearest neighbor and its distance in the tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::array;
    /// use petal_neighbors::{BallTree, distance};
    ///
    /// let points = array![[1., 1.], [1., 2.], [9., 9.]];
    /// let tree = BallTree::euclidean(points).expect("valid array");
    /// let (index, distance) = tree.query_nearest(&[8., 8.]);
    /// assert_eq!(index, 2);  // points[2] is the nearest.
    /// assert!((2_f64.sqrt() - distance).abs() < 1e-8);
    /// ```
    pub fn query_nearest(&self, point: &[A]) -> (usize, A) {
        self.nearest_neighbor_in_subtree(point, 0, A::infinity())
            .expect("0 is a valid index")
    }

    /// Finds the nearest `k` neighbors and their distances in the tree. The
    /// return values are sorted in the ascending order in distance.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::array;
    /// use petal_neighbors::{BallTree, distance};
    ///
    /// let points = array![[1., 1.], [1., 2.], [9., 9.]];
    /// let tree = BallTree::euclidean(points).expect("non-empty input");
    /// let (indices, distances) = tree.query(&[3., 3.], 2);
    /// assert_eq!(indices, &[1, 0]);  // points[1] is the nearest, followed by points[0].
    /// ```
    pub fn query(&self, point: &[A], k: usize) -> (Vec<usize>, Vec<A>) {
        let mut neighbors = BinaryHeap::with_capacity(k);
        self.nearest_k_neighbors_in_subtree(point, 0, A::infinity(), k, &mut neighbors);
        let sorted = neighbors.into_sorted_vec();
        let indices = sorted.iter().map(|v| v.idx).collect();
        let distances = sorted.iter().map(|v| v.distance).collect();
        (indices, distances)
    }

    /// Finds all neighbors whose distances from `point` are less than or equal
    /// to `distance`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::array;
    /// use petal_neighbors::{BallTree, distance};
    ///
    /// let points = array![[1., 0.], [2., 0.], [9., 0.]];
    /// let tree = BallTree::euclidean(points).expect("non-empty input");
    /// let indices = tree.query_radius(&[3., 0.], 1.5);
    /// assert_eq!(indices, &[1]);  // The distance to points[1] is less than 1.5.
    /// ```
    pub fn query_radius(&self, point: &[A], distance: A) -> Vec<usize> {
        self.neighbors_within_radius_in_subtree(point, distance, 0)
    }

    /// Finds the nearest neighbor and its distance in the subtree rooted at `root`.
    ///
    /// # Panics
    ///
    /// Panics if `root` is out of bound.
    fn nearest_neighbor_in_subtree(
        &self,
        point: &[A],
        root: usize,
        radius: A,
    ) -> Option<(usize, A)> {
        let root_node = &self.nodes[root];
        let lower_bound = self.nodes[root].distance_lower_bound(point, self.distance);
        if lower_bound > radius {
            return None;
        }

        if root_node.is_leaf {
            let (min_i, min_dist) = self.idx[root_node.range.clone()].iter().fold(
                (0, A::infinity()),
                |(min_i, min_dist), &i| {
                    let distance = self.distance;
                    let dist = distance(
                        point,
                        self.points.row(i).as_slice().expect("standard layout"),
                    );

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
            let lb1 = self.nodes[child1].distance_lower_bound(point, self.distance);
            let lb2 = self.nodes[child2].distance_lower_bound(point, self.distance);
            let (child1, child2) = if lb1 < lb2 {
                (child1, child2)
            } else {
                (child2, child1)
            };
            match self.nearest_neighbor_in_subtree(point, child1, radius) {
                Some(neighbor1) => {
                    if let Some(neighbor2) =
                        self.nearest_neighbor_in_subtree(point, child2, neighbor1.1)
                    {
                        Some(neighbor2)
                    } else {
                        Some(neighbor1)
                    }
                }
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
        point: &[A],
        root: usize,
        radius: A,
        k: usize,
        neighbors: &mut BinaryHeap<Neighbor<A>>,
    ) {
        let root_node = &self.nodes[root];
        if root_node.distance_lower_bound(point, self.distance) > radius {
            return;
        }

        if root_node.is_leaf {
            self.idx[root_node.range.clone()]
                .iter()
                .filter_map(|&i| {
                    let distance = self.distance;
                    let dist = distance(
                        point,
                        self.points.row(i).as_slice().expect("standard layout"),
                    );

                    if dist < radius {
                        Some(Neighbor::new(i, dist))
                    } else {
                        None
                    }
                })
                .fold(neighbors, |neighbors, n| {
                    if neighbors.len() < k {
                        neighbors.push(n);
                    } else if n < *neighbors.peek().unwrap() {
                        neighbors.pop();
                        neighbors.push(n);
                    }
                    neighbors
                });
        } else {
            let child1 = root * 2 + 1;
            let child2 = child1 + 1;
            let lb1 = self.nodes[child1].distance_lower_bound(point, self.distance);
            let lb2 = self.nodes[child2].distance_lower_bound(point, self.distance);
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
        point: &[A],
        radius: A,
        root: usize,
    ) -> Vec<usize> {
        let mut neighbors = Vec::new();
        let mut subtrees_to_visit = vec![root];

        loop {
            let subroot = subtrees_to_visit.pop().expect("should not be empty");
            let root_node = &self.nodes[subroot];
            let (lb, ub) = root_node.distance_bounds(point, self.distance);

            if lb > radius {
                if subtrees_to_visit.is_empty() {
                    break;
                }
                continue;
            }

            if ub <= radius {
                neighbors.reserve(root_node.range.end - root_node.range.start);
                neighbors.extend(self.idx[root_node.range.clone()].iter().cloned());
            } else if root_node.is_leaf {
                neighbors.extend(self.idx[root_node.range.clone()].iter().filter_map(|&i| {
                    let distance = self.distance;
                    let dist = distance(
                        point,
                        self.points.row(i).as_slice().expect("standard layout"),
                    );
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
}

/// An error returned when an array is not suitable to build a `BallTree`.
#[derive(Clone, Debug)]
struct Neighbor<A>
where
    A: Float,
{
    pub idx: usize,
    pub distance: A,
}

impl<A> Neighbor<A>
where
    A: Float,
{
    #[must_use]
    pub fn new(idx: usize, distance: A) -> Self {
        Self { idx, distance }
    }
}

impl<A> Ord for Neighbor<A>
where
    A: Float,
{
    #[must_use]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<A> PartialOrd for Neighbor<A>
where
    A: Float,
{
    #[must_use]
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl<A> PartialEq for Neighbor<A>
where
    A: Float,
{
    #[must_use]
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl<A> Eq for Neighbor<A> where A: Float {}

/// A node containing a range of points in a ball tree.
#[derive(Clone, Debug)]
struct Node<A> {
    range: Range<usize>,
    centroid: Vec<A>,
    radius: A,
    is_leaf: bool,
}

impl<A> Node<A>
where
    A: Float + Zero + AddAssign + DivAssign + FromPrimitive,
{
    /// Computes the centroid of the node.
    ///
    /// # Panics
    ///
    /// Panics if any row in `points` is not contiguous in memory.
    #[allow(clippy::cast_precision_loss)] // The precision provided by 54-bit-wide mantissa is
                                          // good enough in computing mean.
    fn init(&mut self, points: &CowArray<A, Ix2>, idx: &[usize], distance: Distance<A>) {
        let mut sum = idx
            .iter()
            .fold(vec![A::zero(); points.ncols()], |mut sum, &i| {
                for (s, v) in sum.iter_mut().zip(points.row(i)) {
                    *s += *v;
                }
                sum
            });
        let len = A::from_usize(idx.len()).expect("approximation");
        sum.iter_mut().for_each(|v| *v /= len);
        self.centroid = sum;

        self.radius = idx.iter().fold(A::zero(), |max, &i| {
            A::max(
                distance(&self.centroid, &points.row(i).as_slice().unwrap()),
                max,
            )
        });
    }

    fn distance_bounds(&self, point: &[A], distance: Distance<A>) -> (A, A) {
        let centroid_dist = distance(point, &self.centroid);
        let mut lb = centroid_dist - self.radius;
        if lb < A::zero() {
            lb = A::zero();
        }
        let ub = centroid_dist + self.radius;
        (lb, ub)
    }

    fn distance_lower_bound(&self, point: &[A], distance: Distance<A>) -> A {
        let centroid_dist = distance(point, &self.centroid);
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
    A: Float + Zero,
{
    #[allow(clippy::reversed_empty_ranges)] // An empty range is valid because `centroid` is empty.
    fn default() -> Self {
        Self {
            range: (0..0),
            centroid: Vec::new(),
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
fn build_subtree<A>(
    nodes: &mut [Node<A>],
    idx: &mut [usize],
    points: &CowArray<A, Ix2>,
    root: usize,
    range: Range<usize>,
    distance: Distance<A>,
    reduced_distance: Distance<A>,
) where
    A: Float + AddAssign + DivAssign + FromPrimitive,
{
    let n_nodes = nodes.len();
    let mut root_node = nodes.get_mut(root).expect("root node index out of range");
    root_node.init(
        points,
        &idx.get(range.clone()).expect("invalid subtree range"),
        distance,
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
    build_subtree(
        nodes,
        idx,
        points,
        left,
        range.start..mid,
        distance,
        reduced_distance,
    );
    build_subtree(
        nodes,
        idx,
        points,
        left + 1,
        mid..range.end,
        distance,
        reduced_distance,
    );
}

/// Divides the node index array into two equal-sized parts.
///
/// # Panics
///
/// Panics if `col` is empty.
fn halve_node_indices<A>(idx: &mut [usize], col: &ArrayView1<A>)
where
    A: Float,
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
    A: Float,
    S: Data<Elem = A>,
{
    let mut spread_iter = matrix
        .gencolumns()
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
            if spread
                .partial_cmp(&max_spread)
                .map_or(false, |o| o == cmp::Ordering::Greater)
            {
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
    use super::*;
    use crate::distance;
    use approx;
    use ndarray::{array, aview1, aview2, Array, Axis};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    #[should_panic]
    fn ball_tree_empty() {
        let data: [[f64; 0]; 0] = [];
        let tree = BallTree::new(
            aview2(&data),
            distance::euclidean,
            Some(distance::euclidean_reduced),
        )
        .expect("`data` should not be empty");
        let point = [0., 0.];
        tree.query_nearest(&point);
    }

    #[test]
    fn ball_tree_3() {
        let array = array![[1., 1.], [1., 1.1], [9., 9.]];
        let tree = BallTree::new(
            array,
            distance::euclidean,
            Some(distance::euclidean_reduced),
        )
        .expect("`array` should not be empty");

        let point = [0., 0.];
        let neighbor = tree.query_nearest(&point);
        assert_eq!(neighbor.0, 0);
        assert!(approx::abs_diff_eq!(neighbor.1, 2_f64.sqrt()));
        let (indices, distances) = tree.query(&point, 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(distances.len(), 1);
        assert_eq!(indices[0], neighbor.0);
        assert!(approx::abs_diff_eq!(distances[0], neighbor.1));
        let mut neighbors = tree.query_radius(&point, 2.);
        neighbors.sort_unstable();
        assert_eq!(neighbors, &[0, 1]);

        let point = [1.1, 1.2];
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

        let point = [7., 7.];
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
        let tree = BallTree::new(
            array,
            distance::euclidean,
            Some(distance::euclidean_reduced),
        )
        .expect("`array` should not be empty");

        let point = [1., 2.];
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
        let tree = BallTree::new(
            array,
            distance::euclidean,
            Some(distance::euclidean_reduced),
        )
        .expect("`array` should not be empty");

        let point = [1., 2.];
        let neighbor = tree.query_nearest(&point);
        assert!(approx::abs_diff_eq!(neighbor.1, 1_f64.sqrt()));
    }

    #[test]
    fn ball_tree_query() {
        const DIMENSION: usize = 3;

        let array = Array::random((40, DIMENSION), Uniform::new(0., 1.));
        let bt = BallTree::new(array.view(), distance::euclidean, None)
            .expect("`array` should not be empty");
        for _ in 0..10 {
            let query = Array::random(DIMENSION, Uniform::new(0., 1.));
            let (_, bt_distances) =
                bt.query(query.as_slice().expect("should be contiguous in memory"), 5);
            let naive_neighbors = naive_k_nearest_neighbors(
                &array,
                query.as_slice().expect("should be contiguous in memory"),
                5,
                distance::euclidean,
            );
            for (bt_dist, naive_neighbor) in bt_distances.iter().zip(naive_neighbors.iter()) {
                assert!(approx::abs_diff_eq!(*bt_dist, naive_neighbor.distance));
            }
        }
    }

    #[test]
    fn ball_tree_query_radius() {
        let array = array![[0.], [2.], [3.], [4.], [6.], [8.], [10.]];
        let bt = BallTree::new(
            array,
            distance::euclidean,
            Some(distance::euclidean_reduced),
        )
        .expect("`array` should not be empty");

        let neighbors = bt.query_radius(&[0.1], 1.);
        assert_eq!(neighbors, &[0]);

        let mut neighbors = bt.query_radius(&[3.2], 1.);
        neighbors.sort_unstable();
        assert_eq!(neighbors, &[2, 3]);

        let neighbors = bt.query_radius(&[9.], 0.9);
        assert!(neighbors.is_empty());
    }

    #[test]
    fn node_init() {
        let array = array![[0., 1.], [0., 9.], [0., 2.]];
        let idx: [usize; 3] = [0, 1, 2];
        let mut node = Node::default();
        node.init(&array.view().into(), &idx, distance::euclidean);
        assert_eq!(node.centroid, [0., 4.]);
        assert_eq!(node.radius, 5.);

        let idx: [usize; 2] = [0, 2];
        node.init(&array.into(), &idx, distance::euclidean);
        assert_eq!(node.centroid, [0., 1.5]);
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
    fn naive_k_nearest_neighbors<'a, A, S>(
        neighbors: &'a ArrayBase<S, Ix2>,
        point: &[A],
        k: usize,
        distance: Distance<A>,
    ) -> Vec<Neighbor<A>>
    where
        A: Float,
        S: Data<Elem = A>,
    {
        let mut knn = neighbors
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(i, n)| Neighbor {
                idx: i,
                distance: distance(n.to_slice().unwrap(), point),
            })
            .collect::<Vec<Neighbor<A>>>();
        knn.sort();
        knn[0..k].to_vec()
    }
}
