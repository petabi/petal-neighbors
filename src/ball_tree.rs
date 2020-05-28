use crate::distance::Metric;
use ndarray::{ArrayBase, ArrayView1, CowArray, Data, Ix2};
use std::cmp;
use std::collections::BinaryHeap;
use std::convert::TryFrom;
use std::error;
use std::fmt;
use std::mem::size_of;
use std::ops::Range;

/// A data structure for nearest neighbor search in a multi-dimensional space.
#[derive(Debug)]
pub struct BallTree<'a, M>
where
    M: Metric,
{
    points: CowArray<'a, f64, Ix2>,
    idx: Vec<usize>,
    nodes: Vec<Node>,
    metric: M,
}

impl<'a, M> BallTree<'a, M>
where
    M: Metric,
{
    /// Builds a ball tree using the given distance metric.
    ///
    /// # Errors
    ///
    /// Returns an error if `points` is an empty array.
    pub fn new<T>(points: T, metric: M) -> Result<Self, EmptyArrayError>
    where
        T: Into<CowArray<'a, f64, Ix2>>,
    {
        let points = points.into();
        let n_points: usize = *points
            .shape()
            .first()
            .expect("ArrayView2 should have two dimensions");
        if n_points == 0 {
            return Err(EmptyArrayError {});
        }

        let height = u32::try_from(size_of::<usize>() * 8).expect("smaller than u32::max_value()")
            - n_points.leading_zeros();
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
    /// use ndarray::array;
    /// use petal_neighbors::{BallTree, distance};
    ///
    /// let points = array![[1., 1.], [1., 2.], [9., 9.]];
    /// let tree = BallTree::new(points, distance::EUCLIDEAN).expect("non-empty input");
    /// let (index, distance) = tree.query_nearest(&[8., 8.]);
    /// assert_eq!(index, 2);  // points[2] is the nearest.
    /// assert!((2_f64.sqrt() - distance).abs() < 1e-8);
    /// ```
    pub fn query_nearest<'p, P>(&self, point: P) -> (usize, f64)
    where
        P: Copy + IntoIterator<Item = &'p f64>,
    {
        self.nearest_neighbor_in_subtree(point, 0, std::f64::INFINITY)
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
    /// let tree = BallTree::new(points, distance::EUCLIDEAN).expect("non-empty input");
    /// let (indices, distances) = tree.query(&[3., 3.], 2);
    /// assert_eq!(indices, &[1, 0]);  // points[1] is the nearest, followed by points[0].
    /// ```
    pub fn query<'p, P>(&self, point: P, k: usize) -> (Vec<usize>, Vec<f64>)
    where
        P: Copy + IntoIterator<Item = &'p f64>,
    {
        let mut neighbors = BinaryHeap::with_capacity(k);
        self.nearest_k_neighbors_in_subtree(point, 0, std::f64::INFINITY, k, &mut neighbors);
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
    /// let tree = BallTree::new(points, distance::EUCLIDEAN).expect("non-empty input");
    /// let indices = tree.query_radius(&[3., 0.], 1.5);
    /// assert_eq!(indices, &[1]);  // The distance to points[1] is less than 1.5.
    /// ```
    pub fn query_radius<'p, P>(&self, point: P, distance: f64) -> Vec<usize>
    where
        P: Copy + IntoIterator<Item = &'p f64>,
    {
        self.neighbors_within_radius_in_subtree(point, distance, 0)
    }

    /// Finds the nearest neighbor and its distance in the subtree rooted at `root`.
    ///
    /// # Panics
    ///
    /// Panics if `root` is out of bound.
    fn nearest_neighbor_in_subtree<'p, P>(
        &self,
        point: P,
        root: usize,
        radius: f64,
    ) -> Option<(usize, f64)>
    where
        P: Copy + IntoIterator<Item = &'p f64>,
    {
        let root_node = &self.nodes[root];
        let lower_bound = self.nodes[root].distance_lower_bound(point, &self.metric);
        if lower_bound > radius {
            return None;
        }

        if root_node.is_leaf {
            let (min_i, min_dist) = self.idx[root_node.range.clone()].iter().fold(
                (0, std::f64::INFINITY),
                |(min_i, min_dist), &i| {
                    let dist = self.metric.distance(point, self.points.row(i));

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
    fn nearest_k_neighbors_in_subtree<'p, P>(
        &self,
        point: P,
        root: usize,
        radius: f64,
        k: usize,
        neighbors: &mut BinaryHeap<Neighbor>,
    ) where
        P: Copy + IntoIterator<Item = &'p f64>,
    {
        let root_node = &self.nodes[root];
        if root_node.distance_lower_bound(point, &self.metric) > radius {
            return;
        }

        if root_node.is_leaf {
            self.idx[root_node.range.clone()]
                .iter()
                .filter_map(|&i| {
                    let dist = self.metric.distance(point, self.points.row(i));

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
    fn neighbors_within_radius_in_subtree<'p, P>(
        &self,
        point: P,
        radius: f64,
        root: usize,
    ) -> Vec<usize>
    where
        P: Copy + IntoIterator<Item = &'p f64>,
    {
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
                neighbors.extend(self.idx[root_node.range.clone()].iter().cloned());
            } else if root_node.is_leaf {
                neighbors.extend(self.idx[root_node.range.clone()].iter().filter_map(|&i| {
                    let dist = self.metric.distance(point, self.points.row(i));
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

/// An error returned when an empty array is given as an argument to a function
/// requiring a non-empty array.
#[derive(Debug)]
pub struct EmptyArrayError;

impl fmt::Display for EmptyArrayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "input array is empty")
    }
}

impl error::Error for EmptyArrayError {}

#[derive(Clone, Debug)]
struct Neighbor {
    pub idx: usize,
    pub distance: f64,
}

impl Neighbor {
    #[must_use]
    pub fn new(idx: usize, distance: f64) -> Self {
        Self { idx, distance }
    }
}

impl Ord for Neighbor {
    #[must_use]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl PartialOrd for Neighbor {
    #[must_use]
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl PartialEq for Neighbor {
    #[must_use]
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Neighbor {}

/// A node containing a range of points in a ball tree.
#[derive(Clone, Debug)]
struct Node {
    range: Range<usize>,
    centroid: Vec<f64>,
    radius: f64,
    is_leaf: bool,
}

impl Node {
    /// Computes the centroid of the node.
    #[allow(clippy::cast_precision_loss)] // The precision provided by 54-bit-wide mantissa is
                                          // good enough in computing mean.
    fn init<M>(&mut self, points: &CowArray<f64, Ix2>, idx: &[usize], metric: &M)
    where
        M: Metric,
    {
        let mut sum = idx.iter().fold(vec![0_f64; points.ncols()], |mut sum, &i| {
            for (s, v) in sum.iter_mut().zip(points.row(i)) {
                *s += *v;
            }
            sum
        });
        sum.iter_mut().for_each(|v| *v /= idx.len() as f64);
        self.centroid = sum;

        self.radius = idx.iter().fold(0., |max, &i| {
            f64::max(
                metric.distance(
                    &self.centroid,
                    points.row(i).to_slice().expect("should be contiguous"),
                ),
                max,
            )
        });
    }

    fn distance_bounds<'p, P, M>(&self, point: P, metric: &M) -> (f64, f64)
    where
        P: IntoIterator<Item = &'p f64>,
        M: Metric,
    {
        let centroid_dist = metric.distance(point, &self.centroid);
        let mut lb = centroid_dist - self.radius;
        if lb < 0. {
            lb = 0.;
        }
        let ub = centroid_dist + self.radius;
        (lb, ub)
    }

    fn distance_lower_bound<'p, P, M>(&self, point: P, metric: &M) -> f64
    where
        P: IntoIterator<Item = &'p f64>,
        M: Metric,
    {
        let centroid_dist = metric.distance(point, &self.centroid);
        let lb = centroid_dist - self.radius;
        if lb < 0. {
            0.
        } else {
            lb
        }
    }
}

impl Default for Node {
    fn default() -> Self {
        Self {
            range: (0..0),
            centroid: Vec::new(),
            radius: 0.,
            is_leaf: false,
        }
    }
}

/// Builds a subtree recursively.
///
/// # Panics
///
/// Panics if `root` or `range` is out of range.
fn build_subtree<M>(
    nodes: &mut [Node],
    idx: &mut [usize],
    points: &CowArray<f64, Ix2>,
    root: usize,
    range: Range<usize>,
    metric: &M,
) where
    M: Metric,
{
    let n_nodes = nodes.len();
    let mut root_node = nodes.get_mut(root).expect("root node index out of range");
    root_node.init(
        points,
        &idx.get(range.clone()).expect("invalid subtree range"),
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
fn halve_node_indices(idx: &mut [usize], col: &ArrayView1<f64>) {
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
fn max_spread_column<D>(matrix: &ArrayBase<D, Ix2>, idx: &[usize]) -> usize
where
    D: Data<Elem = f64>,
{
    let mut spread_iter = matrix
        .gencolumns()
        .into_iter()
        .map(|col| {
            let (min, max) = idx
                .iter()
                .skip(1)
                .fold((col[idx[0]], col[idx[0]]), |(min, max), &i| {
                    (f64::min(min, col[i]), f64::max(max, col[i]))
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
        let tree =
            BallTree::new(aview2(&data), distance::EUCLIDEAN).expect("`data` should not be empty");
        let point = [0., 0.];
        tree.query_nearest(&point);
    }

    #[test]
    fn ball_tree_3() {
        let array = array![[1., 1.], [1., 1.1], [9., 9.]];
        let tree = BallTree::new(array, distance::EUCLIDEAN).expect("`array` should not be empty");

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
        let tree = BallTree::new(array, distance::EUCLIDEAN).expect("`array` should not be empty");

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
        let tree = BallTree::new(array, distance::EUCLIDEAN).expect("`array` should not be empty");

        let point = [1., 2.];
        let neighbor = tree.query_nearest(&point);
        assert!(approx::abs_diff_eq!(neighbor.1, 1_f64.sqrt()));
    }

    #[test]
    fn ball_tree_query() {
        const DIMENSION: usize = 3;

        let array = Array::random((40, DIMENSION), Uniform::new(0., 1.));
        let bt =
            BallTree::new(array.view(), distance::EUCLIDEAN).expect("`array` should not be empty");
        for _ in 0..10 {
            let query = Array::random(DIMENSION, Uniform::new(0., 1.));
            let (_, bt_distances) = bt.query(query.as_slice().expect("should be contiguous"), 5);
            let naive_neighbors = naive_k_nearest_neighbors(
                &array,
                query.as_slice().expect("should be contiguous"),
                5,
                distance::EUCLIDEAN,
            );
            for (bt_dist, naive_neighbor) in bt_distances.iter().zip(naive_neighbors.iter()) {
                assert!(approx::abs_diff_eq!(*bt_dist, naive_neighbor.distance));
            }
        }
    }

    #[test]
    fn ball_tree_query_radius() {
        let array = array![[0.], [2.], [3.], [4.], [6.], [8.], [10.]];
        let bt = BallTree::new(array, distance::EUCLIDEAN).expect("`array` should not be empty");

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
        node.init(&array.view().into(), &idx, &distance::EUCLIDEAN);
        assert_eq!(node.centroid, [0., 4.]);
        assert_eq!(node.radius, 5.);

        let idx: [usize; 2] = [0, 2];
        node.init(&array.into(), &idx, &distance::EUCLIDEAN);
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

    fn naive_k_nearest_neighbors<'a, D, M>(
        neighbors: &'a ArrayBase<D, Ix2>,
        point: &[f64],
        k: usize,
        metric: M,
    ) -> Vec<Neighbor>
    where
        D: Data<Elem = f64>,
        M: Metric,
    {
        let mut knn = neighbors
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(i, n)| Neighbor {
                idx: i,
                distance: metric.distance(n.to_slice().unwrap(), point),
            })
            .collect::<Vec<Neighbor>>();
        knn.sort();
        knn[0..k].to_vec()
    }
}
