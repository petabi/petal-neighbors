use ndarray::{Array1, ArrayView1, ArrayView2};
use std::cmp;
use std::collections::BinaryHeap;
use std::convert::TryFrom;
use std::mem::size_of;
use std::ops::Range;

/// A data structure for neighbor search in a multi-dimensional space.
pub struct BallTree<'a> {
    points: ArrayView2<'a, f64>,
    idx: Vec<usize>,
    nodes: Vec<Node>,
    distance: fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
}

impl<'a> BallTree<'a> {
    /// Builds a ball tree containing the given points.
    ///
    /// # Panics
    ///
    /// Panics if `points` is empty.
    pub fn new(points: ArrayView2<'a, f64>) -> Self {
        BallTree::with_distance(points, euclidean_distance)
    }

    pub fn with_distance(
        points: ArrayView2<'a, f64>,
        distance: fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
    ) -> Self {
        let n_points: usize = *points
            .shape()
            .first()
            .expect("ArrayView2 should have two dimensions");
        assert!(
            n_points > 0,
            "A ball tree needs at least one point for initialization."
        );
        let height = u32::try_from(size_of::<usize>() * 8).expect("smaller than u32::max_value()")
            - n_points.leading_zeros();
        let size = 1_usize.wrapping_shl(height) - 1;

        let mut idx: Vec<usize> = (0..n_points).collect();
        let mut nodes = vec![Node::default(); size];
        build_subtree(&mut nodes, &mut idx, &points, 0, 0..n_points, distance);
        BallTree {
            points,
            idx,
            nodes,
            distance,
        }
    }

    /// Finds the nearest neighbor and its distance in the tree.
    ///
    /// # Panics
    ///
    /// Panics if the tree is empty.
    pub fn query_one(&self, point: &ArrayView1<f64>) -> Neighbor {
        self.nearest_neighbor_in_subtree(point, 0, std::f64::INFINITY)
            .unwrap()
    }

    pub fn query(&self, point: &ArrayView1<f64>, k: usize) -> Vec<Neighbor> {
        let mut neighbors = BinaryHeap::with_capacity(k);
        self.nearest_k_neighbors_in_subtree(point, 0, std::f64::INFINITY, k, &mut neighbors);
        neighbors.into_sorted_vec()
    }

    pub fn query_radius(&self, point: &ArrayView1<f64>, distance: f64) -> Vec<usize> {
        self.neighbors_within_radius_in_subtree(point, distance, 0)
    }

    /// Finds the nearest neighbor within the radius in the subtree rooted at `root`.
    ///
    /// # Panics
    ///
    /// Panics if `root` is out of bound.
    fn nearest_neighbor_in_subtree(
        &self,
        point: &ArrayView1<f64>,
        root: usize,
        radius: f64,
    ) -> Option<Neighbor> {
        let root_node = &self.nodes[root];
        let distance = self.distance;
        let lower_bound = self.nodes[root].distance_lower_bound(point, distance);
        if lower_bound > radius {
            return None;
        }

        if root_node.is_leaf {
            let point = ArrayView1::from(point);
            let (min_i, min_dist) = self.idx[root_node.range.clone()].iter().fold(
                (0, std::f64::INFINITY),
                |(min_i, min_dist), &i| {
                    let dist = distance(&point, &self.points.row(i));

                    if dist < min_dist {
                        (i, dist)
                    } else {
                        (min_i, min_dist)
                    }
                },
            );
            if min_dist <= radius {
                Some(Neighbor {
                    idx: min_i,
                    distance: min_dist,
                })
            } else {
                None
            }
        } else {
            let child1 = root * 2 + 1;
            let child2 = child1 + 1;
            let lb1 = self.nodes[child1].distance_lower_bound(point, distance);
            let lb2 = self.nodes[child2].distance_lower_bound(point, distance);
            let (child1, child2) = if lb1 < lb2 {
                (child1, child2)
            } else {
                (child2, child1)
            };
            match self.nearest_neighbor_in_subtree(point, child1, radius) {
                Some(neighbor1) => {
                    if let Some(neighbor2) =
                        self.nearest_neighbor_in_subtree(point, child2, neighbor1.distance)
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
        point: &ArrayView1<f64>,
        root: usize,
        radius: f64,
        k: usize,
        neighbors: &mut BinaryHeap<Neighbor>,
    ) {
        let distance = self.distance;
        let root_node = &self.nodes[root];
        if root_node.distance_lower_bound(point, distance) > radius {
            return;
        }

        if root_node.is_leaf {
            let point = ArrayView1::from(point);
            self.idx[root_node.range.clone()]
                .iter()
                .filter_map(|&i| {
                    let dist = distance(&point, &self.points.row(i));

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
            let lb1 = self.nodes[child1].distance_lower_bound(point, distance);
            let lb2 = self.nodes[child2].distance_lower_bound(point, distance);
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
        point: &ArrayView1<f64>,
        radius: f64,
        root: usize,
    ) -> Vec<usize> {
        let mut neighbors = Vec::new();
        let distance = self.distance;
        let mut subtrees_to_visit = vec![root];

        loop {
            let subroot = subtrees_to_visit.pop().expect("should not be empty");
            let root_node = &self.nodes[subroot];
            let (lb, ub) = root_node.distance_bounds(point, distance);

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
                let point = ArrayView1::from(point);
                neighbors.extend(self.idx[root_node.range.clone()].iter().filter_map(|&i| {
                    let dist = distance(&point, &self.points.row(i));
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

#[derive(Clone, Debug)]
pub struct Neighbor {
    pub idx: usize,
    pub distance: f64,
}

impl Neighbor {
    pub fn new(idx: usize, distance: f64) -> Self {
        Self { idx, distance }
    }

    pub fn approx_eq(&self, other: &Self) -> bool {
        self.idx == other.idx
            && self.distance - std::f64::EPSILON < other.distance
            && other.distance < self.distance + std::f64::EPSILON
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Neighbor {}

/// A node containing a range of points in a ball tree.
#[derive(Clone, Debug)]
struct Node {
    range: Range<usize>,
    centroid: Array1<f64>,
    radius: f64,
    is_leaf: bool,
}

impl Node {
    /// Computes the centroid of the node.
    fn init<D>(&mut self, points: &ArrayView2<f64>, idx: &[usize], distance: D)
    where
        D: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
    {
        self.centroid = idx
            .iter()
            .fold(Array1::<f64>::zeros(points.ncols()), |mut c, &i| {
                c += &points.row(i);
                c
            })
            / idx.len() as f64;

        self.radius = idx.iter().fold(0., |max, &i| {
            f64::max(distance(&self.centroid.view(), &points.row(i)), max)
        });
    }

    fn distance_bounds<D>(&self, point: &ArrayView1<f64>, distance: D) -> (f64, f64)
    where
        D: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
    {
        let centroid_dist = distance(&point, &self.centroid.view());
        let mut lb = centroid_dist - self.radius;
        if lb < 0. {
            lb = 0.;
        }
        let ub = centroid_dist + self.radius;
        (lb, ub)
    }

    fn distance_lower_bound<D>(&self, point: &ArrayView1<f64>, distance: D) -> f64
    where
        D: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
    {
        let centroid_dist = distance(&point, &self.centroid.view());
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
            centroid: Array1::from(vec![]),
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
fn build_subtree<D>(
    nodes: &mut [Node],
    idx: &mut [usize],
    points: &ArrayView2<f64>,
    root: usize,
    range: Range<usize>,
    distance: D,
) where
    D: Copy + Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
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
    build_subtree(nodes, idx, points, left, range.start..mid, distance);
    build_subtree(nodes, idx, points, left + 1, mid..range.end, distance);
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

/// Calculates the squared euclidean distance of two points.
fn euclidean_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    (a - b).mapv(|x| x.powi(2)).sum().sqrt()
}

/// Finds the column with the maximum spread.
///
/// # Panics
///
/// Panics if either `matrix` or `idx` is empty, or any element of `idx` is
/// greater than or equal to the number of rows in `matrix`.
fn max_spread_column(matrix: &ArrayView2<f64>, idx: &[usize]) -> usize {
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
    use ndarray::{aview1, aview2, ArrayView, Axis};
    use rand::prelude::*;

    #[test]
    #[should_panic]
    fn ball_tree_empty() {
        let data: [[f64; 0]; 0] = [];
        let _tree = BallTree::new(aview2(&data));
    }

    #[test]
    fn ball_tree() {
        let data = [[1., 1.], [1., 1.1], [9., 9.]];
        let view = aview2(&data);
        let tree = BallTree::new(view);

        let point = aview1(&[0., 0.]);
        let neighbor = tree.query_one(&point);
        assert!(neighbor.approx_eq(&Neighbor {
            idx: 0,
            distance: 2f64.sqrt()
        }));
        let neighbors = tree.query(&point, 1);
        assert_eq!(neighbors.len(), 1);
        assert!(neighbors[0].approx_eq(&neighbor));
        let mut neighbors = tree.query_radius(&point, 2.);
        neighbors.sort_unstable();
        assert_eq!(neighbors, &[0, 1]);

        let point = aview1(&[1.1, 1.2]);
        let neighbor = tree.query_one(&point);
        assert!(neighbor.approx_eq(&Neighbor {
            idx: 1,
            distance: (2f64 * 0.1f64 * 0.1f64).sqrt()
        }));
        let neighbors = tree.query(&point, 1);
        assert_eq!(neighbors.len(), 1);
        assert!(neighbors[0].approx_eq(&neighbor));

        let point = aview1(&[7., 7.]);
        let neighbor = tree.query_one(&point);
        assert!(neighbor.approx_eq(&Neighbor {
            idx: 2,
            distance: 8f64.sqrt()
        }));
        let neighbors = tree.query(&point, 1);
        assert_eq!(neighbors.len(), 1);
        assert!(neighbors[0].approx_eq(&neighbor));
    }

    #[test]
    fn ball_tree2() {
        let data = vec![
            [1.0, 2.0],
            [1.1, 2.2],
            [0.9, 1.9],
            [1.0, 2.1],
            [-2.0, 3.0],
            [-2.2, 3.1],
        ];
        let view = aview2(&data);
        let tree = BallTree::new(view);

        let point = aview1(&[1., 2.]);
        let neighbor = tree.query_one(&point);
        assert!(neighbor.approx_eq(&Neighbor {
            idx: 0,
            distance: 0f64,
        }));
    }

    #[test]
    fn ball_tree_identical_points() {
        let data = vec![
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ];
        let view = aview2(&data);
        let tree = BallTree::new(view);

        let point = aview1(&[1., 2.]);
        let neighbor = tree.query_one(&point);
        assert_eq!(neighbor.distance, 1f64);
    }

    #[test]
    fn ball_tree_query() {
        const DIMENSION: usize = 3;

        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..40 * DIMENSION).map(|_| rng.gen()).collect();
        let array = ArrayView::from_shape((40, DIMENSION), &data).unwrap();
        let bt = BallTree::new(array.clone());
        for _ in 0..10 {
            let query: Vec<f64> = (0..DIMENSION).map(|_| rng.gen()).collect();
            let bt_neighbors = bt.query(&aview1(&query), 5);
            let naive_neighbors =
                naive_k_nearest_neighbors(&array, &aview1(&query), 5, euclidean_distance);
            for (n_bt, n_naive) in bt_neighbors.iter().zip(naive_neighbors.iter()) {
                assert_eq!(n_bt.distance, n_naive.distance);
            }
        }
    }

    #[test]
    fn ball_tree_query_radius() {
        let data = vec![[0.], [2.], [3.], [4.], [6.], [8.], [10.]];
        let bt = BallTree::new(aview2(&data));

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
        let distance = euclidean_distance;
        let data = [[0., 1.], [0., 9.], [0., 2.]];
        let idx: [usize; 3] = [0, 1, 2];
        let mut node = Node::default();
        node.init(&aview2(&data), &idx, distance);
        assert_eq!(node.centroid, aview1(&[0., 4.]));
        assert_eq!(node.radius, 5.);

        let idx: [usize; 2] = [0, 2];
        node.init(&aview2(&data), &idx, distance);
        assert_eq!(node.centroid, aview1(&[0., 1.5]));
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

    fn naive_k_nearest_neighbors<'a>(
        neighbors: &ArrayView2<'a, f64>,
        point: &ArrayView1<f64>,
        k: usize,
        distance: fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
    ) -> Vec<Neighbor> {
        let mut knn = neighbors
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(i, n)| Neighbor {
                idx: i,
                distance: distance(&n, point),
            })
            .collect::<Vec<Neighbor>>();
        knn.sort();
        knn[0..k].to_vec()
    }
}
