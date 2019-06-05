use ndarray::{ArrayView1, ArrayView2};
use std::cmp;
use std::mem::size_of;
use std::ops::Range;

/// A data structure for neighbor search in a multi-dimensional space.
#[derive(Debug)]
pub struct BallTree<'a> {
    points: &'a ArrayView2<'a, f64>,
    idx: Vec<usize>,
    nodes: Vec<Node>,
}

impl<'a> BallTree<'a> {
    /// Builds a ball tree containing the given points.
    ///
    /// # Panics
    ///
    /// Panics if `points` is empty.
    pub fn new(points: &'a ArrayView2<'a, f64>) -> Self {
        let n_points: usize = *points.shape().first().unwrap();
        let height = (size_of::<usize>() * 8) as u32 - n_points.leading_zeros();
        let size = 1usize.wrapping_shl(height) - 1;

        let mut idx: Vec<usize> = (0..n_points).collect();
        let mut nodes = vec![Node::default(); size];
        build_subtree(&mut nodes, &mut idx, points, 0, 0..n_points);
        BallTree { points, idx, nodes }
    }

    /// Finds the nearest neighbor in the tree.
    pub fn nearest_neighbor(&self, point: &[f64]) -> usize {
        let (i, _) = self
            .nearest_neighbor_in_subtree(point, 0, std::f64::INFINITY)
            .unwrap();
        i
    }

    /// Finds the nearest neighbor within the radius in the subtree rooted at `root`.
    ///
    /// # Panics
    ///
    /// Panics if `root` is out of bound.
    fn nearest_neighbor_in_subtree(
        &self,
        point: &[f64],
        root: usize,
        radius_squared: f64,
    ) -> Option<(usize, f64)> {
        let root_node = &self.nodes[root];
        let lower_bound = self.nodes[root].distance_lower_bound(point);
        if lower_bound * lower_bound > radius_squared {
            return None;
        }

        if root_node.is_leaf {
            let (min_i, min_dist) = self.idx[root_node.range.clone()].iter().fold(
                (0, std::f64::INFINITY),
                |(min_i, min_dist), &i| {
                    let dist_squared =
                        point
                            .iter()
                            .zip(self.points.row(i))
                            .fold(0., |sum, (a, b)| {
                                let diff = a - b;
                                sum + diff * diff
                            });
                    if dist_squared < min_dist {
                        (i, dist_squared)
                    } else {
                        (min_i, min_dist)
                    }
                },
            );
            if min_dist <= radius_squared {
                Some((min_i, min_dist.sqrt()))
            } else {
                None
            }
        } else {
            let child1 = root * 2 + 1;
            let child2 = child1 + 1;
            let lb1 = self.nodes[child1].distance_lower_bound(point);
            let lb2 = self.nodes[child2].distance_lower_bound(point);
            let (child1, child2) = if lb1 < lb2 {
                (child1, child2)
            } else {
                (child2, child1)
            };
            match self.nearest_neighbor_in_subtree(point, child1, radius_squared) {
                Some((i1, dist_squared1)) => {
                    if let Some((i2, dist_squared2)) =
                        self.nearest_neighbor_in_subtree(point, child2, dist_squared1)
                    {
                        Some((i2, dist_squared2))
                    } else {
                        Some((i1, dist_squared1))
                    }
                }
                None => self.nearest_neighbor_in_subtree(point, child2, radius_squared),
            }
        }
    }
}

/// A node containing a range of points in a ball tree.
#[derive(Clone, Debug)]
struct Node {
    range: Range<usize>,
    centroid: Vec<f64>,
    radius_squared: f64,
    is_leaf: bool,
}

impl Node {
    /// Computes the centroid of the node.
    fn init(&mut self, points: &ArrayView2<f64>, idx: &[usize]) {
        let mut centroid = vec![0.; points.shape()[1]];
        for &i in idx {
            for (c, col) in centroid.iter_mut().zip(points.gencolumns()) {
                *c += col[i];
            }
        }
        for c in centroid.iter_mut() {
            *c /= idx.len() as f64;
        }

        let radius_squared = idx.iter().fold(0., |max, &i| {
            let dist_squared = centroid.iter().zip(points.row(i)).fold(0., |sum, (c, p)| {
                let diff = c - p;
                sum + diff * diff
            });
            if dist_squared > max {
                dist_squared
            } else {
                max
            }
        });

        self.centroid = centroid;
        self.radius_squared = radius_squared;
    }

    fn distance_lower_bound(&self, point: &[f64]) -> f64 {
        let centroid_dist = self
            .centroid
            .iter()
            .zip(point.iter())
            .fold(0., |sum, (c, p)| {
                let diff = c - p;
                sum + diff * diff
            })
            .sqrt();
        centroid_dist - self.radius_squared.sqrt()
    }
}

impl Default for Node {
    fn default() -> Self {
        Node {
            range: (0..0),
            centroid: Vec::new(),
            radius_squared: 0.,
            is_leaf: false,
        }
    }
}

/// Builds a subtree recursively.
///
/// # Panics
///
/// Panics if `root` is out of range.
fn build_subtree(
    nodes: &mut [Node],
    idx: &mut [usize],
    points: &ArrayView2<f64>,
    root: usize,
    range: Range<usize>,
) {
    nodes[root].init(points, &idx[range.clone()]);

    let n_nodes = nodes.len();
    if let Some(node) = nodes.get_mut(root) {
        node.range = range.clone();
        if root * 2 + 1 >= n_nodes {
            // A leaf node.
            return;
        }
        node.is_leaf = true;
    } else {
        panic!("node index out of range");
    }

    #[allow(clippy::deref_addrof)]
    let col_idx = max_spread_column(points, &idx[range.clone()]) + range.start;
    debug_assert!(col_idx < points.cols());
    let col = points.column(col_idx);
    halve_node_indices(idx, &col);
    build_subtree(
        nodes,
        idx,
        points,
        root * 2 + 1,
        range.start..(range.start + points.len() / 2),
    );
    build_subtree(
        nodes,
        idx,
        points,
        root * 2 + 2,
        (range.start + points.len() / 2)..range.end,
    );
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
            if spread.partial_cmp(&max_spread).unwrap() == cmp::Ordering::Greater {
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
    use ndarray::{aview1, aview2};

    #[test]
    #[should_panic]
    fn ball_tree_empty() {
        let data: [[f64; 0]; 0] = [];
        let _tree = BallTree::new(&aview2(&data));
    }

    #[test]
    fn ball_tree() {
        let data = [[1., 1.], [1., 1.1], [9., 9.]];
        let view = aview2(&data);
        let tree = BallTree::new(&view);

        let point = [0., 0.];
        assert_eq!(tree.nearest_neighbor(&point), 0);

        let point = [1.1, 1.2];
        assert_eq!(tree.nearest_neighbor(&point), 1);

        let point = [7., 7.];
        assert_eq!(tree.nearest_neighbor(&point), 2);
    }

    #[test]
    fn node_init() {
        let data = [[0., 1.], [0., 9.], [0., 2.]];
        let idx: [usize; 3] = [0, 1, 2];
        let mut node = Node::default();
        node.init(&aview2(&data), &idx);
        assert_eq!(node.centroid, [0., 4.]);
        assert_eq!(node.radius_squared, 25.);

        let idx: [usize; 2] = [0, 2];
        node.init(&aview2(&data), &idx);
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
}
