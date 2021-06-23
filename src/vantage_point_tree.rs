use crate::distance::{self, Metric};
use crate::ArrayError;
use ndarray::{ArrayBase, ArrayView1, CowArray, Data, Ix1, Ix2};
use num_traits::{Float, Zero};
use ordered_float::OrderedFloat;
use std::ops::AddAssign;

/// A data structure for nearest neighbor search in a multi-dimensional space,
/// which is partitioned into two parts for each vantage point: those points
/// closer to the vantage point than a threshold, and those farther.
pub struct VantagePointTree<'a, A, M>
where
    A: Float,
    M: Metric<A>,
{
    points: CowArray<'a, A, Ix2>,
    nodes: Vec<Node<A>>,
    root: usize,
    metric: M,
}

impl<'a, A> VantagePointTree<'a, A, distance::Euclidean>
where
    A: Float + Zero + AddAssign + 'a,
{
    /// Builds a vantage point tree with a euclidean distance metric.
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
        Self::new(points, distance::Euclidean::default())
    }
}

impl<'a, A, M> VantagePointTree<'a, A, M>
where
    A: Float + Zero + AddAssign + 'a,
    M: Metric<A>,
{
    /// Builds a vantage point tree using the given distance metric.
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

        let mut nodes = Vec::with_capacity(n_points);
        let root = Self::create_root(&points, &metric, &mut nodes);
        Ok(VantagePointTree {
            points,
            nodes,
            root,
            metric,
        })
    }

    /// Finds the nearest neighbor and its distance in the tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::{array, aview1};
    /// use petal_neighbors::VantagePointTree;
    ///
    /// let points = array![[1., 1.], [1., 2.], [9., 9.]];
    /// let tree = VantagePointTree::euclidean(points).expect("valid array");
    /// let (index, distance) = tree.query_nearest(&aview1(&[8., 8.]));
    /// assert_eq!(index, 2);  // points[2] is the nearest.
    /// assert!((2_f64.sqrt() - distance).abs() < 1e-8);
    /// ```
    pub fn query_nearest<S>(&self, needle: &ArrayBase<S, Ix1>) -> (usize, A)
    where
        S: Data<Elem = A>,
    {
        let mut nearest = DistanceIndex {
            distance: A::max_value().into(),
            id: NULL,
        };
        self.search_node(&self.nodes[self.root], &needle.view(), &mut nearest);
        (nearest.id, nearest.distance.into_inner())
    }

    fn search_node(&self, node: &Node<A>, needle: &ArrayView1<A>, nearest: &mut DistanceIndex<A>) {
        let distance = self
            .metric
            .distance(&self.points.row(node.vantage_point), needle)
            .into();

        if distance < nearest.distance {
            nearest.distance = distance;
            nearest.id = node.vantage_point;
        }

        if distance < node.radius.into() {
            if let Some(near) = self.nodes.get(node.near) {
                self.search_node(near, needle, nearest);
            }
            if let Some(far) = self.nodes.get(node.far) {
                if distance + nearest.distance > node.radius.into() {
                    self.search_node(far, needle, nearest);
                }
            }
        } else {
            if let Some(far) = self.nodes.get(node.far) {
                self.search_node(far, needle, nearest);
            }
            if let Some(near) = self.nodes.get(node.near) {
                if distance - nearest.distance < node.radius.into() {
                    self.search_node(near, needle, nearest);
                }
            }
        }
    }

    fn create_root<S>(points: &ArrayBase<S, Ix2>, metric: &M, nodes: &mut Vec<Node<A>>) -> usize
    where
        S: Data<Elem = A>,
        M: Metric<A>,
    {
        let mut indexes: Vec<_> = (0..points.nrows())
            .map(|i| DistanceIndex {
                distance: A::max_value().into(),
                id: i,
            })
            .collect();
        Self::create_node(points, metric, &mut indexes, nodes)
    }

    fn create_node<S>(
        points: &ArrayBase<S, Ix2>,
        metric: &M,
        indexes: &mut [DistanceIndex<A>],
        nodes: &mut Vec<Node<A>>,
    ) -> usize
    where
        S: Data<Elem = A>,
    {
        if indexes.is_empty() {
            return NULL;
        }
        if indexes.len() == 1 {
            let id = nodes.len();
            nodes.push(Node {
                near: NULL,
                far: NULL,
                vantage_point: indexes[0].id,
                radius: A::max_value(),
            });
            return id;
        }

        let vp_pos = indexes.len() - 1;
        let vantage_point = indexes[vp_pos].id;
        let rest = &mut indexes[..vp_pos];

        for r in rest.iter_mut() {
            r.distance = metric
                .distance(&points.row(r.id), &points.row(vantage_point))
                .into();
        }
        rest.sort_unstable_by(|a, b| a.distance.cmp(&b.distance));

        let half = rest.len() / 2;
        let (near, far) = rest.split_at_mut(half);
        let radius = far[0].distance;

        let id = nodes.len();
        nodes.push(Node {
            far: NULL,
            near: NULL,
            vantage_point,
            radius: radius.into_inner(),
        });

        let near = Self::create_node(points, metric, near, nodes);
        let far = Self::create_node(points, metric, far, nodes);
        nodes[id].near = near;
        nodes[id].far = far;
        id
    }
}

struct Node<A> {
    far: usize,
    near: usize,
    vantage_point: usize,
    radius: A,
}

const NULL: usize = usize::max_value();

struct DistanceIndex<A> {
    distance: OrderedFloat<A>,
    id: usize,
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::{array, aview1};

    #[test]
    fn euclidian() {
        let points = array![
            [1.0, 2.0],
            [1.1, 2.2],
            [0.9, 1.9],
            [1.0, 2.1],
            [-2.0, 3.0],
            [-2.2, 3.1],
        ];
        let vp = VantagePointTree::euclidean(points).expect("valid array");

        assert_eq!(vp.query_nearest(&aview1(&[0.95, 1.96])).0, 0);
    }
}
