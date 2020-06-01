use crate::distance::{self, Distance};
use std::marker::PhantomData;
use std::ops::Index;

pub trait PointSet<P: ?Sized>: Index<usize, Output = P> {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct Node {
    far: usize,
    near: usize,
    vantage_point: usize,
    radius: f64,
}

const NULL: usize = usize::max_value();

struct DistanceIndex {
    distance: f64,
    id: usize,
}

pub struct VantagePointTree<P, S>
where
    P: ?Sized,
    S: PointSet<P>,
{
    pub data: S,
    pub nodes: Vec<Node>,
    pub root: usize,
    distance: Distance<f64>,
    _phantom: PhantomData<P>,
}

impl<S> VantagePointTree<[f64], S>
where
    S: PointSet<[f64]>,
{
    pub fn new(data: S, distance: Distance<f64>) -> Self {
        let mut nodes = Vec::with_capacity(data.len());
        let root = Self::create_root(&data, distance, &mut nodes);
        VantagePointTree {
            data,
            nodes,
            root,
            distance,
            _phantom: PhantomData,
        }
    }

    /// Builds a vantage point tree with a euclidean distance metric.
    pub fn euclidean<T>(points: S) -> Self {
        Self::new(points, distance::euclidean::<f64>)
    }

    pub fn find_nearest(&self, needle: &[f64]) -> (usize, f64) {
        let mut nearest = DistanceIndex {
            distance: std::f64::MAX,
            id: NULL,
        };
        self.search_node(&self.nodes[self.root], needle, &mut nearest);
        (nearest.id, nearest.distance)
    }

    fn search_node(&self, node: &Node, needle: &[f64], nearest: &mut DistanceIndex) {
        let distance = self.distance;
        let distance = distance(&self.data[node.vantage_point], needle);

        if distance < nearest.distance {
            nearest.distance = distance;
            nearest.id = node.vantage_point;
        }

        if distance < node.radius {
            if let Some(near) = self.nodes.get(node.near) {
                self.search_node(near, needle, nearest);
            }
            if let Some(far) = self.nodes.get(node.far) {
                if node.radius < distance + nearest.distance {
                    self.search_node(far, needle, nearest);
                }
            }
        } else {
            if let Some(far) = self.nodes.get(node.far) {
                self.search_node(far, needle, nearest);
            }
            if let Some(near) = self.nodes.get(node.near) {
                if node.radius + nearest.distance > distance {
                    self.search_node(near, needle, nearest);
                }
            }
        }
    }

    fn create_root(data: &S, distance: Distance<f64>, nodes: &mut Vec<Node>) -> usize {
        let mut indexes: Vec<_> = (0..data.len())
            .map(|i| DistanceIndex {
                distance: std::f64::MAX,
                id: i,
            })
            .collect();
        Self::create_node(data, distance, &mut indexes, nodes)
    }

    fn create_node(
        data: &S,
        distance: Distance<f64>,
        indexes: &mut [DistanceIndex],
        nodes: &mut Vec<Node>,
    ) -> usize {
        if indexes.is_empty() {
            return NULL;
        }
        if indexes.len() == 1 {
            let id = nodes.len();
            nodes.push(Node {
                near: NULL,
                far: NULL,
                vantage_point: indexes[0].id,
                radius: std::f64::MAX,
            });
            return id;
        }

        let vp_pos = indexes.len() - 1;
        let vantage_point = indexes[vp_pos].id;
        let rest = &mut indexes[..vp_pos];

        for r in rest.iter_mut() {
            r.distance = distance(&data[r.id], &data[vantage_point]);
        }
        rest.sort_unstable_by(|a, b| a.distance.partial_cmp(&b.distance).expect("unexpected nan"));

        let half = rest.len() / 2;
        let (near, far) = rest.split_at_mut(half);
        let radius = far[0].distance;

        let id = nodes.len();
        nodes.push(Node {
            far: NULL,
            near: NULL,
            vantage_point,
            radius,
        });

        let near = Self::create_node(data, distance, near, nodes);
        let far = Self::create_node(data, distance, far, nodes);
        nodes[id].near = near;
        nodes[id].far = far;
        id
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::{array, Array2};

    struct Table {
        points: Array2<f64>,
    }

    impl Index<usize> for Table {
        type Output = [f64];

        /// Returns the `i`th point.
        ///
        /// # Panics
        ///
        /// Panics if `i` is out of bound, or if the array's data is not
        /// contiguous or not in standard order.
        fn index(&self, i: usize) -> &Self::Output {
            self.points.row(i).to_slice().unwrap()
        }
    }

    impl PointSet<[f64]> for Table {
        fn len(&self) -> usize {
            self.points.nrows()
        }
    }

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
        let vp = VantagePointTree::euclidean::<f64>(Table { points });

        assert_eq!(vp.find_nearest(&[0.95, 1.96]).0, 0);
    }
}
