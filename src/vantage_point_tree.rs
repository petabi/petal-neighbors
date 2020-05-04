pub trait Metric {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn distance(&self, i: usize, j: usize) -> f64;
    fn distance_to_needle(&self, i: usize, needle: (&Self, usize)) -> f64;
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

pub struct VantagePointTree<D>
where
    D: Metric,
{
    pub data: D,
    pub nodes: Vec<Node>,
    pub root: usize,
}

impl<D> VantagePointTree<D>
where
    D: Metric,
{
    pub fn new(data: D) -> Self {
        let mut nodes = Vec::with_capacity(data.len());
        let root = Self::create_root(&data, &mut nodes);
        VantagePointTree { data, nodes, root }
    }

    pub fn find_nearest(&self, needle: (&D, usize)) -> (usize, f64) {
        let mut nearest = DistanceIndex {
            distance: std::f64::MAX,
            id: NULL,
        };
        self.search_node(&self.nodes[self.root], needle, &mut nearest);
        (nearest.id, nearest.distance)
    }

    fn search_node(&self, node: &Node, needle: (&D, usize), nearest: &mut DistanceIndex) {
        let distance = self.data.distance_to_needle(node.vantage_point, needle);

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

    fn create_root(data: &D, nodes: &mut Vec<Node>) -> usize {
        let mut indexes: Vec<_> = (0..data.len())
            .map(|i| DistanceIndex {
                distance: std::f64::MAX,
                id: i,
            })
            .collect();
        Self::create_node(&data, &mut indexes, nodes)
    }

    fn create_node(data: &D, indexes: &mut [DistanceIndex], nodes: &mut Vec<Node>) -> usize {
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
            r.distance = data.distance(r.id, vantage_point);
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

        let near = Self::create_node(&data, near, nodes);
        let far = Self::create_node(&data, far, nodes);
        nodes[id].near = near;
        nodes[id].far = far;
        id
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::{array, Array, Ix2};

    struct Table {
        points: Array<f64, Ix2>,
    }

    impl Metric for Table {
        fn len(&self) -> usize {
            self.points.nrows()
        }

        fn distance(&self, i: usize, j: usize) -> f64 {
            self.points
                .row(i)
                .into_iter()
                .zip(self.points.row(j).into_iter())
                .fold(0.0, |mut sum, (v1, v2)| {
                    sum += (v1 - v2).powi(2);
                    sum
                })
                .sqrt()
        }

        fn distance_to_needle(&self, i: usize, needle: (&Self, usize)) -> f64 {
            self.points
                .row(i)
                .into_iter()
                .zip(needle.0.points.row(needle.1).into_iter())
                .fold(0.0, |mut sum, (v1, v2)| {
                    sum += (v1 - v2).powi(2);
                    sum
                })
                .sqrt()
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
        let table = Table { points };

        let vp = VantagePointTree::new(table);

        let tester = Table {
            points: array![[0.95, 1.96]],
        };

        assert_eq!(vp.find_nearest((&tester, 0)).0, 0);
    }
}
