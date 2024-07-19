//! Distance metrics.

use std::ops::AddAssign;

use ndarray::{Array2, ArrayView1, ArrayView2};
use num_traits::{Float, Zero};

/// The type of a distance metric function.
pub trait Metric<A> {
    fn distance(&self, _: &ArrayView1<A>, _: &ArrayView1<A>) -> A;
    fn rdistance(&self, _: &ArrayView1<A>, _: &ArrayView1<A>) -> A;
    fn rdistance_to_distance(&self, _: A) -> A;
    fn distance_to_rdistance(&self, _: A) -> A;
}

#[derive(Default, Clone, Debug, Eq, PartialEq)]
pub struct Euclidean {}

unsafe impl Sync for Euclidean {}

impl<A> Metric<A> for Euclidean
where
    A: Float + Zero + AddAssign,
{
    /// Euclidean distance metric.
    fn distance(&self, x1: &ArrayView1<A>, x2: &ArrayView1<A>) -> A {
        x1.iter()
            .zip(x2.iter())
            .fold(A::zero(), |mut sum, (&v1, &v2)| {
                let diff = v1 - v2;
                sum += diff * diff;
                sum
            })
            .sqrt()
    }
    /// Euclidean reduce distance metric.
    fn rdistance(&self, x1: &ArrayView1<A>, x2: &ArrayView1<A>) -> A {
        x1.iter()
            .zip(x2.iter())
            .fold(A::zero(), |mut sum, (&v1, &v2)| {
                let diff = v1 - v2;
                sum += diff * diff;
                sum
            })
    }
    /// Euclidean reduce distance metric.
    fn rdistance_to_distance(&self, d: A) -> A {
        d.sqrt()
    }

    /// Euclidean reduce distance metric.
    fn distance_to_rdistance(&self, d: A) -> A {
        d.powi(2)
    }
}

pub fn pairwise<A: Float + Zero + AddAssign>(
    x: ArrayView2<A>,
    metric: &dyn Metric<A>,
) -> Array2<A> {
    let mut distances = Array2::<A>::zeros((x.nrows(), x.nrows()));
    if x.nrows() < 2 {
        return distances;
    }
    for i in 0..x.nrows() {
        for j in (i + 1)..x.nrows() {
            let d = metric.distance(&x.row(i), &x.row(j));
            distances[[i, j]] = d;
            distances[[j, i]] = d;
        }
    }
    distances
}

#[cfg(test)]
mod test {
    use ndarray::arr2;

    #[test]
    fn pairwise() {
        let x = arr2(&[[3., 4.], [0., 0.]]);
        let distances = super::pairwise(x.view(), &super::Euclidean {});
        assert_eq!(distances, arr2(&[[0., 5.], [5., 0.]]));
    }

    #[test]
    fn pairwise_one() {
        let x = arr2(&[[0.]]);
        let distances = super::pairwise(x.view(), &super::Euclidean {});
        assert_eq!(distances, arr2(&[[0.]]));
    }
}
