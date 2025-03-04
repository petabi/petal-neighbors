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

#[derive(Default, Clone, Debug, Eq, PartialEq)]
pub struct Cosine {}

unsafe impl Sync for Cosine {}

impl<A> Metric<A> for Cosine
where
    A: Float + AddAssign + std::iter::Sum,
{
    /// Cosine distance metric.
    fn distance(&self, x1: &ArrayView1<A>, x2: &ArrayView1<A>) -> A {
        let dot = x1
            .iter()
            .zip(x2.iter())
            .map(|(&v1, &v2)| v1 * v2)
            .sum::<A>();

        let norm1 = x1
            .iter()
            .zip(x1.iter())
            .map(|(&v1, &v2)| v1 * v2)
            .sum::<A>()
            .sqrt();

        let norm2 = x2
            .iter()
            .zip(x2.iter())
            .map(|(&v1, &v2)| v1 * v2)
            .sum::<A>()
            .sqrt();
        A::one() - dot / (norm1 * norm2)
    }

    /// Cosine reduce distance metric.
    fn rdistance(&self, x1: &ArrayView1<A>, x2: &ArrayView1<A>) -> A {
        self.distance(x1, x2)
    }
    /// Cosine reduce distance to distance.
    fn rdistance_to_distance(&self, d: A) -> A {
        d
    }

    /// Cosine distance to reduce distance.
    fn distance_to_rdistance(&self, d: A) -> A {
        d
    }
}

#[cfg(test)]
mod test {
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, arr2};

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

    #[test]
    fn cosine() {
        use super::Metric;

        let metric = super::Cosine::default();
        let x = arr1(&[1., 0.]);
        let y = arr1(&[0., 1.]);
        assert_eq!(metric.distance(&x.view(), &y.view()), 1.);
        assert_eq!(metric.rdistance(&x.view(), &x.view()), 0.);
        assert_eq!(metric.rdistance(&y.view(), &y.view()), 0.);

        // Test case 1: Identical vectors (distance should be 0)
        let v1 = arr1(&[1.0, 2.0, 3.0]);
        let v2 = arr1(&[1.0, 2.0, 3.0]);
        assert_abs_diff_eq!(metric.distance(&v1.view(), &v2.view()), 0.0, epsilon = 1e-6);

        // Test case 2: Orthogonal vectors (normalized) (distance should be 1)
        let v3 = arr1(&[1.0, 0.0]);
        let v4 = arr1(&[0.0, 1.0]);
        assert_abs_diff_eq!(metric.distance(&v3.view(), &v4.view()), 1.0, epsilon = 1e-6);

        // Test case 3: Opposite vectors (should be 2)
        let v5 = arr1(&[1.0, 1.0]);
        let v6 = arr1(&[-1.0, -1.0]);
        assert_abs_diff_eq!(
            metric.rdistance(&v5.view(), &v6.view()),
            2.0_f32,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            metric.distance(&v5.view(), &v6.view()),
            2.0_f32,
            epsilon = 1e-6
        );

        // Test case 4: Random non-trivial vectors
        let v7 = arr1(&[3.0, 4.0]);
        let v8 = arr1(&[6.0, 8.0]);
        assert_abs_diff_eq!(metric.distance(&v7.view(), &v8.view()), 0.0, epsilon = 1e-6);
    }
}
