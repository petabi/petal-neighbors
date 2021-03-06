//! Distance metrics.

use ndarray::ArrayView1;
use num_traits::{Float, Zero};
use std::ops::AddAssign;

/// The type of a distance metric function.
pub trait Metric<A> {
    fn distance(&self, _: &ArrayView1<A>, _: &ArrayView1<A>) -> A;
}

#[derive(Default, Clone)]
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
}
