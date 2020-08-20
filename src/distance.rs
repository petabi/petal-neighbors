//! Distance metrics.

use ndarray::ArrayView1;
use num_traits::{Float, Zero};
use std::ops::AddAssign;

/// The type of a distance metric function.
pub type Distance<A> = fn(&ArrayView1<A>, &ArrayView1<A>) -> A;

/// Euclidean distance before taking a squre root. Used as a lightweight version
/// of [`euclidean`] for relative comparisions.
/// [`euclidean`]: #method.euclidean
#[must_use]
pub fn euclidean_reduced<A>(x1: &ArrayView1<A>, x2: &ArrayView1<A>) -> A
where
    A: Float + Zero + AddAssign,
{
    x1.iter()
        .zip(x2.iter())
        .fold(A::zero(), |mut sum, (&v1, &v2)| {
            let diff = v1 - v2;
            sum += diff * diff;
            sum
        })
}

/// Euclidean distance metric.
#[must_use]
pub fn euclidean<A>(x1: &ArrayView1<A>, x2: &ArrayView1<A>) -> A
where
    A: Float + Zero + AddAssign,
{
    euclidean_reduced(x1, x2).sqrt()
}
