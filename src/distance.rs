//! Distance metrics.

use num_traits::{Float, Zero};
use std::ops::AddAssign;

pub type Distance<A> = fn(&[A], &[A]) -> A;

pub fn euclidean_reduced<A>(x1: &[A], x2: &[A]) -> A
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

pub fn euclidean<A>(x1: &[A], x2: &[A]) -> A
where
    A: Float + Zero + AddAssign,
{
    euclidean_reduced(x1, x2).sqrt()
}
