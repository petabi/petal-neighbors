//! Distance metrics.

use ndarray::ArrayView1;
use num_traits::{Float, Zero};
use std::ops::AddAssign;

/// The type of a distance metric function.
pub trait Distance<A> {
    fn get(&self, _: &ArrayView1<A>, _: &ArrayView1<A>) -> A;
    fn get_reduced(&self, _: &ArrayView1<A>, _: &ArrayView1<A>) -> A;
}

#[derive(Default)]
pub struct Euclidean {}

impl<A> Distance<A> for Euclidean
where
    A: Float + Zero + AddAssign,
{
    /// Euclidean distance before taking a squre root. Used as a lightweight version
    /// of [`euclidean`] for relative comparisions.
    /// [`euclidean`]: #method.euclidean
    fn get_reduced(&self, x1: &ArrayView1<A>, x2: &ArrayView1<A>) -> A {
        x1.iter()
            .zip(x2.iter())
            .fold(A::zero(), |mut sum, (&v1, &v2)| {
                let diff = v1 - v2;
                sum += diff * diff;
                sum
            })
    }

    /// Euclidean distance metric.
    fn get(&self, x1: &ArrayView1<A>, x2: &ArrayView1<A>) -> A {
        self.get_reduced(x1, x2).sqrt()
    }
}

#[derive(Default)]
pub struct EuclideanNoReduced {}

impl<A> Distance<A> for EuclideanNoReduced
where
    A: Float + Zero + AddAssign,
{
    fn get_reduced(&self, x1: &ArrayView1<A>, x2: &ArrayView1<A>) -> A {
        x1.iter()
            .zip(x2.iter())
            .fold(A::zero(), |mut sum, (&v1, &v2)| {
                let diff = v1 - v2;
                sum += diff * diff;
                sum
            })
            .sqrt()
    }

    fn get(&self, x1: &ArrayView1<A>, x2: &ArrayView1<A>) -> A {
        self.get_reduced(x1, x2)
    }
}
