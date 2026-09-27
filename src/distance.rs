//! Distance metrics.

use std::ops::AddAssign;
use std::sync::OnceLock;

use fearless_simd::{dispatch, prelude::*, Level};
use ndarray::{Array2, ArrayView1, ArrayView2};
use num_traits::{Float, Zero};

static SIMD_LEVEL: OnceLock<Level> = OnceLock::new();

fn simd_level() -> Level {
    *SIMD_LEVEL.get_or_init(Level::new)
}

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
    A: Float + Zero + AddAssign + SimdFloatElement,
{
    /// Euclidean distance metric.
    fn distance(&self, x1: &ArrayView1<A>, x2: &ArrayView1<A>) -> A {
        squared_distance(x1, x2).sqrt()
    }
    /// Euclidean reduce distance metric.
    fn rdistance(&self, x1: &ArrayView1<A>, x2: &ArrayView1<A>) -> A {
        squared_distance(x1, x2)
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

#[allow(clippy::inline_always)] // Keep the short scalar path inline at distance call sites.
#[inline(always)]
fn squared_distance<A>(x1: &ArrayView1<A>, x2: &ArrayView1<A>) -> A
where
    A: Float + Zero + AddAssign + SimdFloatElement,
{
    // Dispatch has a fixed cost, so short vectors and strided views use the
    // original scalar reduction.
    if x1.len().min(x2.len()) >= 32 {
        if let (Some(left), Some(right)) = (x1.as_slice(), x2.as_slice()) {
            return squared_distance_contiguous(left, right);
        }
    }
    scalar_squared_distance(x1, x2)
}

#[allow(clippy::inline_always)] // Keep the short scalar reduction inline.
#[inline(always)]
fn scalar_squared_distance<A>(x1: &ArrayView1<A>, x2: &ArrayView1<A>) -> A
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

#[inline(never)]
fn squared_distance_contiguous<A: Float + AddAssign + SimdFloatElement>(
    left: &[A],
    right: &[A],
) -> A {
    dispatch!(simd_level(), simd => squared_distance_simd(simd, left, right))
}

#[allow(clippy::inline_always)] // SIMD operations must be inlined into the dispatched target context.
#[inline(always)]
fn squared_distance_simd<S: Simd, A: Float + AddAssign + SimdFloatElement>(
    simd: S,
    left: &[A],
    right: &[A],
) -> A {
    let len = left.len().min(right.len());
    let lanes = A::Native::<S>::LEN;
    let mut sum = A::Native::<S>::splat(simd, A::zero());
    let mut i = 0;
    while i + lanes <= len {
        let a = A::Native::<S>::from_slice(simd, &left[i..i + lanes]);
        let b = A::Native::<S>::from_slice(simd, &right[i..i + lanes]);
        let diff = a - b;
        sum += diff * diff;
        i += lanes;
    }
    let mut total = sum.reduce_sum();
    for j in i..len {
        let diff = left[j] - right[j];
        total += diff * diff;
    }
    total
}

#[allow(clippy::needless_pass_by_value)] // Silences clippy warning. TODO: Update the parameter type to [`ArrayRef`](https://docs.rs/ndarray/latest/ndarray/struct.ArrayRef.html).
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
    A: Float + AddAssign + std::iter::Sum + SimdFloatElement,
{
    /// Cosine distance metric.
    #[allow(clippy::inline_always)] // Keep the short scalar path inline at call sites.
    #[inline(always)]
    fn distance(&self, x1: &ArrayView1<A>, x2: &ArrayView1<A>) -> A {
        if x1.len() == x2.len() && x1.len() >= 32 {
            if let (Some(left), Some(right)) = (x1.as_slice(), x2.as_slice()) {
                return cosine_distance_contiguous(left, right);
            }
        }
        scalar_cosine_distance(x1, x2)
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

#[allow(clippy::inline_always)] // Keep short vectors on the original scalar reduction.
#[inline(always)]
fn scalar_cosine_distance<A: Float + AddAssign + std::iter::Sum>(
    x1: &ArrayView1<A>,
    x2: &ArrayView1<A>,
) -> A {
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

#[inline(never)]
fn cosine_distance_contiguous<A: Float + AddAssign + SimdFloatElement>(
    left: &[A],
    right: &[A],
) -> A {
    dispatch!(simd_level(), simd => cosine_distance_simd(simd, left, right))
}

#[allow(clippy::inline_always)] // SIMD operations must be inlined into the dispatched target context.
#[inline(always)]
fn cosine_distance_simd<S: Simd, A: Float + AddAssign + SimdFloatElement>(
    simd: S,
    left: &[A],
    right: &[A],
) -> A {
    let lanes = A::Native::<S>::LEN;
    let mut dot = A::Native::<S>::splat(simd, A::zero());
    let mut norm1 = A::Native::<S>::splat(simd, A::zero());
    let mut norm2 = A::Native::<S>::splat(simd, A::zero());
    let mut i = 0;
    while i + lanes <= left.len() {
        let a = A::Native::<S>::from_slice(simd, &left[i..i + lanes]);
        let b = A::Native::<S>::from_slice(simd, &right[i..i + lanes]);
        dot += a * b;
        norm1 += a * a;
        norm2 += b * b;
        i += lanes;
    }
    let mut dot = dot.reduce_sum();
    let mut norm1 = norm1.reduce_sum();
    let mut norm2 = norm2.reduce_sum();
    for j in i..left.len() {
        dot += left[j] * right[j];
        norm1 += left[j] * left[j];
        norm2 += right[j] * right[j];
    }
    A::one() - dot / (norm1.sqrt() * norm2.sqrt())
}

#[cfg(test)]
mod test {
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, arr2, s, Array1};

    use super::Metric;

    #[test]
    #[allow(clippy::cast_possible_truncation)] // The test intentionally compares f32 with f64.
    fn euclidean_simd_matches_scalar_reduction() {
        let metric = super::Euclidean::default();
        for len in [31, 32, 33, 128, 769] {
            let left = Array1::from_iter((0..len).map(|i| f64::from(i % 17) / 7.0));
            let right = Array1::from_iter((0..len).map(|i| f64::from(i % 13) / 11.0));
            let expected = left
                .iter()
                .zip(right.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum::<f64>();
            assert_abs_diff_eq!(
                metric.rdistance(&left.view(), &right.view()),
                expected,
                epsilon = 1e-11
            );

            let strided = left.slice(s![..;2]);
            let right_strided = right.slice(s![..;2]);
            let expected_strided = strided
                .iter()
                .zip(right_strided.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum::<f64>();
            assert_abs_diff_eq!(
                metric.rdistance(&strided, &right_strided),
                expected_strided,
                epsilon = 1e-11
            );

            let left_f32 = left.mapv(|v| v as f32);
            let right_f32 = right.mapv(|v| v as f32);
            let expected_f32 = left_f32
                .iter()
                .zip(right_f32.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum::<f32>();
            assert_abs_diff_eq!(
                metric.rdistance(&left_f32.view(), &right_f32.view()),
                expected_f32,
                epsilon = 1e-3
            );
        }
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)] // The test intentionally compares f32 with f64.
    fn cosine_simd_matches_scalar_reduction() {
        let metric = super::Cosine::default();
        for len in [32, 33, 128, 769] {
            let left = Array1::from_iter((0..len).map(|i| f64::from(i % 17 + 1) / 7.0));
            let right = Array1::from_iter((0..len).map(|i| f64::from(i % 13 + 1) / 11.0));
            assert_abs_diff_eq!(
                metric.distance(&left.view(), &right.view()),
                super::scalar_cosine_distance(&left.view(), &right.view()),
                epsilon = 1e-12
            );
            let left_f32 = left.mapv(|v| v as f32);
            let right_f32 = right.mapv(|v| v as f32);
            assert_abs_diff_eq!(
                metric.distance(&left_f32.view(), &right_f32.view()),
                super::scalar_cosine_distance(&left_f32.view(), &right_f32.view()),
                epsilon = 1e-6
            );

            let strided = left.slice(s![..;2]);
            let right_strided = right.slice(s![..;2]);
            assert_abs_diff_eq!(
                metric.distance(&strided, &right_strided),
                super::scalar_cosine_distance(&strided, &right_strided),
                epsilon = 1e-12
            );
        }
    }

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
        assert_abs_diff_eq!(metric.distance(&x.view(), &y.view()), 1., epsilon = 1e-6);
        assert_abs_diff_eq!(metric.rdistance(&x.view(), &x.view()), 0., epsilon = 1e-6);
        assert_abs_diff_eq!(metric.rdistance(&y.view(), &y.view()), 0., epsilon = 1e-6);

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
