//! Distance metrics.

/// A distance metric for multidimensional points.
pub trait Metric {
    /// Calculates the distance between two points.
    fn distance<'p, 'q, P, Q>(&self, x1: P, x2: Q) -> f64
    where
        P: IntoIterator<Item = &'p f64>,
        Q: IntoIterator<Item = &'q f64>;

    /// Calculates a value that can be used in relative comparison between two
    /// distances. Therefore, if `distance(p1, q1)` is less than `distance(p2,
    /// q2)`, then `reduced_distance(p1, q1)` must be less than
    /// `reduced_distance(p2, q2)`.
    ///
    /// This is to provide a more efficient way of distance comparison when
    /// absolute distance values are not necessary. For example, in the
    /// Euclidean metric, this function may return a squared distance to avoid
    /// the overhead of computing the square root.
    ///
    /// By default, this calls [`distance`].
    ///
    /// [`distance`]: #method.distance
    fn reduced_distance<'p, 'q, P, Q>(&self, x1: P, x2: Q) -> f64
    where
        P: IntoIterator<Item = &'p f64>,
        Q: IntoIterator<Item = &'q f64>;
}

/// The Euclidean distance metric.
#[derive(Debug)]
pub struct Euclidean;

impl Metric for Euclidean {
    /// Calculates the Euclidean distance between two points.
    fn distance<'p, 'q, P, Q>(&self, x1: P, x2: Q) -> f64
    where
        P: IntoIterator<Item = &'p f64>,
        Q: IntoIterator<Item = &'q f64>,
    {
        self.reduced_distance(x1, x2).sqrt()
    }

    /// Calculates the squared Euclidean distance between two points.
    fn reduced_distance<'p, 'q, P, Q>(&self, x1: P, x2: Q) -> f64
    where
        P: IntoIterator<Item = &'p f64>,
        Q: IntoIterator<Item = &'q f64>,
    {
        x1.into_iter()
            .zip(x2.into_iter())
            .fold(0_f64, |mut sum, (v1, v2)| {
                sum += (v1 - v2) * (v1 - v2);
                sum
            })
    }
}

/// An instance of the Euclidean distance metric, to be used as a function
/// argument.
pub const EUCLIDEAN: Euclidean = Euclidean {};
