pub trait Metric {
    fn distance<'a, 'b, P, Q>(&self, x1: P, x2: Q) -> f64
    where
        P: 'a + IntoIterator,
        <P as IntoIterator>::Item: Copy + Into<&'a f64>,
        Q: 'b + IntoIterator,
        <Q as IntoIterator>::Item: Copy + Into<&'b f64>;

    /// Calculates the squared euclidean distance of two points.
    fn reduced_distance<'a, 'b, P, Q>(&self, x1: P, x2: Q) -> f64
    where
        P: 'a + IntoIterator,
        <P as IntoIterator>::Item: Copy + Into<&'a f64>,
        Q: 'b + IntoIterator,
        <Q as IntoIterator>::Item: Copy + Into<&'b f64>;
}

#[derive(Debug)]
pub struct Euclidean;

impl Metric for Euclidean {
    fn distance<'a, 'b, P, Q>(&self, x1: P, x2: Q) -> f64
    where
        P: 'a + IntoIterator,
        <P as IntoIterator>::Item: Copy + Into<&'a f64>,
        Q: 'b + IntoIterator,
        <Q as IntoIterator>::Item: Copy + Into<&'b f64>,
    {
        self.reduced_distance(x1, x2).sqrt()
    }

    fn reduced_distance<'a, 'b, P, Q>(&self, x1: P, x2: Q) -> f64
    where
        P: 'a + IntoIterator,
        <P as IntoIterator>::Item: Copy + Into<&'a f64>,
        Q: 'b + IntoIterator,
        <Q as IntoIterator>::Item: Copy + Into<&'b f64>,
    {
        x1.into_iter()
            .zip(x2.into_iter())
            .fold(0_f64, |mut sum, (v1, v2)| {
                sum += (v1.into() - v2.into()) * (v1.into() - v2.into());
                sum
            })
    }
}

pub const EUCLIDEAN: Euclidean = Euclidean {};
