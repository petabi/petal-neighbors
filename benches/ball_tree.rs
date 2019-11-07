use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::ArrayView;
use petal_neighbors::{distance, BallTree};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn query_radius(c: &mut Criterion) {
    let n = black_box(64);
    let dim = black_box(10);

    let mut rng = StdRng::from_seed(*b"ball tree query_radius test seed");
    let data: Vec<f64> = (0..n * dim).map(|_| rng.gen()).collect();
    let array = ArrayView::from_shape((n, dim), &data).unwrap();
    let tree = BallTree::with_metric(array.clone(), distance::EUCLIDEAN);
    c.bench_function("query_radius", |b| {
        b.iter(|| {
            for i in 0..n {
                let query = &data[i * dim..i * dim + dim];
                tree.query_radius(query, 0.2);
            }
        })
    });
}

criterion_group!(benches, query_radius);
criterion_main!(benches);
