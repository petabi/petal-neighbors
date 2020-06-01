use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::ArrayView;
use ndarray_rand::rand::{rngs::StdRng, Rng, SeedableRng};
use petal_neighbors::BallTree;

fn build(c: &mut Criterion) {
    let n = black_box(128);
    let dim = black_box(10);

    let mut rng = StdRng::from_seed(*b"ball tree build bench test seed ");
    let data: Vec<f64> = (0..n * dim).map(|_| rng.gen()).collect();
    let array = ArrayView::from_shape((n, dim), &data).unwrap();
    c.bench_function("build", |b| {
        b.iter(|| {
            BallTree::euclidean(array).expect("`array` is not empty");
        })
    });
}

fn query_radius(c: &mut Criterion) {
    let n = black_box(64);
    let dim = black_box(10);

    let mut rng = StdRng::from_seed(*b"ball tree query_radius test seed");
    let data: Vec<f64> = (0..n * dim).map(|_| rng.gen()).collect();
    let array = ArrayView::from_shape((n, dim), &data).unwrap();
    let tree = BallTree::euclidean(array).expect("`array` is not empty");
    c.bench_function("query_radius", |b| {
        b.iter(|| {
            for i in 0..n {
                let query = &data[i * dim..i * dim + dim];
                tree.query_radius(query, 0.2);
            }
        })
    });
}

criterion_group!(benches, build, query_radius);
criterion_main!(benches);
