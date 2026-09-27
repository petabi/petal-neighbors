//! Performance benchmarks for distance metrics and nearest-neighbor algorithms.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{ArrayView1, ArrayView2};
use ndarray_rand::rand::{rngs::StdRng, Rng, SeedableRng};
use petal_neighbors::distance::{self, Cosine, Euclidean, Metric};
use petal_neighbors::{BallTree, VantagePointTree};

const DIMENSIONS: [usize; 3] = [10, 128, 768];
const POINTS: usize = 128;
const QUERIES: usize = 16;

macro_rules! benchmark_type {
    ($distance_fn:ident, $tree_fn:ident, $type:ty, $label:literal) => {
        fn $distance_fn(c: &mut Criterion) {
            for dim in DIMENSIONS {
                let mut rng = StdRng::seed_from_u64(42);
                let left: Vec<$type> = (0..dim).map(|_| rng.random()).collect();
                let right: Vec<$type> = (0..dim).map(|_| rng.random()).collect();
                let left = ArrayView1::from(&left);
                let right = ArrayView1::from(&right);

                let mut group = c.benchmark_group(format!("distance/{label}", label = $label));
                group.throughput(Throughput::Elements(dim as u64));
                group.bench_with_input(BenchmarkId::new("euclidean", dim), &dim, |b, _| {
                    let metric = Euclidean::default();
                    b.iter(|| black_box(metric.distance(black_box(&left), black_box(&right))));
                });
                group.bench_with_input(BenchmarkId::new("cosine", dim), &dim, |b, _| {
                    let metric = Cosine::default();
                    b.iter(|| black_box(metric.distance(black_box(&left), black_box(&right))));
                });
                group.finish();
            }
        }

        fn $tree_fn(c: &mut Criterion) {
            for dim in DIMENSIONS {
                let mut rng = StdRng::seed_from_u64(42);
                let data: Vec<$type> = (0..POINTS * dim).map(|_| rng.random()).collect();
                let queries: Vec<$type> = (0..QUERIES * dim).map(|_| rng.random()).collect();
                let points = ArrayView2::from_shape((POINTS, dim), &data).unwrap();
                let queries = ArrayView2::from_shape((QUERIES, dim), &queries).unwrap();
                let tree = BallTree::euclidean(points).unwrap();

                let mut group = c.benchmark_group(format!("ball_tree/{label}", label = $label));
                group.bench_with_input(BenchmarkId::new("build", dim), &dim, |b, _| {
                    b.iter(|| black_box(BallTree::euclidean(black_box(points)).unwrap()));
                });
                group.bench_with_input(
                    BenchmarkId::new("query_nearest_batch", dim),
                    &dim,
                    |b, _| {
                        b.iter(|| {
                            for query in queries.rows() {
                                black_box(tree.query_nearest(&query));
                            }
                        });
                    },
                );
                group.bench_with_input(BenchmarkId::new("query_k_batch", dim), &dim, |b, _| {
                    b.iter(|| {
                        for query in queries.rows() {
                            black_box(tree.query(&query, 5));
                        }
                    });
                });
                let radius = (dim as $type / 6.0).sqrt();
                group.bench_with_input(
                    BenchmarkId::new("query_radius_batch", dim),
                    &dim,
                    |b, _| {
                        b.iter(|| {
                            for query in queries.rows() {
                                black_box(tree.query_radius(&query, radius));
                            }
                        });
                    },
                );
                group.finish();

                let mut group =
                    c.benchmark_group(format!("vantage_point_tree/{label}", label = $label));
                let vp_tree = VantagePointTree::euclidean(points).unwrap();
                group.bench_with_input(BenchmarkId::new("build", dim), &dim, |b, _| {
                    b.iter(|| black_box(VantagePointTree::euclidean(black_box(points)).unwrap()));
                });
                group.bench_with_input(
                    BenchmarkId::new("query_nearest_batch", dim),
                    &dim,
                    |b, _| {
                        b.iter(|| {
                            for query in queries.rows() {
                                black_box(vp_tree.query_nearest(&query));
                            }
                        });
                    },
                );
                group.finish();

                let mut group = c.benchmark_group(format!("pairwise/{label}", label = $label));
                group.bench_with_input(BenchmarkId::new("euclidean", dim), &dim, |b, _| {
                    let metric = Euclidean::default();
                    b.iter(|| black_box(distance::pairwise(black_box(points), &metric)));
                });
                group.finish();
            }
        }
    };
}

benchmark_type!(distance_f32, tree_f32, f32, "f32");
benchmark_type!(distance_f64, tree_f64, f64, "f64");

criterion_group!(benches, distance_f32, distance_f64, tree_f32, tree_f64);
criterion_main!(benches);
