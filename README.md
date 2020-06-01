# petal-neighbors

Nearest neighbor search algorithms including a ball tree and a vantage point tree.

[![crates.io](https://img.shields.io/crates/v/petal-neighbors)](https://crates.io/crates/petal-neighbors)
[![Documentation](https://docs.rs/petal-neighbors/badge.svg)](https://docs.rs/petal-neighbors)
[![Coverage Status](https://codecov.io/gh/petabi/petal-neighbors/branch/master/graphs/badge.svg)](https://codecov.io/gh/petabi/petal-neighbors)

## Requirements

* Rust ≥ 1.37

## Examples

The following example shows how to find two nearest neighbors in a ball tree.

```rust
use ndarray::array;
use petal_neighbors::{BallTree, distance};

let points = array![[1., 1.], [1., 2.], [9., 9.]];
let tree = BallTree::euclidean(points).unwrap();
let (indices, distances) = tree.query(&[3., 3.], 2);
assert_eq!(indices, &[1, 0]);  // points[1] is the nearest, points[0] the next.
```
