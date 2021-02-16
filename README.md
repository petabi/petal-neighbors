# petal-neighbors

Nearest neighbor search algorithms including a ball tree and a vantage point tree.

[![crates.io](https://img.shields.io/crates/v/petal-neighbors)](https://crates.io/crates/petal-neighbors)
[![Documentation](https://docs.rs/petal-neighbors/badge.svg)](https://docs.rs/petal-neighbors)
[![Coverage Status](https://codecov.io/gh/petabi/petal-neighbors/branch/master/graphs/badge.svg)](https://codecov.io/gh/petabi/petal-neighbors)

## Examples

The following example shows how to find two nearest neighbors in a ball tree.

```rust
use ndarray::{array, aview1};
use petal_neighbors::{BallTree, distance};

let points = array![[1., 1.], [1., 2.], [9., 9.]];
let tree = BallTree::euclidean(points).unwrap();
let (indices, distances) = tree.query(&aview1(&[3., 3.]), 2);
assert_eq!(indices, &[1, 0]);  // points[1] is the nearest, points[0] the next.
```

## Minimum Supported Rust Version

This crate is guaranteed to compile on Rust 1.42 and later.

## License

Copyright 2019-2021 Petabi, Inc.

Licensed under [Apache License, Version 2.0][apache-license] (the "License");
you may not use this crate except in compliance with the License.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See [LICENSE](LICENSE) for
the specific language governing permissions and limitations under the License.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the [Apache-2.0
license][apache-license], shall be licensed as above, without any additional
terms or conditions.

[apache-license]: http://www.apache.org/licenses/LICENSE-2.0
