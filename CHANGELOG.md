# Changelog

This file documents recent notable changes to this project. The format of this
file is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and
this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2020-08-20

### Changed

- `BallTree` and `VantagePointTree` accept an ndarray as a point.

### Fixed

- No longer panics when a coordinate is NaN; the distance from a point with
  `NaN` in its coordinate and another point is considered greater than the
  distance between any two points without `NaN` in their coordinates.
- `BallTree::query` returns empty vectors, rather than panics, when `k` is zero.

## [0.4.0] - 2020-06-01

### Added

- A vantage point tree data structure to find nearest points.
- `BallTree` accepts not only an `f64` array but also an `f32`one.
- `BallTree::euclidean` to create a ball tree withoug having to pass a distance
  metric as an argument.

### Changed

- The codinates of each point must be stored in a contiguous area in memory.
- A distance metric is now a function, not a trait.

## [0.3.0] - 2020-04-17

### Changed

- The ownership of the input can be transferred to `BallTree`, which accepts
  both an owned array and a view.
- An error is returned, rather than a panic, if an empty array is given to
  construct a `BallTree`.
- `query_one` has been renamed `query_nearest`.
- `query` returns indices and distances separately, so that data of the same
  type are stored together.

## [0.2.0] - 2020-04-09

### Changed

- `BallTree` takes `ArrayBase` as its input, instead of `ArrayView`, to allow
  more types in ndarray.

## [0.1.0] - 2019-11-20

### Added

- A ball tree data structure to find nearest neighbors.

[0.5.0]: https://github.com/petabi/petal-neighbors/compare/0.4.0...0.5.0
[0.4.0]: https://github.com/petabi/petal-neighbors/compare/0.3.0...0.4.0
[0.3.0]: https://github.com/petabi/petal-neighbors/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/petabi/petal-neighbors/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/petabi/petal-neighbors/tree/0.1.0
