# Changelog

This file documents recent notable changes to this project. The format of this
file is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and
this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/petabi/petal-neighbors/compare/0.2.0...master
[0.2.0]: https://github.com/petabi/petal-neighbors/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/petabi/petal-neighbors/tree/0.1.0
