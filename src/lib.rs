mod ball_tree;
pub mod distance;
mod vantage_point_tree;

pub use ball_tree::BallTree;
use thiserror::Error;
pub use vantage_point_tree::VantagePointTree;

/// The error type for input arrays.
#[derive(Debug, Error)]
pub enum ArrayError {
    #[error("array is empty")]
    Empty,
    #[error("array is not contiguous in memory")]
    NotContiguous,
}
