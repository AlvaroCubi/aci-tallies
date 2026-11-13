//! What the readers of every format share.

pub mod axis;
pub(crate) mod float;
pub mod input;
pub(crate) mod scan;
pub mod select;
pub mod source;
pub(crate) mod values;
pub mod warning;

pub use axis::{Axis, Bins};
pub use input::Input;
pub use select::{OpsError, Select};
pub use source::{ReadError, Source};
pub use warning::Warning;
