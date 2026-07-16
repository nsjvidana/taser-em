pub use taser_em_shaders::math::*;
pub use crate::*;
pub use crate::grid::*;

use khal::backend::GpuBackendError;

pub type GpuResult<T> = core::result::Result<T, GpuBackendError>;