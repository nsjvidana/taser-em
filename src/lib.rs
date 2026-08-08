pub mod fdtd;
pub mod grid;
pub mod consts {
    /// Speed of EM wave in free space
    pub const C_0: f32 = 299792458.0;
    /// Free space permittivity
    pub const EPS_0: f32 = 8.854188e-12;
    /// Free space permeability
    pub const MU_0: f32 = 1.256637e-6;
    /// Free space wave impedance
    pub const IMPEDANCE_0: f32 = 376.73032;
}
pub mod util;
pub mod gpu_util;

pub mod re_exports {
    pub use glamx;
    pub use khal;
    pub use anyhow;
}

pub use taser_em_shaders as shaders;

use khal::re_exports::include_dir::{include_dir, Dir};

pub static SPIRV_DIR: Dir<'static> = include_dir!("$OUT_DIR/shaders-spirv");

// TODO: make an FdtdSolver trait?

pub mod prelude {
    pub use crate::fdtd::*;
    pub use crate::grid::*;
    pub use crate::consts::*;
    pub use crate::util::*;
    pub use taser_em_shaders::math::*;
    pub use khal::backend::*;
    pub use khal::shader::Shader;

    pub type GpuResult<T> = core::result::Result<T, GpuBackendError>;
}