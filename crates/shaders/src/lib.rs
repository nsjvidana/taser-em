#![cfg_attr(target_arch = "spirv", no_std)]
#![allow(clippy::too_many_arguments)]

pub mod fdtd1;
pub mod math;
pub mod fdtd;

use khal_std::glamx::UVec3;

#[inline]
pub fn thread_id_to_3d_grid_index(id3: UVec3) -> UVec3 {
    #[cfg(feature = "dim1")]
    { UVec3::new(0, 0, id3.x) }
    #[cfg(not(feature = "dim1"))]
    id3
}