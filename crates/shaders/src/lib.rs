#![cfg_attr(target_arch = "spirv", no_std)]
#![allow(clippy::too_many_arguments)]

pub mod fdtd1;
pub mod math;
pub mod fdtd;

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

// TODO: remove this test shader
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn add_assign(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] b: &[f32],
) {
    let thread_id = invocation_id.x as usize;
    if thread_id < a.len() && thread_id < b.len() {
        *a.at_mut(thread_id) += b.read(thread_id);
    }
}

#[inline]
pub fn thread_id_to_3d_grid_index(id3: UVec3) -> UVec3 {
    #[cfg(feature = "dim1")]
    { UVec3::new(0, 0, id3.x) }
    #[cfg(not(feature = "dim1"))]
    id3
}