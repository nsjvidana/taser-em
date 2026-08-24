use crate::fdtd::GridParameters;
use crate::math::*;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::*;

#[spirv_bindgen]
#[cfg_attr(feature = "dim1", spirv(compute(threads(1, 1, 64))))]
#[cfg_attr(feature = "dim2", spirv(compute(threads(8, 8, 1))))]
#[cfg_attr(feature = "dim3", spirv(compute(threads(4, 4, 4))))]
pub fn gpu_pec_boundary(
    #[spirv(global_invocation_id)] idx3: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &GridParameters,
    // Vector fields
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] h: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] dn: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] en: &mut [Vec4],
) {
    let cell_idx = GridIndex::from_uvec3(idx3);
    let n_cells = GridIndex::from_uvec3(grid.n_cells3);
    let lo_boundary = cell_idx.cmpeq(GridIndex::ZERO).any();
    let hi_boundary = cell_idx.cmpeq(n_cells - 1).any();
    let out_of_bounds = idx3.cmpge(grid.n_cells3).any();
    if !(lo_boundary || hi_boundary) || out_of_bounds { return; }

    let idx = cell_idx.to_flat_idx(n_cells) as usize;
    h.write(idx, Vec4::ZERO);
    dn.write(idx, Vec4::ZERO);
    en.write(idx, Vec4::ZERO);
}