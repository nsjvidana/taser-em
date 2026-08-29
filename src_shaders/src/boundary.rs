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

macro_rules! pec_boundary {
    ($kernel_name:ident, $axis:ident) => {
        #[spirv_bindgen]
        #[cfg_attr(feature = "dim1", spirv(compute(threads(1, 1, 64))))]
        #[cfg_attr(feature = "dim2", spirv(compute(threads(8, 8, 1))))]
        #[cfg_attr(feature = "dim3", spirv(compute(threads(4, 4, 4))))]
        pub fn $kernel_name(
            #[spirv(global_invocation_id)] cell_idx3: UVec3,
            #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &GridParameters,
            // Vector fields
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] h: &mut [Vec4],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] dn: &mut [Vec4],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] en: &mut [Vec4],
        ) {
            let cell_idx = GridIndex::from_uvec3(cell_idx3);
            let n_cells = GridIndex::from_uvec3(grid.n_cells3);
            let lo_boundary = cell_idx3.$axis == 0;
            let hi_boundary = cell_idx3.$axis == (grid.n_cells3 - 1).$axis;
            let out_of_bounds = cell_idx3.cmpge(grid.n_cells3).any();
            if !(lo_boundary || hi_boundary) || out_of_bounds { return; }
        
            let idx = cell_idx.to_flat_idx(n_cells) as usize;
            h.write(idx, Vec4::ZERO);
            dn.write(idx, Vec4::ZERO);
            en.write(idx, Vec4::ZERO);
        }
    };
}

#[cfg(not(feature = "dim1"))]
pec_boundary!(gpu_pec_boundary_x, x);
#[cfg(not(feature = "dim1"))]
pec_boundary!(gpu_pec_boundary_y, y);
#[cfg(not(feature = "dim2"))]
pec_boundary!(gpu_pec_boundary_z, z);

macro_rules! periodic_boundary {
    ($en_name:ident, $h_name:ident, $axis:ident, $with_axis:ident) => {
        #[spirv_bindgen]
        #[cfg_attr(feature = "dim1", spirv(compute(threads(1, 1, 64))))]
        #[cfg_attr(feature = "dim2", spirv(compute(threads(8, 8, 1))))]
        #[cfg_attr(feature = "dim3", spirv(compute(threads(4, 4, 4))))]
        pub fn $en_name(
            #[spirv(global_invocation_id)] cell_idx3: UVec3,
            #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &GridParameters,
            // Vector fields
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] en: &mut [Vec4],
        ) {
            let last_idx = grid.n_cells3.$axis - 1;
            let hi_boundary = cell_idx3.$axis == last_idx;
            let out_of_bounds = cell_idx3.cmpge(grid.n_cells3).any();
            if !hi_boundary || out_of_bounds { return; }

            let n_cells = GridIndex::from_uvec3(grid.n_cells3);
            let cell_idx = GridIndex::from_uvec3(cell_idx3);
            let idx = cell_idx.to_flat_idx(n_cells) as usize;

            let en_lo = en.read(idx - (grid.flat_idx_incrs.$axis * (last_idx - 1)) as usize);
            en.write(idx, en_lo);
        }
        
        #[spirv_bindgen]
        #[cfg_attr(feature = "dim1", spirv(compute(threads(1, 1, 64))))]
        #[cfg_attr(feature = "dim2", spirv(compute(threads(8, 8, 1))))]
        #[cfg_attr(feature = "dim3", spirv(compute(threads(4, 4, 4))))]
        pub fn $h_name(
            #[spirv(global_invocation_id)] cell_idx3: UVec3,
            #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &GridParameters,
            // Vector fields
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] h: &mut [Vec4],
        ) {
            let lo_boundary = cell_idx3.$axis == 0;
            let out_of_bounds = cell_idx3.cmpge(grid.n_cells3).any();
            if !lo_boundary || out_of_bounds { return; }

            let n_cells = GridIndex::from_uvec3(grid.n_cells3);
            let cell_idx = GridIndex::from_uvec3(cell_idx3);
            let idx = cell_idx.to_flat_idx(n_cells) as usize;

            let h_hi = h.read(idx + (grid.flat_idx_incrs.$axis * (grid.n_cells3.$axis - 2)) as usize);
            h.write(idx, h_hi);
        }
    };
}

#[cfg(not(feature = "dim1"))]
periodic_boundary!(gpu_periodic_x_en, gpu_periodic_x_h, x, with_x);
#[cfg(not(feature = "dim1"))]
periodic_boundary!(gpu_periodic_y_en, gpu_periodic_y_h, y, with_y);
#[cfg(not(feature = "dim2"))]
periodic_boundary!(gpu_periodic_z_en, gpu_periodic_z_h, z, with_z);