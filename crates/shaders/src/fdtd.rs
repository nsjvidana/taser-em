use bytemuck::{Pod, Zeroable};
use khal_std::glamx::{UVec3, Vec3, Vec4};
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use crate::math::{grid_index_to_flat_idx, uvec3_to_grid_index, DIM, Axis, saturating_sub, SpatialAxis};

/// The axes in which "field 1" exist in, depending on the dimension & polarization mode.
/// For example, in 1D with TM polarization
const FIELD_AXES1: core::ops::RangeInclusive<usize> = cfg_select! {
    feature = "dim1" => (Axis::X as usize)..=(Axis::X as usize),
    feature = "dim2" => (Axis::X as usize)..=(Axis::Y as usize),
    feature = "dim3" => (Axis::X as usize)..=(Axis::Z as usize),
};

const FIELD_AXES2: core::ops::RangeInclusive<usize> = cfg_select! {
    feature = "dim1" => (Axis::Y as usize)..=(Axis::Y as usize),
    feature = "dim2" => (Axis::Z as usize)..=(Axis::Z as usize),
    feature = "dim3" => FIELD_AXES1,
};

// TODO: Docs, remove the "2" from the struct names, delete fdtd1 module, try using Vect for vector field
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn fdtd_lossy(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] h: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dn: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] en: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] int_terms: &mut [IntegrationTerms],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] grid_coeffs: &[PmlCoefficients2],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] grid: &GridParameters2,
) {
    let id3 = id;
    if id3.cmpge(grid.n_cells).any() { return; }
    let n_cells = uvec3_to_grid_index(grid.n_cells);
    let boundary_idx3 = grid.n_cells - 1;
    // TODO: use usize indexing once indexing glam vectors is fixed? (see https://github.com/Rust-GPU/rust-gpu/issues/432)

    let id = uvec3_to_grid_index(id3);
    let idx = grid_index_to_flat_idx(id, n_cells) as usize;
    let PmlCoefficients2 {
        h_coeffs,
        dn_coeffs,
        en_coeffs,
    } = grid_coeffs.read(idx);
    let mut int = int_terms.read(idx);
    let en_cell = en.read(idx);

    // H Update
    let h_cell = {
        // TODO: set this to a "if cfg!()", use consts, and compare the performance.
        let h_axes = if grid.polarization_mode == 0 { FIELD_AXES1 } else { FIELD_AXES2 };
        let not_boundary = UVec3::from(id3.cmplt(boundary_idx3)).as_vec3();
        let h_cell = h.read(idx);
        let mut new_h = Vec4::ZERO;
        for axis_idx in h_axes {
            let axis = unsafe { Axis::from_index_unchecked(axis_idx as u32) };
            let axis1 = axis.permute();
            let axis2 = axis1.permute();
            let de2_d1 = if SpatialAxis::is_spatial_axis(axis1) {
                let neighbor_idx = (idx + grid.flat_idx_incrs[axis1] as usize).max(en.len() - 1);
                let en_neighbor = en.read(neighbor_idx) * not_boundary[axis1];
                (en_neighbor[axis2] - en_cell[axis2]) / grid.d[axis1]
            } else { 0. };
            let de1_d2 = if SpatialAxis::is_spatial_axis(axis2) {
                let neighbor_idx = (idx + grid.flat_idx_incrs[axis2] as usize).max(en.len() - 1);
                let en_neighbor = en.read(neighbor_idx) * not_boundary[axis2];
                (en_neighbor[axis1] - en_cell[axis1]) / grid.d[axis2]
            } else { 0. };
            let en_curl_axis = de2_d1 - de1_d2;
            
            // Update Equations
            let coeffs = h_coeffs[axis_idx];
            #[allow(unused_mut)]
            let mut new_h_cmp = coeffs[0] * h_cell[axis] +
                coeffs[1] * en_curl_axis;
            // integration terms
            #[cfg(any(feature = "dim2", feature = "dim3"))]
            {
                int.h[axis_idx][0] += en_curl_axis;
                new_h_cmp += coeffs[2] * int.h[axis_idx][0];
                #[cfg(feature = "dim3")]
                {
                    int.h[axis_idx][1] += h_cell[axis];
                    new_h_cmp += coeffs[3] * int.h[axis_idx][1];
                }
            }
            new_h[axis] = new_h_cmp;
        }
        h.write(idx, new_h);
        new_h
    };

    // Dn Update
    let dn_axes = if grid.polarization_mode == 0 { FIELD_AXES2 } else { FIELD_AXES1 };
    let dn_cell = {
        // TODO: set this to a "if cfg!()", use consts, and compare the performance.
        let not_boundary = UVec3::from(id3.cmpgt(UVec3::ZERO)).as_vec3();
        let dn_cell = dn.read(idx);
        let mut new_dn = Vec4::ZERO;
        for axis_idx in dn_axes.clone() {
            let axis = unsafe { Axis::from_index_unchecked(axis_idx as u32) };
            let axis1 = axis.permute();
            let axis2 = axis1.permute();
            let de2_d1 = if SpatialAxis::is_spatial_axis(axis1) {
                let neighbor_idx = saturating_sub(idx, grid.flat_idx_incrs[axis1] as usize);
                let h_neighbor = h.read(neighbor_idx) * not_boundary[axis1];
                (h_cell[axis2] - h_neighbor[axis2]) / grid.d[axis1]
            } else { 0. };
            let de1_d2 = if SpatialAxis::is_spatial_axis(axis2) {
                let neighbor_idx = saturating_sub(idx, grid.flat_idx_incrs[axis2] as usize);
                let en_neighbor = h.read(neighbor_idx) * not_boundary[axis2];
                (h_cell[axis1] - en_neighbor[axis1]) / grid.d[axis2]
            } else { 0. };
            let h_curl_axis = de2_d1 - de1_d2;

            // Update Equations
            let coeffs = dn_coeffs[axis_idx];
            int.dn[axis_idx][0] += en_cell[axis];
            #[allow(unused_mut)]
            let mut new_dn_cmp = coeffs[0] * dn_cell[axis] +
                coeffs[1] * h_curl_axis +
                coeffs[2] * en_cell[axis] +
                coeffs[3] * int.dn[axis_idx][0];
            // integration terms
            #[cfg(any(feature = "dim2", feature = "dim3"))]
            {
                int.dn[axis_idx][1] += h_curl_axis;
                new_dn_cmp += coeffs[4] * int.dn[axis_idx][1];
                #[cfg(feature = "dim3")]
                {
                    int.dn[axis_idx][2] += dn_cell[axis];
                    new_dn_cmp += coeffs[5] * int.dn[axis_idx][2];
                }
            }
            new_dn[axis] = new_dn_cmp;
        }
        dn.write(idx, new_dn);
        new_dn
    };
    int_terms.write(idx, int);

    // TODO: Source injection (dipole, plane wave, etc.)

    // En Update
    let mut en_cell = en_cell;
    for axis_idx in dn_axes {
        let axis = unsafe { Axis::from_index_unchecked(axis_idx as u32) };
        en_cell[axis] = en_coeffs[axis_idx] * dn_cell[axis];
    }
    en.write(idx, en_cell);

    // TODO: make another update here for TE mode?
}

/// Information describing the grid
#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct GridParameters2 {
    /// Increments by 1 in i/j/k indices for flat indexing. Used for accessing
    /// data from neighboring cells.
    pub flat_idx_incrs: UVec3,
    // TODO: make this use a newtype that ensures the u32 is valid.
    pub polarization_mode: u32,
    pub n_cells: UVec3,
    pub _padding0: u32,
    /// Spatial differentials (cell size)
    pub d: Vec3,
    pub _padding1: u32,
}
/// Update coefficients for H, D, and E fields with a UPML
#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct PmlCoefficients2 {
    pub h_coeffs: [[f32; 2 + DIM - 1]; DIM],
    pub dn_coeffs: [[f32; 4 + DIM - 1]; DIM],
    pub en_coeffs: [f32; DIM],
}

/// Integration terms used in updating H and D fields
///
/// H field technically has zero integration terms in 1D, but we can't have zero-sized arrays in
/// Spir-V, so the H field has one integration term for each dimension.
#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct IntegrationTerms {
    pub h: [[f32; if DIM == 1 { 1 } else { DIM - 1 }]; DIM],
    pub dn: [[f32; DIM]; DIM],
}