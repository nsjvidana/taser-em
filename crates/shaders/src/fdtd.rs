use bytemuck::{Pod, Zeroable};
use khal_std::glamx::{UVec3, UVec4, Vec3, Vec4};
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use crate::math::{grid_index_to_flat_idx, uvec3_to_grid_index, DIM, Axis, saturating_sub, SpatialAxis, MAX_DIM};
use crate::thread_id_to_3d_grid_index;

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
#[cfg_attr(feature = "dim1", spirv(compute(threads(64))))]
#[cfg_attr(feature = "dim2", spirv(compute(threads(8, 8))))]
#[cfg_attr(feature = "dim3", spirv(compute(threads(4, 4, 4))))]
pub fn fdtd_lossy(
    #[spirv(global_invocation_id)] id3: UVec3,
    // Vector fields
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] h: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dn: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] en: &mut [Vec4],
    // Field update terms
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] int_terms: &mut [IntegrationTerms],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] grid_coeffs: &[PmlCoefficients2],
    // Sources
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] dipoles: &[GpuDipole],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] source_vals: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] steps: &mut u32,
    // Uniforms
    #[spirv(uniform, descriptor_set = 0, binding = 8)] grid: &GridParameters2,
) {
    let idx3 = thread_id_to_3d_grid_index(id3);
    if idx3.cmpge(grid.n_cells3).any() { return; }
    let n_cells = uvec3_to_grid_index(grid.n_cells3);
    let boundary_idx3 = grid.n_cells3 - 1;
    // TODO: use usize indexing once indexing glam vectors is fixed? (see https://github.com/Rust-GPU/rust-gpu/issues/432)

    let idx = grid_index_to_flat_idx(uvec3_to_grid_index(idx3), n_cells) as usize;
    let PmlCoefficients2 {
        h_coeffs,
        dn_coeffs,
        en_coeffs,
    } = grid_coeffs.read(idx);
    let mut int = int_terms.read(idx);
    let en_cell = en.read(idx);
    // TODO: make these uniforms?
    let h_axes = if grid.polarization_mode.is_tm() { FIELD_AXES1 } else { FIELD_AXES2 };
    let dn_axes = if grid.polarization_mode.is_tm() { FIELD_AXES2 } else { FIELD_AXES1 };

    // Dipole sources (working)
    let steps_usize = *steps as usize;
    let mut source_term = 0.;
    for i in 0..dipoles.len() {
        let GpuDipole {
            vals_range: [start, end],
            cell_idx,
            t_start,
        } = dipoles.read(i);
        let t_start = t_start as usize;
        let t = saturating_sub(steps_usize, t_start);
        let [start, end] = [start as usize, end as usize];
        let vals_i = start + t;
        let enable = cell_idx as usize == idx && steps_usize >= t_start && vals_i <= end;
        source_term += source_vals.read(vals_i.min(end)) * enable as u32 as f32;
    }
    // TODO: More source injection (plane wave, etc.)

    // H Update
    let h_cell = {
        let not_boundary = UVec3::from(idx3.cmplt(boundary_idx3)).as_vec3();
        let h_cell = h.read(idx);
        let mut new_h = Vec4::ZERO;
        for axis_idx in h_axes.clone() {
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
            // Source injection
            new_h_cmp += source_term * grid.polarization_mode.is_te() as u32 as f32;
            new_h[axis] = new_h_cmp;
        }
        h.write(idx, new_h);
        new_h
    };

    // Dn Update
    let dn_cell = {
        let not_boundary = UVec3::from(idx3.cmpgt(UVec3::ZERO)).as_vec3();
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
            // Source injection
            new_dn_cmp += source_term * grid.polarization_mode.is_tm() as u32 as f32;
            new_dn[axis] = new_dn_cmp;
        }
        dn.write(idx, new_dn);
        new_dn
    };
    int_terms.write(idx, int);

    // En Update
    let mut en_cell = en_cell;
    for axis_idx in dn_axes {
        let axis = unsafe { Axis::from_index_unchecked(axis_idx as u32) };
        en_cell[axis] = en_coeffs[axis_idx] * dn_cell[axis];
    }
    en.write(idx, en_cell);

    if idx3 == UVec3::ZERO {
        *steps += 1;
    }
}

/// Information describing the grid
#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct GridParameters2 {
    /// Increments by 1 in i/j/k indices for flat indexing. Used for accessing
    /// data from neighboring cells.
    pub flat_idx_incrs: UVec3,
    pub _padding0: u32,
    pub n_cells3: UVec3,
    pub _padding1: u32,
    /// Spatial differentials (cell size)
    pub d: Vec3,
    pub _padding2: u32,
    pub polarization_mode: PolarizationModeIndex,
}

#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct PolarizationModeIndex(UVec4);

impl PolarizationModeIndex {
    /// Is Transverse Magnetic
    #[inline]
    pub fn is_tm(&self) -> bool { self.0.x == 0 }
    /// Is Transverse Electric
    #[inline]
    pub fn is_te(&self) -> bool { self.0.x == 1 }

    #[inline]
    pub unsafe fn from_idx_unchecked(idx: u32) -> Self { Self(UVec4::splat(idx)) }
}

#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct AxisIndex(u32);

impl core::ops::Deref for AxisIndex {
    type Target = u32;

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl From<Axis> for AxisIndex {
    #[inline]
    fn from(axis: Axis) -> AxisIndex {
        AxisIndex(axis as u32)
    }
}

/// Update coefficients for H, D, and E fields with a UPML
#[derive(Copy, Clone, Pod, Zeroable, Default, Debug)]
#[repr(C)]
pub struct PmlCoefficients2 {
    pub h_coeffs: [[f32; 2 + DIM - 1]; MAX_DIM],
    pub dn_coeffs: [[f32; 4 + DIM - 1]; MAX_DIM],
    pub en_coeffs: [f32; MAX_DIM],
}

/// An electric dipole source
#[derive(Copy, Clone, Pod, Zeroable, Default, Debug)]
#[repr(C)]
pub struct GpuDipole {
    pub cell_idx: u32,
    pub vals_range: [u32; 2],
    pub t_start: u32,
    // TODO: pub repeat_count: u32,
}

/// Integration terms used in updating H and D fields
///
/// H field technically has zero integration terms in 1D, but we can't have zero-sized arrays in
/// Spir-V, so the H field has one integration term for each dimension.
#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct IntegrationTerms {
    pub h: [[f32; if DIM == 1 { 1 } else { DIM - 1 }]; MAX_DIM],
    pub dn: [[f32; DIM]; MAX_DIM],
}