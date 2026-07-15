#![allow(clippy::needless_range_loop)]

use bytemuck::{Pod, Zeroable};
use khal_std::glamx::{UVec3, Vec3, Vec4};
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use crate::math::{grid_index_to_flat_idx, uvec3_to_grid_index, DIM, Axis, saturating_sub, SpatialAxis, MAX_DIM, Real};

// TODO: Docs, remove the "2" from the struct names, delete fdtd1 module, try using Vect for vector field
#[spirv_bindgen]
#[cfg_attr(feature = "dim1", spirv(compute(threads(1, 1, 64))))]
#[cfg_attr(feature = "dim2", spirv(compute(threads(8, 8))))]
#[cfg_attr(feature = "dim3", spirv(compute(threads(4, 4, 4))))]
pub fn fdtd_lossy(
    #[spirv(global_invocation_id)] idx3: UVec3,
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
    let h_axes = grid.polarization_mode.get_h_axes();
    let dn_axes = grid.polarization_mode.get_dn_axes();

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
        for i in 0..MAX_DIM {
            let axis = h_axes[i];
            if axis == Axis::INVALID { break; }
            let axis_idx = AxisIndex::from_axis(axis);
            let axis1 = axis.permute();
            let axis2 = axis1.permute();
            let de2_d1 = if SpatialAxis::is_spatial_axis(axis1) {
                let neighbor_idx = (idx + grid.flat_idx_incrs[axis1] as usize).min(en.len() - 1);
                let en_neighbor = en.read(neighbor_idx) * not_boundary[axis1];
                (en_neighbor[axis2] - en_cell[axis2]) / grid.d[axis1]
            } else { 0. };
            let de1_d2 = if SpatialAxis::is_spatial_axis(axis2) {
                let neighbor_idx = (idx + grid.flat_idx_incrs[axis2] as usize).min(en.len() - 1);
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
        for i in 0..MAX_DIM {
            let axis = dn_axes[i];
            if axis == Axis::INVALID { break; }
            let axis_idx = AxisIndex::from_axis(axis);
            let axis1 = axis.permute();
            let axis2 = axis1.permute();
            let de2_d1 = if SpatialAxis::is_spatial_axis(axis1) {
                let neighbor_idx = idx.wrapping_sub(grid.flat_idx_incrs[axis1] as usize).min(h.len() - 1);
                let h_neighbor = h.read(neighbor_idx) * not_boundary[axis1];
                (h_cell[axis2] - h_neighbor[axis2]) / grid.d[axis1]
            } else { 0. };
            let de1_d2 = if SpatialAxis::is_spatial_axis(axis2) {
                let neighbor_idx = idx.wrapping_sub(grid.flat_idx_incrs[axis2] as usize).min(h.len() - 1);
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
    for i in 0..MAX_DIM {
        let axis = dn_axes[i];
        if axis == Axis::INVALID { break; }
        let axis_idx = AxisIndex::from_axis(axis);
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
    pub polarization_mode: GpuPolarizationMode,
}

#[derive(Copy, Clone, Pod, Zeroable, Eq, PartialEq)]
#[repr(C)]
pub struct GpuPolarizationMode {
    h_axes: UVec3,
    mode: u32,
    dn_axes: UVec3,
    _p0: u32,
}

impl GpuPolarizationMode {
    const FIELD_AXES1: [Axis; MAX_DIM] = cfg_select! {
        feature = "dim1" => [Axis::X, Axis::INVALID, Axis::INVALID],
        feature = "dim2" => [Axis::X, Axis::Y, Axis::INVALID],
        feature = "dim3" => [Axis::X, Axis::Y, Axis::Z],
    };
    const FIELD_AXES2: [Axis; MAX_DIM] = cfg_select! {
        feature = "dim1" => [Axis::Y, Axis::INVALID, Axis::INVALID],
        feature = "dim2" => [Axis::Z, Axis::INVALID, Axis::INVALID],
        feature = "dim3" => Self::FIELD_AXES1,
    };
    pub const TM: Self = Self {
        h_axes: unsafe { core::mem::transmute::<[Axis; MAX_DIM], UVec3>(Self::FIELD_AXES1) },
        dn_axes: unsafe { core::mem::transmute::<[Axis; MAX_DIM], UVec3>(Self::FIELD_AXES2) },
        mode: 0,
        _p0: 0,
    };
    pub const TE: Self = Self {
        h_axes: unsafe { core::mem::transmute::<[Axis; MAX_DIM], UVec3>(Self::FIELD_AXES2) },
        dn_axes: unsafe { core::mem::transmute::<[Axis; MAX_DIM], UVec3>(Self::FIELD_AXES1) },
        mode: 1,
        _p0: 0
    };

    /// Is Transverse Magnetic
    #[inline]
    pub const fn is_tm(&self) -> bool {
        self.mode == 0
    }

    /// Is Transverse Electric
    #[inline]
    pub const fn is_te(&self) -> bool {
        self.mode == 1
    }

    #[inline]
    pub const fn get_h_axes(&self) -> [Axis; MAX_DIM] {
        unsafe { core::mem::transmute::<[u32; MAX_DIM], [Axis; MAX_DIM]>(self.h_axes.to_array()) }
    }

    #[inline]
    pub const fn get_dn_axes(&self) -> [Axis; MAX_DIM] {
        unsafe { core::mem::transmute::<[u32; MAX_DIM], [Axis; MAX_DIM]>(self.dn_axes.to_array()) }
    }
}

impl Default for GpuPolarizationMode {
    fn default() -> Self {
        Self::TM
    }
}

#[derive(Copy, Clone, Pod, Zeroable, Default, Eq, PartialEq)]
#[repr(C)]
pub struct AxisIndex(u32);

impl AxisIndex {
    pub const INVALID: AxisIndex = AxisIndex(u32::MAX);
    #[inline]
    pub const fn from_axis(axis: Axis) -> Self {
        AxisIndex(axis as u32)
    }
    #[inline]
    pub const fn into_axis(self) -> Axis {
        unsafe { core::mem::transmute::<u32, Axis>(self.0) }
    }

    #[inline]
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl core::ops::Deref for AxisIndex {
    type Target = u32;

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<T, const N: usize> core::ops::Index<AxisIndex> for [T; N] {
    type Output = T;
    fn index(&self, index: AxisIndex) -> &Self::Output {
        &self[index.as_usize()]
    }
}

impl<T, const N: usize> core::ops::IndexMut<AxisIndex> for [T; N] {
    fn index_mut(&mut self, index: AxisIndex) -> &mut Self::Output {
        &mut self[index.as_usize()]
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

/// ===========================================================================================================================
#[spirv_bindgen]
#[spirv(compute(threads(1, 1, 64)))]
pub fn fdtd_lossy_v2(
    #[spirv(global_invocation_id)] idx3: UVec3,
    // Vector fields
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] h: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dn: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] en: &mut [Vec4],
    // Field update terms
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] integrals: &mut [IntegrationTerms],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] grid_coeffs: &[PmlCoefficients2],
    // Sources
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] dipoles: &[GpuDipole],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] source_vals: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] steps: &mut u32,
    // Uniforms
    #[spirv(uniform, descriptor_set = 0, binding = 8)] grid: &GridParameters2,
) {
    if idx3.cmpge(grid.n_cells3).any() { return; }
    let idx = grid_index_to_flat_idx(uvec3_to_grid_index(idx3), uvec3_to_grid_index(grid.n_cells3))
        as usize;
    let boundary_idx3 = grid.n_cells3 - 1;

    // Get the source value at this timestep (working)
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

    let PmlCoefficients2 {
        h_coeffs, dn_coeffs, en_coeffs
    } = grid_coeffs.read(idx);
    let mut int_terms = integrals.read(idx);

    let h_axes = grid.polarization_mode.get_h_axes();
    let dn_axes = grid.polarization_mode.get_dn_axes();

    let en_self = en.read(idx);

    // H update
    let h_self = {
        let mut h_self = h.read(idx);
        let not_boundary = UVec3::from(idx3.cmplt(boundary_idx3)).as_vec3();
        for i in 0..MAX_DIM {
            let h_axis = h_axes[i];
            if h_axis == Axis::INVALID { break; }

            let en_curl = compute_curl::<true>(
                h_axis,
                grid.d,
                idx,
                not_boundary,
                grid.flat_idx_incrs,
                en_self,
                en
            );

            let h_axis_i = h_axis as usize;
            let [m0, m1, ..] = h_coeffs[h_axis_i];
            #[allow(unused_mut)]
            let mut h_cmp_new = m0 * h_self[h_axis] + m1 * en_curl +
                source_term * grid.polarization_mode.is_te() as u32 as f32;
            #[cfg(any(feature = "dim2", feature = "dim3"))]
            {
                int_terms.h[h_axis_i][0] += en_curl;
                h_cmp_new += h_coeffs[h_axis_i][2] * int_terms.h[h_axis_i][0];
                #[cfg(feature = "dim3")]
                {
                    int_terms.h[h_axis_i][1] += h_self[h_axis];
                    h_cmp_new += h_coeffs[h_axis_i][3] * int_terms.h[h_axis_i][1];
                }
            }
            h_self[h_axis] = h_cmp_new;
        }
        h_self
    };
    h.write(idx, h_self);

    let dn_self = {
        let mut dn_self = dn.read(idx);
        let not_boundary = UVec3::from(idx3.cmpgt(UVec3::ZERO)).as_vec3();
        for i in 0..MAX_DIM {
            let dn_axis = dn_axes[i];
            if dn_axis == Axis::INVALID { break; }

            let h_curl = compute_curl::<false>(
                dn_axis,
                grid.d,
                idx,
                not_boundary,
                grid.flat_idx_incrs,
                h_self,
                h
            );

            let dn_axis_i = dn_axis as usize;
            int_terms.dn[dn_axis_i][0] += en_self[dn_axis];
            let [m0, m1, m2, m3, ..] = dn_coeffs[dn_axis_i];
            #[allow(unused_mut)]
            let mut dn_cmp_new = m0 * dn_self[dn_axis] + m1 * h_curl + // regular update terms
                m2 * en_self[dn_axis] + m3 * int_terms.dn[dn_axis_i][0] + // loss terms
                source_term * grid.polarization_mode.is_tm() as u32 as f32;
            #[cfg(any(feature = "dim2", feature = "dim3"))]
            {
                int_terms.dn[dn_axis_i][1] += h_curl;
                dn_cmp_new += dn_coeffs[dn_axis_i][4] * int_terms.dn[dn_axis_i][1];
                #[cfg(feature = "dim3")]
                {
                    int_terms.dn[dn_axis_i][2] += dn_self[dn_axis];
                    dn_cmp_new += dn_coeffs[dn_axis_i][5] * int_terms.dn[dn_axis_i][2];
                }
            }
            dn_self[dn_axis] = dn_cmp_new;
        }
        dn_self
    };
    dn.write(idx, dn_self);

    let mut en_self = en_self;
    for i in 0..MAX_DIM {
        let dn_axis = dn_axes[i];
        if dn_axis == Axis::INVALID { break; }
        en_self[dn_axis] = en_coeffs[dn_axis as usize] * dn_self[dn_axis];
        en.write(idx, en_self);
    }

    integrals.write(idx, int_terms);

    if idx3 == UVec3::ZERO {
        *steps += 1;
    }
}

/// Forwards & backwards component-wise curl operator
#[inline]
fn compute_curl<const FORWARDS: bool>(
    axis: Axis,
    d: Vec3,
    idx: usize,
    not_boundary: Vec3,
    flat_idx_incrs: UVec3,
    vect_self: Vec4,
    vect_field: &[Vec4],
) -> Real {
    let axis1 = axis.permute();
    let axis2 = axis1.permute();
    let curl_term1 = if SpatialAxis::is_spatial_axis(axis1) {
        let neighbor_idx = 
            if FORWARDS { idx + flat_idx_incrs[axis1] as usize }
            else { idx.wrapping_sub(flat_idx_incrs[axis1] as usize) }
                .min(vect_field.len() - 1);
        let vect_neighbor = vect_field.read(neighbor_idx) * not_boundary[axis1];

        if FORWARDS { (vect_neighbor[axis2] - vect_self[axis2]) / d[axis1] }
        else { (vect_self[axis2] - vect_neighbor[axis2]) / d[axis1] }
    } else { 0. };
    let curl_term2 = if SpatialAxis::is_spatial_axis(axis2) {
        let neighbor_idx =
            if FORWARDS { idx + flat_idx_incrs[axis2] as usize }
            else { idx.wrapping_sub(flat_idx_incrs[axis2] as usize) }
                .min(vect_field.len() - 1);
        let vect_neighbor = vect_field.read(neighbor_idx) * not_boundary[axis2];

        if FORWARDS { (vect_neighbor[axis1] - vect_self[axis1]) / d[axis2] }
        else { (vect_self[axis1] - vect_neighbor[axis1]) / d[axis2] }
    } else { 0. };
    curl_term1 - curl_term2
}