#![allow(clippy::needless_range_loop)]

use bytemuck::{Pod, Zeroable};
use khal_std::glamx::{UVec3, Vec3, Vec4};
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use crate::math::{saturating_sub, Axis, GridIndex, GridIndexExt, Real, SpatialAxis, MAX_DIM};

/// Information describing the grid
#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct GridParameters {
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

/// An electric dipole source
#[derive(Copy, Clone, Pod, Zeroable, Default, Debug)]
#[repr(C)]
pub struct GpuDipole {
    pub cell_idx: u32,
    pub vals_range: [u32; 2],
    pub t_start: u32,
    // TODO: pub repeat_count: u32,
}

/// N-dimensional FDTD shader with loss (conductivity)
// TODO: try using Vect for the vector fields
#[spirv_bindgen]
#[cfg_attr(feature = "dim1", spirv(compute(threads(1, 1, 64))))]
#[cfg_attr(feature = "dim2", spirv(compute(threads(8, 8, 1))))]
#[cfg_attr(feature = "dim3", spirv(compute(threads(4, 4, 4))))]
pub fn fdtd_lossy(
    #[spirv(global_invocation_id)] idx3: UVec3,
    // Vector fields
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] h: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dn: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] en: &mut [Vec4],
    // Field update terms
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] integrals: &mut [PmlIntegrals],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] grid_coeffs: &[PmlCoefficients],
    // Sources
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] dipoles: &[GpuDipole],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] source_vals: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] steps: &mut u32,
    // Uniforms
    #[spirv(uniform, descriptor_set = 0, binding = 8)] grid: &GridParameters,
) {
    if idx3.cmpge(grid.n_cells3).any() { return; }
    let idx = GridIndex::from_uvec3(idx3).to_flat_idx(GridIndex::from_uvec3(grid.n_cells3))
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

    let coeffs = grid_coeffs.read(idx);
    let mut ints = integrals.read(idx);

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

            #[allow(unused_mut)]
            let mut h_cmp_new = coeffs.h1[h_axis] * h_self[h_axis] + coeffs.h2[h_axis] * en_curl +
                source_term * grid.polarization_mode.is_te() as u32 as f32;
            #[cfg(any(feature = "dim2", feature = "dim3"))]
            {
                ints.en_curl[h_axis] += en_curl;
                h_cmp_new += coeffs.h3[h_axis] * ints.en_curl[h_axis];
                #[cfg(feature = "dim3")]
                {
                    ints.h[h_axis] += h_self[h_axis];
                    h_cmp_new += coeffs.h4[h_axis] * ints.h[h_axis];
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

            ints.en[dn_axis] += en_self[dn_axis];
            #[allow(unused_mut)]
            let mut dn_cmp_new = coeffs.dn1[dn_axis] * dn_self[dn_axis] + coeffs.dn2[dn_axis] * h_curl + // regular update terms
                coeffs.dn_loss1[dn_axis] * en_self[dn_axis] + coeffs.dn_loss2[dn_axis] * ints.en[dn_axis] + // loss terms
                source_term * grid.polarization_mode.is_tm() as u32 as f32;
            #[cfg(any(feature = "dim2", feature = "dim3"))]
            {
                ints.h_curl[dn_axis] += h_curl;
                dn_cmp_new += coeffs.dn3[dn_axis] * ints.h_curl[dn_axis];
                #[cfg(feature = "dim3")]
                {
                    ints.dn[dn_axis] += dn_self[dn_axis];
                    dn_cmp_new += coeffs.dn4[dn_axis] * ints.dn[dn_axis];
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
        en_self[dn_axis] = coeffs.en1[dn_axis] * dn_self[dn_axis];
        // en_self[dn_axis] = en_coeffs[dn_axis as usize] * dn_self[dn_axis];
    }
    en.write(idx, en_self);

    integrals.write(idx, ints);

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

/// Update coefficients for H, D, and E fields with a UPML
#[derive(Copy, Clone, Pod, Zeroable, Default, Debug)]
#[repr(C)]
pub struct PmlCoefficients {
    pub h1: Vec4,
    pub h2: Vec4,
    #[cfg(any(feature = "dim2", feature = "dim3"))]
    pub h3: Vec4,
    #[cfg(feature = "dim3")]
    pub h4: Vec4,

    pub dn1: Vec4,
    pub dn2: Vec4,
    pub dn_loss1: Vec4,
    pub dn_loss2: Vec4,
    #[cfg(any(feature = "dim2", feature = "dim3"))]
    pub dn3: Vec4,
    #[cfg(feature = "dim3")]
    pub dn4: Vec4,

    pub en1: Vec4,
}

/// Integration terms used in updating H and D fields
///
/// H field technically has zero integration terms in 1D, but we can't have zero-sized arrays in
/// Spir-V, so the H field has one integration term for each dimension.
#[derive(Copy, Clone, Pod, Zeroable, Default, Debug)]
#[repr(C)]
pub struct PmlIntegrals {
    pub en: Vec4, // used for loss

    #[cfg(any(feature = "dim2", feature = "dim3"))]
    pub en_curl: Vec4,
    #[cfg(feature = "dim3")]
    pub h: Vec4,

    #[cfg(any(feature = "dim2", feature = "dim3"))]
    pub h_curl: Vec4,
    #[cfg(feature = "dim3")]
    pub dn: Vec4,
}