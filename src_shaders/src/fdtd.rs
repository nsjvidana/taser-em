#![allow(clippy::needless_range_loop)]
use crate::math::*;
use bytemuck::{Pod, Zeroable};
use khal_std::glamx::{UVec3, Vec3, Vec4};
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

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
    let lo_boundary = cell_idx.cmpeq(GridIndex::zero()).any();
    let hi_boundary = cell_idx.cmpeq(n_cells - 1).any();
    let out_of_bounds = idx3.cmpge(grid.n_cells3).any();
    if !(lo_boundary || hi_boundary) || out_of_bounds { return; }

    let idx = cell_idx.to_flat_idx(GridIndex::from_uvec3(grid.n_cells3)) as usize;
    h.write(idx, Vec4::ZERO);
    dn.write(idx, Vec4::ZERO);
    en.write(idx, Vec4::ZERO);
}

#[spirv_bindgen]
#[cfg_attr(feature = "dim1", spirv(compute(threads(1, 1, 64))))]
#[cfg_attr(feature = "dim2", spirv(compute(threads(8, 8, 1))))]
#[cfg_attr(feature = "dim3", spirv(compute(threads(4, 4, 4))))]
pub fn gpu_compute_source_terms(
    #[spirv(global_invocation_id)] idx3: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &GridParameters,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] steps: &u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] source_terms: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] source_vals: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] dipoles: &[GpuDipole],
    // TODO: #[spirv(storage_buffer, descriptor_set = 0, binding = )] plane_waves: &[GpuPlaneWave],
) {
    let cell_idx = GridIndex::from_uvec3(idx3);
    let n_cells = GridIndex::from_uvec3(grid.n_cells3);
    if skip_update(cell_idx, n_cells, idx3, grid.n_cells3) { return; }

    let idx = cell_idx.to_flat_idx(n_cells) as usize;

    let mut source_term = Vec4::ZERO;
    let curr_step = *steps;

    // Dipoles
    for i in 0..dipoles.len() {
        let dipole = dipoles.read(i);
        let t = curr_step.gpu_saturating_sub(dipole.t_start);
        let vals_i = dipole.vals_start + t;

        let enable = (dipole.cell_idx as usize == idx) &&
            (curr_step >= dipole.t_start) &&
            (vals_i <= dipole.vals_end);
        source_term += source_vals.read(vals_i.min(dipole.vals_end) as usize) *
            enable as u32 as f32 *
            dipole.moment;
    }

    source_terms.write(idx, source_term);
}

/// N-dimensional FDTD shader with loss (conductivity). Works with any polarization mode.
#[spirv_bindgen]
#[cfg_attr(feature = "dim1", spirv(compute(threads(1, 1, 64))))]
#[cfg_attr(feature = "dim2", spirv(compute(threads(8, 8, 1))))]
#[cfg_attr(feature = "dim3", spirv(compute(threads(4, 4, 4))))]
pub fn gpu_lossy_update(
    #[spirv(global_invocation_id)] idx3: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &GridParameters,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] steps: &mut u32,
    // Vector fields
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] h: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dn: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] en: &mut [Vec4],
    // Field update terms
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] integrals: &mut [PmlIntegrals],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] grid_coeffs: &[PmlCoefficients],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] source_terms: &[Vec4],
) {
    let cell_idx = GridIndex::from_uvec3(idx3);
    let n_cells = GridIndex::from_uvec3(grid.n_cells3);
    if skip_update(cell_idx, n_cells, idx3, grid.n_cells3) { return; }

    let idx = cell_idx.to_flat_idx(n_cells) as usize;

    let source_term = source_terms.read(idx);
    let coeffs = grid_coeffs.read(idx);
    let mut ints = integrals.read(idx);

    let en_self = en.read(idx);
    let en_curl = compute_curl::<true>(
        grid.d,
        idx,
        grid.flat_idx_incrs,
        en_self,
        en
    );
    let h_self = h.read(idx);
    let mut h_self_new = coeffs.h1 * h_self + coeffs.h2 * en_curl;
        grid.polarization_mode_index.inject_h_source(&mut h_self_new, source_term);
    #[cfg(any(feature = "dim2", feature = "dim3"))]
    {
        ints.en_curl += en_curl;
        h_self_new += coeffs.h3 * ints.en_curl;
        #[cfg(feature = "dim3")]
        {
            ints.h += h_self;
            h_self_new += coeffs.h4 * ints.h;
        }
    };
    h.write(idx, h_self_new);
    let h_self = h_self_new;

    let h_curl = compute_curl::<false>(
        grid.d,
        idx,
        grid.flat_idx_incrs,
        h_self,
        h
    );
    let dn_self = dn.read(idx);
    ints.en += en_self;
    let mut dn_self_new = coeffs.dn1 * dn_self + coeffs.dn2 * h_curl +
        coeffs.dn_loss1 * en_self + coeffs.dn_loss2 * ints.en;
        grid.polarization_mode_index.inject_dn_source(&mut dn_self_new, source_term);
    #[cfg(any(feature = "dim2", feature = "dim3"))]
    {
        ints.h_curl += h_curl;
        dn_self_new += coeffs.dn3 * ints.h_curl;
        #[cfg(feature = "dim3")]
        {
            ints.dn += dn_self;
            dn_self_new += coeffs.dn4 * ints.dn
        }
    };
    dn.write(idx, dn_self_new);
    let dn_self = dn_self_new;

    let en_self_new = coeffs.en1 * dn_self;
    en.write(idx, en_self_new);

    integrals.write(idx, ints);

    if cell_idx == GridIndex::one() {
        *steps += 1;
    }
}

fn skip_update(cell_idx: GridIndex, n_cells: GridIndex, idx3: UVec3, n_cells3: UVec3) -> bool {
    let lo_boundary = cell_idx.cmpeq(GridIndex::zero()).any();
    let hi_boundary = cell_idx.cmpeq(n_cells - 1).any();
    let out_of_bounds = idx3.cmpge(n_cells3).any();
    lo_boundary || hi_boundary || out_of_bounds
}

/// Forwards & backwards discrete curl
fn compute_curl<const FORWARDS: bool>(
    d: Vec3,
    idx: usize,
    flat_idx_incrs: UVec3,
    v_self: Vec4,
    v: &[Vec4]
) -> Vec4 {
    let mut curl_cmps = [0.; 4];
    for i in 0..MAX_DIM {
        let axis = Axis::ALL_AXES[i];
        if axis == Axis::INVALID { break; } // Removing this causes instability (probably a rust-gpu issue)
        curl_cmps[i] = curl_component::<FORWARDS>(
            axis,
            d,
            idx,
            flat_idx_incrs,
            v_self,
            v,
        );
    }
    Vec4::from(curl_cmps)
}

/// Forwards & backwards component-wise curl operator
fn curl_component<const FORWARDS: bool>(
    axis: Axis,
    d: Vec3,
    idx: usize,
    flat_idx_incrs: UVec3,
    v_self: Vec4,
    v: &[Vec4],
) -> Real {
    let neighbors = get_curl_neighbors::<FORWARDS>(idx, flat_idx_incrs, v);
    let axis1 = axis.permute();
    let axis2 = axis1.permute();

    let curl_term1 = if SpatialAxis::is_spatial_axis(axis1) {
        (neighbors[axis1 as usize][axis2] - v_self[axis2]) / d[axis1]
    } else { 0. };
    let curl_term2 = if SpatialAxis::is_spatial_axis(axis2) {
        (neighbors[axis2 as usize][axis1] - v_self[axis1]) / d[axis2]
    } else { 0. };

    if FORWARDS { curl_term1 - curl_term2 }
        else { curl_term2 - curl_term1 }
}

fn get_curl_neighbors<const FORWARDS: bool>(
    idx: usize,
    flat_idx_incrs: UVec3,
    v: &[Vec4],
) -> [Vec4; 3] {
    macro_rules! get_neighbor {
        ($axis: ident) => {{
            let neighbor_idx =
                if FORWARDS { idx + flat_idx_incrs.$axis as usize }
                else { idx.gpu_saturating_sub(flat_idx_incrs.$axis as usize) };
            v.read(neighbor_idx)
        }};
    }

    [
        cfg_select! {
            not(feature = "dim1") => get_neighbor!(x),
            _ => Vec4::ZERO
        },
        cfg_select! {
            not(feature = "dim1") => get_neighbor!(y),
            _ => Vec4::ZERO
        },
        cfg_select! {
            not(feature = "dim2") => get_neighbor!(z),
            _ => Vec4::ZERO
        },
    ]
}

/// Information describing the grid
#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct GridParameters {
    /// Increments by 1 in i/j/k indices for flat indexing. Used for accessing
    /// data from neighboring cells.
    pub flat_idx_incrs: UVec3,
    pub polarization_mode_index: PolarizationModeIndex,
    pub n_cells3: UVec3,
    pub _padding1: u32,
    /// Spatial differentials (cell size)
    pub d: Vec3,
    pub _padding2: u32,
}

/// A newtype of an index representing the polarization mode.
///
/// Use a `into()`/`from()` conversion or its constants to safely construct this type.
#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(transparent)]
pub struct PolarizationModeIndex(u32);

impl PolarizationModeIndex {
    pub const TM: Self = Self(0);
    pub const TE: Self = Self(1);

    #[inline]
    pub fn inject_h_source(self, v: &mut Vec4, src_term: Vec4) {
        cfg_select! {
            feature = "dim3" => *v += src_term * (self.0 == 1) as u32 as f32,
            _ => *v += src_term
        }
    }

    #[inline]
    pub fn inject_dn_source(self, v: &mut Vec4, src_term: Vec4) {
        cfg_select! {
            feature = "dim3" => *v += src_term * (self.0 == 0) as u32 as f32,
            _ => *v += src_term
        }
    }
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

/// An electric dipole source
#[derive(Copy, Clone, Pod, Zeroable, Default, Debug)]
#[repr(C)]
pub struct GpuDipole {
    pub cell_idx: u32,
    pub vals_start: u32,
    pub vals_end: u32,
    pub t_start: u32,
    pub moment: Vec4,
    // TODO: pub repeat_count: u32,
}