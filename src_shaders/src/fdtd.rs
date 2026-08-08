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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] steps: &u32, // TODO: rename to "t_idx"
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] source_terms: &mut [SourceTerms],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] source_vals: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] dipoles: &[GpuDipole],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] plane_waves: &[GpuPlaneWave],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] plane_wave_coeffs: &[PlaneWaveCoeffs],
) {
    let cell_idx = GridIndex::from_uvec3(idx3);
    let n_cells = GridIndex::from_uvec3(grid.n_cells3);
    if skip_update(cell_idx, n_cells, idx3, grid.n_cells3) { return; }

    let idx = cell_idx.to_flat_idx(n_cells) as usize;

    let mut h_source_term = Vec4::ZERO;
    let mut dn_source_term = Vec4::ZERO;
    let curr_t_idx = *steps;
    let t = curr_t_idx as f32 * grid.dt;

    // Dipoles
    for i in 0..dipoles.len() {
        let dipole = dipoles.read(i);
        let src_t_idx = curr_t_idx.gpu_saturating_sub(dipole.t_start);
        let vals_i = dipole.vals_start + src_t_idx;

        let enable = (dipole.cell_idx as usize == idx) &&
            (curr_t_idx >= dipole.t_start) &&
            (vals_i <= dipole.vals_end);
        let source_term = source_vals.read(vals_i.min(dipole.vals_end) as usize) *
            enable as u32 as f32 *
            dipole.moment;
        grid.polarization_mode_index.inject_h_source(&mut h_source_term, source_term);
        grid.polarization_mode_index.inject_dn_source(&mut dn_source_term, source_term);
    }

    // Plane Waves
    for i in 0..plane_waves.len() {
        let GpuPlaneWave {
            spatial_axis, direction, position_idx, vals_start, vals_end, t_start, ..
        } = plane_waves.read(i);
        let wave_coeffs = plane_wave_coeffs.read(position_idx as usize);
        let polarization = Vec3::X; // TODO: let the user pick wave's polarization
        let axis = Axis::from(spatial_axis);
        let is_positive_dir = direction == 1;
        
        macro_rules! val_idx_and_src_active {
            ($src_t_idx:expr) => {{
                let src_t_idx = $src_t_idx.gpu_saturating_sub(t_start);
                let vals_i = vals_start + src_t_idx;
                let enable = ($src_t_idx >= t_start) && (vals_i <= vals_end);
                (vals_i.min(vals_end) as usize, enable)
            }};
        }

        let cell_pos_idx = cell_idx.dyn_idx(spatial_axis);
        let is_tf = cell_pos_idx == position_idx; // if cell is on total-field edge
        let is_sf = cell_pos_idx as i32 == position_idx as i32 - direction; // if cell is on scattered-field edge
        // Future source value (to simulate total-field quantities when computing curl)
        let fut_t_idx = ((t + wave_coeffs.t_offset[axis]) * grid.inv_dt) as Index; // TODO: try interpolating here?
        let (fut_src_val_idx, fut_src_active) = val_idx_and_src_active!(fut_t_idx);
        let fut_src_val = source_vals.read(fut_src_val_idx) * fut_src_active as u32 as f32;
        // Current source value (to simulate scattered-field quantities for the curl)
        let (curr_src_val_idx, sf_src_active) = val_idx_and_src_active!(curr_t_idx);
        let curr_src_val = source_vals.read(curr_src_val_idx) * sf_src_active as u32 as f32;

        // Compute & write field values
        // TODO: extrapolate from lecture to all axes and test things out.
        //       Things to try: change wavecoeffs axis to `axis1`, `axis`, etc.
        //                      negate half timestep difference for future source time value
        //                      negate t_offset future source time value
        //                      negate wave_coeffs src_coeffs
        let axis1 = axis.permute();
        let axis2 = axis1.permute();
        let pol1 = polarization[axis1];
        let pol2 = polarization[axis2];

        // Update source terms w/ the tf/sf correction terms.
        let inv_d_axis = grid.inv_d[axis];
        if is_positive_dir {
            // En curl corrections (used in H update)
            h_source_term[axis1] += inv_d_axis * (pol1 * curr_src_val) * is_tf as u32 as f32;
            h_source_term[axis2] -= inv_d_axis * (pol2 * curr_src_val) * is_tf as u32 as f32;
            // H curl corrections (used in Dn update)
            dn_source_term[axis1] += inv_d_axis * (pol1 * wave_coeffs.h_curl_coeff[axis2] * fut_src_val) * is_sf as u32 as f32;
            dn_source_term[axis2] -= inv_d_axis * (-pol2 * wave_coeffs.h_curl_coeff[axis1] * fut_src_val) * is_sf as u32 as f32;
        } else {
            // En curl corrections (used in H update)
            h_source_term[axis1] -= inv_d_axis * (-pol1 * wave_coeffs.en_curl_coeff[axis2] * fut_src_val) * is_sf as u32 as f32;
            h_source_term[axis2] += inv_d_axis * (pol2 * wave_coeffs.en_curl_coeff[axis1] * fut_src_val) * is_sf as u32 as f32;
            // H curl corrections (used in Dn update)
            dn_source_term[axis1] -= inv_d_axis * (pol1 * curr_src_val) * is_tf as u32 as f32;
            dn_source_term[axis2] += inv_d_axis * (pol2 * curr_src_val) * is_tf as u32 as f32;
        }
    }

    source_terms.write(idx, SourceTerms {
        h: h_source_term,
        dn: dn_source_term,
    });
}

#[derive(Copy, Clone, Pod, Zeroable, Default, Debug)]
#[repr(C)]
pub struct SourceTerms {
    pub h: Vec4,
    pub dn: Vec4,
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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] source_terms: &[SourceTerms],
) {
    let cell_idx = GridIndex::from_uvec3(idx3);
    let n_cells = GridIndex::from_uvec3(grid.n_cells3);
    if skip_update(cell_idx, n_cells, idx3, grid.n_cells3) { return; }

    let idx = cell_idx.to_flat_idx(n_cells) as usize;

    let SourceTerms {
        h: h_source_term, dn: dn_source_term
    } = source_terms.read(idx);
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
    let mut h_self_new = coeffs.h1 * h_self + coeffs.h2 * en_curl +
        h_source_term;
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
        coeffs.dn_loss1 * en_self + coeffs.dn_loss2 * ints.en +
        dn_source_term;
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
    pub dt: Real,
    /// Spatial differentials (cell size)
    pub d: Vec3,
    pub inv_dt: Real,
    /// Inverse of spatial differential (reciprocated cell size)
    pub inv_d: Vec3,
    pub _padding0: u32
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
            feature = "dim3" => *v += src_term * self.is_te() as u32 as f32,
            _ => *v += src_term
        }
    }

    #[inline]
    pub fn inject_dn_source(self, v: &mut Vec4, src_term: Vec4) {
        cfg_select! {
            feature = "dim3" => *v += src_term * self.is_tm() as u32 as f32,
            _ => *v += src_term
        }
    }

    #[inline]
    pub fn is_tm(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub fn is_te(self) -> bool {
        self.0 == 1
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

/// A dipole source
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

/// A plane wave source
#[derive(Copy, Clone, Pod, Zeroable, Default, Debug)]
#[repr(C)]
pub struct GpuPlaneWave {
    pub spatial_axis: SpatialAxis,
    pub direction: i32, // TODO: turn this into the direction enum by unsafe imples w/ bytemuck traits
    pub position_idx: u32,
    pub vals_start: u32,
    pub vals_end: u32,
    pub t_start: u32,
    pub _padding0: [u32; 2],
    // TODO: pub repeat_count: u32,
}

/// Coefficients for resolving plane wave correction terms
#[derive(Copy, Clone, Pod, Zeroable, Default, Debug)]
#[repr(C)]
pub struct PlaneWaveCoeffs {
    pub t_offset: Vec3, // == (refractive_idx / (2. * C_0)) * d + (dt / 2.)
    pub _padding0: u32,
    /// H curl correction term coefficients
    pub h_curl_coeff: Vec3, // == +-sqrt(eps_r / mu_r)
    pub _padding1: u32,
    /// En curl correction term coefficients
    pub en_curl_coeff: Vec3, // == +-sqrt(eps_r / mu_r)
    pub _padding2: u32,
}