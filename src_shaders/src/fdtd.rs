#![allow(clippy::needless_range_loop)]

use bytemuck::{Pod, Zeroable};
use khal_std::glamx::{UVec3, Vec3, Vec4};
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::workgroup_memory_barrier_with_group_sync;
use crate::math::*;

/// N-dimensional FDTD shader with loss (conductivity). Works with any polarization mode.
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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] source_vals: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] steps: &mut u32,
    // Uniforms
    #[spirv(uniform, descriptor_set = 0, binding = 8)] grid: &GridParameters,
) {
    if idx3.cmpge(grid.n_cells3).any() { return; }
    let idx = GridIndex::from_uvec3(idx3).to_flat_idx(GridIndex::from_uvec3(grid.n_cells3))
        as usize;
    let boundary_idx3 = grid.n_cells3 - 1;

    // Get the source value at this timestep (working)
    let mut source_term = 0.;
    let curr_step = *steps;
    for i in 0..dipoles.len() {
        let GpuDipole {
            vals_range: [start, end],
            cell_idx,
            t_start,
        } = dipoles.read(i);
        let t = curr_step.gpu_saturating_sub(t_start);
        let vals_i = start + t;
        let enable = (cell_idx as usize == idx) && (curr_step >= t_start) && (vals_i <= end);
        source_term += source_vals.read(vals_i.min(end) as usize) * enable as u32 as f32
    }
    #[cfg(not(any(target_arch = "spirv", target_arch = "nvptx64")))]
    {
        if source_term != 0. && idx == 0 {
            khal_std::println!("source still injecting");
        }
        workgroup_memory_barrier_with_group_sync();
    }
    // TODO: inject source in different fields depending on polarization mode
    let source_term = cfg_select! {
        feature = "dim1" => Vec4::new(0., source_term, 0., 0.),
        feature = "dim2" => Vec4::new(0., 0., source_term, 0.),
        feature = "dim3" => Vec4::new(source_term, source_term, source_term, 0.),
    };

    let coeffs = grid_coeffs.read(idx);
    let mut ints = integrals.read(idx);

    let en_self = en.read(idx);

    let not_boundary = UVec3::from(idx3.cmplt(boundary_idx3));
    let en_curl = compute_curl::<true>(
        grid.d,
        idx,
        not_boundary,
        grid.flat_idx_incrs,
        en_self,
        en
    );
    let h_self = h.read(idx);
    let h_self_new = coeffs.h1 * h_self + coeffs.h2 * en_curl +
        source_term;
    #[cfg(any(feature = "dim2", feature = "dim3"))]
    let h_self_new = {
        ints.en_curl += en_curl;
        let h_self_new = h_self_new + coeffs.h3 * ints.en_curl;
        #[cfg(feature = "dim3")]
        let h_self_new = {
            ints.h += h_self;
            h_self_new + coeffs.h4 * ints.h
        };

        h_self_new
    };
    h.write(idx, h_self_new);
    let h_self = h_self_new;

    let not_boundary = UVec3::from(idx3.cmpgt(UVec3::ZERO));
    let h_curl = compute_curl::<false>(
        grid.d,
        idx,
        not_boundary,
        grid.flat_idx_incrs,
        h_self,
        h
    );
    let dn_self = dn.read(idx);
    ints.en += en_self;
    let dn_self_new = coeffs.dn1 * dn_self + coeffs.dn2 * h_curl +
        coeffs.dn_loss1 * en_self + coeffs.dn_loss2 * ints.en +
        source_term;
    #[cfg(any(feature = "dim2", feature = "dim3"))]
    let dn_self_new = {
        ints.h_curl += h_curl;
        let dn_self_new = dn_self_new + coeffs.dn3 * ints.h_curl;
        #[cfg(feature = "dim3")]
        let dn_self_new = {
            ints.dn += dn_self;
            dn_self_new + coeffs.dn4 * ints.dn
        };

        dn_self_new
    };
    dn.write(idx, dn_self_new);
    let dn_self = dn_self_new;

    let en_self_new = coeffs.en1 * dn_self;
    en.write(idx, en_self_new);

    integrals.write(idx, ints);

    if idx3 == UVec3::ZERO {
        *steps += 1;
    }
}

fn compute_curl<const FORWARDS: bool>(
    d: Vec3,
    idx: usize,
    not_boundary: UVec3,
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
            not_boundary,
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
    not_boundary: UVec3,
    flat_idx_incrs: UVec3,
    v_self: Vec4,
    v: &[Vec4]
) -> Real {
    let neighbors = get_curl_neighbors::<FORWARDS>(idx, not_boundary, flat_idx_incrs, v);
    let axis1 = axis.permute();
    let axis2 = axis1.permute();

    let curl_term1 = if SpatialAxis::is_spatial_axis(axis1) {
        if FORWARDS { (neighbors[axis1 as usize][axis2] - v_self[axis2]) / d[axis1] }
        else { (v_self[axis2] - neighbors[axis1 as usize][axis2]) / d[axis1] }
    } else { 0. };
    let curl_term2 = if SpatialAxis::is_spatial_axis(axis2) {
        if FORWARDS { (neighbors[axis2 as usize][axis1] - v_self[axis1]) / d[axis2] }
        else { (v_self[axis1] - neighbors[axis2 as usize][axis1]) / d[axis2] }
    } else { 0. };
    curl_term1 - curl_term2
}

fn get_curl_neighbors<const FORWARDS: bool>(
    idx: usize,
    not_boundary: UVec3,
    flat_idx_incrs: UVec3,
    vect_field: &[Vec4],
) -> [Vec4; 3] {
    macro_rules! get_neighbor {
        ($axis: ident) => {{
            let neighbor_idx =
                if FORWARDS { idx + (flat_idx_incrs.$axis * not_boundary.$axis) as usize }
                else { idx - (flat_idx_incrs.$axis * not_boundary.$axis) as usize };
            vect_field.read(neighbor_idx) * not_boundary.$axis as Real
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
    pub _padding0: u32,
    pub n_cells3: UVec3,
    pub _padding1: u32,
    /// Spatial differentials (cell size)
    pub d: Vec3,
    pub _padding2: u32,
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
    pub vals_range: [u32; 2],
    pub t_start: u32,
    // TODO: pub repeat_count: u32,
}