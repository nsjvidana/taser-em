use crate::fdtd::*;
use crate::math::*;
use bytemuck::{Pod, Zeroable};
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::*;
use khal_std::sync::atomic_add_f32;

#[spirv_bindgen]
#[cfg_attr(feature = "dim1", spirv(compute(threads(1, 1, 64))))]
#[cfg_attr(feature = "dim2", spirv(compute(threads(8, 8, 1))))]
#[cfg_attr(feature = "dim3", spirv(compute(threads(4, 4, 4))))]
pub fn gpu_compute_source_terms(
    #[spirv(global_invocation_id)] cell_idx3: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &GridParameters,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] t_idx: &u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] source_terms: &mut [SourceTerms],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] source_vals: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] dipoles: &[GpuDipole],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] tfsf_sources: &[GpuTfsf],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] tfsf_corrections: &[TfsfSourceValues],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] tfsf_masks: &[TfsfMask],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] pml_coeffs: &[PmlCoefficients],
) {
    let n_cells = GridIndex::from_uvec3(grid.n_cells3);
    let cell_idx = GridIndex::from_uvec3(cell_idx3);
    let outside_problem_space = {
        let min = GridIndex::from_uvec3(grid.problem_space_min);
        let max = GridIndex::from_uvec3(grid.problem_space_max);
        cell_idx.cmplt(min).any() || cell_idx.cmpgt(max).any() || cell_idx3.cmpge(grid.n_cells3).any()
    };
    if outside_problem_space { return; }

    let idx = cell_idx.to_flat_idx(n_cells) as usize;

    let mut h_source_term = Vec4::ZERO;
    let mut dn_source_term = Vec4::ZERO;
    let curr_t_idx = *t_idx;

    // Dipoles
    for i in 0..dipoles.len() {
        let GpuDipole {
            cell_idx, vals_start, vals_end, t_start, moment, dipole_type, ..
        } = dipoles.read(i);
        let src_t_idx = curr_t_idx.gpu_saturating_sub(t_start);
        let vals_i = vals_start + src_t_idx;

        let enable = (cell_idx as usize == idx) &&
            (curr_t_idx >= t_start) &&
            (vals_i <= vals_end);
        let src_val = source_vals.read(vals_i.min(vals_end) as usize);
        match dipole_type {
            DipoleType::Electric => {
                dn_source_term += src_val * enable as u32 as Real * moment;
            }
            DipoleType::Magnetic => {
                // add half-dt advance for current loop
                let next_src_val = source_vals.read((vals_i + 1).min(vals_end) as usize);
                h_source_term += Real::lerp(src_val, next_src_val, 0.5) *
                    enable as u32 as Real *
                    moment;
            }
        };
    }

    // TF/SF sources
    let coeffs = pml_coeffs.read(idx);
    for i in 0..tfsf_sources.len() {
        let GpuTfsf {
            a, a1, a2,
            tf_min_a, vals_start, vals_end,
            corrections_start, num_correction_cells,
            inv_d_a, inv_d_a1, inv_d_a2,
            ..
        } = tfsf_sources.read(i);
        if vals_start == vals_end { continue; } // skip invalid/inactive tfsf sources

        let cell_idx_a = cell_idx3.dyn_idx(a);
        let corrections_end = (corrections_start + num_correction_cells - 1) as usize;
        let correction_idx = ((corrections_start + cell_idx_a.gpu_saturating_sub(tf_min_a.gpu_saturating_sub(1))) as usize)
            .min(corrections_end);
        // plane wave vals at wavefront in this cell
        let src = tfsf_corrections.read(correction_idx);
        // src vals of wavefront just before this cell
        let src_ma = tfsf_corrections.read(correction_idx.gpu_saturating_sub(1).max(corrections_start as _));
        // src vals of wavefront just after this cell
        let src_pa = tfsf_corrections.read((correction_idx + 1).min(corrections_end));

        let mask_idx = i * grid.cell_count as usize + idx;
        let mask = tfsf_masks.read(mask_idx);
        let en_src_a2_pa1 = mask.en_src_a2_pa1 * src.en_a2;
        let en_src_a1_pa2 = mask.en_src_a1_pa2 * src.en_a1;
        let en_src_a2_pa = mask.en_src_a2_pa * src_pa.en_a2;
        let en_src_a1_pa = mask.en_src_a1_pa * src_pa.en_a1;
        let h_src_a2_ma1 = mask.h_src_a2_ma1 * src.h_a2;
        let h_src_a1_ma2 = mask.h_src_a1_ma2 * src.h_a1;
        let h_src_a2_ma = mask.h_src_a2_ma * src_ma.h_a2;
        let h_src_a1_ma = mask.h_src_a1_ma * src_ma.h_a1;

        h_source_term.dyn_insert(a, h_source_term.dyn_idx(a) + coeffs.h2.dyn_idx(a) *
            (-inv_d_a1 * en_src_a2_pa1 + inv_d_a2 * en_src_a1_pa2)
        );
        h_source_term.dyn_insert(a1, h_source_term.dyn_idx(a1) + coeffs.h2.dyn_idx(a1) *
            (inv_d_a * en_src_a2_pa)
        );
        h_source_term.dyn_insert(a2, h_source_term.dyn_idx(a2) + coeffs.h2.dyn_idx(a2) *
            (-inv_d_a * en_src_a1_pa)
        );
        dn_source_term.dyn_insert(a, dn_source_term.dyn_idx(a) + coeffs.dn2.dyn_idx(a) *
            (inv_d_a1 * h_src_a2_ma1 - inv_d_a2 * h_src_a1_ma2)
        );
        dn_source_term.dyn_insert(a1, dn_source_term.dyn_idx(a1) + coeffs.dn2.dyn_idx(a1) *
            (-inv_d_a * h_src_a2_ma)
        );
        dn_source_term.dyn_insert(a2, dn_source_term.dyn_idx(a2) + coeffs.dn2.dyn_idx(a2) *
            (inv_d_a * h_src_a1_ma)
        );
    }

    source_terms.write(idx, SourceTerms {
        h: h_source_term,
        dn: dn_source_term
    });
}

#[spirv_bindgen(spirv_passthrough)] // atomic_add_f32 doesn't work with naga validation, so enable spirv passthrough.
#[spirv(compute(threads(1, 1, 1)))]
pub fn gpu_compute_dipole_terms(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] src_h: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src_dn: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] t_idx: &u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] source_vals: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] dipoles: &[GpuDipole],
) {
    let dipole_idx = id.x as usize; // id.x is dipole idx
    let GpuDipole {
        cell_idx, vals_start, vals_end, t_start, moment, dipole_type, ..
    } = dipoles.read(dipole_idx);
    let curr_t_idx = *t_idx;
    let src_t_idx = curr_t_idx.gpu_saturating_sub(t_start);
    let vals_i = vals_start + src_t_idx;

    let source_is_on = (curr_t_idx >= t_start) && (vals_i <= vals_end);
    if !source_is_on { return; }

    let idx = cell_idx as usize;
    let src_val = source_vals.read(vals_i.min(vals_end) as usize);
    let next_src_val = source_vals.read((vals_i + 1).min(vals_end) as usize);

    let mut dn_source = Vec4::ZERO;
    let mut h_source = Vec4::ZERO;
    match dipole_type {
        DipoleType::Electric => {
            dn_source = src_val * moment;
        }
        DipoleType::Magnetic => {
            // add half-dt advance for magnetic dipoles
            h_source = Real::lerp(src_val, next_src_val, 0.5) * moment;
        }
    };

    let (x_idx, y_idx, z_idx) = dim3_components_flat_idx(idx);
    atomic_add_f32(src_h.at_mut(x_idx), h_source.x);
    atomic_add_f32(src_h.at_mut(y_idx), h_source.y);
    atomic_add_f32(src_h.at_mut(z_idx), h_source.z);
    atomic_add_f32(src_dn.at_mut(x_idx), dn_source.x);
    atomic_add_f32(src_dn.at_mut(y_idx), dn_source.y);
    atomic_add_f32(src_dn.at_mut(z_idx), dn_source.z);
}

pub fn dipole_terms_workgroups(n_dipoles: u32) -> [u32; 3] {
    [n_dipoles, 1, 1]
}

/// Takes workgroup count of entire simulation grid.
#[spirv_bindgen(spirv_passthrough)]
#[cfg_attr(feature = "dim1", spirv(compute(threads(1, 1, 64))))]
#[cfg_attr(feature = "dim2", spirv(compute(threads(8, 8, 1))))]
#[cfg_attr(feature = "dim3", spirv(compute(threads(4, 4, 4))))]
pub fn gpu_compute_tfsf_terms(
    #[spirv(global_invocation_id)] cell_idx3: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &GridParameters,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src_h: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] src_dn: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] tfsf_sources: &[GpuTfsf],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] tfsf_corrections: &[TfsfSourceValues],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] tfsf_masks: &[TfsfMask],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] pml_coeffs: &[PmlCoefficients],
) {
    let n_cells = GridIndex::from_uvec3(grid.n_cells3);
    let cell_idx = GridIndex::from_uvec3(cell_idx3);
    let outside_problem_space = {
        let min = GridIndex::from_uvec3(grid.problem_space_min);
        let max = GridIndex::from_uvec3(grid.problem_space_max);
        cell_idx.cmplt(min).any() || cell_idx.cmpgt(max).any() || cell_idx3.cmpge(grid.n_cells3).any()
    };
    if outside_problem_space { return; }

    let idx = cell_idx.to_flat_idx(n_cells) as usize;

    let mut h_source = Vec4::ZERO;
    let mut dn_source = Vec4::ZERO;

    let coeffs = pml_coeffs.read(idx);
    for i in 0..tfsf_sources.len() {
        let GpuTfsf {
            a, a1, a2,
            tf_min_a, vals_start, vals_end,
            corrections_start, num_correction_cells,
            inv_d_a, inv_d_a1, inv_d_a2,
            ..
        } = tfsf_sources.read(i);
        if vals_start == vals_end { continue; } // skip invalid/inactive tfsf sources

        let cell_idx_a = cell_idx3.dyn_idx(a);
        let corrections_end = (corrections_start + num_correction_cells - 1) as usize;
        let correction_idx = ((corrections_start + cell_idx_a.gpu_saturating_sub(tf_min_a.gpu_saturating_sub(1))) as usize)
            .min(corrections_end);
        // plane wave vals at wavefront in this cell
        let src = tfsf_corrections.read(correction_idx);
        // src vals of wavefront just before this cell
        let src_ma = tfsf_corrections.read(correction_idx.gpu_saturating_sub(1).max(corrections_start as _));
        // src vals of wavefront just after this cell
        let src_pa = tfsf_corrections.read((correction_idx + 1).min(corrections_end));

        let mask_idx = i * grid.cell_count as usize + idx;
        let mask = tfsf_masks.read(mask_idx);
        let en_src_a2_pa1 = mask.en_src_a2_pa1 * src.en_a2;
        let en_src_a1_pa2 = mask.en_src_a1_pa2 * src.en_a1;
        let en_src_a2_pa = mask.en_src_a2_pa * src_pa.en_a2;
        let en_src_a1_pa = mask.en_src_a1_pa * src_pa.en_a1;
        let h_src_a2_ma1 = mask.h_src_a2_ma1 * src.h_a2;
        let h_src_a1_ma2 = mask.h_src_a1_ma2 * src.h_a1;
        let h_src_a2_ma = mask.h_src_a2_ma * src_ma.h_a2;
        let h_src_a1_ma = mask.h_src_a1_ma * src_ma.h_a1;

        h_source.dyn_insert(a, h_source.dyn_idx(a) + coeffs.h2.dyn_idx(a) *
            (-inv_d_a1 * en_src_a2_pa1 + inv_d_a2 * en_src_a1_pa2)
        );
        h_source.dyn_insert(a1, h_source.dyn_idx(a1) + coeffs.h2.dyn_idx(a1) *
            (inv_d_a * en_src_a2_pa)
        );
        h_source.dyn_insert(a2, h_source.dyn_idx(a2) + coeffs.h2.dyn_idx(a2) *
            (-inv_d_a * en_src_a1_pa)
        );
        dn_source.dyn_insert(a, dn_source.dyn_idx(a) + coeffs.dn2.dyn_idx(a) *
            (inv_d_a1 * h_src_a2_ma1 - inv_d_a2 * h_src_a1_ma2)
        );
        dn_source.dyn_insert(a1, dn_source.dyn_idx(a1) + coeffs.dn2.dyn_idx(a1) *
            (-inv_d_a * h_src_a2_ma)
        );
        dn_source.dyn_insert(a2, dn_source.dyn_idx(a2) + coeffs.dn2.dyn_idx(a2) *
            (inv_d_a * h_src_a1_ma)
        );
    }

    let (x_idx, y_idx, z_idx) = dim3_components_flat_idx(idx);
    atomic_add_f32(src_h.at_mut(x_idx), h_source.x);
    atomic_add_f32(src_h.at_mut(y_idx), h_source.y);
    atomic_add_f32(src_h.at_mut(z_idx), h_source.z);
    atomic_add_f32(src_dn.at_mut(x_idx), dn_source.x);
    atomic_add_f32(src_dn.at_mut(y_idx), dn_source.y);
    atomic_add_f32(src_dn.at_mut(z_idx), dn_source.z);
}

#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn update_source_terms(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] src_h: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src_dn: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] source_terms: &mut [SourceTerms],
) {
    if id.x >= source_terms.len() as u32 { return; }
    let idx = id.x as usize;

    let (x_idx, y_idx, z_idx) = dim3_components_flat_idx(idx);
    let h = Vec4::new(
        Real::from_bits(src_h.read(x_idx)),
        Real::from_bits(src_h.read(y_idx)),
        Real::from_bits(src_h.read(z_idx)),
        0.,
    );
    let dn = Vec4::new(
        Real::from_bits(src_dn.read(x_idx)),
        Real::from_bits(src_dn.read(y_idx)),
        Real::from_bits(src_dn.read(z_idx)),
        0.,
    );
    src_h.write(x_idx, Real::to_bits(0.));
    src_h.write(y_idx, Real::to_bits(0.));
    src_h.write(z_idx, Real::to_bits(0.));
    src_dn.write(x_idx, Real::to_bits(0.));
    src_dn.write(y_idx, Real::to_bits(0.));
    src_dn.write(z_idx, Real::to_bits(0.));
    source_terms.write(idx, SourceTerms { h, dn });
}

pub fn update_source_terms_workgroups(n_cells: GridIndex) -> [u32; 3] {
    [n_cells.element_product().div_ceil(64), 1, 1]
}

/// Convert a flat idx to a whole 3D vector into three flat indices, one for each component
/// within an array of individual vector components. `idx` -> `(x_idx, y_idx, z_idx)`
#[inline]
fn dim3_components_flat_idx(idx: usize) -> (usize, usize, usize) {
    let x_idx = idx * 3;
    let y_idx = x_idx + 1;
    let z_idx = y_idx + 1;
    (x_idx, y_idx, z_idx)
}

/// Stores source terms for the current timestep.
///
/// Stored as `u32`s for atomic load/store during the time when source terms are computed.
#[derive(Copy, Clone, Pod, Zeroable, Default, Debug)]
#[repr(C)]
pub struct SourceTerms {
    pub h: Vec4,
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
    pub dipole_type: DipoleType,
    pub _padding0: [u32; 3]
    // TODO: pub repeat_count: u32,
}

#[derive(Copy, Clone, Debug, Default)]
#[repr(u32)]
pub enum DipoleType {
    /// An oscillating point charge
    #[default]
    Electric = 0,
    /// An infinitesimal current loop
    Magnetic = 1,
}

// SAFETY: DipoleType has a zero variant.
unsafe impl Zeroable for DipoleType {}
// SAFETY: DipoleType has u32 representation, and u32 is also POD.
unsafe impl Pod for DipoleType {}

pub const INIT_TFSF_WORKGROUP_SIZE: UVec3 = cfg_select! {
    feature = "dim1" => UVec3::new(1, 1, 64),
    feature = "dim2" => UVec3::new(8, 8, 1),
    feature = "dim3" => UVec3::new(4, 4, 4),
};

#[spirv_bindgen]
#[cfg_attr(feature = "dim1", spirv(compute(threads(1, 1, 64))))]
#[cfg_attr(feature = "dim2", spirv(compute(threads(8, 8, 1))))]
#[cfg_attr(feature = "dim3", spirv(compute(threads(4, 4, 4))))]
pub fn init_tfsf_masks(
    #[spirv(global_invocation_id)] cell_idx3: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &GridParameters,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] tfsf_sources: &[GpuTfsf],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] tfsf_masks: &mut [TfsfMask],
) {
    let out_of_bounds = cfg_select! {
        feature = "dim1" => cell_idx3.cmpge(grid.n_cells3.with_x(tfsf_sources.len() as u32)).any(),
        feature = "dim2" => cell_idx3.cmpge(grid.n_cells3.with_z(tfsf_sources.len() as u32)).any(),
        feature = "dim3" => cell_idx3.cmpge(grid.n_cells3.with_z(tfsf_sources.len() as u32 * grid.n_cells3.z)).any(),
    };
    if out_of_bounds { return; }

    let n_cells = GridIndex::from_uvec3(grid.n_cells3);
    let cell_idx = cfg_select! {
        feature = "dim3" => GridIndex::from_uvec3(cell_idx3.with_z(cell_idx3.z % grid.n_cells3.z)),
        _ => GridIndex::from_uvec3(cell_idx3)
    };
    let src_idx = cfg_select! {
        feature = "dim1" => cell_idx3.x as usize,
        feature = "dim2" => cell_idx3.z as usize,
        feature = "dim3" => (cell_idx3.z / grid.n_cells3.z) as usize,
    };
    let idx = src_idx * grid.cell_count as usize + cell_idx.to_flat_idx(n_cells) as usize;
    let GpuTfsf {
        a, a1, a2,
        #[cfg_attr(not(feature = "dim1"), allow(unused_variables))]
        direction,
        tf_min_a, tf_min_a1,tf_min_a2,
        tf_max_a, tf_max_a1, tf_max_a2,
        vals_start, vals_end, ..
    } = tfsf_sources.read(src_idx);

    if vals_start == vals_end { return; } // skip invalid/inactive tfsf sources

    let cell_idx_a = cell_idx3.dyn_idx(a);
    let cell_idx_a1 = cell_idx3.dyn_idx(a1);
    let cell_idx_a2 = cell_idx3.dyn_idx(a2);
    let a1_spatial = SpatialAxis::is_spatial_axis(a1);
    let a2_spatial = SpatialAxis::is_spatial_axis(a2);

    let inside_tf = (cell_idx_a >= tf_min_a && cell_idx_a <= tf_max_a) &&
        (!a1_spatial || (cell_idx_a1 >= tf_min_a1 && cell_idx_a1 <= tf_max_a1)) &&
        (!a2_spatial || (cell_idx_a2 >= tf_min_a2 && cell_idx_a2 <= tf_max_a2));
    let at_max_tf_edge_a = [
        inside_tf && (cell_idx_a == tf_max_a),
        inside_tf && (cell_idx_a1 == tf_max_a1),
        inside_tf && (cell_idx_a2 == tf_max_a2),
    ];
    let at_min_tf_edge_a = [
        inside_tf && (cell_idx_a == tf_min_a),
        inside_tf && (cell_idx_a1 == tf_min_a1),
        inside_tf && (cell_idx_a2 == tf_min_a2),
    ];

    let sf_edge_or_in_tf = (cell_idx_a+1 >= tf_min_a && cell_idx_a <= tf_max_a+1) &&
        (!a1_spatial || (cell_idx_a1+1 >= tf_min_a1 && cell_idx_a1 <= tf_max_a1+1)) &&
        (!a2_spatial || (cell_idx_a2+1 >= tf_min_a2 && cell_idx_a2 <= tf_max_a2+1));
    let at_max_sf_edge_a = [
        sf_edge_or_in_tf && (cell_idx_a == tf_max_a+1),
        sf_edge_or_in_tf && (cell_idx_a1 == tf_max_a1+1),
        sf_edge_or_in_tf && (cell_idx_a2 == tf_max_a2+1),
    ];
    let at_min_sf_edge_a = [
        sf_edge_or_in_tf && (cell_idx_a+1 == tf_min_a),
        sf_edge_or_in_tf && (cell_idx_a1+1 == tf_min_a1),
        sf_edge_or_in_tf && (cell_idx_a2+1 == tf_min_a2),
    ];

    // Skip corrections at opposite end of the source in 1D since the entire plane wave is
    // guaranteed to hit the object.
    #[cfg(feature = "dim1")]
    {
        if (direction == Direction::Positive && (at_max_sf_edge_a[0] || at_max_tf_edge_a[0])) ||
            (direction == Direction::Negative && (at_min_sf_edge_a[0] || at_min_tf_edge_a[0]))
        {
            return;
        }
    }

    let not_sf_corner = cfg_select! {
        feature = "dim1" => true,
        _ => {{
            let mut at_sf_edge_count = 0;
            for i in 0..3 {
                at_sf_edge_count += at_min_sf_edge_a[i] as usize + at_max_sf_edge_a[i] as usize;
            }
            at_sf_edge_count < 2
        }}
    };

    let neg = if inside_tf { -1. } else { 1. };
    tfsf_masks.write(idx, TfsfMask {
        en_src_a2_pa1: (not_sf_corner && (at_max_tf_edge_a[1] || at_min_sf_edge_a[1])) as u32 as Real * neg,
        en_src_a1_pa2: (not_sf_corner && (at_max_tf_edge_a[2] || at_min_sf_edge_a[2])) as u32 as Real * neg,
        en_src_a2_pa: (not_sf_corner && (at_max_tf_edge_a[0] || at_min_sf_edge_a[0])) as u32 as Real * neg,
        en_src_a1_pa: (not_sf_corner && (at_max_tf_edge_a[0] || at_min_sf_edge_a[0])) as u32 as Real * neg,
        h_src_a2_ma1: (not_sf_corner && (at_min_tf_edge_a[1] || at_max_sf_edge_a[1])) as u32 as Real * neg,
        h_src_a1_ma2: (not_sf_corner && (at_min_tf_edge_a[2] || at_max_sf_edge_a[2])) as u32 as Real * neg,
        h_src_a2_ma: (not_sf_corner && (at_min_tf_edge_a[0] || at_max_sf_edge_a[0])) as u32 as Real * neg,
        h_src_a1_ma: (not_sf_corner && (at_min_tf_edge_a[0] || at_max_sf_edge_a[0])) as u32 as Real * neg,
    });
}

pub fn init_tfsf_masks_workgroups(n_tfsf_sources: u32, n_cells3: UVec3) -> [u32; 3] {
    let threads = cfg_select! {
        feature = "dim1" => n_cells3.with_x(n_tfsf_sources).to_array(),
        feature = "dim2" => n_cells3.with_z(n_tfsf_sources).to_array(),
        feature = "dim3" => n_cells3.with_z(n_tfsf_sources * n_cells3.z).to_array(),
    };
    core::array::from_fn(|i| threads[i].div_ceil(INIT_TFSF_WORKGROUP_SIZE[i]))
}

pub const AUX_GRID_WORKGROUP_SIZE: UVec3 = UVec3::new(1, 1, 64);

#[spirv_bindgen]
#[spirv(compute(threads(1, 1, 64)))]
pub fn aux_grid_update(
    #[spirv(global_invocation_id)] cell_idx3: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] tfsf_sources: &[GpuTfsf],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] t_idx: &Index,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] corrections: &mut [TfsfSourceValues],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] source_vals: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] auxgr_coeffs: &[AuxGridPmlCoeffs],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] h: &mut [AuxVect],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] dn: &mut [AuxVect],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] en: &mut [AuxVect],
) {
    let wave_idx = cell_idx3.x as usize;
    let GpuTfsf {
        direction,
        grid_start, vals_start, vals_end, t_start, n_cells,
        polarization_a1, polarization_a2,
        corrections_start, num_correction_cells,
        inv_d_a, ..
    } = tfsf_sources.read(wave_idx);
    if vals_start == vals_end { return; } // skip invalid/inactive tfsf sources

    if cell_idx3.z >= n_cells { return; }

    let last_idx_local = n_cells as usize - 1;
    let idx_local = cell_idx3.z as usize;
    let idx_local_inv = last_idx_local - idx_local;

    let is_positive_dir = direction == Direction::Positive;
    let idx_offset = grid_start as usize;
    let idx = idx_offset + idx_local;
    let m = auxgr_coeffs.read(idx);

    // Resolve dipole source
    let is_source = (is_positive_dir && idx_local == 0) ||
        (!is_positive_dir && idx_local == last_idx_local);
    let vals_i = vals_start + t_idx.gpu_saturating_sub(t_start);
    let src_enable = (*t_idx >= t_start && (vals_i <= vals_end)) as u32;
    let src_val = source_vals.read((vals_i * src_enable) as usize) * src_enable as Real;
    let src_vect = AuxVect::new(polarization_a1 * src_val, polarization_a2 * src_val);

    let en_self = en.read(idx);
    let en_self_corr = en_self; // En curl is used before E gets its update.

    let mut h_self = h.read(idx);
    let not_boundary = (idx_local < last_idx_local) as u32 as Real;
    let mut neighbor = en.read((idx + 1).min(en.len() - 1)) * not_boundary;
    let en_curl_a1 = -(neighbor.y - en_self.y) * inv_d_a;
    let en_curl_a2 = (neighbor.x - en_self.x) * inv_d_a;
    h_self.x = m.h1.x * h_self.x + m.h2.x * en_curl_a1;
    h_self.y = m.h1.y * h_self.y + m.h2.y * en_curl_a2;
    h.write(idx, h_self);
    let h_self_corr = h_self; // H terms are half-dt ahead in time.

    let mut dn_self = dn.read(idx);
    let not_boundary = (idx_local > 0) as u32 as Real;
    neighbor = h.read(idx.gpu_saturating_sub(1)) * not_boundary;
    let h_curl_a1 = -(h_self.y - neighbor.y) * inv_d_a;
    let h_curl_a2 = (h_self.x - neighbor.x) * inv_d_a;
    dn_self.x = m.dn1.x * dn_self.x + m.dn2.x * h_curl_a1;
    dn_self.y = m.dn1.y * dn_self.y + m.dn2.y * h_curl_a2;
    dn_self = if is_source { src_vect } else { dn_self };
    dn.write(idx, dn_self);

    let en_self = AuxVect::new(
        m.en1.x * dn_self.x,
        m.en1.y * dn_self.y,
    );
    en.write(idx, en_self);

    let num_correction_cells = num_correction_cells as usize;
    let dir_local_idx = if is_positive_dir { idx_local } else { idx_local_inv };
    let is_correction_cell = dir_local_idx > 0 && dir_local_idx <= num_correction_cells;
    if !is_correction_cell { return; }
    let corr_idx_offset = if is_positive_dir { dir_local_idx - 1 } else { num_correction_cells - dir_local_idx };
    let correction_idx = corrections_start as usize + corr_idx_offset;
    corrections.write(correction_idx, TfsfSourceValues {
        en_a1: en_self_corr.x,
        en_a2: en_self_corr.y,
        h_a1: h_self_corr.x,
        h_a2: h_self_corr.y,
    });
}

pub fn aux_grid_update_workgroups(n_tfsf_sources: u32, aux_grid_n_cells_max: u32) -> [u32; 3] {
    let threads = [n_tfsf_sources, 1, aux_grid_n_cells_max];
    core::array::from_fn(|i| threads[i].div_ceil(AUX_GRID_WORKGROUP_SIZE[i]))
}

/// A plane wave source (TF/SF)
///
/// Immediately after the element at `vals_end` in `source_vals` buffer are the plane wave values at the TF/SF boundaries:
/// ```
/// let source_vals = [..., src_0, src_1, ..., src_n, h_src_start, en_src_start, h_src_end, en_src_end, ...]
///                 //        ^                         ^
///                 // 1D source values                 |
///                 //                            tf/sf boundary field values start here
/// ```
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
#[repr(C)]
pub struct GpuTfsf {
    pub a: Axis,
    pub a1: Axis,
    pub a2: Axis,

    pub direction: Direction,
    /// The smallest index component of a cell that is fully inside the TF/SF boundary.
    ///
    /// (component of cell idx is along `GpuTfsf.a` direction)
    pub tf_min_a: u32,
    pub tf_min_a1: u32,
    pub tf_min_a2: u32,

    /// The largest index component of a cell that is half-inside the TF/SF boundary.
    /// Only the En components of the cell are within the TF/SF boundary.
    ///
    /// (component of cell idx is along `a` direction)
    pub tf_max_a: u32,
    pub tf_max_a1: u32,
    pub tf_max_a2: u32,
    pub grid_start: u32,

    pub vals_start: u32,
    pub vals_end: u32,
    pub t_start: u32,
    pub n_cells: u32,

    pub polarization_a1: Real,
    pub polarization_a2: Real,
    pub corrections_start: u32,
    pub num_correction_cells: u32,

    pub inv_d_a: Real,
    pub inv_d_a1: Real,
    pub inv_d_a2: Real,
    // TODO: pub repeat_count: u32,
}

/// A mask that enables and/or negates TFSF source values used in the correction terms
#[derive(Copy, Clone, Pod, Zeroable, Default, Debug)]
#[repr(C)]
pub struct TfsfMask {
    pub en_src_a2_pa1: Real,
    pub en_src_a1_pa2: Real,
    pub en_src_a2_pa: Real,
    pub en_src_a1_pa: Real,
    pub h_src_a2_ma1: Real,
    pub h_src_a1_ma2: Real,
    pub h_src_a2_ma: Real,
    pub h_src_a1_ma: Real,
}

pub type AuxVect = Vec2;

#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct AuxGridPmlCoeffs {
    pub h1: Vec2,
    pub h2: Vec2,
    pub dn1: Vec2,
    pub dn2: Vec2,
    pub en1: Vec2,
}

#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
#[repr(C)]
pub struct TfsfSourceValues {
    pub en_a1: Real,
    pub en_a2: Real,
    pub h_a1: Real,
    pub h_a2: Real,
}