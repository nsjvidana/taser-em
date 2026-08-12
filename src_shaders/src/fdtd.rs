#![allow(clippy::needless_range_loop)]
use crate::math::*;
use bytemuck::{Pod, Zeroable};
use khal_std::glamx::{UVec3, Vec3, Vec4};
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

#[allow(unused_imports)]
use khal_std::num_traits::Float;

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

    let mut h_source_term = Vec4::ZERO;
    let mut dn_source_term = Vec4::ZERO;
    let t_idx = *t_idx;
    let t = t_idx as Real * grid.dt;

    let coeffs = pml_coeffs.read(idx);
    for i in 0..tfsf_sources.len() {
        // TODO: make sure the code here works for a "zeroed" tfsf source
        todo!()
    }

    source_terms.write(idx, SourceTerms {
        h: h_source_term,
        dn: dn_source_term
    });
}

#[spirv_bindgen]
#[spirv(compute(threads(1, 1, 64)))]
pub fn aux_grid_update(
    #[spirv(global_invocation_id)] cell_idx3: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] tfsf_sources: &[GpuTfsf],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] t_idx: &Index,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] corrections: &mut [TfsfCorrections],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] source_vals: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] auxgr_coeffs: &[AuxGridPmlCoeffs],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] h: &mut [[Real; 2]],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] dn: &mut [[Real; 2]],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] en: &mut [[Real; 2]],
) {
    const A1: usize = 0;
    const A2: usize = 1;
    let wave_idx = cell_idx3.x as usize;
    let GpuTfsf {
        prop_axis, direction, boundary_min, boundary_max, vals_start, vals_end,
        t_start, n_cells, polarization_a1, polarization_a2, inv_d_a, corrections_start, num_correction_cells,
        grid_start: coeffs_start
    } = tfsf_sources.read(wave_idx);
    let a = Axis::from(prop_axis);
    let a1 = a.permute();
    let a2 = a1.permute();
    if cell_idx3.z >= n_cells { return; }

    let last_idx_local = n_cells as usize - 1;
    let idx_local = cell_idx3.z as usize;
    let idx_local_inv = last_idx_local - idx_local;

    let is_positive_dir = direction == WaveDirection::Positive;
    let idx_offset = coeffs_start as usize;
    let idx = idx_offset + idx_local;
    let m = auxgr_coeffs.read(idx);

    // Resolve dipole source
    let is_source = (is_positive_dir && idx_local == 0) ||
        (!is_positive_dir && idx_local == last_idx_local);
    let vals_i = vals_start + t_idx.gpu_saturating_sub(t_start);
    let src_enable = (*t_idx >= t_start && (vals_i <= vals_end)) as u32;
    let src_val = source_vals.read((vals_i * src_enable) as usize) * src_enable as Real;
    let src_vect = [polarization_a1 * src_val, polarization_a2 * src_val];

    let en_self = en.read(idx);

    let mut h_self = h.read(idx);
    let not_boundary = (idx_local < last_idx_local) as u32 as Real;
    let mut neighbor = en.read((idx + 1).min(en.len() - 1));
    let en_curl_a1 = -(neighbor[A2] * not_boundary - en_self[A2]) * inv_d_a;
    let en_curl_a2 = (neighbor[A1] * not_boundary - en_self[A1]) * inv_d_a;
    h_self[A1] = m.h1[a1] * h_self[A1] + m.h2[a1] * en_curl_a1;
    h_self[A2] = m.h1[a2] * h_self[A2] + m.h2[a2] * en_curl_a2;
    h.write(idx, h_self);

    let mut dn_self = dn.read(idx);
    let not_boundary = (idx_local > 0) as u32 as Real;
    neighbor = en.read(idx.gpu_saturating_sub(1));
    let h_curl_a1 = -(h_self[A2] * not_boundary - neighbor[A2]) * inv_d_a;
    let h_curl_a2 = (h_self[A1] * not_boundary - neighbor[A1]) * inv_d_a;
    dn_self[A1] = m.dn1[a1] * dn_self[A1] + m.dn2[a1] * h_curl_a1;
    dn_self[A2] = m.dn1[a2] * dn_self[A2] + m.dn2[a2] * h_curl_a2;
    dn_self = if is_source { src_vect } else { dn_self };
    dn.write(idx, dn_self);

    let en_self = [
        m.en1[a1] * dn_self[A1],
        m.en1[a2] * dn_self[A2],
    ];
    en.write(idx, en_self);

    let num_correction_cells = num_correction_cells as usize;
    let dir_local_idx = if is_positive_dir { idx_local } else { idx_local_inv };
    let is_correction_cell = dir_local_idx > 0 && dir_local_idx <= num_correction_cells;
    if !is_correction_cell { return; }
    let corr_idx_offset = if is_positive_dir { dir_local_idx - 1 } else { num_correction_cells - dir_local_idx };
    let correction_idx = corrections_start as usize + corr_idx_offset;
    corrections.write(correction_idx, TfsfCorrections {
        h_a1: h_self[A1],
        h_a2: h_self[A2],
        en_a1: en_self[A1],
        en_a2: en_self[A2],
    });
}

#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct AuxGridPmlCoeffs {
    pub h1: Vec4,
    pub h2: Vec4,
    pub dn1: Vec4,
    pub dn2: Vec4,
    pub en1: Vec4,
}

#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct TfsfCorrections {
    pub h_a1: Real,
    pub h_a2: Real,
    pub en_a1: Real,
    pub en_a2: Real,
}

#[spirv_bindgen]
#[cfg_attr(feature = "dim1", spirv(compute(threads(1, 1, 64))))]
#[cfg_attr(feature = "dim2", spirv(compute(threads(8, 8, 1))))]
#[cfg_attr(feature = "dim3", spirv(compute(threads(4, 4, 4))))]
pub fn gpu_compute_source_terms_old(
    #[spirv(global_invocation_id)] cell_idx3: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &GridParameters,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] t_idx: &u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] source_terms: &mut [SourceTerms],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] source_vals: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] dipoles: &[GpuDipole],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] plane_waves: &[GpuTfsf],
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

    let curr_t_idx = *t_idx;
    let mut h_source_term = Vec4::ZERO;
    let mut dn_source_term = Vec4::ZERO;
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

    // Plane waves
    let pml_coeffs = pml_coeffs.read(idx);
    for i in 0..plane_waves.len() {
        todo!()
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

#[spirv_bindgen]
#[cfg_attr(feature = "dim1", spirv(compute(threads(1, 1, 64))))]
#[cfg_attr(feature = "dim2", spirv(compute(threads(8, 8, 1))))]
#[cfg_attr(feature = "dim3", spirv(compute(threads(4, 4, 4))))]
pub fn gpu_lossy_update(
    #[spirv(global_invocation_id)] cell_idx3: UVec3,
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
    let n_cells = GridIndex::from_uvec3(grid.n_cells3);
    let cell_idx = GridIndex::from_uvec3(cell_idx3);
    let boundary_or_out_of_bounds = cell_idx.cmpeq(GridIndex::ZERO).any() ||
        cell_idx.cmpeq(n_cells - 1).any() ||
        cell_idx3.cmpge(grid.n_cells3).any();
    if boundary_or_out_of_bounds { return; }

    let idx = cell_idx.to_flat_idx(n_cells) as usize;

    let m = grid_coeffs.read(idx);
    let mut ints = integrals.read(idx);

    let en_self = en.read(idx);
    let src = source_terms.read(idx);

    // H update
    let mut h_self = h.read(idx);
    let en_curl = Vec4::new(
        (en.read(idx + grid.flat_idx_incrs.y as usize).z - en_self.z) * grid.inv_d.y,
        -(en.read(idx + grid.flat_idx_incrs.x as usize).z - en_self.z) * grid.inv_d.x,
        (en.read(idx + grid.flat_idx_incrs.x as usize).y - en_self.y) * grid.inv_d.x -
            (en.read(idx + grid.flat_idx_incrs.y as usize).x - en_self.x) * grid.inv_d.y,
        0.
    );
    ints.en_curl += en_curl;
    ints.h += h_self;
    h_self.x = m.h1.x * h_self.x + m.h2.x * en_curl.x + m.h3.x * ints.en_curl.x;
    h_self.y = m.h1.y * h_self.y + m.h2.y * en_curl.y + m.h3.y * ints.en_curl.y;
    h_self.z = m.h1.z * h_self.z + m.h2.z * en_curl.z + m.h4.z * ints.h.z;
    h_self += src.h;
    h.write(idx, h_self);

    // Dn update
    let mut dn_self = dn.read(idx);
    let h_curl = Vec4::new(
        (h_self.z - h.read(idx - grid.flat_idx_incrs.y as usize).z) * grid.inv_d.y,
        -(h_self.z - h.read(idx - grid.flat_idx_incrs.x as usize).z) * grid.inv_d.x,
        (h_self.y - h.read(idx - grid.flat_idx_incrs.x as usize).y) * grid.inv_d.x -
            (h_self.x - h.read(idx - grid.flat_idx_incrs.y as usize).x) * grid.inv_d.y,
        0.
    );
    ints.h_curl += h_curl;
    ints.en += en_self;
    ints.dn += dn_self;

    dn_self.x = m.dn1.x * dn_self.x + m.dn2.x * h_curl.x + m.dn3.x * ints.h_curl.x +
        m.dn_loss1.x * en_self.x + m.dn_loss2.x * ints.en.x;
    dn_self.y = m.dn1.y * dn_self.y + m.dn2.y * h_curl.y + m.dn3.y * ints.h_curl.y +
        m.dn_loss1.y * en_self.y + m.dn_loss2.y * ints.en.y;
    dn_self.z = m.dn1.z * dn_self.z + m.dn2.z * h_curl.z + m.dn4.z * ints.dn.z +
        m.dn_loss1.z * en_self.z + m.dn_loss2.z * ints.en.z;
    dn_self += src.dn;
    dn.write(idx, dn_self);

    en.write(idx, m.en1 * dn_self);

    if cell_idx == GridIndex::ONE {
        *steps += 1;
    }
}

#[spirv_bindgen]
#[cfg_attr(feature = "dim1", spirv(compute(threads(1, 1, 64))))]
#[cfg_attr(feature = "dim2", spirv(compute(threads(8, 8, 1))))]
#[cfg_attr(feature = "dim3", spirv(compute(threads(4, 4, 4))))]
pub fn gpu_lossy_update_old(
    #[spirv(global_invocation_id)] cell_idx3: UVec3,
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
    let n_cells = GridIndex::from_uvec3(grid.n_cells3);
    let cell_idx = GridIndex::from_uvec3(cell_idx3);

    let is_boundary_or_out_of_bounds = cell_idx.cmpeq(GridIndex::ZERO).any() ||
        cell_idx.cmpge(n_cells - 1).any() || cell_idx3.cmpge(grid.n_cells3).any();
    if is_boundary_or_out_of_bounds { return; }

    let idx = cell_idx.to_flat_idx(n_cells) as usize;
    let pml_coeffs = grid_coeffs.read(idx);
    let mut pml_ints = integrals.read(idx);
    let en_self = en.read(idx);
    let source_terms = source_terms.read(idx);

    fn curl<const FWD: bool>(
        idx: usize,
        v_self: Vec4,
        v: &[Vec4],
        grid: &GridParameters
    ) -> Vec4 {
        let mut curl1;
        let mut curl2;
        let mut neighbor_idx;

        let mut curl = Vec4::ZERO;
        for i in 0..Axis::ALL_AXES.len() {
            let axis = Axis::ALL_AXES[i];
            if axis as u32 == u32::MAX { break; } // Removing this causes instability (probably a rust-gpu issue)
            let axis1 = axis.permute();
            let axis2 = axis1.permute();
            curl1 = if SpatialAxis::is_spatial_axis(axis1) {
                if FWD {
                    neighbor_idx = idx + grid.flat_idx_incrs[axis1] as usize;
                    (v.read(neighbor_idx)[axis2] - v_self[axis2]) * grid.inv_d[axis1]
                } else {
                    neighbor_idx = idx - grid.flat_idx_incrs[axis1] as usize;
                    (v_self[axis2] - v.read(neighbor_idx)[axis2]) * grid.inv_d[axis1]
                }
            } else { 0. };
            curl2 = if SpatialAxis::is_spatial_axis(axis2) {
                if FWD {
                    neighbor_idx = idx + grid.flat_idx_incrs[axis2] as usize;
                    (v.read(neighbor_idx)[axis1] - v_self[axis1]) * grid.inv_d[axis2]
                } else {
                    neighbor_idx = idx - grid.flat_idx_incrs[axis2] as usize;
                    (v_self[axis1] - v.read(neighbor_idx)[axis1]) * grid.inv_d[axis2]
                }
            } else { 0. };
            curl.dyn_insert(axis, curl1 - curl2);
        }
        curl
    }

    let h_self = {
        let old_self = h.read(idx);

        let en_curl = curl::<true>(idx, en_self, en, grid);

        let mut new_h = pml_coeffs.h1 * old_self + pml_coeffs.h2 * en_curl;
        cfg_select! {
            feature = "dim1" => {
                pml_ints.en_curl.z += en_curl.z;
                new_h.z += pml_coeffs.h3.z * pml_ints.en_curl.z;
            }
            feature = "dim2" => {
                pml_ints.en_curl.x += en_curl.x;
                new_h.x += pml_coeffs.h3.x * pml_ints.en_curl.x;
                pml_ints.en_curl.y += en_curl.y;
                new_h.y += pml_coeffs.h3.y * pml_ints.en_curl.y;

                pml_ints.h.z += old_self.z;
                new_h.z += pml_coeffs.h4.z * pml_ints.h.z;
            }
            feature = "dim3" => {
                pml_ints.en_curl += en_curl;
                pml_ints.h += old_self;
                new_h += pml_coeffs.h3 * pml_ints.en_curl + pml_coeffs.h4 * pml_ints.h;
            }
        }
        new_h + source_terms.h
    };
    h.write(idx, h_self);

    let dn_self = {
        let old_self = dn.read(idx);

        let h_curl = curl::<false>(idx, h_self, h, grid);

        let mut new_dn = pml_coeffs.dn1 * old_self + pml_coeffs.dn2 * h_curl +
            pml_coeffs.dn_loss1 * en_self;
        // 2nd loss term
        cfg_select! {
            feature = "dim1" => {
                pml_ints.en.z += en_self.z;
                new_dn.z += pml_coeffs.dn_loss2.z * pml_ints.en.z;
            }
            feature = "dim2" => {
                pml_ints.en.x += en_self.x;
                new_dn.x += pml_coeffs.dn_loss2.x * pml_ints.en.x;
                pml_ints.en.y += en_self.y;
                new_dn.y += pml_coeffs.dn_loss2.y * pml_ints.en.y;
            }
            feature = "dim3" => {
                pml_ints.en += en_self;
                new_dn += pml_coeffs.dn_loss2 * pml_ints.en;
            }
        }
        // last two integral terms
        cfg_select! {
            feature = "dim1" => {
                pml_ints.h_curl.z += h_curl.z;
                new_dn.z += pml_coeffs.dn3.z * pml_ints.h_curl.z;
            }
            feature = "dim2" => {
                pml_ints.h_curl.x += h_curl.x;
                new_dn.x += pml_coeffs.dn3.x * pml_ints.h_curl.x;
                pml_ints.h_curl.y += h_curl.y;
                new_dn.y += pml_coeffs.dn3.y * pml_ints.h_curl.y;

                pml_ints.dn.z += old_self.z;
                new_dn.z += pml_coeffs.dn4.z * pml_ints.dn.z;
            }
            feature = "dim3" => {
                pml_ints.h_curl += h_curl;
                pml_ints.dn += old_self;
                new_dn += pml_coeffs.dn3 * pml_ints.h_curl + pml_coeffs.dn4 * pml_ints.dn;
            }
        }
        new_dn + source_terms.dn
    };
    dn.write(idx, dn_self);

    en.write(idx, pml_coeffs.en1 * dn_self);

    if cell_idx == GridIndex::ONE {
        *steps += 1;
    }
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
    pub _padding0: u32,
    pub problem_space_min: UVec3,
    pub _padding1: u32,
    pub problem_space_max: UVec3,
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
    pub h3: Vec4,
    #[cfg(any(feature = "dim2", feature = "dim3"))]
    pub h4: Vec4,

    pub dn1: Vec4,
    pub dn2: Vec4,
    pub dn_loss1: Vec4,
    pub dn_loss2: Vec4,
    pub dn3: Vec4,
    #[cfg(any(feature = "dim2", feature = "dim3"))]
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
    pub en_curl: Vec4,
    #[cfg(any(feature = "dim2", feature = "dim3"))]
    pub h: Vec4,
    pub h_curl: Vec4,
    #[cfg(any(feature = "dim2", feature = "dim3"))]
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

/// A plane wave source (TF/SF)
///
/// Immediately after the element at `vals_end` in `source_vals` buffer are the plane wave values at the TF/SF boundaries:
/// ```
/// let source_vals = [..., src_0, src_1, ..., src_n, h_src_start, en_src_start, h_src_end, en_src_end, ...]
///                 //        ^                         ^
///                 // 1D source values                 |
///                 //                            tf/sf boundary field values start here
/// ```
// TODO: allow user to disable tf/sf boundary at the end of the plane wave's propagation
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
#[repr(C)]
pub struct GpuTfsf {
    pub prop_axis: SpatialAxis,
    pub direction: WaveDirection, // TODO: turn this into the direction enum using unsafe impls w/ bytemuck traits
    pub boundary_min: u32, // TODO: make 3d version of this for 3d implementation
    pub boundary_max: u32,

    pub vals_start: u32,
    pub vals_end: u32,
    pub t_start: u32,
    pub n_cells: u32,

    pub polarization_a1: Real,
    pub polarization_a2: Real,
    pub inv_d_a: Real,
    pub corrections_start: u32,

    pub num_correction_cells: u32,
    pub grid_start: u32
    // TODO: pub repeat_count: u32,
}