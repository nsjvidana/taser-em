use bytemuck::{Pod, Zeroable};
use khal_std::glamx::{UVec3, Vec3, Vec4, Vec4Swizzles};
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use crate::fdtd::{GpuDipole, GridParameters, IntegrationTerms};
use crate::math::{saturating_sub, GridIndex, GridIndexExt, Vect};

#[spirv_bindgen]
#[cfg_attr(feature = "dim1", spirv(compute(threads(1, 1, 64))))]
#[cfg_attr(feature = "dim2", spirv(compute(threads(8, 8, 1))))]
#[cfg_attr(feature = "dim3", spirv(compute(threads(4, 4, 4))))]
fn fdtd_lossy_v2(
    #[spirv(global_invocation_id)] idx3: UVec3,
    // Vector fields
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] h: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dn: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] en: &mut [Vec4],
    // Field update terms
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] integrals: &mut [PmlIntegrals2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] grid_coeffs: &[PmlCoefficients2],
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
    let mut dipole_src_term = 0.;
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
        dipole_src_term += source_vals.read(vals_i.min(end)) * enable as u32 as f32;
    }

    let coeffs = grid_coeffs.read(idx);
    let mut ints = integrals.read(idx);
    let en_self = en.read(idx);

    // H Update
    let h_self = {
        let not_boundary = UVec3::from(idx3.cmplt(boundary_idx3)).as_vec3();
        let en_curl = compute_curl::<true>(
            idx,
            grid.flat_idx_incrs,
            grid.d,
            not_boundary,
            en,
            en_self
        );
        let old_h = h.read(idx);
        #[allow(unused_mut)]
        let mut h_self = coeffs.h1 * old_h + coeffs.h2 * en_curl;
        #[cfg(any(feature = "dim2", feature = "dim3"))]
        {
            ints.en_curl += en_curl;
            h_self += coeffs.h3 * ints.en_curl;
            #[cfg(feature = "dim3")]
            {
                ints.h += old_h;
                h_self += coeffs.h4 * ints.h;
            }
        }
        h.write(idx, h_self);
        h_self
    };

    // Dn Update
    let dn_self = {
        let not_boundary = UVec3::from(idx3.cmpgt(UVec3::ZERO)).as_vec3();
        let h_curl = compute_curl::<false>(
            idx,
            grid.flat_idx_incrs,
            grid.d,
            not_boundary,
            h,
            h_self,
        );
        let old_dn = dn.read(idx);
        ints.en += en_self;
        #[allow(unused_mut)]
        let mut dn_self = coeffs.dn1 * old_dn + coeffs.dn2 * h_curl + // regular update terms
            coeffs.dn_loss1 * en_self + coeffs.dn_loss2 * ints.en + // loss terms
            dipole_src_term;
        #[cfg(any(feature = "dim2", feature = "dim3"))]
        {
            ints.h_curl += h_curl;
            dn_self += coeffs.dn3 * ints.h_curl;
            #[cfg(feature = "dim3")]
            {
                ints.dn += old_dn;
                dn_self += coeffs.dn4 * ints.dn;
            }
        }
        dn.write(idx, dn_self);
        dn_self
    };

    let en_self = coeffs.en1 * dn_self;
    en.write(idx, en_self);

    integrals.write(idx, ints);

    if idx3 == UVec3::ZERO {
        *steps += 1;
    }
}

/// `POL_MODE == false` => TM mode
/// `POL_MODE == true` => TE mode
#[inline(always)]
fn compute_curl<const FORWARDS: bool>(
    idx: usize,
    flat_idx_incrs: UVec3,
    d: Vec3,
    not_boundary: Vec3,
    v_field: &[Vec4],
    v_self: Vec4
) -> Vec4 {
    macro_rules! curl_diff {
        ($field_elem:ident, $diff_elem:ident) => {{
            let neighbor_idx =
                if FORWARDS { idx + flat_idx_incrs.$diff_elem as usize }
                else { idx.wrapping_sub(flat_idx_incrs.$diff_elem as usize) }
                    .min(v_field.len() - 1);
            let neighbor = v_field.read(neighbor_idx).$field_elem * not_boundary.$diff_elem;
            if FORWARDS { (neighbor - v_self.$field_elem) / d.$diff_elem }
            else { (v_self.$field_elem - neighbor) / d.$diff_elem }
        }};
    }

    let mut v_curl = Vec4::ZERO;

    v_curl.x = cfg_select! {
        feature = "dim1" => -curl_diff!(y, z),
        feature = "dim2" => curl_diff!(z, y),
        feature = "dim3" => curl_diff!(z, y) - curl_diff!(y, z),
    };
    v_curl.y = cfg_select! {
        feature = "dim1" => curl_diff!(x, z),
        feature = "dim2" => -curl_diff!(z, x),
        feature = "dim3" => curl_diff!(x, z) - curl_diff!(z, x),
    };
    v_curl.z = cfg_select! {
        feature = "dim1" => 0.,
        any(feature = "dim2", feature = "dim3") => curl_diff!(y, x) - curl_diff!(x, y),
    };

    v_curl
}

#[derive(Copy, Clone, Pod, Zeroable, Default, Debug)]
#[repr(C)]
pub struct PmlCoefficients2 {
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

#[derive(Copy, Clone, Pod, Zeroable, Default, Debug)]
#[repr(C)]
pub struct PmlIntegrals2 {
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