use bytemuck::{Pod, Zeroable};
use khal_std::glamx::{UVec3, Vec2, Vec4};
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

/// Lossless 1-dimensional FDTD kernel in "Ey" mode.
///
/// Only the x and y components of E and H fields, respectively, are simulated.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn fdtd1_dn_y(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] h_x: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dn_y: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] en_y: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] int_en_y: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] coeffs: &[PmlCoefficients],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] grid: &GridParameters,
) {
    let idx = id.x as usize;
    if idx >= dn_y.len() { return; }
    let boundary_idx = dn_y.len() - 1;
    let dz = grid.dz;

    let PmlCoefficients {
        h_x_coeffs,
        dn_y_coeffs,
        en_y_coeff,
        ..
    } = coeffs.read(idx);
    let en_y_k = en_y.read(idx);

    let not_boundary = (idx < boundary_idx) as u32 as f32;
    let en_y_k1 = en_y.read((idx + 1).max(boundary_idx)) * not_boundary;
    let e_curl_x = -(en_y_k1 - en_y_k) / dz;
    let new_h_x_k = h_x_coeffs.dot(Vec2::new(
        h_x.read(idx), e_curl_x
    ));
    *h_x.at_mut(idx) = new_h_x_k;

    let new_int_en_y_k = int_en_y.read(idx) + en_y_k;

    let not_boundary = (idx > 0) as u32 as f32;
    let h_x_km1 = h_x.read(idx.wrapping_sub(1).clamp(0, boundary_idx)) * not_boundary;
    let h_curl_y = (new_h_x_k - h_x_km1) / dz;
    let new_dn_y_k = dn_y_coeffs.dot(Vec4::new(
        dn_y.read(idx), h_curl_y, en_y_k, new_int_en_y_k
    ));
    *dn_y.at_mut(idx) = new_dn_y_k;
    *int_en_y.at_mut(idx) = new_int_en_y_k;

    *en_y.at_mut(idx) = en_y_coeff * new_dn_y_k;
}

#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct GridParameters {
    pub dz: f32
}

#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct PmlCoefficients {
    pub h_x_coeffs: Vec2,
    // pub h_x1: f32,
    // pub h_x2: f32,

    pub en_y_coeff: f32,
    pub _padding0: u32,

    pub dn_y_coeffs: Vec4,
    // pub dn_y1: f32,
    // pub dn_y2: f32,
    // pub dn_y5: f32,
    // pub dn_y6: f32,

}