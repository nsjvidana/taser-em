#![cfg_attr(target_arch = "spirv", no_std)]
#![allow(clippy::too_many_arguments)]

use crate::dft::ComplexVec3;
use crate::math::*;
use khal_std::index::MaybeIndexUnchecked;

pub mod math;
pub mod fdtd;
pub mod boundary;
pub mod dft;
pub mod sensor;
pub mod source;

#[macro_export]
#[doc(hidden)]
macro_rules! cfg_gpu {
    ($($input:tt)*) => {
        cfg_select! {
            any(target_arch = "spirv", target_arch = "nvptx64") => {
                $($input)*
            }
            _ => {}
        }
    };
}

/// Conditionally compiles anything passed into the macro for CPU targets only.
#[macro_export]
#[doc(hidden)]
macro_rules! cfg_cpu {
    { $($input:tt)* } => {
        cfg_select! {
            not(any(target_arch = "spirv", target_arch = "nvptx64")) => {
                $($input)*
            }
            _ => {}
        }
    };
}

/// Creates workgroup counts from thread counts, given the workgroup size as an array.
#[macro_export]
#[doc(hidden)]
macro_rules! workgroup_counts {
    ($threads:expr, $wg_size:expr) => {{
        let t = &$threads;
        core::array::from_fn(|i| t[i].div_ceil($wg_size[i]))
    }};
}

macro_rules! en_spatial_avg_impl {
    ($name:ident, $v:ident) => {
        pub(crate) fn $name<const SAFE: bool>(en: &[$v], en_self: $v, idx: usize, flat_idx_incrs: UVec3, cell_count: usize) -> $v {
            let idx_pyz = idx + (flat_idx_incrs.y + flat_idx_incrs.z) as usize;
            let idx_pzx = idx + (flat_idx_incrs.z + flat_idx_incrs.x) as usize;
            let idx_pxy = idx + (flat_idx_incrs.x + flat_idx_incrs.y) as usize;
            let (en_pyz, en_pzx, en_pxy) =
                if SAFE {
                    let idx_pyz_enable = idx_pyz < cell_count;
                    let idx_pzx_enable = idx_pzx < cell_count;
                    let idx_pxy_enable = idx_pxy < cell_count;
                    let en_pyz = en.read(idx_pyz * idx_pyz_enable as usize);
                    let en_pzx = en.read(idx_pzx * idx_pzx_enable as usize);
                    let en_pxy = en.read(idx_pxy * idx_pxy_enable as usize);
                    (
                        if idx_pyz_enable { en_pyz } else { en_self },
                        if idx_pzx_enable { en_pzx } else { en_self },
                        if idx_pxy_enable { en_pxy } else { en_self }
                    )
                }
                else { (en.read(idx_pyz), en.read(idx_pzx), en.read(idx_pxy)) };
            $v {
                x: (en_self.x + en_pyz.x) * 0.5,
                y: (en_self.y + en_pzx.y) * 0.5,
                z: (en_self.z + en_pxy.z) * 0.5,
            }
        }
    };
}

en_spatial_avg_impl!(en_spatial_avg_cmplx, ComplexVec3);
// TODO: en_spatial_avg_impl!(en_spatial_avg, Vec3); ...use this in instantaneous flux

macro_rules! h_spatial_avg_impl {
    ($name:ident, $v:ident) => {
        pub(crate) fn $name<const SAFE: bool>(h: &[$v], h_self: $v, idx: usize, flat_idx_incrs: UVec3, cell_count: usize) -> $v {
            let idx_px = idx + flat_idx_incrs.x as usize;
            let idx_py = idx + flat_idx_incrs.y as usize;
            let idx_pz = idx + flat_idx_incrs.z as usize;
            let (h_px, h_py, h_pz) =
                if SAFE {
                    let idx_px_enable = idx_px < cell_count;
                    let idx_py_enable = idx_py < cell_count;
                    let idx_pz_enable = idx_pz < cell_count;
                    let h_px = h.read(idx_px * idx_px_enable as usize);
                    let h_py = h.read(idx_py * idx_py_enable as usize);
                    let h_pz = h.read(idx_pz * idx_pz_enable as usize);
                    (
                        if idx_px_enable { h_px } else { h_self },
                        if idx_py_enable { h_py } else { h_self },
                        if idx_pz_enable { h_pz } else { h_self }
                    )
                }
                else { (h.read(idx_px), h.read(idx_py), h.read(idx_pz)) };
            $v {
                x: (h_self.x + h_px.x) * 0.5,
                y: (h_self.y + h_py.y) * 0.5,
                z: (h_self.z + h_pz.z) * 0.5,
            }
        }
    };
}

h_spatial_avg_impl!(h_spatial_avg_cmplx, ComplexVec3);
// TODO: h_spatial_avg_impl!(h_spatial_avg, Vec3); ...use this in instantaneous flux