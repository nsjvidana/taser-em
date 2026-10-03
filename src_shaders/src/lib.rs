#![cfg_attr(target_arch = "spirv", no_std)]
#![allow(clippy::too_many_arguments)]

pub mod math;
pub mod fdtd;
pub mod boundary;
pub mod dft;
pub mod monitor;
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

#[macro_export]
#[doc(hidden)]
macro_rules! workgroup_counts {
    ($threads:expr, $wg_size:expr) => {{
        let t = &$threads;
        core::array::from_fn(|i| t[i].div_ceil($wg_size[i]))
    }};
}