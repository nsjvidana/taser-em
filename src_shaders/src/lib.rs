#![cfg_attr(target_arch = "spirv", no_std)]
#![allow(clippy::too_many_arguments)]

pub mod math;
pub mod fdtd;

#[macro_export]
#[doc(hidden)]
macro_rules! cfg_gpu {
    ($($input:tt)*) => {
        #[cfg(any(target_arch = "spirv", target_arch = "nvptx64"))]
        $($input)*
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! cfg_cpu {
    ($($input:tt)*) => {
        #[cfg(not(any(target_arch = "spirv", target_arch = "nvptx64")))]
        $($input)*
    };
}