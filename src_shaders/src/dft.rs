use crate::fdtd::GridParameters;
use crate::math::*;
use bytemuck::{Pod, Zeroable};
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::*;
use num_complex::Complex32;

pub const DFT_WORKGROUP_SIZE: UVec3 = UVec3::new(64, 1, 1);

/// Calculates DFT kernels that will be used by the DFT shader. You only need to dispatch this
/// once before using `dft_kernels` in a DFT algorithm.
///
/// Don't confuse "kernels" with "shader kernels" here. Kernels in the case of DFTs refers to the constant
/// complex number associated with each frequency.
///
/// # Arguments
/// - `grid` - used for the delta-time only.
/// - `dft_infos` - Describes every DFT the kernel will compute.
/// - `dft_frequencies` - the frequencies of every DFT.
/// - `dft_kernels` - the kernels of every DFT.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn gpu_compute_dft_kernels(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &GridParameters,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dft_infos: &[DftInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] dft_frequencies: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dft_kernels: &mut [Complex32],
) {
    // X idx is kernel index, Z idx is the DFT index.
    let dft_idx = id.z as usize;
    let dft = dft_infos.read(dft_idx);
    let kernel_idx = id.x;
    if kernel_idx >= dft.kernel_idx_range() { return; }

    let global_kernel_idx = dft.global_kernel_idx(kernel_idx) as usize;
    let f = dft_frequencies.read(global_kernel_idx);
    let kernel = Complex32::from_polar(1., -core::f32::consts::TAU * f * grid.dt);
    dft_kernels.write(global_kernel_idx, kernel);
}

/// Discrete Fourier Transform algorithm that runs in series with FDTD steps.
///
/// Don't confuse "kernels" with "shader kernels" here. Kernels in the case of DFTs refers to the constant
/// complex number associated with each frequency.
///
/// # Arguments
/// - `time_step` - the index of current time step.
/// - `functions` - each element is the instantaneous value (occurring at `time_step` time step) of
///                 a function that a DFT is being run on.
/// - `dft_infos` - Descriptors of each DFT this shader will compute (one DFT for each function).
/// - `dft_kernels` - the kernels of every frequency, of every DFT.
/// - `dfts` - the DFT values of every frequency, of every function.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn gpu_dft(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] time_step: &u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] functions: &[Real], // parallel w/ dft_infos
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] dft_infos: &[DftInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dft_kernels: &[Complex32], // parallel w/ dfts
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] dfts: &mut [Complex32],
) {
    let dft_idx = id.z as usize;
    let kernel_idx = id.x;
    let dft = dft_infos.read(dft_idx);
    if kernel_idx >= dft.kernel_idx_range() { return; }

    let global_kernel_idx = dft.global_kernel_idx(kernel_idx) as usize;
    let k = dft_kernels.read(global_kernel_idx).powu(*time_step);
    *dfts.at_mut(global_kernel_idx) += k * functions.read(dft_idx);
}

/// Compute workgroup count for DFT kernel for multiple DFTs.
///
/// # Arguments
/// - `max_n_kernels` is the maximum number of frequencies that a single DFT will resolve across all the DFTs.
///   This is necessary to allow running multiple DFTs under one dispatch.
/// - `n_functions` is the number of functions that will have their DFTs resolved in one dispatch.
pub fn dft_workgroups(max_n_kernels: u32, n_functions: u32) -> [u32; 3] {
    [
        max_n_kernels.div_ceil(DFT_WORKGROUP_SIZE.x),
        1u32.div_ceil(DFT_WORKGROUP_SIZE.y),
        n_functions.div_ceil(DFT_WORKGROUP_SIZE.z)
    ]
}

#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct DftInfo {
    pub kernels_start: u32,
    pub kernels_end: u32,
}

impl DftInfo {
    #[inline]
    pub fn kernel_idx_range(&self) -> u32 {
        self.kernels_end - self.kernels_start
    }

    #[inline]
    pub fn global_kernel_idx(&self, kernel_idx: u32) -> u32 {
        self.kernels_start + kernel_idx
    }
}