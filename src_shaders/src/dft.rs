cfg_cpu! {
    use khal::ShaderArgsType;
}

use crate::fdtd::GridParameters;
use crate::math::*;
use bytemuck::{Pod, Zeroable};
use derive_more::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::*;
use num_complex::{Complex, Complex32};
use crate::{cfg_cpu, workgroup_counts};

/// Calculates DFT kernels that will be used by the DFT shader. You only need to dispatch this
/// once before using `dft_kernels` in a DFT algorithm.
///
/// Don't confuse "kernels" with "shader kernels" here. Kernels in the case of DFTs refers to the constant
/// complex number associated with each frequency.
///
/// # Arguments
/// - `grid` - used for the delta-time only.
/// - `func_dfts` - Descriptors of every function's DFT.
/// - `dft_frequencies` - the frequencies of every DFT in one array.
/// - `dft_kernels` - the kernels of every DFT in one array.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn gpu_compute_dft_kernels(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &GridParameters,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] func_dfts: &[GpuFunctionDft],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] dft_frequencies: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dft_kernels: &mut [Complex32],
) {
    // X idx is kernel index, Z idx is the function index.
    let func_idx = id.z as usize;
    let func_dft = func_dfts.read(func_idx);
    let kernel_idx = id.x;
    if kernel_idx >= func_dft.kernel_count() { return; }

    let global_kernel_idx = func_dft.global_kernel_idx(kernel_idx) as usize;
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
/// - `function_values` - the instantaneous value (occurring at `time_step` time step) of
///                 every function whose DFT is being computed (parallel w/ `func_dfts`).
/// - `func_dfts` - Descriptors of each function DFT this shader will compute.
/// - `dft_kernels` - the kernels of every frequency, of every DFT (parallel w/ `dfts`).
/// - `dft_outputs` - the DFT values of every frequency, of every function.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn gpu_dft_shader(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] time_step: &u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] function_value_positions: &[Index],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] function_values: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] func_dfts: &[GpuFunctionDft],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] dft_kernels: &[Complex32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] dft_outputs: &mut [Complex32],
) {
    let dft_idx = id.z as usize;
    let kernel_idx = id.x;
    let dft = func_dfts.read(dft_idx);
    if kernel_idx >= dft.kernel_count() { return; }

    let global_kernel_idx = dft.global_kernel_idx(kernel_idx) as usize;
    let k = dft_kernels.read(global_kernel_idx).powu(*time_step);
    let f_val_idx = function_value_positions.read(dft_idx) as usize;
    *dft_outputs.at_mut(global_kernel_idx) += k * function_values.read(f_val_idx);
}

cfg_cpu! {
    /// Compute workgroup count for DFT kernel for multiple DFTs.
    ///
    /// # Arguments
    /// - `max_n_kernels` is the maximum number of frequencies that a single DFT will resolve across all the DFTs.
    ///   This is necessary to allow running multiple DFTs under one dispatch.
    /// - `n_functions` is the number of functions that will have their DFTs resolved in one dispatch.
    pub fn dft_workgroups(max_n_kernels: u32, n_functions: u32) -> [u32; 3] {
        workgroup_counts!([max_n_kernels, 1u32, n_functions, ], GpuDftShaderArgs::WORKGROUP_SIZE)
    }
}

/// Describes the DFT of a function for the [`GpuDft`] shader.
#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct GpuFunctionDft {
    pub kernels_start: u32,
    pub kernels_end: u32,
}

impl GpuFunctionDft {
    #[inline]
    pub fn kernel_count(&self) -> u32 {
        self.kernels_end - self.kernels_start
    }

    #[inline]
    pub fn global_kernel_idx(&self, kernel_idx: u32) -> u32 {
        self.kernels_start + kernel_idx
    }
}

#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn gpu_init_dft_regions(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &GpuDftRegionsParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] regions: &mut [GpuDftRegion],
) {
    let region_idx = id.x as usize;
    if region_idx >= regions.len() { return; }
    let region = regions.at_mut(region_idx);
    region.kernel = Complex32::from_polar(1., -core::f32::consts::TAU * region.freq * params.dt);
}

cfg_cpu! {
    pub fn init_dft_workgroups(regions_params: &GpuDftRegionsParams) -> [u32; 3] {
        workgroup_counts!([regions_params.n_regions, 1, 1], GpuInitDftRegionsArgs::WORKGROUP_SIZE)
    }
}

// One GpuDftRegion per region frequency.
// the grid in this shader is a maximum region grid repeated on the Z axis for each gpu region.
#[spirv_bindgen]
#[cfg_attr(feature = "dim1", spirv(compute(threads(1, 1, 64))))]
#[cfg_attr(feature = "dim2", spirv(compute(threads(8, 8, 1))))]
#[cfg_attr(feature = "dim3", spirv(compute(threads(4, 4, 4))))]
pub fn gpu_dft_regions_update(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &GpuDftRegionsParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] time_step: &u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] h: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] en: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] regions: &[GpuDftRegion],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] h_dfts: &mut [ComplexVec3],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] en_dfts: &mut [ComplexVec3],
) {
    let region_idx = id.z / params.max_region_n_cells.z;
    let region = regions.read(region_idx as usize);
    let local_cell_idx = id.with_z(id.z - region_idx * params.max_region_n_cells.z);
    if local_cell_idx.cmpge(region.n_cells3).any() { return; }

    let idx_dft = (region.dft_start_idx + grid_idx3_to_flat_idx(local_cell_idx, region.n_cells3)) as usize;
    let cell_idx = GridIndex::from_uvec3(region.cell_idx_min + local_cell_idx);
    let idx = cell_idx.to_flat_idx(GridIndex::from_uvec3(params.n_cells3)) as usize;

    let k = region.kernel.powu(*time_step);
    // TODO: post-process H DFTs with e^(-jw*dt/2)
    *h_dfts.at_mut(idx_dft) += k * ComplexVec3::from_real(h.read(idx).xyz());
    *en_dfts.at_mut(idx_dft) += k * ComplexVec3::from_real(en.read(idx).xyz());
}

cfg_cpu! {
    pub fn dft_regions_update_workgroups(regions_params: &GpuDftRegionsParams) -> [u32; 3] {
        let thread_count = regions_params.max_region_n_cells
            .with_z(regions_params.max_region_n_cells.z * regions_params.n_regions)
            .to_array();
        workgroup_counts!(thread_count, GpuDftRegionsUpdateArgs::WORKGROUP_SIZE)
    }
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct GpuDftRegionsParams {
    pub max_region_n_cells: UVec3,
    pub n_regions: u32,
    /// Same as [`GridParameters::n_cells3`]
    pub n_cells3: UVec3,
    /// Same as [`GridParameters::dt`]
    pub dt: Real
}

impl GpuDftRegionsParams {
    pub fn new(n_cells3: UVec3, dt: Real, dft_regions: &[GpuDftRegion]) -> Self {
        let mut max_region_n_cells = UVec3::MIN;
        for region in dft_regions {
            max_region_n_cells = max_region_n_cells.max(region.n_cells3);
        }
        Self {
            max_region_n_cells,
            n_regions: dft_regions.len() as u32,
            n_cells3,
            dt,
        }
    }
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct GpuDftRegion {
    pub cell_idx_min: UVec3,
    pub dft_start_idx: u32,
    pub n_cells3: UVec3,
    pub freq: Real,
    pub kernel: Complex32,
    pub cell_count: u32,
    pub _padding0: u32,
}

bitflags::bitflags! {
    #[derive(Copy, Clone, Pod, Zeroable, Default, Debug)]
    #[repr(C)]
    pub struct DftFields: u32 {
        const H = 1 << 0;
        // TODO: const DN = 1 << 1;
        const EN = 1 << 2;
        // TODO: const SN = Self::H.bits() | Self::EN.bits() | (1 << 3); ...power.
        //       don't forget to update DftMask functions after adding these, if necessary
    }
}

impl DftFields {
    #[inline]
    pub fn enabled(&self) -> bool {
        self.intersects(Self::H | Self::EN)
    }
}

// TODO: move this to math module
#[derive(Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign)]
#[mul(forward)]
#[mul_assign(forward)]
#[div(forward)]
#[div_assign(forward)]
#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct ComplexVec3 {
    pub x: Complex32,
    pub y: Complex32,
    pub z: Complex32,
}

impl ComplexVec3 {
    /// Complex cross product
    pub fn cross(self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - rhs.y * self.z,
            y: self.z * rhs.x - rhs.z * self.x,
            z: self.x * rhs.y - rhs.x * self.y
        }
    }

    /// Complex conjugate of vector
    pub fn conj(self) -> Self {
        Self {
            x: self.x.conj(),
            y: self.y.conj(),
            z: self.z.conj(),
        }
    }
}

impl ComplexVec3 {
    pub const ZERO: Self = Self::splat(Complex32::ZERO);

    #[inline]
    pub const fn splat(v: Complex<f32>) -> Self {
        Self { x: v, y: v, z: v, }
    }

    #[inline]
    pub const fn from_real(real: Vec3) -> Self {
        Self {
            x: Complex32::new(real.x, 0.),
            y: Complex32::new(real.y, 0.),
            z: Complex32::new(real.z, 0.),
        }
    }
}

impl core::ops::Mul<Complex32> for ComplexVec3 {
    type Output = Self;

    fn mul(self, rhs: Complex32) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl core::ops::Mul<ComplexVec3> for Complex32 {
    type Output = ComplexVec3;

    fn mul(self, rhs: ComplexVec3) -> Self::Output {
        ComplexVec3 {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
        }
    }
}

impl core::ops::Mul<Real> for ComplexVec3 {
    type Output = Self;

    fn mul(self, rhs: Real) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl core::ops::Mul<ComplexVec3> for Real {
    type Output = ComplexVec3;

    fn mul(self, rhs: ComplexVec3) -> Self::Output {
        ComplexVec3 {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
        }
    }
}

impl GpuDynamicIndex<Axis> for ComplexVec3 {
    type Output = Complex32;

    fn dyn_idx(&self, index: Axis) -> Self::Output {
        match index {
            Axis::X => self.x,
            Axis::Y => self.y,
            Axis::Z => self.z,
        }
    }

    fn dyn_insert(&mut self, index: Axis, val: Self::Output) {
        match index {
            Axis::X => self.x = val,
            Axis::Y => self.y = val,
            Axis::Z => self.z = val,
        }
    }
}