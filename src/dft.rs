use crate::gpu_util::CreateGpuBuffer;
use crate::prelude::TaserResult;
use khal::backend::{Buffer, DispatchGrid, GpuBackend, GpuBuffer, GpuPass};
use taser_em_shaders::dft::*;
use taser_em_shaders::fdtd::GridParameters;
use taser_em_shaders::math::*;

/// Describes the DFT of some function.
#[derive(Clone, Debug)]
pub struct Dft {
    /// The frequencies of the DFT in Hz
    frequencies: Vec<Real>,
}

impl Dft {
    pub fn new(frequencies: Vec<Real>) -> TaserResult<Self> {
        if frequencies.is_empty() { return Err(DftError::NoFrequencies.into()) };
        Ok(Self { frequencies })
    }

    pub fn get_frequencies(&self) -> &Vec<Real> {
        &self.frequencies
    }
}

pub struct DftPipeline {
    init_kernels: GpuComputeDftKernels,
    dft_shader: GpuDftShader
}

impl DftPipeline {
    pub fn new(backend: &GpuBackend) -> TaserResult<Self> {
        Ok(Self {
            init_kernels: GpuComputeDftKernels::from_dir(backend, &crate::SPIRV_DIR)?,
            dft_shader: GpuDftShader::from_dir(backend, &crate::SPIRV_DIR)?,
        })
    }

    pub fn initialize_states(
        &self,
        pass: &mut GpuPass,
        grid: &GpuBuffer<GridParameters>,
        dft_states: &mut DftStates
    ) -> TaserResult<()> {
        self.init_kernels.call(
            pass,
            DispatchGrid::Grid(dft_states.workgroups),
            grid,
            &dft_states.func_dfts,
            &dft_states.dft_frequencies,
            &mut dft_states.dft_kernels
        )?;
        Ok(())
    }

    /// Dispatch DFT step in time.
    ///
    /// # Arguments
    /// - `time_step` - buffer of current time step index.
    /// - `functions` - buffer of the instantaneous values (occurring at `time_step` time step) of
    ///                 every function whose DFT is being computed (parallel w/ `func_dfts`).
    /// - `dft_states` - states of function DFTs.
    pub fn dispatch_step(
        &self,
        pass: &mut GpuPass,
        time_step: &GpuBuffer<u32>,
        functions: &GpuBuffer<Real>,
        dft_states: &mut DftStates
    ) -> TaserResult<()> {
        self.dft_shader.call(
            pass,
            DispatchGrid::Grid(dft_states.workgroups),
            time_step,
            functions,
            &dft_states.func_dfts,
            &dft_states.dft_kernels,
            &mut dft_states.dft_outputs
        )?;
        Ok(())
    }
}

/// The states of multiple DFTs
pub struct DftStates {
    pub func_dfts: GpuBuffer<GpuFunctionDft>,
    pub dft_frequencies: GpuBuffer<Real>,
    pub dft_kernels: GpuBuffer<Complex32>,
    pub dft_outputs: GpuBuffer<Complex32>,
    pub workgroups: [u32; 3]
}

impl DftStates {
    /// Creates new zeroed-out DFT buffers.
    ///
    /// Use [`DftPipeline::initialize_states`] to populate buffers with non-zero kernels.
    pub fn new_zeroed(backend: &GpuBackend, dfts: &Vec<Dft>) -> TaserResult<Self> {
        let mut kernels_start = 0;
        let func_dfts = dfts.iter()
            .map(|dft| {
                let kernels_end = kernels_start + dft.frequencies.len() as u32 - 1;
                let gpu_ver = GpuFunctionDft {
                    kernels_start,
                    kernels_end,
                };
                kernels_start = kernels_end + 1;
                gpu_ver
            })
            .collect::<Vec<_>>()
            .create_gpu_buffer(backend)?;
        let dft_frequencies = dfts.iter()
            .map(|d| &d.frequencies)
            .flatten()
            .copied()
            .collect::<Vec<_>>()
            .create_gpu_buffer(backend)?;
        let dft_kernels_zeroed = vec![Complex32::ZERO; dft_frequencies.len()]
            .create_gpu_buffer(backend)?;
        let dft_outputs_zeroed = vec![Complex32::ZERO; dft_frequencies.len()]
            .create_gpu_buffer(backend)?;

        let max_n_kernels = dfts.iter()
            .map(|d| d.frequencies.len())
            .max()
            .expect("Dft type can't be initialized with an empty frequencies vec") as u32;
        let n_functions = func_dfts.len() as u32;

        Ok(Self {
            func_dfts,
            dft_frequencies,
            dft_kernels: dft_kernels_zeroed,
            dft_outputs: dft_outputs_zeroed,
            workgroups: dft_workgroups(max_n_kernels, n_functions),
        })
    }
}

/// Helper function for creating a list of frequencies for a DFT.
///
/// # Arguments
/// - `range` - range of frequencies to resolve.
/// - `resolution` - splits `range` by this resolution such that the output [`Vec`] will have
///   `resolution + 1` elements.
pub fn frequencies_from_range(range: core::ops::RangeInclusive<Real>, resolution: usize) -> Vec<Real> {
    let mut frequencies = vec![0.; resolution + 1];
    let df = (range.end() - range.start()) / resolution as Real;
    for (i, f) in frequencies.iter_mut().enumerate() {
        *f = range.start() + df * i as Real;
    }
    frequencies
}

#[derive(thiserror::Error, Debug)]
pub enum DftError {
    #[error("Expected at least one frequency in DFT but none were provided")]
    NoFrequencies,
}