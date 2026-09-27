use std::ops::Deref;
use crate::gpu_util::*;
use crate::prelude::TaserResult;
use khal::backend::{Backend, Buffer, DispatchGrid, Encoder, GpuBackend, GpuBuffer, GpuEncoder, GpuPass};
use std::sync::Arc;
use khal::BufferUsages;
use taser_em_shaders::dft::*;
use taser_em_shaders::fdtd::GridParameters;
use taser_em_shaders::math::*;

/// Describes the DFT of some function.
///
/// Includes all frequencies to be resolved in DFT
#[derive(Clone)]
pub struct Dft<Func: DftFunction + ?Sized> {
    /// The frequencies of the DFT in Hz
    frequencies: Vec<Real>,
    function: Arc<Func>,
}

impl<Func: DftFunction> Dft<Func> {
    pub fn new(frequencies: Vec<Real>, function: Arc<Func>) -> TaserResult<Self> {
        if frequencies.is_empty() { return Err(DftError::NoFrequencies.into()) };
        Ok(Self {
            frequencies,
            function,
        })
    }

    pub fn get_frequencies(&self) -> &Vec<Real> { &self.frequencies }

    pub fn get_function(&self) -> &Arc<Func> { &self.function }
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

    pub fn initialize_states<Func: DftFunction>(
        &self,
        pass: &mut GpuPass,
        grid: &GpuBuffer<GridParameters>,
        dft_states: &mut DftStates<Func>
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

    /// Updates `dft_states` buffers with proper function values and
    pub fn encode_steps_full<Func: DftFunction>(
        &self,
        encoder: &mut GpuEncoder,
        time_step: &GpuBuffer<u32>,
        dft_states: &mut DftStates<Func>,
    ) -> TaserResult<()> {
        Self::encode_copy(encoder, dft_states)?;
        let mut pass = encoder.begin_pass("", None);
        self.dispatch_step(
            &mut pass,
            time_step,
            dft_states
        )
    }

    pub fn encode_copy<Func: DftFunction>(encoder: &mut GpuEncoder, dft_states: &mut DftStates<Func>) -> TaserResult<()> {
        for (off, func) in dft_states.dft_funcs.iter().enumerate() {
            func.copy_to_dft_buffer(encoder, &mut dft_states.function_values, off)?;
        }
        Ok(())
    }

    /// Dispatch DFT step in time.
    ///
    /// # Arguments
    /// - `time_step` - buffer of current time step index.
    /// - `functions` - buffer of the instantaneous values (occurring at `time_step` time step) of
    ///   every function whose DFT is being computed (parallel w/ `func_dfts`).
    /// - `dft_states` - states of function DFTs.
    pub fn dispatch_step<Func: DftFunction>(
        &self,
        pass: &mut GpuPass,
        time_step: &GpuBuffer<u32>,
        dft_states: &mut DftStates<Func>
    ) -> TaserResult<()> {
        self.dft_shader.call(
            pass,
            DispatchGrid::Grid(dft_states.workgroups),
            time_step,
            &dft_states.function_values,
            &dft_states.func_dfts,
            &dft_states.dft_kernels,
            &mut dft_states.dft_outputs
        )?;
        Ok(())
    }
}

/// The states of multiple DFTs that will be evaluated in one shader dispatch
pub struct DftStates<Func: DftFunction> {
    /// The functions stored on CPU side.
    pub dft_funcs: Vec<Arc<Func>>,

    // Buffers / GPU data
    pub function_values: GpuBuffer<Real>,
    pub func_dfts: GpuBuffer<GpuFunctionDft>,
    pub dft_frequencies: GpuBuffer<Real>,
    pub dft_kernels: GpuBuffer<Complex32>,
    pub dft_outputs: GpuBuffer<Complex32>,
    pub workgroups: [u32; 3]
}

impl<Func: DftFunction> DftStates<Func> {
    /// Creates new zeroed-out DFT buffers.
    ///
    /// Use [`DftPipeline::initialize_states`] to populate buffers with non-zero kernels.
    pub fn new_zeroed(backend: &GpuBackend, dfts: &[Dft<Func>]) -> TaserResult<Self> {
        if dfts.is_empty() { return Err(DftError::NoDfts.into()) };

        let dft_funcs = dfts.iter()
            .map(|dft| dft.function.clone())
            .collect::<Vec<_>>();

        let function_values = backend.init_buffer(
            vec![0.; dft_funcs.len()].as_slice(),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        )?;

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
            .flat_map(|d| &d.frequencies)
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
            dft_funcs,
            function_values,
            func_dfts,
            dft_frequencies,
            dft_kernels: dft_kernels_zeroed,
            dft_outputs: dft_outputs_zeroed,
            workgroups: dft_workgroups(max_n_kernels, n_functions),
        })
    }
}

pub trait DftFunction {
    /// Convert this [`DftFunction`] into a [`Dft`] with the specified `frequencies`.
    fn to_dft(self, frequencies: Vec<Real>) -> TaserResult<Dft<Self>>;

    /// Get the buffer where this function's instantaneous value is stored.
    fn get_buffer(&self) -> impl Deref<Target=GpuBuffer<Real>>;

    /// The index of this function's instantaneous value in [`Self::get_buffer`] buffer.
    fn get_value_position(&self) -> usize;

    /// Used in copying the function's value into DFT shader buffer before running the DFT shader.
    fn copy_to_dft_buffer(&self, encoder: &mut GpuEncoder, dft_buf: &mut GpuBuffer<Real>, dft_buf_offset: usize) -> TaserResult<()> {
        encoder.copy_buffer_to_buffer(
            &*self.get_buffer(),
            self.get_value_position(),
            dft_buf,
            dft_buf_offset,
            1
        )?;
        Ok(())
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
    #[error("Expected at least one DFT")]
    NoDfts
}