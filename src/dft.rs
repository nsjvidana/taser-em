use std::ops::Range;
use crate::gpu_util::*;
use crate::prelude::TaserResult;
use khal::backend::{Backend, Buffer, DispatchGrid, Encoder, GpuBackend, GpuBuffer, GpuPass, GpuReadback};
use std::sync::Arc;
use taser_em_shaders::dft::*;
use taser_em_shaders::fdtd::GridParameters;
use taser_em_shaders::math::*;
use crate::fdtd::FdtdLossyState;

/// Describes the DFT of some function.
///
/// Includes all frequencies to be resolved in DFT
#[derive(Clone)]
pub struct Dft<Func: ToDft + ?Sized> {
    /// The frequencies of the DFT in Hz
    frequencies: Vec<Real>,
    function: Arc<Func>,
}

impl<Func: ToDft> Dft<Func> {
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

    pub fn initialize_states<Func: ToDft>(
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

    /// Dispatch DFT step in time.
    ///
    /// # Arguments
    /// - `time_step` - buffer of current time step index.
    /// - `functions` - buffer of the instantaneous values (occurring at `time_step` time step) of
    ///   every function whose DFT is being computed (parallel w/ `func_dfts`).
    /// - `dft_states` - states of function DFTs.
    pub fn dispatch_step<Func: ToDft>(
        &self,
        pass: &mut GpuPass,
        time_step: &GpuBuffer<u32>,
        function_state: &Func::GpuStateType,
        dft_states: &mut DftStates<Func>
    ) -> TaserResult<()> {
        self.dft_shader.call(
            pass,
            DispatchGrid::Grid(dft_states.workgroups),
            time_step,
            &dft_states.function_value_positions,
            Func::get_value_buffer(function_state),
            &dft_states.func_dfts,
            &dft_states.dft_kernels,
            &mut dft_states.dft_outputs
        )?;
        Ok(())
    }
}

/// The states of multiple DFTs that will be evaluated in one shader dispatch
pub struct DftStates<Func: ToDft> {
    /// The DFTs stored on CPU side.
    pub dfts: Vec<Dft<Func>>,

    // Buffers / GPU data
    pub function_value_positions: GpuBuffer<Index>,
    pub func_dfts: GpuBuffer<GpuFunctionDft>,
    pub dft_frequencies: GpuBuffer<Real>,
    pub dft_kernels: GpuBuffer<Complex32>,
    pub dft_outputs: GpuBuffer<Complex32>,
    pub workgroups: [u32; 3]
}

impl<Func: ToDft> DftStates<Func> {
    /// Creates new DFT buffers that are properly initialized
    pub fn new(
        backend: &GpuBackend,
        dfts: impl Into<Vec<Dft<Func>>>,
        sim_state: &FdtdLossyState,
        dft_pipeline: &DftPipeline,
    ) -> TaserResult<Self> {
        let mut selff = Self::new_zeroed(backend, dfts)?;
        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("__dft_init", None);
        dft_pipeline.initialize_states(
            &mut pass,
            &sim_state.grid,
            &mut selff
        )?;
        drop(pass);
        backend.submit(encoder)?;
        Ok(selff)
    }

    /// Creates new zeroed-out DFT buffers.
    ///
    /// Use [`DftPipeline::initialize_states`] to populate buffers with non-zero kernels, or use
    /// [`DftStates::new`] instead to submit an initialize command right away.
    pub fn new_zeroed(backend: &GpuBackend, dfts: impl Into<Vec<Dft<Func>>>) -> TaserResult<Self> {
        let dfts = dfts.into();
        if dfts.is_empty() { return Err(DftError::NoDfts.into()) };

        let function_value_positions = dfts.iter()
            .map(|dft| dft.function.get_value_position() as Index)
            .collect::<Vec<_>>()
            .create_gpu_buffer(backend)?;

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
            dfts,
            function_value_positions,
            func_dfts,
            dft_frequencies,
            dft_kernels: dft_kernels_zeroed,
            dft_outputs: dft_outputs_zeroed,
            workgroups: dft_workgroups(max_n_kernels, n_functions),
        })
    }
}

pub trait ToDft {
    type GpuStateType;

    /// Convert this [`ToDft`] into a [`Dft`] with the specified `frequencies`.
    fn to_dft(self, frequencies: Vec<Real>) -> TaserResult<(Dft<Self>, Arc<Self>)>;

    fn get_value_position(&self) -> usize;

    fn get_value_buffer(state: &Self::GpuStateType) -> &GpuBuffer<Real>;
}

pub struct DftReadback<Func: ToDft> {
    func_ranges: Vec<(Arc<Func>, Range<usize>)>,
    frequencies: Vec<Real>,
    dft_outputs: Vec<Complex32>,
    dft_outputs_read: GpuReadback<Complex32>
}

impl<Func: ToDft> DftReadback<Func> {
    pub async fn new(backend: &GpuBackend, dft_states: &DftStates<Func>) -> TaserResult<Self> {
        let mut gpu_dfts = vec![GpuFunctionDft::default(); dft_states.func_dfts.len()];
        backend.slow_read_buffer(&dft_states.func_dfts, &mut gpu_dfts).await?;
        let func_ranges = gpu_dfts.into_iter()
            .zip(dft_states.dfts.iter())
            .map(|(gpu_dft, dft)| {
                (dft.function.clone(), (gpu_dft.kernels_start as usize)..(gpu_dft.kernels_end as usize))
            })
            .collect::<Vec<_>>();

        let mut frequencies = vec![0.; dft_states.dft_frequencies.len()];
        backend.slow_read_buffer(&dft_states.dft_frequencies, &mut frequencies).await?;

        Ok(Self {
            func_ranges,
            frequencies,
            dft_outputs: vec![Complex32::ZERO; dft_states.dft_outputs.len()],
            dft_outputs_read: GpuReadback::new(backend, dft_states.dft_outputs.len())?,
        })
    }

    pub fn request_copy(&mut self, backend: &GpuBackend, state: &DftStates<Func>) -> TaserResult<()> {
        if self.dft_outputs_read.is_idle() {
        self.dft_outputs_read.request_copy(backend, &state.dft_outputs, 0)?;
        }
        Ok(())
    }

    pub fn read_back(&mut self, backend: &GpuBackend) -> TaserResult<()> {
        backend.synchronize()?;
        self.try_read_back(backend);
        Ok(())
    }

    pub fn try_read_back(&mut self, backend: &GpuBackend) -> bool {
        self.dft_outputs_read.try_take(backend, &mut self.dft_outputs)
    }

    pub fn get_frequencies(&self, dft_function: &Arc<Func>) -> Option<Vec<Real>> {
        self.func_ranges.iter()
            .find_map(|(m, r)|
                Arc::ptr_eq(m, dft_function)
                    .then(||
                        self.frequencies[r.clone()].to_vec()
                    )
            )
    }

    pub fn get_dft(&self, dft_function: &Arc<Func>) -> Option<Vec<Complex32>> {
        self.func_ranges.iter()
            .find_map(|(m, r)|
                Arc::ptr_eq(m, dft_function)
                    .then(||
                        self.dft_outputs[r.clone()].to_vec()
                    )
            )
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DftDataPoint {
    pub frequency: Real,
    pub complex: Complex32,
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
    NoDfts,
    #[error("Couldn't find DFT function")]
    CannotFindFunction,
}