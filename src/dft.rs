use crate::gpu_util::CreateGpuBuffer;
use crate::prelude::*;
use taser_em_shaders::dft::{DftInfo, GpuComputeDftKernels, GpuDft, dft_workgroups};
use taser_em_shaders::math::Real;

pub struct DftPipeline {
    compute_dft_kernels: GpuComputeDftKernels,
    dft: GpuDft
}

impl DftPipeline {
    pub fn new(backend: &GpuBackend) -> TaserResult<Self> {
        Ok(Self {
            compute_dft_kernels: GpuComputeDftKernels::from_dir(backend, &crate::SPIRV_DIR)?,
            dft: GpuDft::from_dir(backend, &crate::SPIRV_DIR)?,
        })
    }

    pub fn initialize(&self, pass: &mut GpuPass, sim_state: &mut FdtdLossyState) -> TaserResult<()> {
        let Some(state) = &mut sim_state.dft_states else { return Ok(()) };

        self.compute_dft_kernels.call(
            pass,
            DispatchGrid::Grid(state.workgroup_count),
            &sim_state.grid_params,
            &state.dft_infos,
            &state.dft_frequencies,
            &mut state.dft_kernels
        )?;

        Ok(())
    }

    pub fn dispatch_steps(&mut self, _pass: &mut GpuPass, sim_state: &mut FdtdLossyState) -> TaserResult<()> {
        let Some(_state) = &mut sim_state.dft_states else { return Ok(()) };
        todo!()
    }
}

pub struct DftStates {
    pub dft_infos: GpuBuffer<DftInfo>,
    pub dft_frequencies: GpuBuffer<Real>,
    pub dft_kernels: GpuBuffer<Complex32>,
    pub dfts: GpuBuffer<Complex32>,
    pub functions: GpuBuffer<Real>,
    pub workgroup_count: [u32; 3]
}

impl DftStates {
    pub fn new(backend: &GpuBackend, sim: &FdtdLossySimulation) -> TaserResult<Option<Self>> {
        if dft_frequencies_iter(sim).any(|fs| fs.is_empty()) {
            return Err(Error::EmptyFrequenciesList)
        }

        let frequencies = dft_frequencies_iter(sim)
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        if frequencies.is_empty() { return Ok(None); }

        let mut kernels_start = 0;
        let dft_infos = dft_frequencies_iter(sim)
            .map(|freqs| {
                let kernels_end = kernels_start + freqs.len() as u32;
                let dft_info = DftInfo { kernels_start, kernels_end };
                kernels_start = kernels_end;
                dft_info
            })
            .collect::<Vec<_>>();

        let dft_kernels = vec![Complex32::ZERO; frequencies.len()];

        let max_n_freqs = dft_frequencies_iter(sim)
            .map(|freqs| freqs.len())
            .max()
            .unwrap_or(0) as u32;

        Ok(Some(Self {
            dft_infos: dft_infos.create_gpu_buffer(backend)?,
            dft_frequencies: frequencies.create_gpu_buffer(backend)?,
            dft_kernels: dft_kernels.create_gpu_buffer(backend)?,
            dfts: dft_kernels.create_gpu_buffer(backend)?,
            functions: vec![0.; dft_infos.len()].create_gpu_buffer(backend)?,
            workgroup_count: dft_workgroups(max_n_freqs, dft_infos.len() as u32),
        }))
    }
}

/// Helper function for iterating over DFT frequencies in a simulation (used internally).
pub fn dft_frequencies_iter(sim: &FdtdLossySimulation) -> impl Iterator<Item = &Vec<Real>> {
    sim.power_flux_monitors.iter()
        .filter_map(|m| m.dft_frequencies.as_ref())
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