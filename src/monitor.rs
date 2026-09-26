use crate::fdtd::*;
use khal::backend::{Backend, Buffer, DispatchGrid, GpuBackend, GpuBuffer, GpuPass, GpuReadback};
use taser_em_shaders::math::*;
use taser_em_shaders::monitor::*;
use crate::gpu_util::CreateGpuBuffer;
use crate::prelude::TaserResult;

/// Measure Poynting Flux flowing through a plane perpendicular to `axis`, stretching to the edges
/// of the grid
///
/// Cuts into spacer regions, meaning spacer regions won't wrap around [`PowerFluxMonitor`]s.
/// If the monitor ends up outside the grid, an error will be returned.
#[derive(Clone, Debug)]
pub struct PowerFluxMonitor {
    pub axis: SpatialAxis,
    /// Position of plane along `axis` in world-space.
    pub position: Real,
    /// The direction to measure flow in.
    ///
    /// This matters when measuring transmittance / reflection.
    pub direction: Direction,
    // TODO: /// All the frequencies of a DFT that will be run on this monitor's recorded values
    // TODO: pub dft_frequencies: Option<Vec<Real>>,
}

pub struct PowerFluxReadback {
    monitor_power: Vec<Real>,
    monitor_power_read: GpuReadback<Real>
}

impl PowerFluxReadback {
    pub fn new(backend: &GpuBackend, state: &FdtdLossyState) -> TaserResult<Option<Self>> {
        let Some(flux_state) = &state.power_flux_states else { return Ok(None); };
        let n_powers = flux_state.monitor_power.len();
        Ok(Some(Self {
            monitor_power: vec![0.; n_powers],
            monitor_power_read: GpuReadback::new(backend, n_powers)?,
        }))
    }

    pub fn request_copy(&mut self, backend: &GpuBackend, state: &FdtdLossyState) -> TaserResult<()> {
        let Some(flux_state) = &state.power_flux_states else { return Ok(()); };
        self.monitor_power_read.request_copy(backend, &flux_state.monitor_power, 0)?;
        Ok(())
    }

    pub fn read_back(&mut self, backend: &GpuBackend) -> TaserResult<()> {
        backend.synchronize()?;
        self.try_read_back(backend);
        Ok(())
    }

    pub fn try_read_back(&mut self, backend: &GpuBackend) -> bool {
        self.monitor_power_read.try_take(backend, &mut self.monitor_power)
    }

    pub fn get_power(&self, monitor_idx: usize) -> Real {
        self.monitor_power[monitor_idx]
    }
}

pub struct PowerFluxPipeline {
    power_flux_kernel: GpuPowerFlux
}

impl PowerFluxPipeline {
    pub fn new(backend: &GpuBackend) -> TaserResult<Self> {
        Ok(Self {
            power_flux_kernel: GpuPowerFlux::from_dir(backend, &crate::SPIRV_DIR)?,
        })
    }

    pub fn dispatch_steps(&self, pass: &mut GpuPass, sim_state: &mut FdtdLossyState) -> TaserResult<()> {
        let Some(flux_state) = &mut sim_state.power_flux_states else {
            return Ok(());
        };

        self.power_flux_kernel.call(
            pass,
            DispatchGrid::Grid(flux_state.workgroups),
            &sim_state.grid_params,
            &sim_state.h_previous,
            &sim_state.h,
            &sim_state.en,
            &flux_state.flux_monitors,
            &mut flux_state.monitor_power,
            &mut flux_state.wg_summations,
        )?;

        Ok(())
    }
}

pub struct PowerFluxStates {
    pub flux_monitors: GpuBuffer<GpuPowerFluxMonitor>,
    pub monitor_power: GpuBuffer<Real>,
    pub wg_summations: GpuBuffer<Real>,
    pub workgroups: [u32; 3],
}

impl PowerFluxStates {
    pub fn new(
        backend: &GpuBackend,
        sim: &FdtdLossySimulation,
        n_cells: GridIndex,
        regions_offset: &Vec3
    ) -> TaserResult<Option<Self>> {
        if sim.power_flux_monitors.is_empty() {
            return Ok(None);
        }

        let workgroups = flux_workgroups(n_cells, sim.power_flux_monitors.len() as _);

        let cell_size3_one = sim.fdtd_parameters.cell_size
            .to_3d(Vec3::ONE);

        let flux_monitors = sim.power_flux_monitors.iter()
            .map(|monitor| {
                let axis = Axis::from(monitor.axis);
                let axis1 = axis.permute();
                let axis2 = axis1.permute();

                let da = axis.to_vec3() * monitor.direction as i32 as Real;
                let position = ((regions_offset[axis] + monitor.position) / cell_size3_one[axis]) as u32;
                GpuPowerFluxMonitor {
                    da,
                    axis,
                    axis1,
                    axis2,
                    position,
                    _padding0: 0,
                }
            })
            .collect::<Vec<_>>()
            .create_gpu_buffer(backend)?;

        let monitor_power = vec![0.; sim.power_flux_monitors.len()]
            .create_gpu_buffer(backend)?;
        let wg_summations = vec![0.; workgroups.iter().product::<u32>() as usize]
            .create_gpu_buffer(backend)?;

        Ok(Some(Self {
            flux_monitors,
            monitor_power,
            wg_summations,
            workgroups,
        }))
    }
}

// TODO: probe monitor (measures at one point)