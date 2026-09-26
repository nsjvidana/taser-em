use crate::fdtd::*;
use khal::backend::{GpuBackend, GpuBuffer};
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
    /// All the frequencies of a DFT that will be run on this monitor's recorded values
    // TODO: pub dft_frequencies: Option<Vec<Real>>,
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
    ) -> TaserResult<Self> {
        let workgroups = flux_workgroups(n_cells, sim.power_flux_monitors.len() as _);

        let cell_size3_one = sim.fdtd_parameters.cell_size
            .to_3d(Vec3::ONE);

        let flux_monitors = sim.power_flux_monitors.iter()
            .map(|monitor| {
                let axis = Axis::from(monitor.axis);
                let axis1 = axis.permute();
                let axis2 = axis1.permute();
                let mut da = Vec3::ZERO; // TODO: turn this into an Axis fn?
                    da[axis] = 1. * monitor.direction as i32 as Real;
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
        let wg_summations = vec![0.; workgroups.iter().product::<usize>()]
            .create_gpu_buffer(backend)?;

        Ok(Self {
            flux_monitors,
            monitor_power,
            wg_summations,
            workgroups,
        })
    }
}

// TODO: probe monitor (measures at one point)