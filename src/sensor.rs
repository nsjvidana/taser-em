use std::sync::Arc;
use crate::fdtd::*;
use khal::backend::{Backend, Buffer, DispatchGrid, GpuBackend, GpuBuffer, GpuPass, GpuReadback};
use taser_em_shaders::dft::DftFields;
use taser_em_shaders::math::*;
use taser_em_shaders::sensor::*;
use crate::dft::{Dft, DftError, DftRegion, DftRegionStates, DftRegionsHook, ToDft};
use crate::gpu_util::CreateGpuBuffer;
use crate::prelude::TaserResult;

#[derive(Clone, Debug)]
pub struct PowerFluxDftSensor {
    /// Axis that's normal to the flux measurement plane
    pub axis: SpatialAxis,
    /// Position of the center of measurement plane in world coordinates
    pub position: Vect,
    /// Size of power flux measurement plane. Make this [`None`] if you want to measure
    /// along an entire slice of the grid.
    ///
    /// This field does nothing in 1D.
    pub size: Option<Vect>,
    /// The direction to measure flow in.
    ///
    /// This matters when measuring transmittance / reflection.
    pub direction: Direction,
    pub frequencies: Vec<Real>,
}

// TODO: interop w/ external DftRegionsHook instead?
//        maybe works with a DftRegionsHook builder struct...
pub struct PowerFluxDftHook {
    // TODO: /// If all [`PowerFluxDftSensor`]s should be updated in the next step.
    //       pub update_dfts: bool,
    #[allow(dead_code)]
    region_objs: Vec<Arc<DftRegion>>,
    states: PowerFluxDftStates,
    update: GpuPowerFluxDft,
    dft_regions_hook: DftRegionsHook,
}

impl PowerFluxDftHook {
    pub fn new(backend: &GpuBackend, sim: &FdtdLossyState, sensors: &[Arc<PowerFluxDftSensor>]) -> TaserResult<Self> {
        if sensors.is_empty() { return Err(PowerFluxError::NoSensors.into()) }
        let sim_offset = Vect::from_vec3(sim.grid_sizing.sim_offset);
        let grid_extents = sim.n_cells.as_vect() * sim.cell_size;

        let region_objs = sensors.iter()
            .map(|sensor| {
                // snap pos to grid on axis
                let mut pos = sensor.position + sim_offset;
                pos[sensor.axis] = ((pos[sensor.axis] / sim.cell_size[sensor.axis]).floor() + 0.5)
                    * sim.cell_size[sensor.axis];
                pos -= sim_offset;
                // clamp size to grid edges, with thickness of 1 cell
                let mut size = sensor.size.unwrap_or(grid_extents).abs();
                size[sensor.axis] = sim.cell_size[sensor.axis];
                let plane_min = (pos - size * 0.5).max(Vect::ZERO);
                let plane_max = (pos + size * 0.5).min(grid_extents);

                Arc::new(DftRegion {
                    frequencies: sensor.frequencies.clone(),
                    bb_min: plane_min,
                    bb_max: plane_max,
                    fields: DftFields::SN,
                })
            })
            .collect::<Vec<_>>();
        let dft_region_states = DftRegionStates::new(backend, sim, &region_objs)?;

        let flux_sensors = sensors.iter()
            .map(|sensor| {
                let axis = Axis::from(sensor.axis);
                let axis1 = axis.permute();
                let axis2 = axis1.permute();
                // TODO: use a dft_region_offset uniform field instead of storing region indices
                let gpu_sensor = GpuPowerFluxDftSensor {
                    axis,
                    axis1,
                    axis2,
                    direction: sensor.direction,
                    gpu_dft_region_idx: u32::MAX, // temporary value.
                };
                (!sensor.frequencies.is_empty())
                    .then(|| {
                        std::iter::repeat_n(gpu_sensor, sensor.frequencies.len())
                            .collect::<Vec<_>>()
                    })
                    .ok_or(DftError::NoFrequencies.into())
            })
            .collect::<TaserResult<Vec<_>>>()?
            .into_iter()
            .flatten()
            .enumerate()
            .map(|(i, s)| GpuPowerFluxDftSensor { gpu_dft_region_idx: i as u32, ..s })
            .collect::<Vec<_>>();

        let sensor_power = vec![Complex32::ZERO; flux_sensors.len()];

        let gpu_regions = dft_region_states.get_all_region_descs();
        let flux_dft_workgroups = flux_dft_workgroups(&flux_sensors, &gpu_regions);
        let wg_summations = vec![Complex32::ZERO; flux_dft_workgroups.iter().product::<u32>() as usize];

        Ok(Self {
            region_objs,
            states: PowerFluxDftStates {
                flux_sensors: flux_sensors.create_gpu_buffer(backend)?,
                sensor_power: sensor_power.create_gpu_buffer(backend)?,
                wg_summations: wg_summations.create_gpu_buffer(backend)?,
                flux_dft_workgroups,
            },
            update: GpuPowerFluxDft::from_dir(backend, &crate::SPIRV_DIR)?,
            dft_regions_hook: DftRegionsHook::new(backend, dft_region_states)?,
        })
    }
}

impl FdtdLossyHook for PowerFluxDftHook {
    fn initialize(&mut self, pass: &mut GpuPass, state: &mut FdtdLossyState) -> TaserResult<()> {
        self.dft_regions_hook.initialize(pass, state)
    }

    fn dispatch_step(&mut self, pass: &mut GpuPass, state: &mut FdtdLossyState) -> TaserResult<()> {
        self.dft_regions_hook.dispatch_step(pass, state)?;

        self.update.call(
            pass,
            DispatchGrid::Grid(self.states.flux_dft_workgroups),
            &state.grid,
            self.dft_regions_hook.state().h_dfts(),
            self.dft_regions_hook.state().en_dfts(),
            self.dft_regions_hook.state().regions(),
            &self.states.flux_sensors,
            &mut self.states.sensor_power,
            &mut self.states.wg_summations,
        )?;
        Ok(())
    }
}

pub struct PowerFluxDftStates {
    flux_sensors: GpuBuffer<GpuPowerFluxDftSensor>,
    sensor_power: GpuBuffer<Complex32>,
    wg_summations: GpuBuffer<Complex32>,
    flux_dft_workgroups: [u32; 3],
}

// TODO: rename this to instantaneous
/// Measure Poynting Flux flowing through a plane perpendicular to `axis`, stretching to the edges
/// of the grid
///
/// Cuts into spacer regions, meaning spacer regions won't wrap around [`PowerFluxSensor`]s.
/// If the sensor ends up outside the grid, an error will be returned.
#[derive(Clone, Debug)]
pub struct PowerFluxSensor {
    pub axis: SpatialAxis,
    /// Position of plane along `axis` in world-space.
    pub position: Real,
    /// The direction to measure flow in.
    ///
    /// This matters when measuring transmittance / reflection.
    pub direction: Direction,
}

#[derive(Clone)]
pub struct PowerFluxFunction {
    obj: Arc<PowerFluxSensor>,
    sensor_idx: usize,
}

impl PowerFluxFunction {
    pub fn new(sensor: &Arc<PowerFluxSensor>, flux_state: &PowerFluxStates) -> TaserResult<Self> {
        let (sensor_idx, _) = flux_state.sensors.iter()
            .enumerate()
            .find(|(_, m)| Arc::ptr_eq(m, sensor))
            .ok_or_else(|| PowerFluxError::MissingSensor(sensor.clone()))?;

        Ok(Self {
            obj: sensor.clone(),
            sensor_idx,
        })
    }

    pub fn get_sensor_ref(&self) -> &Arc<PowerFluxSensor> { &self.obj }

    pub fn sensor_idx(&self) -> usize { self.sensor_idx }
}

impl ToDft for PowerFluxFunction {
    fn to_dft(self, frequencies: Vec<Real>) -> TaserResult<(Dft<Self>, Arc<Self>)> {
        let ptr = Arc::new(self);
        Ok((Dft::new(frequencies, ptr.clone())?, ptr))
    }

    fn get_value_position(&self) -> usize { self.sensor_idx() }

    fn get_value_buffer(state: &FdtdLossyState) -> TaserResult<&GpuBuffer<Real>> {
        let flux_states = state.power_flux_states
            .as_ref()
            .ok_or(PowerFluxError::NoSensors)?;
        Ok(&flux_states.sensor_power)
    }
}

pub struct PowerFluxReadback {
    sensors: Vec<Arc<PowerFluxSensor>>,
    sensor_power: Vec<Real>,
    sensor_power_read: GpuReadback<Real>
}

impl PowerFluxReadback {
    pub fn new(backend: &GpuBackend, state: &FdtdLossyState) -> TaserResult<Option<Self>> {
        let Some(flux_state) = &state.power_flux_states else { return Ok(None); };
        let n_powers = flux_state.sensor_power.len();
        Ok(Some(Self {
            sensors: flux_state.sensors.clone(),
            sensor_power: vec![0.; n_powers],
            sensor_power_read: GpuReadback::new(backend, n_powers)?,
        }))
    }

    pub fn request_copy(&mut self, backend: &GpuBackend, state: &FdtdLossyState) -> TaserResult<()> {
        let Some(flux_state) = &state.power_flux_states else { return Ok(()); };
        if self.sensor_power_read.is_idle() {
        self.sensor_power_read.request_copy(backend, &flux_state.sensor_power, 0)?;
        }
        Ok(())
    }

    pub fn read_back(&mut self, backend: &GpuBackend) -> TaserResult<()> {
        backend.synchronize()?;
        self.try_read_back(backend);
        Ok(())
    }

    pub fn try_read_back(&mut self, backend: &GpuBackend) -> bool {
        self.sensor_power_read.try_take(backend, &mut self.sensor_power)
    }

    pub fn get_power(&self, sensor: &Arc<PowerFluxSensor>) -> Option<Real> {
        self.sensors.iter()
            .zip(self.sensor_power.iter())
            .find(|(m, _)| Arc::ptr_eq(m, sensor))
            .map(|(_, pwr)| *pwr)
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
            &sim_state.grid,
            &sim_state.h_previous,
            &sim_state.h,
            &sim_state.en,
            &flux_state.flux_sensors,
            &mut flux_state.sensor_power,
            &mut flux_state.wg_summations,
        )?;

        Ok(())
    }
}

pub struct PowerFluxStates {
    pub sensors: Vec<Arc<PowerFluxSensor>>,
    pub flux_sensors: GpuBuffer<GpuPowerFluxSensor>,
    pub sensor_power: GpuBuffer<Real>,
    pub wg_summations: GpuBuffer<Real>,
    pub workgroups: [u32; 3],
}

impl PowerFluxStates {
    pub fn new(
        backend: &GpuBackend,
        sim: &FdtdLossySimulation,
        n_cells: GridIndex,
        sim_offset: &Vec3
    ) -> TaserResult<Option<Self>> {
        if sim.power_flux_sensors.is_empty() {
            return Ok(None);
        }

        let workgroups = flux_workgroups(n_cells, sim.power_flux_sensors.len() as _);

        let cell_size3_one = sim.fdtd_parameters.cell_size
            .to_3d(Vec3::ONE);

        let (sensors, flux_sensors) = sim.power_flux_sensors.iter()
            .map(|sensor| {
                let axis = Axis::from(sensor.axis);
                let axis1 = axis.permute();
                let axis2 = axis1.permute();

                let da = axis.to_vec3() * sensor.direction as i32 as Real;
                let position = ((sim_offset[axis] + sensor.position) / cell_size3_one[axis]) as u32;
                let gpu_sensor = GpuPowerFluxSensor {
                    da,
                    axis,
                    axis1,
                    axis2,
                    position,
                    _padding0: 0,
                };
                (sensor.clone(), gpu_sensor)
            })
            .collect::<(Vec<_>, Vec<_>)>();

        let sensor_power = vec![0.; sim.power_flux_sensors.len()].create_gpu_buffer(backend)?;
        let wg_summations = vec![0.; workgroups.iter().product::<u32>() as usize]
            .create_gpu_buffer(backend)?;

        Ok(Some(Self {
            sensors,
            flux_sensors: flux_sensors.create_gpu_buffer(backend)?,
            sensor_power,
            wg_summations,
            workgroups,
        }))
    }
}

// TODO: probe sensor (measures at one point)

#[derive(thiserror::Error, Debug)]
pub enum PowerFluxError {
    #[error("The following sensor couldn't be found in the simulation: {0:?}")]
    MissingSensor(Arc<PowerFluxSensor>),
    #[error("Tried accessing data of a sensor but none are in the simulation")]
    NoSensors
}