cfg_cpu! {
    use khal::ShaderArgsType;
}

use crate::fdtd::GridParameters;
use crate::math::*;
use bytemuck::{Pod, Zeroable};
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::*;
use khal_std::sync::workgroup_memory_barrier_with_group_sync;
use crate::{cfg_cpu, workgroup_counts};
use crate::dft::{ComplexVec3, GpuDftRegion};

pub const FLUX_WORKGROUP_SIZE: UVec3 = UVec3::new(8, 8, 1);

#[spirv_bindgen]
#[spirv(compute(threads(8, 8, 1)))]
pub fn gpu_power_flux(
    #[spirv(global_invocation_id)] thread_id: UVec3,
    #[spirv(local_invocation_id)] local_idx3: UVec3,
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(num_workgroups)] n_workgroups: UVec3,
    #[spirv(workgroup)] local_power: &mut [
        Real; (FLUX_WORKGROUP_SIZE.x * FLUX_WORKGROUP_SIZE.y * FLUX_WORKGROUP_SIZE.z) as usize
    ],
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &GridParameters,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] h_previous: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] h: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] en: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] flux_sensors: &[GpuPowerFluxSensor],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] sensor_power: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] wg_summations: &mut [Real], // flat 3d array, element for every workgroup
) {
    let sensor_idx = thread_id.z as usize;
    let sensor = flux_sensors.read(sensor_idx);

    // TODO: try a "cell flat idx buffer" instead of calculating flat idx each invocation
    //       also compare performance to see if it's worth the memory...
    let mut cell_idx3 = UVec3::ZERO;
    cell_idx3.dyn_insert(sensor.axis, sensor.position);
    cell_idx3.dyn_insert(sensor.axis1, thread_id.x);
    cell_idx3.dyn_insert(sensor.axis2, thread_id.y);

    let n_cells = GridIndex::from_uvec3(grid.n_cells3);
    let cell_idx = GridIndex::from_uvec3(cell_idx3);

    let is_hi_boundary_or_out_of_bounds =
        cell_idx.cmpge(n_cells - 1).any() || cell_idx3.cmpge(grid.n_cells3).any();
    let idx_enable = !is_hi_boundary_or_out_of_bounds as usize;

    let idx = cell_idx.to_flat_idx(n_cells) as usize * idx_enable;

    let en_self = en.read(idx);
    let h_self = h.read(idx);
    let p = compute_poynting(grid, h_previous, h, en, idx, &en_self, &h_self);
    let power = p.dot(sensor.da);

    let enable_f = idx_enable as Real;
    let local_pwr_idx = grid_idx3_to_flat_idx(local_idx3, FLUX_WORKGROUP_SIZE) as usize;
    local_power.write(local_pwr_idx, power * enable_f);
    workgroup_memory_barrier_with_group_sync();

    // Sum up power at each workgroup
    if local_pwr_idx != 0 { return; }; // one invocation per workgroup
    let wg_flat_idx = grid_idx3_to_flat_idx(workgroup_id, n_workgroups) as usize;
    let mut wg_power_sum = power;
    for i in 1..local_power.len() { // workgroup sums
        wg_power_sum += local_power.read(i);
    }
    wg_summations.write(wg_flat_idx, wg_power_sum);

    // Merge workgroup sums for each sensor
    if thread_id.x != 0 || thread_id.y != 0 { return; }; // one invocation per sensor
    let mut power_integral = wg_power_sum;
    for plane_wg_idx in wg_flat_idx..(wg_flat_idx + (n_workgroups.x * n_workgroups.y) as usize) {
        power_integral += wg_summations.read(plane_wg_idx);
    }
    sensor_power.write(sensor_idx, power_integral);
}

cfg_cpu! {
    pub fn flux_workgroups(n_cells: GridIndex, n_flux_sensors: Index) -> [u32; 3] {
        let max_n = n_cells.max_element();
        workgroup_counts!([max_n, max_n, n_flux_sensors], GpuPowerFluxArgs::WORKGROUP_SIZE)
    }
}

#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct GpuPowerFluxSensor {
    /// Magnitude-direction of `da` vector used in power flux integral.
    pub da: Vec3,
    pub axis: Axis,
    pub axis1: Axis,
    pub axis2: Axis,
    pub position: u32,
    pub _padding0: u32,
}

fn compute_poynting(
    grid: &GridParameters,
    h_previous: &[Vec4],
    h: &[Vec4],
    en: &[Vec4],
    idx: usize,
    en_self: &Vec4,
    h_self: &Vec4,
) -> Vec3 {
    let en_pyz = en.read(idx + (grid.flat_idx_incrs.y + grid.flat_idx_incrs.z) as usize);
    let en_pzx = en.read(idx + (grid.flat_idx_incrs.z + grid.flat_idx_incrs.x) as usize);
    let en_pxy = en.read(idx + (grid.flat_idx_incrs.x + grid.flat_idx_incrs.y) as usize);
    let en_avg = Vec3 {
        x: (en_self.x + en_pyz.x) * 0.5,
        y: (en_self.y + en_pzx.y) * 0.5,
        z: (en_self.z + en_pxy.z) * 0.5,
    };

    fn h_avg(grid: &GridParameters, idx: usize, h: &[Vec4], h_self: &Vec4) -> Vec3 {
        let h_px = h.read(idx + grid.flat_idx_incrs.x as usize);
        let h_py = h.read(idx + grid.flat_idx_incrs.y as usize);
        let h_pz = h.read(idx + grid.flat_idx_incrs.z as usize);
        Vec3 {
            x: (h_self.x + h_px.x) * 0.5,
            y: (h_self.y + h_py.y) * 0.5,
            z: (h_self.z + h_pz.z) * 0.5,
        }
    }

    // Compute power at cell
    let h_avg_prev = h_avg(grid, idx, h_previous, h_self);
    let h_avg_fut = h_avg(grid, idx, h, h_self);
    let h_avg = (h_avg_prev + h_avg_fut) * 0.5;
    en_avg.cross(h_avg)
}

#[spirv_bindgen]
#[spirv(compute(threads(8, 8, 1)))]
pub fn gpu_power_flux_dft(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(local_invocation_id)] local_idx3: UVec3,
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(num_workgroups)] n_workgroups: UVec3,
    #[spirv(workgroup)] local_s: &mut [
        Complex32; (FLUX_WORKGROUP_SIZE.x * FLUX_WORKGROUP_SIZE.y * FLUX_WORKGROUP_SIZE.z) as usize
    ],
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &GridParameters,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] h_dfts: &[ComplexVec3],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] en_dfts: &[ComplexVec3],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] flux_sensors: &[GpuPowerFluxDftSensor],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] dft_regions: &[GpuDftRegion],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] sensor_power: &mut [Complex32], // one elem for each sensor
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] wg_summations: &mut [Complex32], // flat 3d array, element for every workgroup
) {
    let sensor_idx = id.z as usize;
    let sensor = flux_sensors.read(sensor_idx);
    let region = dft_regions.read(sensor.gpu_dft_region_idx as usize);

    let mut local_cell_idx3 = UVec3::ZERO;
    local_cell_idx3.dyn_insert(sensor.axis1, id.x);
    local_cell_idx3.dyn_insert(sensor.axis2, id.y);

    let cell_idx3 = region.cell_idx_min + local_cell_idx3;
    let n_cells = GridIndex::from_uvec3(grid.n_cells3);
    let cell_idx = GridIndex::from_uvec3(cell_idx3);

    let idx = cell_idx.to_flat_idx(n_cells) as usize;
    let out_of_bounds = local_cell_idx3.cmpge(region.n_cells3).any();
    let enable = (idx < grid.cell_count as usize || out_of_bounds) as usize;
    let idx = idx * enable;
    let idx_dft = (region.dft_start_idx + grid_idx3_to_flat_idx(local_cell_idx3, region.n_cells3)) as usize;

    // Power calculation for this cell (includes DFT finalization terms)
    let en_self = en_dfts.read(idx);
    let en_m = en_cmplx_spatial_avg(en_dfts, en_self, idx_dft, grid.flat_idx_incrs)
        * grid.dt;
    let h_self = h_dfts.read(idx);
    let h_m = h_cmplx_spatial_avg(h_dfts, h_self, idx_dft, grid.flat_idx_incrs)
        * Complex32::from_polar(grid.dt, -core::f32::consts::TAU * (region.freq * 0.5 * grid.dt));
    let ds = Complex32::new(0.5, 0.) * en_m.cross(h_m.conj());
    let ds_dot = ds.dyn_idx(sensor.axis) * sensor.direction as i32 as Real;

    let local_pwr_idx = grid_idx3_to_flat_idx(local_idx3, FLUX_WORKGROUP_SIZE) as usize;
    local_s.write(local_pwr_idx, ds_dot * enable as Real);
    workgroup_memory_barrier_with_group_sync();

    // Sum up power at each workgroup
    if local_pwr_idx != 0 { return; }; // one invocation per workgroup
    let wg_flat_idx = grid_idx3_to_flat_idx(workgroup_id, n_workgroups) as usize;
    let mut wg_power_sum = ds_dot;
    for i in 1..local_s.len() { // workgroup sums
        wg_power_sum += local_s.read(i);
    }
    wg_summations.write(wg_flat_idx, wg_power_sum);

    // Merge workgroup sums for each sensor
    if id.x != 0 || id.y != 0 { return; }; // one invocation per sensor
    let mut power_integral = wg_power_sum;
    for plane_wg_idx in wg_flat_idx..(wg_flat_idx + (n_workgroups.x * n_workgroups.y) as usize) {
        power_integral += wg_summations.read(plane_wg_idx);
    }
    sensor_power.write(sensor_idx, power_integral);
}

cfg_cpu! {
    pub fn flux_dft_workgroups(flux_dft_sensors: &[GpuPowerFluxDftSensor], dft_regions: &[GpuDftRegion]) -> [u32; 3] {
        let max_n = flux_dft_sensors.iter()
            .map(|s| dft_regions.get(s.gpu_dft_region_idx as usize)
                .expect("dft_region_idx must be a valid index")
                .n_cells3
                .max_element()
            )
            .max()
            .expect("Must have at least one sensor in flux_dft_sensors");
        let n_flux_sensors = flux_dft_sensors.len() as u32;
        workgroup_counts!([max_n, max_n, n_flux_sensors], GpuPowerFluxArgs::WORKGROUP_SIZE)
    }
}

/// GPU version of power flux DFT. One instance of this struct for every frequency, for every flux sensor.
/// Just like [`GpuDftRegion`].
#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct GpuPowerFluxDftSensor {
    pub axis: Axis,
    pub axis1: Axis,
    pub axis2: Axis,
    pub direction: Direction,
    pub gpu_dft_region_idx: u32,
}

macro_rules! en_spatial_avg_impl {
    ($name:ident, $v:ident) => {
        fn $name(en: &[$v], en_self: $v, idx: usize, flat_idx_incrs: UVec3) -> $v {
            let en_pyz = en.read(idx + (flat_idx_incrs.y + flat_idx_incrs.z) as usize);
            let en_pzx = en.read(idx + (flat_idx_incrs.z + flat_idx_incrs.x) as usize);
            let en_pxy = en.read(idx + (flat_idx_incrs.x + flat_idx_incrs.y) as usize);
            $v {
                x: (en_self.x + en_pyz.x) * 0.5,
                y: (en_self.y + en_pzx.y) * 0.5,
                z: (en_self.z + en_pxy.z) * 0.5,
            }
        }
    };
}

en_spatial_avg_impl!(en_cmplx_spatial_avg, ComplexVec3);
// TODO: en_spatial_avg_impl!(en_spatial_avg, Vec3); ...use this in instantaneous flux

macro_rules! h_spatial_avg_impl {
    ($name:ident, $v:ident) => {
        fn $name(h: &[$v], h_self: $v, idx: usize, flat_idx_incrs: UVec3) -> $v {
            let h_px = h.read(idx + grid.flat_idx_incrs.x as usize);
            let h_py = h.read(idx + grid.flat_idx_incrs.y as usize);
            let h_pz = h.read(idx + grid.flat_idx_incrs.z as usize);
            $v {
                x: (h_self.x + h_px.x) * 0.5,
                y: (h_self.y + h_py.y) * 0.5,
                z: (h_self.z + h_pz.z) * 0.5,
            }
        }
    };
}

h_spatial_avg_impl!(h_cmplx_spatial_avg, ComplexVec3);
// TODO: h_spatial_avg_impl!(h_spatial_avg, Vec3); ...use this in instantaneous flux