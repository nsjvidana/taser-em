use bytemuck::{Pod, Zeroable};
use crate::fdtd::GridParameters;
use crate::math::*;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::*;

pub const FLUX_WORKGROUP_SIZE: UVec3 = UVec3::new(8, 8, 1);

#[spirv_bindgen]
#[spirv(compute(threads(8, 8, 1)))]
pub fn gpu_power_flux(
    #[spirv(global_invocation_id)] thread_id: UVec3,
    #[allow(unused_variables)] #[spirv(local_invocation_id)] local_idx3: UVec3,
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(num_workgroups)] n_workgroups: UVec3,
    #[spirv(workgroup)] local_power: &mut [
        Real; (FLUX_WORKGROUP_SIZE.x * FLUX_WORKGROUP_SIZE.y * FLUX_WORKGROUP_SIZE.z) as usize
    ],
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &GridParameters,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] h_previous: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] h: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] en: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] flux_monitors: &[GpuPowerFluxMonitor],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] monitor_power: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] wg_summations: &mut [Real], // flat 3d array, element for every workgroup
) {
    let monitor_idx = thread_id.z as usize;
    let monitor = flux_monitors.read(monitor_idx);

    // TODO: try a "cell flat idx buffer" instead of calculating flat idx each invocation
    //       also compare performance to see if it's worth the memory...
    let mut cell_idx3 = UVec3::ZERO;
    cell_idx3.dyn_insert(monitor.axis, monitor.position);
    cell_idx3.dyn_insert(monitor.axis1, thread_id.x);
    cell_idx3.dyn_insert(monitor.axis2, thread_id.y);
    if cell_idx3.cmpge(grid.n_cells3).any() { return; }

    let n_cells = GridIndex::from_uvec3(grid.n_cells3);
    let cell_idx = GridIndex::from_uvec3(cell_idx3);
    let idx = cell_idx.to_flat_idx(n_cells) as usize;

    let en_self = en.read(idx);
    let en_pyz = en.read(idx + (grid.flat_idx_incrs.y + grid.flat_idx_incrs.z) as usize);
    let en_pzx = en.read(idx + (grid.flat_idx_incrs.z + grid.flat_idx_incrs.x) as usize);
    let en_pxy = en.read(idx + (grid.flat_idx_incrs.x + grid.flat_idx_incrs.y) as usize);
    let en_avg = Vec3 {
        x: (en_self.x + en_pyz.x) * 0.5,
        y: (en_self.y + en_pzx.y) * 0.5,
        z: (en_self.z + en_pxy.z) * 0.5,
    };

    fn h_avg(grid: &GridParameters, idx: usize, h: &[Vec4]) -> Vec3 {
        let h_self = h.read(idx);
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
    let h_avg_prev = h_avg(grid, idx, h_previous);
    let h_avg_fut = h_avg(grid, idx, h);
    let h_avg = (h_avg_prev + h_avg_fut) * 0.5;
    let p = en_avg.cross(h_avg);
    let power = p.dot(monitor.da);

    let local_pwr_idx = grid_idx3_to_flat_idx(local_idx3, FLUX_WORKGROUP_SIZE) as usize;
    local_power.write(local_pwr_idx, power);

    // Sum up power at each workgroup
    if local_pwr_idx != 0 { return; }; // one invocation per workgroup
    let wg_flat_idx = grid_idx3_to_flat_idx(workgroup_id, n_workgroups) as usize;
    let mut wg_power_sum = power;
    for i in 1..local_power.len() { // workgroup sums
        wg_power_sum += local_power.read(i);
    }
    wg_summations.write(wg_flat_idx, wg_power_sum);

    // Merge workgroup sums for each monitor
    if thread_id.x != 0 || thread_id.y != 0 { return; }; // one invocation per monitor
    let mut power_integral = wg_power_sum;
    for plane_wg_idx in wg_flat_idx..(wg_flat_idx + (n_workgroups.x * n_workgroups.y) as usize) {
        power_integral += wg_summations.read(plane_wg_idx);
    }
    monitor_power.write(monitor_idx, power_integral);
}

pub fn flux_workgroups(n_cells: GridIndex, n_flux_monitors: Index) -> [u32; 3] {
    let max_n = n_cells.max_element();
    [
        max_n.div_ceil(FLUX_WORKGROUP_SIZE.x),
        max_n.div_ceil(FLUX_WORKGROUP_SIZE.y),
        n_flux_monitors
    ]
}

// TODO: move this to math module
pub fn grid_idx3_to_flat_idx(grid_idx3: UVec3, n_cells3: UVec3) -> u32 {
    grid_idx3.z * n_cells3.x * n_cells3.y +
        grid_idx3.y * n_cells3.x +
        grid_idx3.x
}

// TODO: move this to math module
impl Axis {
    pub const fn to_vec3(&self) -> Vec3 {
        match self {
            Axis::X => Vec3::X,
            Axis::Y => Vec3::Y,
            Axis::Z => Vec3::Z,
        }
    }
}

#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct GpuPowerFluxMonitor {
    /// Magnitude-direction of `da` vector used in power flux integral.
    pub da: Vec3,
    pub axis: Axis,
    pub axis1: Axis,
    pub axis2: Axis,
    pub position: u32,
    pub _padding0: u32,
}