use bytemuck::{Pod, Zeroable};
use crate::fdtd::GridParameters;
use crate::math::*;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::*;
use taser_em_macros::clone_replaced_axes;

pub const FLUX_WORKGROUP_SIZE: UVec3 = cfg_select! {
    feature = "dim1" => UVec3::new(1, 1, 1),
    feature = "dim2" => UVec3::new(64, 1, 1),
    feature = "dim3" => UVec3::new(8, 8, 1),
};

#[cfg_attr(not(feature = "dim1"), clone_replaced_axes(axis = x, axis1 = y, axis2 = z))]
#[cfg_attr(not(feature = "dim1"), clone_replaced_axes(axis = y, axis1 = z, axis2 = x))]
#[cfg_attr(not(feature = "dim2"), clone_replaced_axes(axis = z, axis1 = x, axis2 = y))]
#[spirv_bindgen]
#[cfg_attr(feature = "dim1", spirv(compute(threads(1, 1, 1))))] // Only recording at "points" in 1D
#[cfg_attr(feature = "dim2", spirv(compute(threads(64, 1, 1))))] // "lines" in 2D
#[cfg_attr(feature = "dim3", spirv(compute(threads(8, 8, 1))))] // full planes in 3D
pub fn gpu_power_flux(
    #[spirv(global_invocation_id)] idx3: UVec3,
    #[allow(unused_variables)] #[spirv(local_invocation_id)] local_idx3: UVec3,
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(workgroup)] local_power: &mut [
        Real; (FLUX_WORKGROUP_SIZE.x * FLUX_WORKGROUP_SIZE.y * FLUX_WORKGROUP_SIZE.x) as usize
    ],
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &GridParameters,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] flux_monitors: &[GpuPowerFluxMonitor],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] monitor_power: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] h_previous: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] h: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] en: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] wg_summations: &mut [Real], // len() = num_workgroups element product
) {
    // Guarantee zeroed-out wg_power since early-exited threads might leave their
    // local_power element uninitialized (UB)
    let local_pwr_idx = grid_idx3_to_flat_idx(local_idx3, FLUX_WORKGROUP_SIZE) as usize;
    local_power.write(local_pwr_idx, 0.);

    let mut cell_idx3 = UVec3::ZERO;
    cell_idx3.axis1 = idx3.x;
    cell_idx3.axis2 = idx3.y;
    if cell_idx3.axis1 >= grid.n_cells3.axis1 || cell_idx3.axis2 >= grid.n_cells3.axis2 { return; }

    let monitor_idx = idx3.z as usize;
    let monitor = flux_monitors.read(monitor_idx);
    cell_idx3.axis = monitor.cell_idx_a;
    let idx = GridIndex::from_uvec3(cell_idx3)
        .to_flat_idx(GridIndex::from_uvec3(grid.n_cells3)) as usize;

    let en_self = en.read(idx);
    let en_pa1a = en.read(idx + (grid.flat_idx_incrs.axis1 + grid.flat_idx_incrs.axis) as usize);
    let en_pa2a = en.read(idx + (grid.flat_idx_incrs.axis2 + grid.flat_idx_incrs.axis) as usize);
    let en_pa1a2 = en.read(idx + (grid.flat_idx_incrs.axis1 + grid.flat_idx_incrs.axis2) as usize);
    let mut en_avg = Vec3::ZERO;
    en_avg.axis = (en_self.axis + en_pa1a2.axis) * 0.5;
    en_avg.axis1 = (en_self.axis1 + en_pa2a.axis1) * 0.5;
    en_avg.axis2 = (en_self.axis2 + en_pa1a.axis2) * 0.5;
    
    fn h_avg(grid: &GridParameters, idx: usize, h: &[Vec4]) -> Vec3 {
        let h_self = h.read(idx);
        let h_pa = h.read(idx + grid.flat_idx_incrs.axis as usize);
        let h_pa1 = h.read(idx + grid.flat_idx_incrs.axis1 as usize);
        let h_pa2 = h.read(idx + grid.flat_idx_incrs.axis2 as usize);
        let mut h_avg = Vec3::ZERO;
        h_avg.axis = (h_self.axis + h_pa.axis) * 0.5;
        h_avg.axis1 = (h_self.axis1 + h_pa1.axis1) * 0.5;
        h_avg.axis2 = (h_self.axis2 + h_pa2.axis2) * 0.5;
        h_avg
    }

    // Compute power at cell
    let h_avg_prev = h_avg(grid, idx, h_previous);
    let h_avg_fut = h_avg(grid, idx, h);
    let h_avg = (h_avg_prev + h_avg_fut) * 0.5;
    let p = en_avg.cross(h_avg);
    let power_self = p.axis * monitor.da;
    local_power.write(local_pwr_idx, power_self);

    // Sum up power using a concurrent merge algorithm
    if local_pwr_idx != 0 { return; };
    let wg_flat_idx = grid_idx3_to_flat_idx(workgroup_id, num_workgroups) as usize;
    let mut wg_power_sum = power_self;
    for i in 1..local_power.len() { // workgroup sums
        wg_power_sum += local_power.read(i);
    }
    wg_summations.write(wg_flat_idx, wg_power_sum);
    if !(cell_idx3.axis1 == 0 && cell_idx3.axis2 == 0) { return; }
    let mut power_integral = wg_power_sum;
    for i in 1..wg_summations.len() { // merge workgroup sums into one sum.
        power_integral += wg_summations.read(i);
    }
    monitor_power.write(monitor_idx, power_integral);
}

// TODO: move this to math module & use it in to_flat_idx
fn grid_idx3_to_flat_idx(grid_idx3: UVec3, n_cells: UVec3) -> Index {
    grid_idx3.z * n_cells.x * n_cells.y +
        grid_idx3.y * n_cells.x +
        grid_idx3.x
}

/// Computes workgroup count for a flux monitor kernel (kernels are per-axis)
pub fn flux_workgroups_axis(_n_cells: GridIndex, n_flux_monitors: u32) -> [u32; 3] {
    #[cfg(not(feature = "dim1"))]
    let max_n = _n_cells.max_element();
    cfg_select! {
        feature = "dim1" => [1, 1, n_flux_monitors],
        feature = "dim2" => [max_n.div_ceil(FLUX_WORKGROUP_SIZE.x), 1, n_flux_monitors],
        feature = "dim3" => [
            max_n.div_ceil(FLUX_WORKGROUP_SIZE.x),
            max_n.div_ceil(FLUX_WORKGROUP_SIZE.y),
            n_flux_monitors
        ],
    }
}

#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct GpuPowerFluxMonitor {
    /// Magnitude of `da` vector used in power flux integral.
    pub da: Real,
    /// Grid index component of measurement plane along the axis perpendicular to it
    pub cell_idx_a: u32,
}