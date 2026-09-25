use bytemuck::{Pod, Zeroable};
use crate::fdtd::GridParameters;
use crate::math::*;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::*;
use taser_em_macros::replace_idents;

pub const FLUX_WORKGROUP_SIZE: UVec3 = cfg_select! {
    feature = "dim1" => UVec3::new(1, 1, 1),
    feature = "dim2" => UVec3::new(64, 1, 1),
    feature = "dim3" => UVec3::new(8, 8, 1),
};

#[cfg_attr(feature = "dim1", replace_idents(suffix = "z", axis = z, axis1 = x, axis2 = y))]
#[cfg_attr(
    feature = "dim2",
    replace_idents(suffix = "x", axis = x, axis1 = y, axis2 = z),
    replace_idents(suffix = "y", axis = y, axis1 = x, axis2 = z)
)]
#[cfg_attr(
    feature = "dim3",
    replace_idents(suffix = "x", axis = x, axis1 = y, axis2 = z),
    replace_idents(suffix = "y", axis = y, axis1 = z, axis2 = x),
    replace_idents(suffix = "z", axis = z, axis1 = x, axis2 = y)
)]
#[spirv_bindgen]
#[cfg_attr(feature = "dim1", spirv(compute(threads(1, 1, 1))))] // Only recording at "points" in 1D
#[cfg_attr(feature = "dim2", spirv(compute(threads(64, 1, 1))))] // "lines" in 2D
#[cfg_attr(feature = "dim3", spirv(compute(threads(8, 8, 1))))] // full planes in 3D
pub fn gpu_power_flux(
    #[spirv(global_invocation_id)] idx3: UVec3,
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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] wg_summations: &mut [Real], // an element for every workgroup
) {
    // Guarantee zeroed-out wg_power since early-exited threads might leave their
    // local_power element uninitialized (UB)
    let local_pwr_idx = grid_idx3_to_flat_idx(local_idx3, FLUX_WORKGROUP_SIZE) as usize;
    local_power.write(local_pwr_idx, 0.);

    let mut cell_idx3 = UVec3::ZERO;
    cell_idx3.axis1 = idx3.x;
    cell_idx3.axis2 = idx3.y;
    if cell_idx3.axis1 >= grid.n_cells3.axis1 || cell_idx3.axis2 >= grid.n_cells3.axis2 {
        return;
    }

    let monitor_idx = idx3.z as usize; // thread indices z component is just monitor idx.
    let monitor = flux_monitors.read(monitor_idx);
    cell_idx3.axis = monitor.cell_idx_a;

    let cell_idx = GridIndex::from_uvec3(cell_idx3);
    let n_cells = GridIndex::from_uvec3(grid.n_cells3);
    let idx = cell_idx.to_flat_idx(n_cells) as usize;

    // Skip boundary cells
    if cell_idx.cmpeq(GridIndex::ZERO).any() || cell_idx.cmpeq(n_cells - 1).any() {
        return;
    }

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

    // Sum up power at each workgroup
    // Restrict to local indices that touch workgroup low-boundary planes/lines/points
    const LOCAL_1: UVec2 = UVec2::new(
        if FLUX_WORKGROUP_SIZE.x > 1 { 1 } else { 0 },
        if FLUX_WORKGROUP_SIZE.y > 1 { 1 } else { 0 },
    );
    if local_idx3.x != LOCAL_1.x || local_idx3.y != LOCAL_1.y { return; };
    let wg_flat_idx = grid_idx3_to_flat_idx(workgroup_id, n_workgroups) as usize;
    let mut wg_power_sum = 0.;
    for i in 0..local_power.len() { // workgroup sums
        wg_power_sum += local_power.read(i);
    }
    wg_summations.write(wg_flat_idx, wg_power_sum);

    // Merge workgroup sums for each monitor
    // Restrict to one invocation per monitor
    if cell_idx3.axis1 != LOCAL_1.x || cell_idx3.axis2 != LOCAL_1.y { return; }
    let mut power_integral = wg_power_sum;
    for plane_wg_idx in (wg_flat_idx+1)..wg_summations.len() {
        power_integral += wg_summations.read(plane_wg_idx);
    }
    monitor_power.write(monitor_idx, power_integral);
}

/// Computes workgroup count for a flux monitor kernel.
#[cfg_attr(feature = "dim1", replace_idents(suffix = "z", axis = z, axis1 = x, axis2 = y))]
#[cfg_attr(
    feature = "dim2",
    replace_idents(suffix = "x", axis = x, axis1 = y, axis2 = z),
    replace_idents(suffix = "y", axis = y, axis1 = x, axis2 = z)
)]
#[cfg_attr(
    feature = "dim3",
    replace_idents(suffix = "x", axis = x, axis1 = y, axis2 = z),
    replace_idents(suffix = "y", axis = y, axis1 = z, axis2 = x),
    replace_idents(suffix = "z", axis = z, axis1 = x, axis2 = y)
)]
pub fn flux_num_workgroups(n_cells: GridIndex, n_flux_monitors_axis: u32) -> [u32; 3] {
    let n_cells3 = n_cells.n_cells_to_3d();
    [
        n_cells3.axis1.div_ceil(FLUX_WORKGROUP_SIZE.x),
        n_cells3.axis2.div_ceil(FLUX_WORKGROUP_SIZE.y),
        n_flux_monitors_axis
    ]
}

#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct GpuPowerFluxMonitor {
    /// Magnitude-direction of `da` vector used in power flux integral.
    pub da: Real,
    /// Grid index component of measurement plane along the axis perpendicular to it
    pub cell_idx_a: u32,
}