use std::num::NonZeroU32;
use taser_em1d::prelude::*;
use taser_em1d::re_exports::glamx::glam::*;
use taser_em1d::re_exports::khal::backend::{Backend, Buffer, Encoder, GpuBackend, WebGpu};
use taser_em_testbed1d::FdtdTestbedViewer;
use taser_em_testbed1d::re_exports::anyhow;

#[kiss3d::main]
async fn main() {
    cube().await.unwrap()
}

pub async fn cube() -> anyhow::Result<()> {
    let webgpu = WebGpu::default().await?;
    let backend = GpuBackend::WebGpu(webgpu);

    // Gaussian pulse maximum frequency
    let f_max = 2.4e9; // 2.4 GHz

    // Simulation parameters w/ default stability values.
    let stability = FdtdStability::default();
    let cell_size = stability.cell_size_from_min_wavelength(f_max);
    let dt = stability.cfl_condition(cell_size)
        .min(stability.dt_from_gaussian_freq(f_max));

    // Compute slab dimensions
    let wavelen = C_0 / f_max;
    let slab_extents = [Vect::splat(-20.), Vect::splat(-20. + wavelen * 2.)];

    // Compute source position and gaussian curve data points
    let source = Source::Dipole {
        position: slab_extents[0] - wavelen * 3.,
        t_start: 0.,
        vals: Source::gaussian_max_f(f_max, 1., dt)
    };

    // Construct the slab shape
    let mut mat_regions = MaterialRegions::new();
    let mat = ElectricMaterial {
        eps_r: Vec3::splat(7.),
        mu_r: Vec3::splat(1.),
        sig: Vec3::splat(0.3),
    };
    mat_regions.fill_region(slab_extents[0], slab_extents[1], mat);

    // Discretize slab in the grid
    let grid = YeeGrid::new(
        mat_regions,
        cell_size,
        PolarizationMode::TransverseMagnetic,
        &stability
    );
    let mut solver = FdtdSolver::new(&backend, grid, dt)?;
    solver.add_source(source);
    let n_cells = solver.grid_n_cells();
    let (regions_offset, coeffs) = solver.compute_pml_coeffs();
    let mut buffers = solver.create_shader_data(&backend, &coeffs, regions_offset)?;

    // Set up viewer
    let grid_extents = n_cells.as_vect() * cell_size;
    let mut testbed = FdtdTestbedViewer::new().await?;
    testbed.set_clipping_planes(cell_size / 3., cell_size * 1000.)
        .camera
        .look_at(
            Vec3::new(-grid_extents, 0., grid_extents / 2.),
            Vec3::Z * grid_extents / 2.
        );
    testbed.add_region_meshes(&solver.grid.material_regions, regions_offset);
    let mut dn_field = vec![Vec4::ZERO; buffers.dn.buffer.len()];
    while testbed.render_frame(&dn_field, n_cells, cell_size).await {
        backend.synchronize()?;
        buffers.dn.read(&backend, &mut dn_field).await?;
        // Submit simulation step
        {
            let mut encoder = backend.begin_encoding();
            let mut pass = encoder.begin_pass("fdtd vis", None);
            solver.submit_step(&mut buffers, &mut pass)?;
            drop(pass);
            buffers.dn.encode_copy_cmd(&mut encoder)?;
            backend.submit(encoder)?;
        }
    }
    Ok(())
}