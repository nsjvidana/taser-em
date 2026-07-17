use std::num::{NonZeroU32, NonZeroUsize};
use taser_em1d::prelude::*;
use taser_em1d::re_exports::glamx::glam::*;
use taser_em1d::re_exports::khal::backend::{Backend, Buffer, GpuBackend, WebGpu};
use taser_em_testbed1d::FdtdTestbedViewer;
use taser_em_testbed1d::re_exports::anyhow;

#[kiss3d::main]
async fn main() {
    single_slab().await.unwrap()
}

pub async fn single_slab() -> anyhow::Result<()> {
    // Gaussian pulse maximum frequency
    let f_max = 2.4e9; // 2.4 GHz

    // Simulation parameters w/ default stability values.
    let stability = FdtdStability::default();
    let cell_size = stability.cell_size_from_min_wavelength(f_max);
    let dt = stability.cfl_condition(cell_size)
        .min(stability.dt_from_gaussian_freq(f_max));
    let parameters = FdtdParameters {
        cell_size,
        dt,
        material_discretization: MaterialDiscretization::Smooth {
            resolution: NonZeroU32::new(3).unwrap()
        },
        polarization_mode: PolarizationMode::TransverseMagnetic
    };
    let mut simulation = FdtdLossySimulation::new(parameters, PmlParameters::new(dt));

    // Compute slab dimensions
    let wavelen = C_0 / f_max;
    let slab_extents = [Vect::splat(-20.), Vect::splat(-20. + wavelen * 2.)];

    // Compute source position and gaussian curve data points
    let source = Source::Dipole {
        position: slab_extents[0] - wavelen * 3.,
        t_start: 0.,
        vals: Source::gaussian_max_f(f_max, 1., dt)
    };

    // Construct the slab
    let mat = ElectricMaterial {
        eps_r: Vec3::splat(7.),
        mu_r: Vec3::splat(1.),
        sig: Vec3::splat(0.3),
    };
    simulation.material_regions.fill_region(slab_extents[0], slab_extents[1], mat);
    simulation.add_source(source);

    // TODO: prevent user from having to compute these for rendering...
    let n_cells = simulation.compute_n_cells(&stability);
    let regions_offset = YeeGridMaterials::compute_regions_offset(
        &simulation.material_regions,
        n_cells,
        cell_size,
    );

    // Set up buffers and pipeline
    let webgpu = WebGpu::default().await?;
    let backend = GpuBackend::WebGpu(webgpu);
    let mut gpu_data = simulation.finalize(&backend, &stability)?;
    let pipeline = FdtdLossyPipeline::new(&backend, NonZeroUsize::new(1).unwrap())?;

    // Set up viewer and camera
    let grid_extents = n_cells.as_vect() * cell_size;
    let mut testbed = FdtdTestbedViewer::new().await?;
    testbed.set_clipping_planes(cell_size / 3., cell_size * 1000.)
        .camera
        .look_at(
            Vec3::new(-grid_extents, 0., grid_extents / 2.),
            Vec3::Z * grid_extents / 2.
        );
    testbed.add_region_meshes(&simulation.material_regions, regions_offset);

    // Render simulation
    let mut dn_field = vec![Vec4::ZERO; gpu_data.dn.buffer.len()];
    while testbed.render_frame(&dn_field, n_cells, cell_size).await {
        backend.synchronize()?;
        gpu_data.dn.read(&backend, &mut dn_field).await?;
        pipeline.submit_steps(&backend, &mut gpu_data, None, |encoder, gpu_data| {
            gpu_data.dn.encode_copy_cmd(encoder)
        })?;
    }
    Ok(())
}