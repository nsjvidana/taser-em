use std::num::{NonZeroU32, NonZeroUsize};
use kiss3d::prelude::Light;
use kiss3d::window::NumSamples;
use taser_em_testbed3d::{FdtdTestbedViewer, re_exports::anyhow, VisualizationMode};
use taser_em3d::prelude::*;
use taser_em3d::re_exports::khal::backend::{Backend, Buffer, GpuBackend, WebGpu};
use taser_em3d::re_exports::glamx::glam::*;

#[kiss3d::main]
async fn main() {
    cube().await.unwrap()
}

pub async fn cube() -> anyhow::Result<()> {
    // Gaussian pulse maximum frequency
    let f_max = 2.4e9; // 2.4 GHz
    let sim_speed = NonZeroUsize::new(12).unwrap();

    // Simulation parameters w/ default stability values.
    let stability = FdtdStability {
        dt_safety_factor: 10.,
        cells_per_wavelength: 10,
        material_resolution: NonZeroU32::new(5).unwrap(),
        ..Default::default()
    };
    let cell_size = stability.cell_size_from_min_wavelength(f_max);
    let dt = stability.cfl_condition(cell_size)
        .min(stability.dt_from_gaussian_freq(f_max));
    let fdtd_params = FdtdParameters {
        cell_size,
        dt,
        material_discretization: MaterialDiscretization::Smooth {
            resolution: stability.material_resolution
        },
        // material_discretization: MaterialDiscretization::Rough,
        polarization_mode: PolarizationMode::TransverseMagnetic
    };
    let pml_params = PmlParameters::new(dt);
    let mut simulation = FdtdLossySimulation::new(fdtd_params, pml_params);

    // Compute cube dimensions
    let wavelen = C_0 / f_max;
    let cube_extents = [Vect::splat(-20.), Vect::splat(-20. + wavelen * 0.1)];
    // construct the cube
    let mat = ElectricMaterial {
        eps_r: Vec3::splat(7.),
        mu_r: Vec3::splat(1.),
        // sig: Vec3::splat(0.3),
        sig: Vec3::splat(0.),
    };
    simulation.material_regions.fill_region(cube_extents[0], cube_extents[1], mat);

    // Compute source position and gaussian curve data points
    let source = Source::Dipole {
        position: cube_extents[0] - Vect::Z * wavelen * 0.25,
        t_start: 0.,
        vals: Source::gaussian_max_f(f_max, 1., dt)
    };
    simulation.add_source(source);

    let sim_bb = simulation.compute_bounding_box();
    println!("n_cells: {}", simulation.compute_n_cells(&sim_bb, &stability));

    // Set up buffers and pipeline
    let webgpu = WebGpu::default().await?;
    let backend = GpuBackend::WebGpu(webgpu);
    let mut gpu_data = simulation.finalize(&backend, &stability)?;
    let pipeline = FdtdLossyPipeline::new(&backend, sim_speed)?;

    // Create viewer and set up camera
    let n_cells = gpu_data.n_cells;
    let grid_extents = n_cells.as_vect() * cell_size;
    let grid_center = grid_extents / 2.;
    let mut testbed = FdtdTestbedViewer::new(
        &simulation,
        &stability,
        VisualizationMode::default()
    ).await?;
    testbed.window.set_samples(NumSamples::Four);
    testbed.window.set_ambient(0.5);
    testbed.set_clipping_planes(cell_size.smallest_element() / 100., cell_size.largest_element() * 1000.)
        .camera
        .look_at(grid_extents * 2., grid_center);
    testbed.camera.set_up_axis_dir(Vec3::Z);

    testbed.scene
        .add_light(
            Light::directional(Vec3::new(-0.5, -0.8, -0.4))
                .with_intensity(2.2)
        )
        .set_position(grid_extents * 2.);

    // Render simulation
    let mut dn_field = vec![Vec4::ZERO; gpu_data.dn.buffer.len()];
    while testbed.render_frame(&dn_field).await {
        backend.synchronize()?;
        gpu_data.dn.read(&backend, &mut dn_field).await?;
        pipeline.submit_steps(&backend, &mut gpu_data, None, |encoder, gpu_data| {
            gpu_data.dn.encode_copy_cmd(encoder)
        })?;
    }

    Ok(())
}