use kiss3d::color::*;
use std::num::NonZeroU32;
use taser_em3d::prelude::*;
use taser_em3d::re_exports::glamx::glam::*;
use taser_em3d::re_exports::khal;
use taser_em3d::re_exports::khal::backend::{Backend, Buffer, GpuBackend};
use taser_em3d::re_exports::khal::Shader;
use taser_em_testbed3d::{re_exports::anyhow, ColorMode, FdtdTestbedViewer, VisualizationMode};

#[kiss3d::main]
async fn main() {
    cube().await.unwrap()
}

pub async fn cube() -> anyhow::Result<()> {
    // Gaussian pulse maximum frequency
    let f_max = 2.4e9; // 2.4 GHz
    let sim_speed = 6;

    // Simulation parameters w/ default stability values.
    let stability = FdtdStability {
        dt_safety_factor: 8.,
        material_resolution: NonZeroU32::new(3).unwrap(),
        ..Default::default()
    };
    let cell_size = stability.cell_size_from_min_wavelength(f_max);
    let dt = stability.cfl_condition(cell_size)
        .min(stability.dt_from_gaussian_freq(f_max));
    let fdtd_params = FdtdParameters {
        cell_size,
        dt,
        // material_discretization: MaterialDiscretization::Smooth {
        //     resolution: stability.material_resolution
        // },
        material_discretization: MaterialDiscretization::Rough,
        polarization_mode: PolarizationMode::TransverseElectric
    };
    let pml_params = PmlParameters::new(dt);
    let mut simulation = FdtdLossySimulation::new(fdtd_params, pml_params);

    // Compute cube dimensions
    let wavelen = C_0 / f_max;
    let box_min = Vect::splat(-20.);
    let box_max = box_min + Vect::splat(wavelen);
    let box_extents = box_max - box_min;
    let box_center = (box_min + box_max) / 2.;
    // construct the cube
    let mat = ElectricMaterial {
        eps_r: Vec3::splat(3.),
        mu_r: Vec3::splat(1.),
        // sig: Vec3::splat(0.3),
        sig: Vec3::splat(0.),
    };
    simulation.material_regions.fill_region(box_min, box_max, mat);

    // Compute source position and gaussian curve data points
    let source = Source::Dipole {
        position: box_center - (box_extents.with_yz(Vec2::ZERO) / 2.) - Vec3::ZERO.with_x(wavelen),
        t_start: 0.,
        vals: Source::gaussian_max_f(f_max, 1., dt),
        moment: Vec3::Z
    };
    simulation.add_source(source);

    let sim_bb = simulation.compute_bounding_box();
    println!("n_cells: {}", simulation.compute_n_cells(&sim_bb, &stability));

    // Set up buffers and pipeline
    let (backend, backend_name) = cfg_select! {
        feature = "webgpu" => {{
            let webgpu = khal::backend::WebGpu::default().await?;
            (GpuBackend::WebGpu(webgpu), "WebGPU")
        }}
        feature = "metal" => {todo!()}
        feature = "cpu" => { (GpuBackend::Cpu, "CPU") }
        feature = "cuda" => {todo!()}
    };
    println!("Running on backend: {backend_name}");
    let mut gpu_data = simulation.finalize(&backend, &stability)?;
    let boundary_condition = PECBoundary::from_backend(&backend)?;
    let mut pipeline = FdtdLossyPipeline::new(&backend, boundary_condition, sim_speed)?;

    // Create viewer and set up camera
    let mut testbed = FdtdTestbedViewer::new(
        &simulation,
        &stability,
        VisualizationMode::default().with_color(
            ColorMode::FixedRange {
                v_min: 0.,
                v_max: 0.05,
                color_min: TRANSPARENT,
                color_max: RED
            }
        )
    ).await?;
    testbed.window.set_ambient(0.5);
    let n_cells = gpu_data.n_cells;
    let grid_extents = n_cells.as_vect() * cell_size;
    let grid_center = grid_extents / 2.;
    testbed
        .set_clipping_planes(cell_size.min_element() / 100., cell_size.max_element() * 1000.)
        .camera
        .look_at(
            Vec3::new(-grid_extents.x * 2., -grid_extents.y * 2., grid_extents.z * 2.),
            grid_center
        );
    testbed.camera.set_up_axis_dir(Vec3::Z);

    // Render simulation
    let mut dn_field = vec![Vec4::ZERO; gpu_data.dn.buffer.len()];
    while testbed.render_frame(&dn_field).await {
        backend.synchronize()?;
        gpu_data.dn.read(&backend, &mut dn_field).await?;
        pipeline.submit_steps(&backend, &mut gpu_data, None, |encoder, gpu_data| {
            gpu_data.dn.encode_copy_cmd(encoder)?;
            gpu_data.steps.encode_copy_cmd(encoder)
        })?;
    }

    let mut steps = vec![0];
    gpu_data.steps.read(&backend, &mut steps).await?;
    println!("Simulated time: {} ns", dt * steps[0] as f32 * 1e9);
    println!("steps: {}", steps[0]);

    Ok(())
}