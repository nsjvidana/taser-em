use std::num::NonZeroU32;
use taser_em2d::prelude::*;
use taser_em2d::re_exports::glamx::glam::*;
use taser_em2d::re_exports::khal;
use taser_em2d::re_exports::khal::backend::*;
use taser_em2d::re_exports::khal::Shader;
use taser_em_testbed2d::{re_exports::anyhow, FdtdTestbedViewer, VisualizationMode};

#[kiss3d::main]
async fn main() {
    single_rod().await.unwrap()
}

pub async fn single_rod() -> anyhow::Result<()> {
    // Gaussian pulse maximum frequency
    let f_max = 2.4e9; // 2.4 GHz
    let sim_speed = 5;

    // Simulation parameters w/ default stability values.
    let stability = FdtdStability {
        dt_safety_factor: 10.,
        cells_per_wavelength: 10,
        material_resolution: NonZeroU32::new(10).unwrap(),
        ..Default::default()
    };
    let cell_size = stability.cell_size_from_min_wavelength(f_max);
    let dt = stability.cfl_condition(cell_size)
        .min(stability.dt_from_gaussian_freq(f_max));
    let parameters = FdtdParameters {
        cell_size,
        dt,
        material_discretization: MaterialDiscretization::Smooth {
            resolution: stability.material_resolution
        },
        // material_discretization: MaterialDiscretization::Rough,
        polarization_mode: PolarizationMode::TransverseMagnetic
    };
    let mut simulation = FdtdLossySimulation::new(parameters, PmlParameters::new(dt));

    // Compute slab dimensions
    let wavelen = C_0 / f_max;
    let slab_extents = [Vect::splat(-20.), Vect::splat(-20. + wavelen * 0.5)];
    // construct the slab
    let mat = ElectricMaterial {
        eps_r: Vec3::splat(7.),
        mu_r: Vec3::splat(1.),
        // sig: Vec3::splat(0.3),
        sig: Vec3::splat(0.),
    };
    simulation.material_regions.fill_region(slab_extents[0], slab_extents[1], mat);

    // Compute source position and gaussian curve data points
    let source = Source::Dipole {
        position: slab_extents[0] - wavelen * 0.5,
        t_start: 0.,
        vals: Source::gaussian_max_f(f_max, 1., dt),
        moment: Vec3::Z
    };
    simulation.add_source(source);
    
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
    let n_cells = gpu_data.n_cells;
    let grid_extents = n_cells.as_vect() * cell_size;
    let grid_center = grid_extents / 2.;
    let vis_mode = VisualizationMode::default();
    let mut testbed = FdtdTestbedViewer::new(&simulation, &stability, vis_mode).await?;
    testbed.window.set_ambient(0.5);
    testbed.set_clipping_planes(cell_size.min_element() / 3., cell_size.max_element() * 1000.)
        .camera
        .look_at(
            grid_center.to_3d(Vec3::Z * grid_extents.length() * 3.),
            grid_center.to_3d(Vec3::ZERO)
        );
    testbed.camera.set_up_axis(Vec3::Z);
        
    // Render simulation
    let mut dn_field = vec![Vec4::ZERO; gpu_data.dn.buffer.len()];
    while testbed.render_frame(&dn_field).await {
        backend.synchronize()?;
        gpu_data.dn.read(&backend, &mut dn_field).await?;

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("2d fdtd example", None);
        pipeline.dispatch_steps(&mut pass, &mut gpu_data)?;
        drop(pass);
        gpu_data.dn.encode_copy_cmd(&mut encoder)?;
        gpu_data.steps.encode_copy_cmd(&mut encoder)?;
        backend.submit(encoder)?;
    }
    let mut steps = vec![0];
    gpu_data.steps.read(&backend, &mut steps).await?;
    println!("simulated time: {:?} ns", steps[0] as Real * dt * 1e9);
    println!("steps: {:?}", steps[0]);

    Ok(())
}