use kiss3d::prelude::RED;
use taser_em2d::prelude::*;
use taser_em2d::shaders::fdtd::{AuxVect, TfsfCorrections};
use taser_em_testbed2d::{re_exports::anyhow, ColorMode, FdtdTestbedViewer, VisualizationMode};

#[kiss3d::main]
async fn main() {
    single_rod().await.unwrap()
}

pub async fn single_rod() -> anyhow::Result<()> {
    // Gaussian pulse maximum frequency
    let f_max = 2.4e9; // 2.4 GHz
    let sim_speed = 3;

    // Simulation parameters w/ default stability values.
    let stability = FdtdStability {
        dt_safety_factor: 10.,
        cells_per_wavelength: 10,
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
    let quad_min = Vect::splat(-20.);
    let quad_max = Vect::splat(-20. + wavelen * 0.5);
    // construct the slab
    let mat = ElectricMaterial {
        eps_r: Vec3::splat(7.),
        mu_r: Vec3::splat(1.),
        // sig: Vec3::splat(0.3),
        sig: Vec3::splat(0.),
    };
    simulation.material_regions.fill_region(quad_min, quad_max, mat);

    // Compute source position and gaussian curve data points
    let quad_center = (quad_min + quad_max) / 2.;
    let source = Source::TFSF {
        spatial_axis: SpatialAxis::X,
        direction: WaveDirection::Positive,
        // position: quad_center.x + ((quad_extents.x / 2.) + wavelen),
        // direction: WaveDirection::Negative,
        t_start: 0.0,
        vals: Source::gaussian_max_f(f_max, 1., dt),
        polarization: Vec3::Z,
        tfsf_buffer_width: LayerWidths::splat_spatial(5),
    };
    simulation.add_source(source);
    
    // Set up buffers and pipeline
    let backend = create_backend().await?;
    let backend_name = backend_name(&backend);
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
            grid_center.to_3d(Vec3::Z * grid_extents.length()),
            grid_center.to_3d(Vec3::ZERO)
        );
    testbed.camera.set_up_axis(Vec3::Z);
        
    // Render simulation
    let mut dn_field = vec![Vec4::ZERO; gpu_data.dn.buffer.len()];
    let mut aux_en = vec![Vec2::ZERO; gpu_data.tfsf_dispatch_data.en.buffer.len()];
    while testbed.render_frame(&dn_field).await {
        let mut prev_pos = Vec3::ZERO + Vec3::new(0., aux_en[0].y, 0.);
        for (i, v) in aux_en.iter().enumerate().skip(1) {
            let cell_pos = Vec3::new(i as Real * cell_size.x, 0., 0.);
            let pos2 = cell_pos + Vec3::new(0., v.y * cell_size.x, 0.);
            testbed.window.draw_line(prev_pos, pos2, RED, 2., false);
            prev_pos = pos2;
        }
        backend.synchronize()?;
        gpu_data.tfsf_dispatch_data.en.read(&backend, &mut aux_en).await?;
        gpu_data.dn.read(&backend, &mut dn_field).await?;

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("2d fdtd example", None);
        pipeline.dispatch_steps(&mut pass, &mut gpu_data)?;
        drop(pass);
        gpu_data.dn.encode_copy_cmd(&mut encoder)?;
        gpu_data.t_idx.encode_copy_cmd(&mut encoder)?;
        gpu_data.tfsf_dispatch_data.en.encode_copy_cmd(&mut encoder)?;
        backend.submit(encoder)?;
    }
    let mut steps = vec![0];
    gpu_data.t_idx.read(&backend, &mut steps).await?;
    println!("simulated time: {:?} ns", steps[0] as Real * dt * 1e9);
    println!("steps: {:?}", steps[0]);

    Ok(())
}