use kiss3d::glamx::Vec3;
use taser_em2d::prelude::*;
use taser_em_testbed2d::{re_exports::anyhow, FdtdTestbedViewer, VisualizationMode};

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
        cells_per_wavelength: 15,
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

    // Construct device
    let wavelen = C_0 / f_max;
    let mat = ElectricMaterial {
        eps_r: Vec3::splat(7.),
        mu_r: Vec3::splat(1.),
        // sig: Vec3::splat(0.3),
        sig: Vec3::splat(0.),
    };
    simulation.material_regions.load_trimesh_regions(
        mat,
        "assets/suzanne.obj",
        Vec3::splat(wavelen)
    )?;

    // Compute source position and gaussian curve data points
    let source_values = Source::gaussian_max_f(f_max, 1., dt);
    // simulation.add_source(Source::TFSF {
    //     spatial_axis: SpatialAxis::X,
    //     // direction: WaveDirection::Positive,
    //     direction: WaveDirection::Negative,
    //     t_start: 0.0,
    //     vals: source_values.clone(),
    //     polarization: Vec3::Z,  // Transverse Magnetic Z mode
    //     tfsf_buffer_width: LayerWidths::splat_spatial(3)
    //         .with_axis_widths(SpatialAxis::X, LoHiWidths::splat(8)),
    // });
    simulation.add_source(Source::Dipole {
        dipole_type: DipoleType::Electric,
        position: Vect::from_vec3(simulation.compute_bounding_box().mins - wavelen),
        t_start: 0.0,
        vals: source_values,
        moment: Vec3::Z,
    });
    
    // Set up buffers and pipeline
    let backend = create_backend().await?;
    let backend_name = backend_name(&backend);
    println!("Running on backend: {backend_name}");
    let boundary_conditions = BoundaryConditions::new(
        PeriodicBoundaryX::from_backend(&backend)?,
        PeriodicBoundaryY::from_backend(&backend)?,
        // PECBoundaryY::from_backend(&backend)?,
    );
    simulation.pml_parameters.widths = simulation.pml_parameters.widths
        .with_axis_widths(SpatialAxis::X, LoHiWidths::splat(0))
        .with_axis_widths(SpatialAxis::Y, LoHiWidths::splat(0));
    let mut state = simulation.finalize(&backend, &stability)?;
    let mut pipeline = FdtdLossyPipeline::new_initialized(
        &backend,
        boundary_conditions,
        sim_speed,
        &mut state
    )?;
    let mut readback = FdtdStateReadback::new(&backend, &state, FdtdSimulationMode::TransverseMagneticZ)?;
    
    // Create viewer and set up camera
    let vis_mode = VisualizationMode::default();
    let mut testbed = FdtdTestbedViewer::new(&simulation, &stability, vis_mode).await?;
    testbed.window.set_ambient(0.5);
        
    // Render simulation
    while testbed.render_frame(&readback.get_dn_field()).await {
        readback.read_back_dn(&backend)?;

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("2d fdtd example", None);
        pipeline.dispatch_steps(&mut pass, &mut state)?;
        drop(pass);
        backend.submit(encoder)?;
        readback.request_copy_dn(&backend, &state)?;
    }
    
    readback.request_copy_t_idx(&backend, &state)?;
    readback.read_back_t_idx(&backend)?;
    let n_steps = readback.get_t_idx();
    println!("simulated time: {:?} ns", n_steps as Real * dt * 1e9);
    println!("steps: {:?}", n_steps);

    Ok(())
}