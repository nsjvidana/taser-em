use kiss3d::glamx::Vec3;
use taser_em2d::prelude::*;
use taser_em_testbed2d::{re_exports::anyhow, ColorMode, FdtdTestbedViewer, VectorFieldVisual, VisualizationMode};

#[kiss3d::main]
async fn main() {
    let example = Example::DipoleAntenna;
    match example {
        Example::Suzanne => suzanne_cross_section().await.unwrap(),
        Example::DipoleAntenna => dipole_antenna().await.unwrap(),
    }
}

enum Example {
    Suzanne,
    DipoleAntenna
}

pub async fn suzanne_cross_section() -> anyhow::Result<()> {
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
    };
    let mut simulation = FdtdLossySimulation::new(parameters, PmlParameters::new(dt));

    // Construct device
    let wavelen = C_0 / f_max;
    let mat = ElectricMaterial {
        eps_r: Vec3::splat(7.),
        mu_r: Vec3::splat(1.),
        sig: Vec3::INFINITY,
        // sig: Vec3::splat(0.),
    };
    simulation.material_regions.load_trimesh_regions(
        mat,
        "assets/suzanne.obj",
        Vec3::splat(wavelen)
    )?;

    // Compute source position and gaussian curve data points
    let source_values = Source::gaussian_max_f(f_max, 1., dt);
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
        PECBoundaryX::from_backend(&backend)?,
        PECBoundaryY::from_backend(&backend)?,
    );
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
    let mut testbed = FdtdTestbedViewer::new(&simulation, &stability, vis_mode, VectorFieldVisual::H).await?;
    testbed.window.set_ambient(0.5);
        
    // Render simulation
    while testbed.render_frame(&backend, &state, &mut readback).await? {
        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("2d suzanne example", None);
        pipeline.dispatch_steps(&mut pass, &mut state)?;
        drop(pass);
        backend.submit(encoder)?;
    }
    
    readback.request_copy_t_idx(&backend, &state)?;
    readback.read_back_t_idx(&backend)?;
    let n_steps = readback.get_t_idx();
    println!("simulated time: {:?} ns", n_steps as Real * dt * 1e9);
    println!("steps: {:?}", n_steps);

    Ok(())
}

pub async fn dipole_antenna() -> anyhow::Result<()> {
    // Gaussian pulse maximum frequency
    let freq = 2.4e9; // 2.4 GHz
    let dft_resolution = 10;
    let sim_speed = 3;

    // Simulation parameters w/ default stability values.
    let stability = FdtdStability {
        dt_safety_factor: 15.,
        cells_per_wavelength: 30,
        spacer_region_widths: LayerWidths::splat_spatial(30),
        ..Default::default()
    };
    let cell_size = stability.cell_size_from_min_wavelength(freq);
    let dt = stability.cfl_condition(cell_size);
    let parameters = FdtdParameters {
        cell_size,
        dt,
        material_discretization: MaterialDiscretization::Smooth {
            resolution: stability.material_resolution
        },
    };
    let mut simulation = FdtdLossySimulation::new(parameters, PmlParameters::new(dt));

    // Construct dipole antenna
    let antenna_len = C_0 / (freq * 2.);
    let elem_thickness = cell_size.y;
    let feed_gap = cell_size.y * 2.;
    let half_len = antenna_len / 2.0;
    let pec = ElectricMaterial::PEC;
    simulation
        .fill_region(
            Vect::ZERO,
            Vect::new(elem_thickness, -half_len),
            pec
        )
        .fill_region(
            Vect::new(0., feed_gap),
            Vect::new(0., feed_gap) + Vect::new(elem_thickness, half_len),
            pec
        );

    // Source injection in antenna feed gap
    let source_values = Source::sin_cycle(freq, dt).repeat(10);
    simulation
        .add_source(Source::Dipole {
            dipole_type: DipoleType::Electric,
            position: Vect::new(elem_thickness, feed_gap) / 2.,
            t_start: 0.0,
            vals: source_values.clone(),
            moment: Vec3::Y,
        });

    // Power flux monitor for reading back power flux
    let flux_monitor = simulation.add_flux_monitor(
        PowerFluxMonitor {
            axis: SpatialAxis::X,
            position: -(stability.spacer_region_widths[SpatialAxis::X].hi as Real * cell_size.x) * 0.90,
            direction: Direction::Negative,
        }
    );

    // Create instance of backend we will run the simulation with
    let backend = create_backend().await?;
    let backend_name = backend_name(&backend);
    println!("Running on backend: {backend_name}");
    let boundary_conditions = BoundaryConditions::new(
        PECBoundaryX::from_backend(&backend)?,
        PECBoundaryY::from_backend(&backend)?,
    );

    // Create GPU simulation state
    let mut state = simulation.finalize(&backend, &stability)?;

    // Power flux DFT state
    let frequencies = frequencies_from_range((freq * 0.5)..=(freq * 1.5), dft_resolution);
    let dft = PowerFluxFunction::new(&flux_monitor, state.power_flux_states.as_ref().unwrap())?
        .to_dft(frequencies.clone())?;
    let flux_func = dft.get_function().clone();
    let mut flux_dft_states = DftStates::new_zeroed(
        &backend,
        vec![dft],
        state.power_flux_states.as_ref().unwrap().monitor_power.clone()
    )?;

    // Create and initialize pipelines
    let mut pipeline = FdtdLossyPipeline::new(
        &backend,
        boundary_conditions,
        sim_speed
    )?;
    let mut dft_pipeline = DftPipeline::new(&backend)?;
    let mut encoder = backend.begin_encoding();
    let mut pass = encoder.begin_pass("2D pipeline initialization", None);
    pipeline.initialize(&mut pass, &mut state)?;
    dft_pipeline.initialize_states(&mut pass, &state.grid_params, &mut flux_dft_states)?;
    drop(pass);
    backend.submit(encoder)?;

    // Set up readback
    let mut readback = FdtdStateReadback::new(&backend, &state, FdtdSimulationMode::TransverseElectricZ)?;
    let mut power_readback = PowerFluxReadback::new(&backend, &state)?
        .expect("we added a flux monitor to the simulation so readback must be possible");
    let mut dft_readback = DftReadback::new(&backend, &flux_dft_states).await?;

    // Create viewer and set up camera
    let vis_mode = VisualizationMode::default()
        .with_color_mode(ColorMode::default().to_fixed_range(0.0..0.25));
    let mut testbed = FdtdTestbedViewer::new(&simulation, &stability, vis_mode, VectorFieldVisual::H).await?;

    // Render simulation
    while testbed.render_frame(&backend, &state, &mut readback).await? {
        power_readback.read_back(&backend)?;
        power_readback.request_copy(&backend, &state)?;
        let instantaneous_flux = power_readback.get_power(&flux_monitor).unwrap();
        println!("Instantaneous power flux: {instantaneous_flux}");

        dft_readback.read_back(&backend)?;
        dft_readback.request_copy(&backend, &flux_dft_states)?;
        let dft_vals = dft_readback.get_dft(&flux_func).unwrap();
        for (f, cmplx) in frequencies.iter().zip(dft_vals.into_iter()) {
            println!("{f} Hz: {}", cmplx.norm());
        }

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("2d dipole antenna example", None);
        pipeline.dispatch_steps_aux(&mut pass, &mut state, |pass, state|
            dft_pipeline.dispatch_step(
                pass,
                &state.t_idx,
                &mut flux_dft_states
            )
        )?;
        drop(pass);
        backend.submit(encoder)?;
    }

    readback.request_copy_t_idx(&backend, &state)?;
    readback.read_back_t_idx(&backend)?;
    let n_steps = readback.get_t_idx();
    println!("simulated time: {:?} ns", n_steps as Real * dt * 1e9);
    println!("steps: {:?}", n_steps);

    Ok(())
}