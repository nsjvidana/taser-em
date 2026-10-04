use kiss3d::egui::Color32;
use kiss3d::glamx::Vec3;
use taser_em2d::prelude::*;
use taser_em_testbed2d::plot::{DftPlotLine, PlotLine, PlotWindow};
use taser_em_testbed2d::{ColorMode, FdtdTestbedViewer, VectorFieldVisual, VisualizationMode, re_exports::anyhow};

#[kiss3d::main]
async fn main() {
    let example = Example::DipoleAntenna;
    match example {
        Example::Suzanne => suzanne_cross_section().await.unwrap(),
        Example::DipoleAntenna => dipole_antenna().await.unwrap(),
        Example::BenchAll => benchmark_all().await.unwrap(),
    }
}

enum Example {
    Suzanne,
    DipoleAntenna,
    BenchAll,
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
    simulation.add_dipole(
        Dipole {
            dipole_type: DipoleType::Electric,
            position: Vect::from_vec3(simulation.compute_bounding_box().mins - wavelen),
            t_start: 0.0,
            moment: Vec3::Z,
        },
        source_values
    );

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
    let dft_resolution = 100;
    let sim_speed = 1;

    // Simulation parameters w/ default stability values.
    let stability = FdtdStability {
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
    let dipole = Dipole {
        dipole_type: DipoleType::Electric,
        position: Vect::new(elem_thickness, feed_gap) / 2.,
        t_start: 0.0,
        moment: Vec3::Y,
    };
    let dipole = simulation.add_dipole(dipole, source_values);

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

    // Set up power flux DFT
    let frequencies = frequencies_from_range(0.0..=(freq * 2.), dft_resolution);
    let (dft, flux_func) = PowerFluxFunction::new(&flux_monitor, state.power_flux_states.as_ref().unwrap())?
        .to_dft(frequencies.clone())?;
    let mut flux_dft_states = DftStates::new_zeroed(&backend, vec![dft])?;

    // Set up source DFT
    let (dft, src_func) = SourceFunction::new(&dipole, &state.source_states)?
        .to_dft(frequencies)?;
    let mut src_dft_states = DftStates::new_zeroed(&backend, vec![dft])?;

    // Create and initialize pipelines
    let mut pipeline = FdtdLossyPipeline::new(
        &backend,
        boundary_conditions,
        sim_speed
    )?;
    let dft_pipeline = DftPipeline::new(&backend)?;
    let mut encoder = backend.begin_encoding();
    let mut pass = encoder.begin_pass("2D pipeline initialization", None);
    pipeline.initialize(&mut pass, &mut state)?;
    dft_pipeline.initialize_states(&mut pass, &state.grid_params, &mut flux_dft_states)?;
    dft_pipeline.initialize_states(&mut pass, &state.grid_params, &mut src_dft_states)?;
    drop(pass);
    backend.submit(encoder)?;

    // Set up readback
    let mut readback = FdtdStateReadback::new(&backend, &state, FdtdSimulationMode::TransverseElectricZ)?;
    let mut power_readback = PowerFluxReadback::new(&backend, &state)?
        .expect("we added a flux monitor to the simulation so readback must be possible");
    let mut flux_dft_read = DftReadback::new(&backend, &flux_dft_states).await?;
    let mut src_dft_read = DftReadback::new(&backend, &src_dft_states).await?;

    // Create viewer and set up camera
    let vis_mode = VisualizationMode::default()
        .with_color_mode(ColorMode::default().to_fixed_range(0.0..0.25));
    let mut testbed = FdtdTestbedViewer::new(&simulation, &stability, vis_mode, VectorFieldVisual::H).await?;

    // Set up DFT plot
    let mut plot_window = PlotWindow::new(
        "Power Flux DFT",
        Some("Frequency (GHz)"),
        None,
    );
    let mut flux_plot = DftPlotLine::new(
        "Power Flux",
        &flux_func,
        &flux_dft_read,
        Some(1e-9) // Frequency is in GHz, so multiply by 1E-9 for a cleaner X axis
    )?.with_color(Color32::RED);
    let mut src_plot = DftPlotLine::new(
        "Source",
        &src_func,
        &src_dft_read,
        Some(1e-9) // Frequency is in GHz, so multipl by 1E-9 for a cleaner X axis
    )?.with_color(Color32::GREEN);

    // Render simulation
    while testbed.render_frame(&backend, &state, &mut readback).await? {
        power_readback.read_back(&backend)?;
        power_readback.request_copy(&backend, &state)?;
        let instantaneous_flux = power_readback.get_power(&flux_monitor).unwrap();
        println!("Instantaneous power flux: {instantaneous_flux}");

        flux_dft_read.read_back(&backend)?;
        flux_dft_read.request_copy(&backend, &flux_dft_states)?;
        flux_plot.update_points(&flux_dft_read)?;
        src_dft_read.read_back(&backend)?;
        src_dft_read.request_copy(&backend, &src_dft_states)?;
        src_plot.update_points(&src_dft_read)?;
        plot_window.show(&mut testbed, vec![src_plot.create_line(), flux_plot.create_line()])?;

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("2d dipole antenna example", None);
        pipeline.dispatch_steps_aux(&mut pass, &mut state, |pass, state| {
            dft_pipeline.dispatch_step(
                pass,
                &state.t_idx,
                state.power_flux_states.as_ref().unwrap(),
                &mut flux_dft_states,
            )?;
            dft_pipeline.dispatch_step(
                pass,
                &state.t_idx,
                &state.source_states,
                &mut src_dft_states,
            )
        })?;
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

pub async fn benchmark_all() -> anyhow::Result<()> {
    const WARM_UP: u32 = 10;
    const BENCH: u32 = 3000;
    const SIM_SPEED: usize = 1;

    // Gaussian pulse maximum frequency
    let freq = 2.4e9; // 2.4 GHz
    let dft_resolution = 100;

    // Simulation parameters w/ default stability values.
    let stability = FdtdStability {
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

    // Source injection in antenna feed gap, and a TF/SF source
    let source_values = Source::sin_cycle(freq, dt).repeat(10);
    simulation
        .add_dipole(
            Dipole {
                dipole_type: DipoleType::Electric,
                position: Vect::new(elem_thickness, feed_gap) / 2.,
                t_start: 0.0,
                moment: Vec3::Y,
            },
            source_values,
        );
    simulation.add_tfsf(
        Tfsf {
            spatial_axis: SpatialAxis::Y,
            direction: Direction::Negative,
            t_start: 0.0,
            polarization: Vec3::Z,
            tfsf_buffer_width: LayerWidths::splat_spatial(3),
        },
        Source::gaussian_max_f(freq, 1., dt)
    );

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
    let boundary_conditions = BoundaryConditions::new(
        PECBoundaryX::from_backend(&backend)?,
        PECBoundaryY::from_backend(&backend)?,
    );

    // Create GPU simulation state
    let mut state = simulation.finalize(&backend, &stability)?;

    // Power flux DFT state
    let frequencies = frequencies_from_range(0.0..=(freq * 2.), dft_resolution);
    let (dft, _) = PowerFluxFunction::new(&flux_monitor, state.power_flux_states.as_ref().unwrap())?
        .to_dft(frequencies.clone())?;
    let mut flux_dft_states = DftStates::new_zeroed(&backend, vec![dft])?;

    // Create and initialize pipelines
    let mut pipeline = FdtdLossyPipeline::new(
        &backend,
        boundary_conditions,
        SIM_SPEED
    )?;
    let dft_pipeline = DftPipeline::new(&backend)?;
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

    let mut run_sim = || -> anyhow::Result<()> {
        // Dummy readbacks (also synchronizes the backend)
        power_readback.read_back(&backend)?;
        power_readback.request_copy(&backend, &state)?;
        dft_readback.try_read_back(&backend); // already synced at this point, so trying is enough.
        dft_readback.request_copy(&backend, &flux_dft_states)?;

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("2d benchmark example", None);
        pipeline.dispatch_steps_aux(&mut pass, &mut state, |pass, state|
            dft_pipeline.dispatch_step(
                pass,
                &state.t_idx,
                state.power_flux_states.as_ref().unwrap(),
                &mut flux_dft_states,
            )
        )?;
        drop(pass);
        backend.submit(encoder)?;
        Ok(())
    };

    // Warm up
    for _ in 0..WARM_UP {
        run_sim()?;
    }
    // Run bench
    let start = std::time::Instant::now();
    for _ in 0..BENCH {
        run_sim()?;
    }
    let elapsed = start.elapsed();

    readback.request_copy_t_idx(&backend, &state)?;
    readback.read_back_t_idx(&backend)?;
    let n_steps = readback.get_t_idx() - WARM_UP;
    let avg_per_step = elapsed / n_steps;
    let backend_name = backend_name(&backend);
    println!("===============2D FDTD BENCHMARK===============");
    println!("Backend: {backend_name}");
    println!("Average time per step: {avg_per_step:?}");
    println!("Number of steps steps: {n_steps}");
    println!("Steps per GPU submission (simulation speed): {SIM_SPEED}");

    Ok(())
}