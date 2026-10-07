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
    let mut pipeline = FdtdLossyPipeline::new(&backend, boundary_conditions, sim_speed)?;
    let mut state = simulation.finalize(&backend, &stability, &mut pipeline)?;
    let mut readback = FdtdStateReadback::new(&backend, &state, FdtdSimulationMode::TransverseMagneticZ)?;

    // Create viewer and set up camera
    let vis_mode = VisualizationMode::default();
    let mut testbed = FdtdTestbedViewer::new(&simulation, &stability, vis_mode, VectorFieldVisual::H).await?;
    testbed.window.set_ambient(0.5);

    // Render simulation
    while testbed.render_frame(&backend, &state, &mut readback).await? {
        pipeline.simulate(&backend, &mut state)?;
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
    let dft_resolution = 200;
    let sim_speed = 2;

    // Simulation parameters w/ 30 cells-per-wavelength
    let stability = FdtdStability::from_cpw(30);
    let cell_size = stability.cell_size_from_min_wavelength(freq);
    let dt = stability.cfl_condition(cell_size);
    let parameters = FdtdParameters {
        cell_size,
        dt,
        material_discretization: MaterialDiscretization::smooth_from_stability(&stability)
    };
    let mut simulation = FdtdLossySimulation::new(parameters, PmlParameters::new(dt));

    // Construct dipole antenna
    let antenna_len = C_0 / (freq * 2.);
    let elem_thickness = cell_size.y;
    let feed_gap = cell_size.y * 2.;
    let half_len = antenna_len / 2.0;
    simulation
        .fill_region(Vect::ZERO, Vect::new(elem_thickness, -half_len), ElectricMaterial::PEC)
        .fill_region(
            Vect::new(0., feed_gap),
            Vect::new(0., feed_gap) + Vect::new(elem_thickness, half_len),
            ElectricMaterial::PEC
        );

    // Source injection in antenna feed gap
    let source_values = Source::gaussian_max_f(freq, 1., dt);
    let dipole = Dipole::electric(Vect::new(elem_thickness, feed_gap) / 2., Vec3::Y);
    let dipole = simulation.add_dipole(dipole, source_values);

    // Power flux monitor for reading back power flux
    let flux_monitor = simulation.add_flux_monitor(
        PowerFluxMonitor {
            axis: SpatialAxis::X,
            position: -(stability.spacer_region_widths[SpatialAxis::X].hi as Real * cell_size.x) * 0.90,
            direction: Direction::Negative,
        }
    );

    // Preparing simulation
    // Create instance of backend we will run the simulation with
    let backend = create_backend().await?;

    // Pipeline with boundary conditions
    let boundary_conditions = BoundaryConditions::new(
        PECBoundaryX::from_backend(&backend)?,
        PECBoundaryY::from_backend(&backend)?,
    );
    let mut pipeline = FdtdLossyPipeline::new(&backend, boundary_conditions, sim_speed)?;

    // Create GPU simulation state
    let mut state = simulation.finalize(&backend, &stability, &mut pipeline)?;

    // Set up power flux and source DFTs
    let frequencies = frequencies_from_range(0.0..=(freq * 2.), dft_resolution);
    let (flux_dft, flux_func) = PowerFluxFunction::new(&flux_monitor, state.power_flux_states.as_ref().unwrap())?
        .to_dft(frequencies.clone())?;
    let flux_dft_hook = DftHook::new(&backend, [flux_dft], &state)?;
    let (src_dft, src_func) = SourceFunction::new(&dipole, &state.source_states)?
        .to_dft(frequencies)?;
    let src_dft_hook = DftHook::new(&backend, [src_dft], &state)?;

    // Set up readback
    let mut readback = FdtdStateReadback::new(&backend, &state, FdtdSimulationMode::TransverseElectricZ)?;
    let mut power_readback = PowerFluxReadback::new(&backend, &state)?
        .expect("we added a flux monitor to the simulation, so readback must be possible");
    let mut flux_dft_read = DftReadback::new(&backend, &flux_dft_hook).await?;
    let mut src_dft_read = DftReadback::new(&backend, &src_dft_hook).await?;

    // Apply DFT hooks to pipeline to update DFTs with the main simulation
    let mut pipeline = apply_pipeline_hooks(pipeline, (flux_dft_hook, src_dft_hook));

    // Running simulation
    // Set up DFT plots
    let frequency_scale = Some(1e-9); // Frequency is in GHz, so multiply by 1E-9 for a cleaner X axis;
    let mut plot_window = PlotWindow::new("Power Flux DFT", Some("Frequency (GHz)"), None);
    let mut flux_plot = DftPlotLine::new("Power Flux", &flux_func, &flux_dft_read, frequency_scale)?
        .with_color(Color32::RED);
    let mut src_plot = DftPlotLine::new("Source", &src_func, &src_dft_read, frequency_scale)?
        .with_color(Color32::GREEN);

    // Run & render simulation with viewer
    println!("Running on backend: {}", backend_name(&backend));
    let vis_mode = VisualizationMode::default()
        .with_color_mode(ColorMode::default().to_fixed_range(0.0..0.25));
    let mut testbed = FdtdTestbedViewer::new(&simulation, &stability, vis_mode, VectorFieldVisual::H).await?;
    while testbed.render_frame(&backend, &state, &mut readback).await? {
        power_readback.read_back(&backend)?;
        power_readback.request_copy(&backend, &state)?;
        let instantaneous_flux = power_readback.get_power(&flux_monitor).unwrap();
        println!("Instantaneous power flux: {instantaneous_flux}");

        let (flux_dft, src_dft) = &pipeline.hooks;
        src_plot.update_dft_and_points(&backend, src_dft.states(), &mut src_dft_read)?;
        flux_plot.update_dft_and_points(&backend, flux_dft.states(), &mut flux_dft_read)?;
        plot_window.show(&mut testbed, vec![src_plot.create_line(), flux_plot.create_line()])?;

        pipeline.simulate(&backend, &mut state)?;
    }

    readback.request_copy_t_idx(&backend, &state)?;
    readback.read_back_t_idx(&backend)?;
    let n_steps = readback.get_t_idx();
    println!("simulated time: {:?} ns", n_steps as Real * dt * 1e9);
    println!("steps: {:?}", n_steps);

    Ok(())
}

pub async fn benchmark_all() -> anyhow::Result<()> {
    const WARM_UP: u32 = 100;
    const BENCH: u32 = 3000;
    const SIM_SPEED: usize = 1;

    // Gaussian pulse maximum frequency
    let freq = 2.4e9; // 2.4 GHz
    let dft_resolution = 100;

    // Simulation parameters w/ 30 cells-per-wavelength
    let stability = FdtdStability::from_cpw(30);
    let cell_size = stability.cell_size_from_min_wavelength(freq);
    let dt = stability.cfl_condition(cell_size);
    let parameters = FdtdParameters {
        cell_size,
        dt,
        material_discretization: MaterialDiscretization::smooth_from_stability(&stability)
    };
    let mut simulation = FdtdLossySimulation::new(parameters, PmlParameters::new(dt));

    // Construct dipole antenna
    let antenna_len = C_0 / (freq * 2.);
    let elem_thickness = cell_size.y;
    let feed_gap = cell_size.y * 2.;
    let half_len = antenna_len / 2.0;
    simulation
        .fill_region(Vect::ZERO, Vect::new(elem_thickness, -half_len), ElectricMaterial::PEC)
        .fill_region(
            Vect::new(0., feed_gap),
            Vect::new(0., feed_gap) + Vect::new(elem_thickness, half_len),
            ElectricMaterial::PEC
        );

    // Source injection in antenna feed gap
    let source_values = Source::sin_cycle(freq, dt).repeat(10);
    let dipole = Dipole::electric(Vect::new(elem_thickness, feed_gap) / 2., Vec3::Y);
    let dipole = simulation.add_dipole(dipole, source_values);

    // Power flux monitor for reading back power flux
    let flux_monitor = simulation.add_flux_monitor(
        PowerFluxMonitor {
            axis: SpatialAxis::X,
            position: -(stability.spacer_region_widths[SpatialAxis::X].hi as Real * cell_size.x) * 0.90,
            direction: Direction::Negative,
        }
    );

    // Preparing simulation
    // Create instance of backend we will run the simulation with
    let backend = create_backend().await?;

    // Create pipelines and boundary conditions
    let boundary_conditions = BoundaryConditions::new(
        PECBoundaryX::from_backend(&backend)?,
        PECBoundaryY::from_backend(&backend)?,
    );
    let mut pipeline = FdtdLossyPipeline::new(&backend, boundary_conditions, SIM_SPEED)?;

    // Create GPU simulation state
    let mut state = simulation.finalize(&backend, &stability, &mut pipeline)?;

    // Set up power flux & source DFT
    let frequencies = frequencies_from_range(0.0..=(freq * 2.), dft_resolution);
    let (dft, _flux_func) = PowerFluxFunction::new(&flux_monitor, state.power_flux_states.as_ref().unwrap())?
        .to_dft(frequencies.clone())?;
    let flux_dft_hook = DftHook::new(&backend, [dft], &state)?;
    let (dft, _src_func) = SourceFunction::new(&dipole, &state.source_states)?
        .to_dft(frequencies)?;
    let src_dft_hook = DftHook::new(&backend, [dft], &state)?;

    // Set up readback
    let mut readback = FdtdStateReadback::new(&backend, &state, FdtdSimulationMode::TransverseElectricZ)?;
    let mut power_readback = PowerFluxReadback::new(&backend, &state)?
        .expect("we added a flux monitor to the simulation, so readback must be possible");
    let mut flux_dft_read = DftReadback::new(&backend, &flux_dft_hook).await?;
    let mut src_dft_read = DftReadback::new(&backend, &src_dft_hook).await?;

    // Apply DFT hooks
    let mut pipeline = apply_pipeline_hooks(pipeline, (flux_dft_hook, src_dft_hook));

    // Run simulation
    println!("Running on backend: {}", backend_name(&backend));
    let mut run_sim = || -> TaserResult<()> {
        // Dummy readbacks (synchronizes backend too)
        let (flux_dft, src_dft) = &pipeline.hooks;
        power_readback.read_back(&backend)?;
        power_readback.request_copy(&backend, &state)?;
        flux_dft_read.try_read_back(&backend);
        flux_dft_read.request_copy(&backend, flux_dft.states())?;
        src_dft_read.try_read_back(&backend);
        src_dft_read.request_copy(&backend, src_dft.states())?;

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("2d benchmark example", None);
        pipeline.dispatch_steps_aux(&mut pass, &mut state)?;
        drop(pass);
        backend.submit(encoder)?;
        Ok(())
    };

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
    let n_steps = readback.get_t_idx() - WARM_UP * SIM_SPEED as u32;
    let avg_per_step = elapsed / n_steps;
    let backend_name = backend_name(&backend);
    println!("===============2D FDTD BENCHMARK===============");
    println!("Backend: {backend_name}");
    println!("Average time per step: {avg_per_step:?}");
    println!("Number of steps: {n_steps}");
    println!("Steps per GPU submission (simulation speed): {SIM_SPEED}");

    Ok(())
}