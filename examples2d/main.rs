mod binary_grating;

use kiss3d::glamx::Vec3;
use std::sync::Arc;
use taser_em2d::prelude::*;
use taser_em_testbed2d::plot::*;
use taser_em_testbed2d::{ColorMode, FdtdTestbedViewer, VectorFieldVisual, VisualizationMode, re_exports::anyhow};

#[kiss3d::main]
async fn main() {
    let example = Example::BinaryGrating;
    match example {
        Example::Suzanne => suzanne_cross_section().await.unwrap(),
        Example::DipoleAntenna => dipole_antenna().await.unwrap(),
        Example::BinaryGrating => binary_grating::binary_grating().await.unwrap(),
        Example::BenchAll => benchmark_all().await.unwrap(),
    }
}

enum Example {
    Suzanne,
    DipoleAntenna,
    BinaryGrating,
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
    simulation.add_tfsf(
        Tfsf {
            spatial_axis: SpatialAxis::Y,
            direction: Direction::Negative,
            t_start: 0.0,
            polarization: Vec3::Z,
            tfsf_buffer_width: LayerWidths::splat_spatial(3),
            injection_plane_only: false,
        },
        source_values,
    );

    // Set up buffers and pipeline
    let backend = create_backend().await?;
    let backend_name = backend_name(&backend);
    println!("Running on backend: {backend_name}");
    let boundary_conditions = BoundaryConditions::new(
        PECBoundaryX::from_backend(&backend)?,
        PECBoundaryY::from_backend(&backend)?,
    );
    let pipeline = FdtdLossyPipeline::new(&backend, boundary_conditions, sim_speed)?;
    let mut state = simulation.finalize(&backend, &stability, &pipeline)?;
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
    let freq = 2.4e9; // 2.4 GHz
    let dft_resolution = 50;
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
    let _dipole = simulation.add_dipole(
        Dipole::electric(Vect::new(elem_thickness, feed_gap) / 2., Vec3::Y),
        source_values
    );

    // Instantaneous and time-averaged power flux sensors
    let flux = simulation.add_flux_sensor(
        PowerFluxSensor {
            axis: SpatialAxis::X,
            position: -(stability.spacer_region_widths[SpatialAxis::X].hi as Real * cell_size.x) * 0.90,
            direction: Direction::Negative,
        }
    );
    let flux_dft = Arc::new(PowerFluxDftSensor {
        axis: SpatialAxis::X,
        position: Vect::new(
            -(stability.spacer_region_widths[SpatialAxis::X].hi as Real * cell_size.x) * 0.90,
            0.
        ),
        size: None,
        direction: Direction::Negative,
        frequencies: frequencies_from_range((freq * 0.5)..=(freq * 1.5), dft_resolution),
    });
    let all_flux_dfts = std::slice::from_ref(&flux_dft);

    // Preparing simulation
    let backend = create_backend().await?;
    // flux DFT state
    let args = simulation.compute_state_construction_args(&stability);
    let flux_dft_state = PowerFluxDftStates::new(&backend, args, all_flux_dfts)?;
    // pipeline with boundary conditions and power flux hook
    let boundary_conditions = BoundaryConditions::new(
        PECBoundaryX::from_backend(&backend)?,
        PECBoundaryY::from_backend(&backend)?,
    );
    let pipeline = apply_pipeline_hooks(
        FdtdLossyPipeline::new(&backend, boundary_conditions, sim_speed)?,
        PowerFluxDftHook::new(&backend)?
    );
    // create simulation state and readbacks
    let mut state = simulation.finalize_with_hooks(&backend, &stability, &pipeline, flux_dft_state)?;
    let mut readback = FdtdStateReadback::new(&backend, &state, FdtdSimulationMode::TransverseElectricZ)?;
    let mut power_readback = PowerFluxReadback::new(&backend, &state)?
        .expect("we added a flux sensor to the simulation, so readback must be possible");
    let mut flux_dft_read = PowerFluxDftReadback::new(&backend, &state.hooks_states)?;

    // Source-only simulation for normalizing flux DFT
    let src_sim = simulation.sources_only();
    let args = src_sim.compute_state_construction_args(&stability);
    let flux_dft_state_src = PowerFluxDftStates::new(&backend, args, all_flux_dfts)?;
    let mut state_src = src_sim.finalize_with_hooks(&backend, &stability, &pipeline, flux_dft_state_src)?;
    let mut src_flux_dft_read = PowerFluxDftReadback::new(&backend, &state_src.hooks_states)?;

    // Running simulation
    // Set up DFT plots
    let frequency_scale = Some(1e-9); // Frequency is in GHz, so multiply by 1E-9 for a cleaner X axis;
    let mut plot_window = PlotWindow::new("Power Flux DFT", Some("Frequency (GHz)"), None);
    let mut flux_plot = ComplexPlot::new("Power Flux", &flux_dft.frequencies, frequency_scale, None)?;
    let mut src_flux_plot = ComplexPlot::new("Power Flux (Source)", &flux_dft.frequencies, frequency_scale, None)?;
    let mut normalized_plot = ComplexPlot::new("Power Flux Normalized", &flux_dft.frequencies, frequency_scale, None)?;

    // Run & render simulation with viewer
    println!("Running on backend: {}", backend_name(&backend));
    let vis_mode = VisualizationMode::default()
        .with_color_mode(ColorMode::default().to_fixed_range(0.0..0.25));
    let mut testbed = FdtdTestbedViewer::new(&simulation, &stability, vis_mode, VectorFieldVisual::H).await?;
    while testbed.render_frame(&backend, &state, &mut readback).await? {
        power_readback.read_back(&backend)?;
        power_readback.request_copy(&backend, &state)?;
        let instantaneous_flux = power_readback.get_power(&flux).unwrap();
        println!("Instantaneous power flux: {instantaneous_flux}");

        let (flux_dft_state, flux_dft_state_src) = (&state.hooks_states, &state_src.hooks_states);
        flux_dft_read.sync(&backend, flux_dft_state)?;
        src_flux_dft_read.sync(&backend, flux_dft_state_src)?;
        let new_dfts = flux_dft_read.get_flux_dfts(&flux_dft, flux_dft_state).unwrap();
        let src_dfts = src_flux_dft_read.get_flux_dfts(&flux_dft, flux_dft_state_src).unwrap();
        let mut dfts_normalized = new_dfts.clone();
        dfts_normalized
            .zip_mut_with(&src_dfts, |to_normalize, src| *to_normalize /= src);
        flux_plot.set_y(new_dfts.as_slice().unwrap());
        src_flux_plot.set_y(src_dfts.as_slice().unwrap());
        if dfts_normalized.iter().all(|dft| dft.is_finite()) {
            normalized_plot.set_y(dfts_normalized.as_slice().unwrap());
        }
        plot_window.show(
            &mut testbed,
            vec![flux_plot.create_line(), src_flux_plot.create_line(), normalized_plot.create_line()]
        )?;

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("2D dipole antenna", None);
        pipeline.dispatch_steps(&mut pass, &mut state)?;
        pipeline.dispatch_steps(&mut pass, &mut state_src)?;
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
    let _dipole = simulation.add_dipole(dipole, source_values);

    // Preparing simulation
    // Create instance of backend we will run the simulation with
    let backend = create_backend().await?;

    // Create pipelines and boundary conditions
    let boundary_conditions = BoundaryConditions::new(
        PECBoundaryX::from_backend(&backend)?,
        PECBoundaryY::from_backend(&backend)?,
    );
    let pipeline = FdtdLossyPipeline::new(&backend, boundary_conditions, SIM_SPEED)?;

    // Create GPU simulation state
    let mut state = simulation.finalize(&backend, &stability, &pipeline)?;

    // TODO: Set up power flux DFT for source and main sim
    let _frequencies = frequencies_from_range(0.0..=(freq * 2.), dft_resolution);

    // Set up readback
    let mut readback = FdtdStateReadback::new(&backend, &state, FdtdSimulationMode::TransverseElectricZ)?;
    let mut power_readback = PowerFluxReadback::new(&backend, &state)?
        .expect("we added a flux sensor to the simulation, so readback must be possible");

    // Run simulation
    println!("Running on backend: {}", backend_name(&backend));
    let mut run_sim = || -> TaserResult<()> {
        // Dummy readbacks (synchronizes backend too)
        power_readback.read_back(&backend)?;
        power_readback.request_copy(&backend, &state)?;

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("2d benchmark example", None);
        pipeline.dispatch_steps(&mut pass, &mut state)?;
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