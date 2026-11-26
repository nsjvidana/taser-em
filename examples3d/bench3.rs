use taser_em3d::prelude::*;
use taser_em3d::re_exports::anyhow;
use taser_em3d::re_exports::khal::re_exports::wgpu;

const WARMUP: usize = 10;
const BENCH: usize = 200;

/// Benchmark of [`yagi_uda_antenna`](super::yagi::yagi_uda_antenna) example
pub async fn benchmark() -> anyhow::Result<()> {
    // Gaussian pulse maximum frequency
    let freq = 2.4e9; // 2.4 GHz
    let sim_speed = 1;

    // Simulation parameters w/ default stability values.
    let cpw = 120;
    let stability = FdtdStability {
        cells_per_wavelength: cpw,
        spacer_region_widths: LayerWidths::splat_spatial(cpw/4),
        ..Default::default()
    };
    let cell_size = stability.cell_size_from_min_wavelength(freq);
    let dt = stability.cfl_condition(cell_size)
        .min(stability.dt_from_gaussian_freq(freq));
    let parameters = FdtdParameters {
        cell_size,
        dt,
        // material_discretization: MaterialDiscretization::smooth_from_stability(&stability),
        material_discretization: MaterialDiscretization::Rough,
    };
    let mut simulation = FdtdLossySimulation::new(parameters, PmlParameters::new(dt));

    // Construct antenna
    let elem_thickness = 5e-3;
    let pec = ElectricMaterial::PEC;
    // reflector
    let refl_len = 59e-3;
    simulation.fill_region(
        Vect::ZERO.with_y(refl_len/2.),
        Vect::new(elem_thickness, -refl_len/2., elem_thickness),
        pec
    );
    // dipole
    let dipole_half_len = 60e-3 / 2.;
    let dipole_pos = 25e-3;
    let feed_gap = 0.6e-3 * 3.;
    let feed_center = Vect::new(dipole_pos, 0., 0.);
    let start = feed_center + Vect::new(0., feed_gap / 2. + dipole_half_len, 0.);
    simulation.fill_region(
        start,
        start + Vect::new(elem_thickness, -dipole_half_len, elem_thickness),
        pec
    );
    let start = feed_center + Vect::new(0., -(feed_gap / 2. + dipole_half_len), 0.);
    simulation.fill_region(
        start,
        start + Vect::new(elem_thickness, dipole_half_len, elem_thickness),
        pec
    );
    // directors
    let dir_lens = [49e-3, 49e-3, 49e-3];
    let dir_positions = [50e-3, 75e-3, 100e-3];
    for (pos, dir_len) in dir_positions.into_iter().zip(dir_lens) {
        let start = Vect::new(pos, dir_len/2., 0.);
        simulation.fill_region(
            start,
            start + Vect::new(elem_thickness, -dir_len, elem_thickness),
            pec
        );
    }

    // Source injection in antenna feed gap
    let dipole = Dipole {
        dipole_type: DipoleType::Electric,
        position: feed_center + Vec3::new(1., 0., 1.) * (elem_thickness / 2.),
        t_start: 0.0,
        moment: Vec3::Y,
    };
    simulation.add_dipole(dipole, Source::sin_cycle(freq, dt).repeat(20));

    // Set up buffers and pipeline
    let max_buffer_size = 512 << 20;
    let backend = create_backend_custom( // request larger buffer limits for large simulation
        None,
        Some(wgpu::Limits {
            max_buffer_size,
            max_storage_buffer_binding_size: max_buffer_size,
            ..Default::default()
        })
    ).await?;
    let boundary_conditions = BoundaryConditions::new(
        PECBoundaryX::from_backend(&backend)?,
        PECBoundaryY::from_backend(&backend)?,
        PECBoundaryZ::from_backend(&backend)?,
    );
    let pipeline = FdtdLossyPipeline::new(&backend, boundary_conditions, sim_speed)?;
    let mut state = simulation.finalize(&backend, &stability, &pipeline)?;
    let mut readback = FdtdStateReadback::new(&backend, &state)?;

    // Run simulation
    for _ in 0..WARMUP {
        pipeline.simulate(&backend, &mut state)?;
        backend.synchronize()?;
    }

    readback.request_copy_t_idx(&backend, &state)?;
    readback.read_back_t_idx(&backend)?;
    let n_steps_warmup = readback.get_t_idx();

    let start = std::time::Instant::now();
    for _ in 0..BENCH {
        pipeline.simulate(&backend, &mut state)?;
        backend.synchronize()?;
    }
    let elapsed = start.elapsed();

    readback.request_copy_t_idx(&backend, &state)?;
    readback.read_back_t_idx(&backend)?;
    let n_steps = readback.get_t_idx() - n_steps_warmup;

    let avg_per_step = elapsed / n_steps;
    let backend_name = backend_name(&backend);
    println!("===============3D FDTD BENCHMARK===============");
    println!("Backend: {backend_name}");
    println!("Average time per step: {avg_per_step:?}");
    println!("Number of steps: {n_steps}");
    println!("Steps per GPU submission (simulation speed): {sim_speed}");
    println!("Number of cells: {}", state.grid_sizing.n_cells.element_product());

    Ok(())
}