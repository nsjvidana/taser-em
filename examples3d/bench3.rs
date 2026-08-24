use std::num::NonZeroU32;
use taser_em3d::prelude::*;
use taser_em3d::re_exports::anyhow;

const WARMUP: usize = 10;
const BENCH: usize = 1000;

pub async fn benchmark() -> anyhow::Result<()> {
    let f_max = 2.4e9; // 2.4 GHz
    let sim_speed = 1;

    // Params
    let stability = FdtdStability {
        dt_safety_factor: 16.,
        cells_per_wavelength: 10,
        material_resolution: NonZeroU32::new(3).unwrap(),
        spacer_region_widths: LayerWidths::splat_spatial(10)
            .with_axis_widths(SpatialAxis::X, LoHiWidths::splat(5))
            .with_axis_widths(SpatialAxis::Y, LoHiWidths::splat(5)),
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
        polarization_mode: PolarizationMode::TransverseElectric
    };
    let pml_params = PmlParameters::new(dt);
    let mut simulation = FdtdLossySimulation::new(fdtd_params, pml_params);

    // Device
    let mat = ElectricMaterial {
        eps_r: Vec3::splat(4.),
        mu_r: Vec3::splat(1.),
        // sig: Vec3::splat(0.3),
        sig: Vec3::splat(0.),
    };
    let wavelen = C_0 / f_max;
    let box_min = Vect::splat(-20.);
    let box_max = box_min + Vect::splat(wavelen);
    simulation.material_regions.fill_region(box_min, box_max, mat);

    // Source
    let source = Source::TFSF {
        spatial_axis: SpatialAxis::Z,
        direction: WaveDirection::Positive,
        t_start: 0.0,
        vals: Source::gaussian_max_f(f_max, 1., dt),
        polarization: Vec3::new(1., 1., 0.).normalize(),
        tfsf_buffer_width: LayerWidths::splat_spatial(3),
    };
    simulation.add_source(source);

    // Set up buffers and pipeline
    let backend = create_backend().await?;
    let mut state = simulation.finalize(&backend, &stability)?;
    let boundary_condition = PECBoundary::from_backend(&backend)?;
    let mut pipeline = FdtdLossyPipeline::new_initialized(&backend, boundary_condition, sim_speed, &mut state)?;
    let mut readback = FdtdStateReadback::new(&backend, &state)?;

    macro_rules! get_n_steps {
        () => {{
            readback.request_copy_t_idx(&backend, &state)?;
            readback.read_back_t_idx(&backend)?;
            readback.get_t_idx()
        }};
    }

    // Run simulation
    for _ in 0..WARMUP {
        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("3d fdtd bench", None);
        pipeline.dispatch_steps(&mut pass, &mut state)?;
        drop(pass);
        backend.submit(encoder)?;
        backend.synchronize()?;
    }

    let n_steps_warmup = get_n_steps!();

    let start = std::time::Instant::now();
    for _ in 0..BENCH {
        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("3d fdtd bench", None);
        pipeline.dispatch_steps(&mut pass, &mut state)?;
        drop(pass);
        backend.submit(encoder)?;
        backend.synchronize()?;
    }
    let elapsed = start.elapsed();

    let n_steps = get_n_steps!() - n_steps_warmup;

    let avg_per_step = elapsed / n_steps;
    let backend_name = backend_name(&backend);
    println!("===============3D FDTD BENCHMARK===============");
    println!("Backend: {backend_name}");
    println!("Average time per step: {avg_per_step:?}");
    println!("Number of steps steps: {n_steps}");
    println!("Steps per GPU submission (simulation speed): {sim_speed}");
    println!("Number of cells: {}", state.n_cells.element_product());

    Ok(())
}