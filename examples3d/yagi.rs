use kiss3d::prelude::*;
use taser_em3d::prelude::*;
use taser_em3d::re_exports::khal::re_exports::wgpu;
use taser_em_testbed3d::{re_exports::anyhow, ColorMode, FdtdTestbedViewer, VectorFieldVisual, VisualizationMode};

/// Yagi-Uda antenna example.
///
/// Measurements were calculated with [changpuak](https://www.changpuak.ch/)
pub async fn yagi_uda_antenna() -> anyhow::Result<()> {
    // Gaussian pulse maximum frequency
    let freq = 145e6; // 145 MHz
    let sim_speed = 3;

    // Simulation parameters w/ default stability values.
    let stability = FdtdStability {
        cells_per_wavelength: 60,
        spacer_region_widths: LayerWidths::splat_spatial(60/4)
            .with_axis_widths(SpatialAxis::X, LoHiWidths::splat(60/4).with_hi(60)),
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
    // Yagi-Uda antenna design by www.changpuak.ch
    let elem_thickness = cell_size.min_element();
    println!("elem_thickness: {}", elem_thickness);
    let pec = ElectricMaterial::PEC;
    // reflector
    let refl_len = 997e-3;
    simulation.fill_region(
        Vect::ZERO.with_y(refl_len/2.),
        Vect::new(elem_thickness, -refl_len/2., elem_thickness),
        pec
    );
    // dipole
    let dipole_half_len = 490.4e-3;
    let dipole_pos = 497e-3;
    let feed_gap = (dipole_half_len) / 5.;
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
    let dir_positions = [652e-3, 1024e-3, 1469e-3];
    let dir_lens = [940e-3, 931e-3, 923e-3];
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
        position: feed_center,
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

    // Create viewer and set up camera
    let vis_mode = VisualizationMode::default()
        .with_color_mode(ColorMode::default().to_fixed_range(0.0..0.005));
    let mut testbed = FdtdTestbedViewer::new(&simulation, &stability, vis_mode, VectorFieldVisual::En).await?;
    testbed.set_pml_alpha(0.);

    // Render simulation
    println!("Running on backend: {}", backend_name(&backend));
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