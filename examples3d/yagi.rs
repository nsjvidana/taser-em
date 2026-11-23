use kiss3d::prelude::*;
use taser_em3d::prelude::*;
use taser_em3d::re_exports::khal::re_exports::wgpu;
use taser_em_testbed3d::{re_exports::anyhow, ColorMode, FdtdTestbedViewer, VectorFieldVisual, VisualizationMode};

/// Yagi-Uda antenna example.
///
/// Measurements were calculated with [changpuak](https://www.changpuak.ch/)
pub async fn yagi_uda_antenna() -> anyhow::Result<()> {
    // Gaussian pulse maximum frequency
    let freq = 2.4e9; // 2.4 GHz
    let sim_speed = 2;

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
    // **** YAGI UDA ANTENNA ****
    // Design by www.changpuak.ch
    // ---------------------------------------------------
    // Frequency  :  2400  MHz
    // Wavelength :  125  mm
    // d/lambda   :  0.04    ( min.: 0.001 , max.: 0.04 )
    // D/lambda   :  0.002    ( min.: 0.002 , max.: 0.04 )
    // Boomlength :  100  mm
    // Elements   :  5
    // Gain       :  8.8  dB    (approx.)
    // ---------------------------------------------------
    // Reflector Length [mm] : 59
    // Reflector Position [mm] : 0
    // ---------------------------------
    // Dipole Length [mm] : 60
    // Dipole Position [mm] : 25
    // ---------------------------------
    // Director Length [mm] : 49
    // Director Position [mm] : 50
    // ---------------------------------
    // Director Length [mm] : 49
    // Director Position [mm] : 75
    // ---------------------------------
    // Director Length [mm] : 49
    // Director Position [mm] : 100
    // ---------------------------------
    // Calculations based on NBS TECHNICAL NOTE 688
    // Length might be slightly too long.
    // Manufacturing Tolerances : < 0  mm
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

    // Create viewer and set up camera
    let vis_mode = VisualizationMode::default()
        .with_color_mode(ColorMode::default().to_fixed_range(0.0..0.020))
        .with_alpha(AlphaMode::Mask(0.1), 1.);
    let mut testbed = FdtdTestbedViewer::new(&simulation, &stability, vis_mode, VectorFieldVisual::H).await?;
    testbed.set_pml_alpha(0.);
    testbed.visualization_mode.slice_visual(SpatialAxis::Z, elem_thickness / 2.);

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