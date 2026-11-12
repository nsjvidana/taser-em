use kiss3d::prelude::*;
use taser_em3d::prelude::*;
use taser_em_testbed3d::{re_exports::anyhow, ColorMode, FdtdTestbedViewer, VectorFieldVisual, VisualizationMode};

/// Yagi-Uda antenna example.
///
/// Measurements were taken from [Wikipedia](https://commons.wikimedia.org/wiki/File:Yagi_3_element.svg)
pub async fn yagi_uda_antenna() -> anyhow::Result<()> {
    // Gaussian pulse maximum frequency
    let freq = 2.4e9; // 2.4 GHz
    let sim_speed = 3;

    // Simulation parameters w/ default stability values.
    let stability = FdtdStability {
        cells_per_wavelength: 20,
        spacer_region_widths: LayerWidths::splat_spatial(20),
        ..Default::default()
    };
    let cell_size = stability.cell_size_from_min_wavelength(freq);
    let dt = stability.cfl_condition(cell_size);
    let parameters = FdtdParameters {
        cell_size,
        dt,
        material_discretization: MaterialDiscretization::Rough,
    };
    let mut simulation = FdtdLossySimulation::new(parameters, PmlParameters::new(dt));

    // Construct antenna
    let feed_gap = cell_size.y * 3.;
    let elem_thickness = cell_size.y;
    let wavelen = C_0 / freq;
    let driven_len = 0.44 * wavelen;
    let driven_half = driven_len / 2.0;
    let feed_half = feed_gap / 2.;
    let feed_center = Vect::Y * cell_size;
    let pec = ElectricMaterial::PEC;
    let start = feed_center + Vect::Y * feed_half;
    let start2 = feed_center - Vect::Y * feed_half;
    simulation
        .fill_region(
            start,
            start + Vect::new(elem_thickness, driven_half, elem_thickness),
            pec
        )
        .fill_region(
            start2,
            start2 + Vect::new(elem_thickness, -driven_half, elem_thickness),
            pec
        );
    // reflector
    let refl_len = 0.5 * wavelen;
    let refl_half = refl_len / 2.;
    let feed_refl_gap = 0.25 * wavelen;
    let start = feed_center + Vect::new(-feed_refl_gap, refl_half, 0.);
    simulation
        .fill_region(
            start,
            start + Vect::new(elem_thickness, -refl_len, elem_thickness),
            pec
        );
    // director
    let director_len = 0.43 * wavelen;
    let feed_director_gap = 0.13 * wavelen;
    let director_half = director_len / 2.;
    let start = feed_center + Vect::new(feed_director_gap, director_half, 0.);
    simulation
        .fill_region(
            start,
            start + Vect::new(elem_thickness, -director_len, elem_thickness),
            pec
        );

    // Source injection in antenna feed gap
    let dipole = Dipole {
        dipole_type: DipoleType::Electric,
        position: feed_center,
        t_start: 0.0,
        moment: Vec3::Y,
    };
    simulation
        .add_dipole(dipole, Source::sin_cycle(freq, dt).repeat(10));

    // Set up buffers and pipeline
    let backend = create_backend().await?;
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
        .with_color_mode(ColorMode::default().to_fixed_range(0.0..0.025));
    let mut testbed = FdtdTestbedViewer::new(&simulation, &stability, vis_mode, VectorFieldVisual::H).await?;
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