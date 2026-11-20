mod bench3;
mod yagi;

use kiss3d::prelude::*;
use std::num::NonZeroU32;
use taser_em3d::prelude::*;
use taser_em3d::re_exports::khal::re_exports::wgpu;
use taser_em_testbed3d::{re_exports::anyhow, ColorMode, FdtdTestbedViewer, VectorFieldVisual, VisualizationMode};

#[kiss3d::main]
async fn main() {
    const EXAMPLE: Examples = Examples::Yagi;
    match EXAMPLE {
        Examples::SuzanneTheScatterer => mesh_scatterer().await.unwrap(),
        Examples::Dipole => dipole_antenna().await.unwrap(),
        Examples::Yagi => yagi::yagi_uda_antenna().await.unwrap(),
        Examples::Bench => bench3::benchmark().await.unwrap(),
    }
}

enum Examples {
    SuzanneTheScatterer,
    Dipole,
    Yagi,
    Bench
}

pub async fn mesh_scatterer() -> anyhow::Result<()> {
    // Gaussian pulse maximum frequency
    let f_max = 2.4e9; // 2.4 GHz
    let sim_speed = 2;

    // Simulation parameters w/ default stability values.
    let stability = FdtdStability {
        cells_per_wavelength: 10,
        material_resolution: NonZeroU32::new(2).unwrap(),
        spacer_region_widths: LayerWidths::splat_spatial(10)
            .with_axis_widths(SpatialAxis::Z, LoHiWidths::splat(10).with_lo(20)),
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
    };
    let mut simulation = FdtdLossySimulation::new(fdtd_params, PmlParameters::new(dt));

    // Construct scatterer from mesh
    let mat = ElectricMaterial {
        eps_r: Vec3::splat(4.),
        mu_r: Vec3::splat(1.),
        sig: Vec3::splat(0.),
    };
    let wavelen = C_0 / f_max;
    simulation.material_regions.load_trimesh_regions(
        mat,
        "assets/suzanne.obj",
        Vec3::splat(wavelen)
    )?;

    // Compute source position and gaussian curve data points
    let source = Tfsf {
        spatial_axis: SpatialAxis::Z,
        direction: Direction::Positive,
        t_start: 0.0,
        polarization: Vec3::X,
        tfsf_buffer_width: LayerWidths::splat_spatial(8)
            .with_axis_widths(SpatialAxis::Z, LoHiWidths::splat(8).with_hi(4)),
        injection_plane_only: false,
    };
    simulation.add_tfsf(source, Source::gaussian_max_f(f_max, 1., dt));

    // Set up buffers and pipeline
    let backend = create_backend().await?;
    let boundary_condition = BoundaryConditions::new(
        PECBoundaryX::from_backend(&backend)?,
        PECBoundaryY::from_backend(&backend)?,
        PECBoundaryZ::from_backend(&backend)?,
    );
    let pipeline = FdtdLossyPipeline::new(&backend, boundary_condition, sim_speed)?;
    let mut state = simulation.finalize(&backend, &stability, &pipeline)?;
    let mut readback = FdtdStateReadback::new(&backend, &state)?;

    // Create viewer and set up camera
    let vis_mode = VisualizationMode::default();
    let mut testbed = FdtdTestbedViewer::new(
        &simulation,
        &stability,
        vis_mode,
        VectorFieldVisual::H
    ).await?;
    testbed.set_pml_alpha(0.);

    // Render simulation
    println!("Running on backend: {}", backend_name(&backend));
    while testbed.render_frame(&backend, &state, &mut readback).await? {
        pipeline.simulate(&backend, &mut state)?;
    }

    readback.request_copy_t_idx(&backend, &state)?;
    readback.read_back_t_idx(&backend)?;
    let n_steps = readback.get_t_idx();
    println!("Simulated time: {} ns", dt * n_steps as f32 * 1e9);
    println!("steps: {}", n_steps);

    Ok(())
}

pub async fn dipole_antenna() -> anyhow::Result<()> {
    // Gaussian pulse maximum frequency
    let freq = 2.4e9; // 2.4 GHz
    let sim_speed = 3;

    // Simulation parameters w/ default stability values.
    let cpw = 120;
    let stability = FdtdStability {
        cells_per_wavelength: cpw,
        spacer_region_widths: LayerWidths::splat_spatial(cpw/4),
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

    // Construct dipole antenna
    // Measurements are from https://www.changpuak.ch/electronics/Dipole_straight.php
    // Frequency = 2400MHz, Diameter D = 7mm.
    let elem_thickness = 5e-3; // diameter
    let dipole_half_len = 28.8e-3;
    let pec = ElectricMaterial::PEC;
    let feed_gap = 0.6e-3 * 2.;
    let feed_center = Vect::new(0., 0., 0.);
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
        .with_color_mode(ColorMode::default().to_fixed_range(0.0..0.030));
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