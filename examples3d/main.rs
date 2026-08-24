mod bench3;

use kiss3d::prelude::*;
use std::num::NonZeroU32;
use taser_em3d::prelude::*;
use taser_em_testbed3d::{re_exports::anyhow, ColorMode, FdtdTestbedViewer, VisualizationMode};

#[kiss3d::main]
async fn main() {
    // cube().await.unwrap()
    bench3::benchmark().await.unwrap()
}

pub async fn cube() -> anyhow::Result<()> {
    // Gaussian pulse maximum frequency
    let f_max = 2.4e9; // 2.4 GHz
    let sim_speed = 3;

    // Simulation parameters w/ default stability values.
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
        // material_discretization: MaterialDiscretization::Rough,
        polarization_mode: PolarizationMode::TransverseElectric
    };
    let pml_params = PmlParameters::new(dt);
    let mut simulation = FdtdLossySimulation::new(fdtd_params, pml_params);

    // Compute cube dimensions
    // construct the cube
    let mat = ElectricMaterial {
        eps_r: Vec3::splat(4.),
        mu_r: Vec3::splat(1.),
        // sig: Vec3::splat(0.3),
        sig: Vec3::splat(0.),
    };
    let wavelen = C_0 / f_max;
    simulation.material_regions.load_trimesh_regions(
        mat,
        "assets/suzanne.obj",
        Vec3::splat(wavelen)
    )?;

    // Compute source position and gaussian curve data points
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
    let backend_name = backend_name(&backend);
    println!("Running on backend: {backend_name}");
    let mut state = simulation.finalize(&backend, &stability)?;
    let boundary_condition = BoundaryConditions::new(
        PECBoundaryX::from_backend(&backend)?,
        PECBoundaryY::from_backend(&backend)?,
        PECBoundaryZ::from_backend(&backend)?,
    );
    let mut pipeline = FdtdLossyPipeline::new_initialized(&backend, boundary_condition, sim_speed, &mut state)?;
    let mut readback = FdtdStateReadback::new(&backend, &state)?;

    // Create viewer and set up camera
    let vis_mode = VisualizationMode::default()
        .with_color_mode(
            ColorMode::FixedRange {
                v_min: 0.,
                v_max: 0.5,
                color_min: TRANSPARENT,
                color_max: RED
            }
        );
    let mut testbed = FdtdTestbedViewer::new(
        &simulation,
        &stability,
        vis_mode
    ).await?;
    testbed.window.set_ambient(0.5);

    // Render simulation
    while testbed.render_frame(readback.get_dn_field()).await {
        readback.read_back_dn(&backend)?;

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("3d fdtd example", None);
        pipeline.dispatch_steps(&mut pass, &mut state)?;
        drop(pass);
        backend.submit(encoder)?;
        readback.request_copy_dn(&backend, &state)?;
    }

    readback.request_copy_t_idx(&backend, &state)?;
    readback.read_back_t_idx(&backend)?;
    let n_steps = readback.get_t_idx();
    println!("Simulated time: {} ns", dt * n_steps as f32 * 1e9);
    println!("steps: {}", n_steps);

    Ok(())
}