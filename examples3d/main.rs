use kiss3d::prelude::*;
use std::num::NonZeroU32;
use taser_em3d::prelude::*;
use taser_em_testbed3d::{re_exports::anyhow, ColorMode, FdtdTestbedViewer, VisualizationMode};

#[kiss3d::main]
async fn main() {
    cube().await.unwrap()
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
        eps_r: Vec3::splat(2.),
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
    let boundary_condition = PECBoundary::from_backend(&backend)?;
    let mut pipeline = FdtdLossyPipeline::new_initialized(&backend, boundary_condition, sim_speed, &mut state)?;

    // Create viewer and set up camera
    let vis_mode = VisualizationMode::default()
        .with_color_mode(
            ColorMode::FixedRange {
                v_min: 0.,
                v_max: 0.4,
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
    let mut dn_field = vec![Vec4::ZERO; state.dn.buffer.len()];
    while testbed.render_frame(&dn_field).await {
        state.dn.read(&backend, &mut dn_field).await?;
        backend.synchronize()?;

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("3d fdtd example", None);
        pipeline.dispatch_steps(&mut pass, &mut state)?;
        drop(pass);
        state.dn.encode_copy_cmd(&mut encoder)?;
        state.t_idx.encode_copy_cmd(&mut encoder)?;
        backend.submit(encoder)?;
    }
    let mut steps = vec![0];
    state.t_idx.read(&backend, &mut steps).await?;
    println!("Simulated time: {} ns", dt * steps[0] as f32 * 1e9);
    println!("steps: {}", steps[0]);

    Ok(())
}