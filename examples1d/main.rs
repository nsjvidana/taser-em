use std::num::NonZeroU32;
use taser_em1d::prelude::*;
use taser_em_testbed1d::{FdtdTestbedViewer, VisualizationMode};
use taser_em_testbed1d::re_exports::anyhow;

#[kiss3d::main]
async fn main() {
    single_slab().await.unwrap()
}

pub async fn single_slab() -> anyhow::Result<()> {
    // Gaussian pulse maximum frequency
    let f_max = 2.4e9; // 2.4 GHz
    let sim_speed = 5;

    // Simulation parameters
    let stability = FdtdStability {
        dt_safety_factor: 10.,
        cells_per_wavelength: 10,
        spacer_region_widths: LayerWidths::splat_spatial(20),
        ..Default::default()
    };
    let cell_size = stability.cell_size_from_min_wavelength(f_max);
    let dt = stability.cfl_condition(cell_size)
        .min(stability.dt_from_gaussian_freq(f_max));
    let parameters = FdtdParameters {
        cell_size,
        dt,
        material_discretization: MaterialDiscretization::Smooth {
            resolution: NonZeroU32::new(3).unwrap()
        },
        // material_discretization: MaterialDiscretization::Rough,
        polarization_mode: PolarizationMode::TransverseMagnetic
    };
    let mut simulation = FdtdLossySimulation::new(parameters, PmlParameters::new(dt));

    // Construct the slab
    let mat = ElectricMaterial {
        eps_r: Vec3::splat(7.),
        mu_r: Vec3::splat(1.),
        // sig: Vec3::splat(0.3),
        sig: Vec3::splat(0.),
    };
    simulation.material_regions.load_path_as_region(
        mat,
        "assets/suzanne.obj",
        &MeshConverter::ConvexDecomposition,
        Vec3::ONE
    )?;

    // Compute source position and gaussian curve data points
    simulation.add_source(Source::TFSF {
        spatial_axis: SpatialAxis::Z,
        direction: WaveDirection::Negative,
        t_start: 0.,
        vals: Source::gaussian_max_f(f_max, 1., dt),
        polarization: Vec3::Y,
        tfsf_buffer_width: LayerWidths::splat_spatial(3),
    });

    // Set up buffers and pipeline
    let backend = create_backend().await?;
    let backend_name = backend_name(&backend);
    println!("Running on backend: {backend_name}");
    let mut state = simulation.finalize(&backend, &stability)?;
    let boundary_condition = PECBoundary::from_backend(&backend)?;
    let mut pipeline = FdtdLossyPipeline::new_initialized(&backend, boundary_condition, sim_speed, &mut state)?;

    // Create viewer and set up camera
    let mut testbed = FdtdTestbedViewer::new(&simulation, &stability, VisualizationMode::default()).await?;

    // Render simulation
    let mut dn_field = vec![Vec4::ZERO; state.dn.buffer.len()];
    while testbed.render_frame(&dn_field).await {
        backend.synchronize()?;
        state.dn.read(&backend, &mut dn_field).await?;

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("1d fdtd example", None);
        pipeline.dispatch_steps(&mut pass, &mut state)?;
        drop(pass);
        state.dn.encode_copy_cmd(&mut encoder)?;
        backend.submit(encoder)?;
    }
    Ok(())
}