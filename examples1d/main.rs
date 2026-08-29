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
    let sim_speed = 10;

    // Simulation parameters
    let stability = FdtdStability {
        dt_safety_factor: 18.,
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
    simulation.material_regions.load_trimesh_regions(
        mat,
        "assets/suzanne.obj",
        Vec3::ONE
    )?;

    // Compute source position and gaussian curve data points
    simulation.add_source(Source::TFSF {
        spatial_axis: SpatialAxis::Z,
        direction: WaveDirection::Negative,
        t_start: 0.,
        vals: Source::gaussian_max_f(f_max, 1., dt),
        polarization: Vec3::Y, // Ey/Hx mode
        tfsf_buffer_width: LayerWidths::splat_spatial(3),
    });

    // Set up buffers and pipeline
    let backend = create_backend().await?;
    let backend_name = backend_name(&backend);
    println!("Running on backend: {backend_name}");
    let boundary_condition = BoundaryConditions::new(
        // PECBoundaryZ::from_backend(&backend)?,
        PeriodicBoundaryZ::from_backend(&backend)?,
    );
    simulation.pml_parameters.widths = simulation.pml_parameters.widths
        .with_axis_widths(SpatialAxis::Z, LoHiWidths::splat(0));
    let mut state = simulation.finalize(&backend, &stability)?;
    let mut pipeline = FdtdLossyPipeline::new_initialized(&backend, boundary_condition, sim_speed, &mut state)?;
    let mut readback = FdtdStateReadback::new(&backend, &state, FdtdSimulationMode::EyHx)?;

    // Create viewer and set up camera
    let mut testbed = FdtdTestbedViewer::new(&simulation, &stability, VisualizationMode::default()).await?;

    // Render simulation
    while testbed.render_frame(readback.get_dn_field()).await {
        readback.read_back_dn(&backend)?;

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("1d fdtd example", None);
        pipeline.dispatch_steps(&mut pass, &mut state)?;
        drop(pass);
        backend.submit(encoder)?;
        readback.request_copy_dn(&backend, &state)?;
    }

    readback.request_copy_t_idx(&backend, &state)?;
    readback.read_back_t_idx(&backend)?;
    let n_steps = readback.get_t_idx();
    println!("simulated time: {:?} ns", n_steps as Real * dt * 1e9);
    println!("steps: {:?}", n_steps);

    Ok(())
}