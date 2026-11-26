use std::num::NonZeroU32;
use taser_em1d::prelude::*;
use taser_em_testbed1d::re_exports::anyhow;
use taser_em_testbed1d::{FdtdTestbedViewer, VisualizationMode};

#[kiss3d::main]
async fn main() {
    suzanne().await.unwrap()
}

pub async fn suzanne() -> anyhow::Result<()> {
    // Gaussian pulse maximum frequency
    let freq = 2.4e9; // 2.4 GHz
    let sim_speed = 1;

    // Simulation parameters
    let cpw = 20;
    let stability = FdtdStability {
        spacer_region_widths: LayerWidths::splat_spatial(cpw),
        ..FdtdStability::from_cpw(cpw)
    };
    let cell_size = stability.cell_size_from_min_wavelength(freq);
    let dt = stability.cfl_condition(cell_size)
        .min(stability.dt_from_gaussian_freq(freq));
    let parameters = FdtdParameters {
        cell_size,
        dt,
        material_discretization: MaterialDiscretization::Smooth {
            resolution: NonZeroU32::new(3).unwrap()
        },
        // material_discretization: MaterialDiscretization::Rough,
    };
    let mut simulation = FdtdLossySimulation::new(parameters, PmlParameters::new(dt));

    // Construct the slab
    let mat = ElectricMaterial {
        eps_r: Vec3::splat(7.),
        ..ElectricMaterial::FREE_SPACE
    };
    simulation.material_regions.load_trimesh_regions(
        mat,
        "assets/suzanne.obj",
        Vec3::splat(C_0 / freq)
    )?;

    // Compute TF/SF source with gaussian curve data points
    let tfsf = Tfsf {
        spatial_axis: SpatialAxis::Z,
        direction: Direction::Negative,
        t_start: 0.,
        polarization: Vec3::Y, // Ey/Hx mode
        tfsf_buffer_width: LayerWidths::splat_spatial(4),
        injection_plane_only: false,
    };
    simulation.add_tfsf(tfsf, Source::gaussian_max_f(freq, 1., dt));

    // Set up buffers and pipeline
    let backend = create_backend().await?;
    let pipeline = FdtdLossyPipeline::new(
        &backend,
        BoundaryConditions::new(PECBoundaryZ::from_backend(&backend)?),
        sim_speed
    )?;
    let mut state = simulation.finalize(&backend, &stability, &pipeline)?;

    // Create viewer and set up camera
    let mut viewer = FdtdTestbedViewer::new(
        &backend,
        &state,
        FdtdSimulationMode::EyHx,
        VisualizationMode::default(),
        VectorField::En
    ).await?;

    // Render simulation
    let backend_name = backend_name(&backend);
    println!("Running on backend: {backend_name}");
    while viewer.render_frame().await? {
        viewer.sync(&backend, &state)?;
        pipeline.simulate(&backend, &mut state)?;
    }

    // Print duration of simulation
    viewer.readback.request_copy_t_idx(&backend, &state)?;
    viewer.readback.read_back_t_idx(&backend)?;
    let n_steps = viewer.readback.get_t_idx();
    println!("simulated time: {:?} ns", n_steps as Real * dt * 1e9);
    println!("steps: {:?}", n_steps);

    Ok(())
}