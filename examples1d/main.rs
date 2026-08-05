use std::num::{NonZeroU32, NonZeroUsize};
use taser_em1d::prelude::*;
use taser_em1d::re_exports::glamx::glam::*;
use taser_em1d::re_exports::khal;
use taser_em1d::re_exports::khal::backend::{Backend, Buffer, GpuBackend};
use taser_em_testbed1d::{FdtdTestbedViewer, VisualizationMode};
use taser_em_testbed1d::re_exports::anyhow;

#[kiss3d::main]
async fn main() {
    single_slab().await.unwrap()
}

pub async fn single_slab() -> anyhow::Result<()> {
    // Gaussian pulse maximum frequency
    let f_max = 2.4e9; // 2.4 GHz
    let sim_speed = NonZeroUsize::new(12).unwrap();

    // Simulation parameters w/ default stability values.
    let stability = FdtdStability {
        dt_safety_factor: 10.,
        cells_per_wavelength: 30,
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

    // Compute slab dimensions
    let wavelen = C_0 / f_max;
    let slab_extents = [Vect::from_element(-20.), Vect::from_element(-20. + wavelen * 2.)];
    // construct the slab
    let mat = ElectricMaterial {
        eps_r: Vec3::splat(7.),
        mu_r: Vec3::splat(1.),
        // sig: Vec3::splat(0.3),
        sig: Vec3::splat(0.),
    };
    simulation.material_regions.fill_region(slab_extents[0], slab_extents[1], mat);

    // Compute source position and gaussian curve data points
    let source = Source::Dipole {
        position: slab_extents[0] - wavelen * 3.,
        t_start: 0.,
        vals: Source::gaussian_max_f(f_max, 1., dt)
    };
    simulation.add_source(source);

    // Set up buffers and pipeline
    let (backend, backend_name) = cfg_select! {
        feature = "webgpu" => {{
            let webgpu = khal::backend::WebGpu::default().await?;
            (GpuBackend::WebGpu(webgpu), "WebGPU")
        }}
        feature = "metal" => {todo!()}
        feature = "cpu" => { (GpuBackend::Cpu, "CPU") }
        feature = "cuda" => {todo!()}
    };
    println!("Running on backend: {backend_name}");
    let mut gpu_data = simulation.finalize(&backend, &stability)?;
    let pipeline = FdtdLossyPipeline::new(&backend, sim_speed)?;

    // Create viewer and set up camera
    let n_cells = gpu_data.n_cells;
    let grid_extents = n_cells.as_vect() * cell_size;
    let mut testbed = FdtdTestbedViewer::new(&simulation, &stability, VisualizationMode::default()).await?;
    testbed.set_clipping_planes(cell_size / 3., cell_size * 1000.)
        .camera
        .look_at(
            Vec3::new(-grid_extents, 0., grid_extents / 2.),
            Vec3::Z * grid_extents / 2.
        );

    // Render simulation
    let mut dn_field = vec![Vec4::ZERO; gpu_data.dn.buffer.len()];
    while testbed.render_frame(&dn_field).await {
        backend.synchronize()?;
        gpu_data.dn.read(&backend, &mut dn_field).await?;
        pipeline.submit_steps(&backend, &mut gpu_data, None, |encoder, gpu_data| {
            gpu_data.dn.encode_copy_cmd(encoder)
        })?;
    }
    Ok(())
}