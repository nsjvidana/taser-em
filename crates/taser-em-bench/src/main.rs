use taser_em2d::prelude::*;
use taser_em2d::re_exports::anyhow;
use taser_em2d::re_exports::khal::backend::{Backend, Encoder};
use taser_em2d::re_exports::khal::Shader;

const WARM_UP: u32 = 10;
const BENCH: u32 = 2000;

fn main() -> anyhow::Result<()> {
    pollster::block_on(bench())
}

async fn bench() -> anyhow::Result<()> {
    let stability = FdtdStability::default();
    let sim_params = FdtdParameters {
        cell_size: Vect::ONE,
        dt: 1.,
        polarization_mode: PolarizationMode::TransverseElectric,
        material_discretization: MaterialDiscretization::Rough
    };
    let pml_params = PmlParameters::new(1.);
    let sim = FdtdLossySimulation::new(sim_params, pml_params);

    let backend = taser_em2d::prelude::create_backend().await?;

    let start = std::time::Instant::now();
    let mut gpu_data = sim.finalize(&backend, &stability)?;
    let startup_dur = start.elapsed();

    let boundary_condition = PECBoundary::from_backend(&backend)?;
    let mut pipeline = FdtdLossyPipeline::new(&backend, boundary_condition, 1)?;
    for _ in 0..WARM_UP {
        backend.synchronize()?;
        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("2d fdtd bench", None);
        pipeline.dispatch_steps(&mut pass, &mut gpu_data)?;
        drop(pass);
        backend.submit(encoder)?;
    }

    let start = std::time::Instant::now();
    for _ in 0..BENCH {
        backend.synchronize()?;
        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("2d fdtd bench", None);
        pipeline.dispatch_steps(&mut pass, &mut gpu_data)?;
        drop(pass);
        backend.submit(encoder)?;
    }
    let dur = start.elapsed();

    println!("==========2D FDTD Benchmark==========");
    let backend_name = backend_name(&backend);
    println!("Running on backend: {backend_name}");
    println!("Rayon: {}", cfg!(feature = "rayon"));
    println!("Num cells: {}", gpu_data.n_cells.n_cells_to_3d().element_product());
    println!("Num trials: {BENCH}");
    println!("Startup time: {:?}", startup_dur);
    println!("Average time per step: {:?}", dur / BENCH);
    Ok(())
}