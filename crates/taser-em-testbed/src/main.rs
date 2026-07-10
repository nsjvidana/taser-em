use khal::backend::{Backend, Encoder, GpuBackend, WebGpu};
use khal::Shader;
use kiss3d::glamx::Vec4;
use taser_em1d::gpu_util::CreateGpuBuffer;
use taser_em1d::shaders::fdtd::{GridParameters2, IntegrationTerms, PmlCoefficients2};
use taser_em1d::FdtdWithLoss;
use taser_em1d::prelude::GpuResult;

const WARMUP_ITERS: usize = 10;
const BENCH_ITERS: usize = 1000;

#[kiss3d::main]
async fn main() {
    let webgpu = WebGpu::default().await.unwrap();
    let backend = GpuBackend::WebGpu(webgpu);

    let avg_time = benchmark(&backend).unwrap();

    println!("FDTD Benchmark (1D)");
    println!("------------------------------");
    println!("Average execution time: {}ms", avg_time * 1000.);
    println!("Warmup iterations {}", WARMUP_ITERS);
    println!("Total iterations {}", BENCH_ITERS);
}

fn benchmark(backend: &GpuBackend) -> GpuResult<f32> {
    let n_cells = 100;
    let mut h = vec![Vec4::default(); n_cells].create_gpu_buffer(backend)?;
    let mut dn = vec![Vec4::default(); n_cells].create_gpu_buffer(backend)?;
    let mut en = vec![Vec4::default(); n_cells].create_gpu_buffer(backend)?;
    let mut int_terms = vec![IntegrationTerms::default(); n_cells].create_gpu_buffer(backend)?;
    let grid_coeffs = vec![PmlCoefficients2::default(); n_cells].create_gpu_buffer(backend)?;
    let grid = GridParameters2::default().create_gpu_uniform(backend)?;
    let fdtd = FdtdWithLoss::from_backend(backend)?;

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("fdtd warmup", None);
        fdtd.call(
            &mut pass,
            n_cells,
            &mut h,
            &mut dn,
            &mut en,
            &mut int_terms,
            &grid_coeffs,
            &grid,
        )?;
        drop(pass);
        backend.submit(encoder)?;
        backend.synchronize()?;
    }

    // Timed iters
    let start = std::time::Instant::now();
    for _ in 0..BENCH_ITERS {
        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("fdtd", None);
        fdtd.call(
            &mut pass,
            n_cells,
            &mut h,
            &mut dn,
            &mut en,
            &mut int_terms,
            &grid_coeffs,
            &grid,
        )?;
        drop(pass);
        backend.submit(encoder)?;
        backend.synchronize()?;
    }
    Ok(start.elapsed().as_secs_f32() / BENCH_ITERS as f32)
}