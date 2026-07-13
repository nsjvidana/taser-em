use std::num::NonZeroU32;
use glamx::Vec3;
use khal::backend::{Backend, Buffer, Encoder, GpuBackend, WebGpu};
use taser_em1d::{ElectricMaterial, FdtdSolver, FdtdStability, Source, C_0};
use taser_em1d::grid::{LayerWidths, MaterialRegions, PolarizationMode, YeeGrid};
use taser_em1d::prelude::GpuResult;
use taser_em1d::shaders::math::{Vect, VectExt};

const WARMUP_ITERS: usize = 10;
const BENCH_ITERS: usize = 1000;

#[kiss3d::main]
async fn main() {
    let webgpu = WebGpu::default().await.unwrap();
    let backend = GpuBackend::WebGpu(webgpu);

    let avg_time = benchmark(&backend).await.unwrap();

    println!("FDTD Benchmark (1D)");
    println!("------------------------------");
    println!("Average execution time: {}ms", avg_time * 1000.);
    println!("Warmup iterations {}", WARMUP_ITERS);
    println!("Total iterations {}", BENCH_ITERS);
}

async fn benchmark(backend: &GpuBackend) -> GpuResult<f32> {
    let stability = FdtdStability::default();
    let f_max = 2.4e9; // 2.4 GHz
    let cell_size = stability.cell_size_from_min_wavelength(f_max);
    let dt = stability.cfl_condition(cell_size);
    let wavelen = C_0 / f_max;

    let slab_extents = [Vect::splat(1.), Vect::splat(1. + wavelen * 2.)];
    let source = Source::Dipole {
        position: 1. - wavelen,
        t_start: 0.,
        vals: Source::gaussian_max_f(f_max, 1., dt)
    };

    let mut mat_regions = MaterialRegions::new();
    let mat = ElectricMaterial {
        eps_r: Vec3::splat(7.),
        mu_r: Vec3::splat(1.),
        sig: Vec3::splat(1e-15),
    };
    mat_regions.fill_region(slab_extents[0], slab_extents[1], mat);
    let grid = YeeGrid::new(
        cell_size,
        PolarizationMode::TransverseMagnetic,
        mat_regions,
        NonZeroU32::new(3).unwrap(),
        LayerWidths::splat(10)
    );
    let mut solver = FdtdSolver::new(
        backend,
        grid,
        dt,
    )?;
    solver.add_source(source);

    // Warmup
    let mut buffers = solver.compute_and_create_buffers(backend)?;
    for _ in 0..WARMUP_ITERS {
        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("fdtd warmup", None);
        solver.submit_step(&mut buffers, &mut pass)?;
        drop(pass);
        backend.submit(encoder)?;
        backend.synchronize()?;
    }

    // Timed iters
    let start = std::time::Instant::now();
    for _ in 0..BENCH_ITERS {
        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("fdtd bench", None);
        solver.submit_step(&mut buffers, &mut pass)?;
        drop(pass);
        backend.submit(encoder)?;
        backend.synchronize()?;
    }
    let time = start.elapsed().as_secs_f32() / BENCH_ITERS as f32;
    let mut out = vec![glamx::Vec4::ZERO; buffers.h.buffer.len()];
    backend.slow_read_buffer(&buffers.h.buffer, &mut out).await?;
    println!("{:?}", out);
    Ok(time)
}