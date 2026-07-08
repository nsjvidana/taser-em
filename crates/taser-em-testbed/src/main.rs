use khal::backend::{Backend, Encoder, GpuBackend, WebGpu};
use khal::Shader;
use kiss3d::glamx::Vec4;
use taser_em1d::gpu_util::CreateGpuBuffer;
use taser_em1d::shaders::fdtd::{GridParameters2, IntegrationTerms, PmlCoefficients2};
use taser_em1d::FdtdWithLoss;

#[kiss3d::main]
async fn main() {
    let webgpu = WebGpu::default().await.unwrap();
    let backend = GpuBackend::WebGpu(webgpu);

    let n_cells = 100;
    let mut h = vec![Vec4::default(); n_cells].create_gpu_buffer(&backend).unwrap();
    let mut dn = vec![Vec4::default(); n_cells].create_gpu_buffer(&backend).unwrap();
    let mut en = vec![Vec4::default(); n_cells].create_gpu_buffer(&backend).unwrap();
    let mut int_terms = vec![IntegrationTerms::default(); n_cells].create_gpu_buffer(&backend).unwrap();
    let grid_coeffs = vec![PmlCoefficients2::default(); n_cells].create_gpu_buffer(&backend).unwrap();
    let grid = GridParameters2::default().create_gpu_uniform(&backend).unwrap();
    let fdtd = FdtdWithLoss::from_backend(&backend).unwrap();
    let mut total = std::time::Duration::default();
    let num_trials = 1000;
    for _ in 0..num_trials {
        let mut encoder = backend.begin_encoding();
        {
            let mut pass = encoder.begin_pass("fdtd", None);
            for _ in 0..100 {
                fdtd.call(
                    &mut pass,
                    n_cells,
                    &mut h,
                    &mut dn,
                    &mut en,
                    &mut int_terms,
                    &grid_coeffs,
                    &grid,
                ).unwrap();
            }
        }
        let start = std::time::Instant::now();
        backend.submit(encoder).unwrap();
        backend.synchronize().unwrap();
        total += start.elapsed();
    }
    println!("Average execution time: {:?}", total / num_trials);
}