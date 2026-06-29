use khal::backend::{Backend, GpuBackend, WebGpu};
use khal::Shader;
use kiss3d::prelude::*;
use taser_em1d::shaders::fdtd1::GridParameters;
use taser_em1d::{Fdtd1, Fdtd1Runner};

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("taser FDTD test bed").await;
    let mut camera = OrbitCamera3d::default();
    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(0.0, 2.0, -2.0));

    let webgpu = WebGpu::default().await.unwrap();
    let backend = GpuBackend::WebGpu(webgpu);
    let kernel = Fdtd1::from_backend(&backend).unwrap();
    let mut runner = Fdtd1Runner::new(&backend, 100, GridParameters { dz: 1. }).unwrap();

    let mut total = std::time::Duration::default();
    let mut count = 0;
    let max_count = 100;
    while window.render_3d(&mut scene, &mut camera).await {
        let _ = runner.submit(&backend, &kernel, None).unwrap();
        let start = std::time::Instant::now();
        backend.synchronize().unwrap();
        let elapsed = start.elapsed();
        total += elapsed;
        count += 1;
        if count >= max_count {
            println!("{}ms", total.as_secs_f64() * 1000. / max_count as f64);
            total = std::time::Duration::default();
            count = 0;
        }
    }
}