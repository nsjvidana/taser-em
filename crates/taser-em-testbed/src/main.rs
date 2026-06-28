use khal::backend::{Backend, GpuBackend, WebGpu};
use khal::Shader;
use kiss3d::prelude::*;
use taser_em1d::{AddAssign, AddAssignRunner, Fdtd1, Fdtd1Runner};
use taser_em1d::shaders::fdtd1::GridParameters;

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
    let mut runner = Fdtd1Runner::new(100, GridParameters { dz: 1. }, &backend).unwrap();

    while window.render_3d(&mut scene, &mut camera).await {
        let _ = runner.submit(&kernel, None, &backend).unwrap();
        let start = std::time::Instant::now();
        backend.synchronize().unwrap();
        let elapsed = start.elapsed();
        runner.buffers.en_y.read(&backend, &mut runner.en_y)
            .await.unwrap();
        println!("{:?}", elapsed);
    }
}