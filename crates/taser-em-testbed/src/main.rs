use khal::backend::{Backend, GpuBackend, WebGpu};
use khal::Shader;
use kiss3d::prelude::*;
use taser_em1d::{AddAssign, AddAssignRunner};

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("taser FDTD test bed").await;
    let mut camera = OrbitCamera3d::default();
    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(0.0, 2.0, -2.0));

    let mut a = (0..10000).map(|i| i as f32).collect::<Vec<_>>();
    let b = (0..10000).map(|i| i as f32 * 10.0).collect::<Vec<_>>();

    let webgpu = WebGpu::default().await.unwrap();
    let backend = GpuBackend::WebGpu(webgpu);
    let add_assign = AddAssign::from_backend(&backend).unwrap()
        .add_assign;
    let mut runner = AddAssignRunner::new(a, b, &backend).unwrap();

    while window.render_3d(&mut scene, &mut camera).await {
        let timestamp = runner.submit(&add_assign, &backend).unwrap();
        backend.synchronize().unwrap();
        let t = timestamp.read(&backend).await.unwrap()[0].clone();
        runner.buffers.a.read(&backend, &mut runner.a)
            .await.unwrap();
        println!("{}: {}ms", t.label, t.duration_ms);
    }
}