/// Attempt to generate a mesh from an arbitrary shape
#[cfg(feature = "render")]
pub fn generate_mesh(shape: &dyn parry3d::shape::Shape) -> Option<(Vec<glamx::Vec3>, Vec<[u32; 3]>)> {
    if let Some(ball) = shape.as_ball() {
        Some(ball.to_trimesh(8, 8))
    }
    else if let Some(cuboid) = shape.as_cuboid() {
        Some(cuboid.to_trimesh())
    }
    else if let Some(caps) = shape.as_capsule() {
        Some(caps.to_trimesh(8, 8))
    }
    else if let Some(mesh) = shape.as_trimesh() {
        Some((mesh.vertices().to_vec(), mesh.indices().to_vec()))
    }
    else { None }
}