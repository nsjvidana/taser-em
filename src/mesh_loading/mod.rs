//! This entire module directory consists of code copied
//! and adapted from [rapier](https://github.com/dimforge/rapier/)
//! Commit: ccd01cd294e84945da55d61548002a5b6f73df31 (2026-08-16)
//! License: Apache-2.0
//!
//! See LICENSE_rapier file in this directory for rapier's copyright notice.
//!
//! Modifications:
//! - Removed all imports of rapier.
//! - Bundled MeshConverter into same module directory as the copied/adapted rapier3d_meshloader code.

pub mod mesh_converter;

use std::path::Path;
pub use mesh_loader;
pub use mesh_converter::*;

use mesh_loader::{Material, Mesh};
use parry3d::math::{Pose, Vector};
use parry3d::shape::SharedShape;

/// The result of loading a shape.
pub struct LoadedShape {
    /// The shape loaded from the file and converted by the [`MeshConverter`].
    pub shape: SharedShape,
    /// The shape's pose.
    pub pose: Pose,
    /// The raw mesh read from the file without any modification.
    pub raw_mesh: Mesh,
    /// The material declared in the source asset for this mesh. For an
    /// OBJ this is the `.mtl` material referenced by the mesh's
    /// `usemtl` group (loaded by `mesh-loader` when the `.mtl` sits
    /// alongside the `.obj`). STL files don't carry materials, so this
    /// is the default-constructed `Material`.
    pub material: Material,
}

/// Error while loading an STL file.
#[derive(thiserror::Error, Debug)]
pub enum MeshLoaderError {
    /// An error triggered by rapier’s [`MeshConverter`].
    #[error(transparent)]
    MeshConverter(#[from] MeshConverterError),
    /// A generic IO error.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Loads parry shapes from a file.
///
/// # Parameters
/// - `path`: the file’s path.
/// - `converter`: controls how the shapes are computed from the content. In particular, it lets
///   you specify if the computed [`SharedShape`] is a triangle mesh, its convex hull,
///   bounding box, etc.
/// - `scale`: the scaling factor applied to the geometry input to the `converter`. This scale will
///   affect at the geometric level the [`LoadedShape::shape`]. Note that raw mesh value stored
///   in [`LoadedShape::raw_mesh`] remains unscaled.
pub fn load_from_path(
    path: impl AsRef<Path>,
    converter: &MeshConverter,
    scale: Vector,
) -> Result<Vec<Result<LoadedShape, MeshConverterError>>, MeshLoaderError> {
    let loader = mesh_loader::Loader::default();
    let mut colliders = vec![];
    let scene = loader.load(path)?;
    // mesh-loader's OBJ backend aligns `scene.materials[i]` with
    // `scene.meshes[i]` (it expands per-mesh from the material_index),
    // so zipping is correct here. STL produces empty materials; pair
    // each mesh with a default material in that case.
    let mut materials = scene.materials.into_iter();
    for raw_mesh in scene.meshes.into_iter() {
        let material = materials.next().unwrap_or_default();
        let shape = load_from_raw_mesh(&raw_mesh, converter, scale);

        colliders.push(shape.map(|(shape, pose)| LoadedShape {
            shape,
            pose,
            raw_mesh,
            material,
        }));
    }
    Ok(colliders)
}

/// Loads an file as a shape from a preloaded raw [`mesh_loader::Mesh`].
///
/// # Parameters
/// - `raw_mesh`: the raw mesh.
/// - `converter`: controls how the shape is computed from the STL content. In particular, it lets
///   you specify if the computed [`SharedShape`] is a triangle mesh, its convex hull,
///   bounding box, etc.
/// - `scale`: the scaling factor applied to the geometry input to the `converter`. This scale will
///   affect at the geometric level the [`LoadedShape::shape`]. Note that raw mesh value stored
///   in [`LoadedShape::raw_mesh`] remains unscaled.
pub fn load_from_raw_mesh(
    raw_mesh: &Mesh,
    converter: &MeshConverter,
    scale: Vector,
) -> Result<(SharedShape, Pose), MeshConverterError> {
    let vertices: Vec<_> = raw_mesh
        .vertices
        .iter()
        .map(|xyz| Vector::new(xyz[0], xyz[1], xyz[2]) * scale)
        .collect();
    let indices: Vec<_> = raw_mesh.faces.clone();
    converter.convert(vertices, indices)
}