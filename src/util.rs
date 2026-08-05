#[macro_export]
#[doc(hidden)]
macro_rules! to_parallel {
    ($iter:expr) => {
        cfg_select! {
            feature = "rayon" => ParallelBridge::par_bridge($iter),
            _ => $iter,
        }
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! into_par_iter {
    ($coll:expr) => {
        cfg_select! {
            feature = "rayon" => $coll.into_par_iter(),
            _ => $coll.into_iter()
        }
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! par_iter_mut {
    ($coll:expr) => {
        cfg_select! {
            feature = "rayon" => $coll.par_iter_mut(),
            _ => $coll.iter_mut()
        }
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! par_iter {
    ($coll:expr) => {
        cfg_select! {
            feature = "rayon" => $coll.par_iter(),
            _ => $coll.iter()
        }
    };
}

/// Attempt to generate a mesh from an arbitrary shape
#[cfg(feature = "render")]
pub fn generate_mesh(shape: &dyn parry3d::shape::Shape) -> Option<(Vec<glamx::Vec3>, Vec<[u32; 3]>)> {
    shape.as_ball()
        .map(|ball| ball.to_trimesh(8, 8))
        .or_else(|| shape.as_cuboid()
            .map(|cuboid| cuboid.to_trimesh())
        )
        .or_else(|| shape.as_capsule()
            .map(|caps| caps.to_trimesh(8, 8))
        )
        .or_else(|| shape.as_trimesh()
            .map(|mesh| (mesh.vertices().to_vec(), mesh.indices().to_vec()))
        )
}