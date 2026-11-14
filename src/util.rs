use khal::backend::GpuBackend;
use taser_em_shaders::math::*;

/// Does rayon `into_par_iter` if `rayon` feature is enabled, `into_iter` otherwise.
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

/// Does rayon `par_iter_mut` if `rayon` feature is enabled, `iter_mut` otherwise.
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

/// Utility function for picking a backend and constructing it, depending on which features
/// are enabled.
///
/// For to specify `wgpu` features and limits, use [`create_backend_custom`]. However, `webgpu` feature
/// must be enabled for to customize features and limits.
///
/// # Warning
/// Only use this function if exactly **one** backend feature is selected, otherwise
/// you might not get the backend you actually want.
pub async fn create_backend() -> anyhow::Result<GpuBackend> {
    create_backend_custom(
        #[cfg(feature = "webgpu")]
        None,
        #[cfg(feature = "webgpu")]
        None,
    ).await
}

pub async fn create_backend_custom(
    #[cfg(feature = "webgpu")]
    features: Option<khal::re_exports::wgpu::Features>,
    #[cfg(feature = "webgpu")]
    limits: Option<khal::re_exports::wgpu::Limits>,
) -> anyhow::Result<GpuBackend> {
    let backend = cfg_select! {
        feature = "cpu" => GpuBackend::Cpu,
        feature = "metal" => GpuBackend::Metal(khal::backend::Metal::new()?),
        feature = "cuda" => GpuBackend::Cuda(khal::backend::Cuda::new(0)?),
        feature = "webgpu" => {{
            GpuBackend::WebGpu(khal::backend::WebGpu::new(
                features.unwrap_or_default(),
                limits.unwrap_or_default()
            ).await?)
        }},
    };
    Ok(backend)
}

/// Get the name of the [`GpuBackend`] that `backend` is.
pub fn backend_name(backend: &GpuBackend) -> &'static str {
    match backend {
        #[cfg(feature = "webgpu")]
        GpuBackend::WebGpu(..) => "WebGPU",
        #[cfg(feature = "cuda")]
        GpuBackend::Cuda(..) => "CUDA",
        #[cfg(feature = "metal")]
        GpuBackend::Metal(..) => "Metal",
        #[cfg(feature = "cpu")]
        GpuBackend::Cpu => "CPU",
        _ => "UNKNOWN",
    }
}

/// Constructs an iterator of all cell positions in a grid of dimensions `n_cells`.
/// The cell positions are given as tuples but AREN'T IN ORDER
pub fn grid_cells_iter(n_cells: GridIndex) -> impl Iterator<Item = GridIndex> {
    let iter = cfg_select! {
        feature = "dim1" => itertools::iproduct!(0..n_cells),
        feature = "dim2" => itertools::iproduct!(0..n_cells[SpatialAxis::X], 0..n_cells[SpatialAxis::Y]),
        feature = "dim3" => itertools::iproduct!(0..n_cells[SpatialAxis::X], 0..n_cells[SpatialAxis::Y], 0..n_cells[SpatialAxis::Z]),
    };
    iter.map(|tuple| GridIndex::from_index_array(tuple.into()))
}