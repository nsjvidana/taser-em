pub use taser_em_shaders::math::*;
pub use crate::*;
pub use crate::grid::*;

use khal::backend::GpuBackendError;

pub type GpuResult<T> = core::result::Result<T, GpuBackendError>;

/// Utility function for picking a backend and constructing it, depending on which features
/// are enabled.
///
/// # Warning
/// Only use this function if exactly **one** backend feature is selected, otherwise
/// you might not get the backend you actually want.
pub async fn create_backend() -> anyhow::Result<GpuBackend> {
    let backend = cfg_select! {
        feature = "metal" => GpuBackend::Metal(khal::backend::Metal::new()?),
        feature = "cpu" => GpuBackend::Cpu,
        feature = "cuda" => GpuBackend::Cuda(khal::backend::Cuda::new(0)?),
        feature = "webgpu" => GpuBackend::WebGpu(khal::backend::WebGpu::default().await?),
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