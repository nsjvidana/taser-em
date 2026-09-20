use khal::backend::GpuBackendError;
use crate::mesh_loading::{MeshConverterError, MeshLoaderError};
use crate::monitor::PowerFluxMonitor;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    MeshLoader(#[from] MeshLoaderError),
    #[error(transparent)]
    MeshConversion(#[from] MeshConverterError),
    #[error(transparent)]
    GpuBackend(#[from] GpuBackendError),
    #[error("Power flux monitor outside simulation grid: {0:?}")]
    OutOfBoundsFluxMonitor(PowerFluxMonitor),
}