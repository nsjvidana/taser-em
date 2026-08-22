use khal::backend::GpuBackendError;
use crate::mesh_loading::{MeshConverterError, MeshLoaderError};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    MeshLoader(#[from] MeshLoaderError),
    #[error(transparent)]
    MeshConversion(#[from] MeshConverterError),
    #[error(transparent)]
    GpuBackend(#[from] GpuBackendError)
}