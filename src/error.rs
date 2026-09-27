use khal::backend::GpuBackendError;
use crate::dft::DftError;
use crate::mesh_loading::{MeshConverterError, MeshLoaderError};
use crate::monitor::PowerFluxError;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    MeshLoader(#[from] MeshLoaderError),
    #[error(transparent)]
    MeshConversion(#[from] MeshConverterError),
    #[error(transparent)]
    GpuBackend(#[from] GpuBackendError),
    #[error(transparent)]
    Dft(#[from] DftError),
    #[error(transparent)]
    PowerFlux(#[from] PowerFluxError),
}