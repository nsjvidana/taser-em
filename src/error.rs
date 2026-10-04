use khal::backend::GpuBackendError;
use crate::prelude::*;

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
    #[error(transparent)]
    SourceError(#[from] SourceError)
}