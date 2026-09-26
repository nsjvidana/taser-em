use crate::mesh_loading::{MeshConverterError, MeshLoaderError};
use glamx::Vec3;
use khal::backend::GpuBackendError;
use taser_em_shaders::math::{Real, SpatialAxis};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    MeshLoader(#[from] MeshLoaderError),
    #[error(transparent)]
    MeshConversion(#[from] MeshConverterError),
    #[error(transparent)]
    GpuBackend(#[from] GpuBackendError),
    #[error("Power flux monitor outside simulation grid. Monitor Axis: {axis:?}, Position: {position:?}")]
    OutOfBoundsFluxMonitor {
        axis: SpatialAxis,
        position: Real,
    },
}