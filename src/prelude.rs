use khal::backend::GpuBackendError;

pub type GpuResult<T> = core::result::Result<T, GpuBackendError>;