use crate::prelude::*;
use khal::backend::{Backend, DeviceValue, GpuBackend, GpuBuffer};
use khal::re_exports::bytemuck::NoUninit;
use khal::BufferUsages;

pub type GpuResult<T> = core::result::Result<T, GpuBackendError>;

pub trait CreateGpuBuffer<T: DeviceValue + NoUninit> {
    /// Create a [`GpuBuffer`] storage buffer with [`BufferUsages::COPY_SRC`] usage.
    fn create_gpu_buffer(&self, backend: &GpuBackend) -> GpuResult<GpuBuffer<T>>;

    #[allow(unused_variables)]
    fn create_gpu_uniform(&self, backend: &GpuBackend) -> GpuResult<GpuBuffer<T>> {
        panic!("Unimplemented for this type or cannot be a uniform!")
    }
}

impl<T: DeviceValue + NoUninit> CreateGpuBuffer<T> for Vec<T> {
    fn create_gpu_buffer(&self, backend: &GpuBackend) -> GpuResult<GpuBuffer<T>> {
        backend.init_buffer(
            self.as_slice(),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )
    }
}

impl<T: DeviceValue + NoUninit> CreateGpuBuffer<T> for T {
    fn create_gpu_buffer(&self, backend: &GpuBackend) -> GpuResult<GpuBuffer<T>> {
        backend.init_buffer(
            &[*self],
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )
    }
    fn create_gpu_uniform(&self, backend: &GpuBackend) -> GpuResult<GpuBuffer<T>> {
        backend.init_buffer(
            &[*self],
            BufferUsages::UNIFORM,
        )
    }
}