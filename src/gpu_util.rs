use crate::prelude::*;
use khal::backend::{Backend, Buffer, DeviceValue, Encoder, GpuBackend, GpuBuffer, GpuEncoder};
use khal::re_exports::bytemuck::{AnyBitPattern, NoUninit};
use khal::BufferUsages;

pub struct GpuBufferReadable<T: DeviceValue + NoUninit + AnyBitPattern> {
    pub buffer: GpuBuffer<T>,
    pub readback: GpuBuffer<T>,
}

impl<T: DeviceValue + NoUninit + AnyBitPattern> GpuBufferReadable<T> {
    pub fn encode_copy_cmd(&mut self, encoder: &mut GpuEncoder) -> GpuResult<()> {
        encoder.copy_buffer_to_buffer(
            &self.buffer,
            0,
            &mut self.readback,
            0,
            self.buffer.len()
        )
    }

    pub async fn read(&self, backend: &GpuBackend, out: &mut [T]) -> GpuResult<()> {
        backend.read_buffer(&self.readback, out).await
    }
}

pub trait CreateGpuBuffer<T: DeviceValue + NoUninit> {
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
            BufferUsages::STORAGE,
        )
    }
}

impl<T: DeviceValue + NoUninit> CreateGpuBuffer<T> for T {
    fn create_gpu_buffer(&self, backend: &GpuBackend) -> GpuResult<GpuBuffer<T>> {
        backend.init_buffer(
            &[*self],
            BufferUsages::STORAGE,
        )
    }
    fn create_gpu_uniform(&self, backend: &GpuBackend) -> GpuResult<GpuBuffer<T>> {
        backend.init_buffer(
            &[*self],
            BufferUsages::UNIFORM,
        )
    }
}

pub trait CreateGpuBufferReadable<T: DeviceValue + NoUninit + AnyBitPattern> {
    fn create_gpu_buffer_readable(&self, backend: &GpuBackend) -> GpuResult<GpuBufferReadable<T>>;
}

impl<T: DeviceValue + NoUninit + AnyBitPattern> CreateGpuBufferReadable<T> for Vec<T> {
    fn create_gpu_buffer_readable(&self, backend: &GpuBackend) -> GpuResult<GpuBufferReadable<T>> {
        let buffer = backend.init_buffer(
            self.as_slice(),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )?;
        let readback = backend.uninit_buffer(
            buffer.len(),
            BufferUsages::COPY_DST | BufferUsages::MAP_READ
        )?;
        Ok(
            GpuBufferReadable { buffer, readback }
        )
    }
}

impl<T: DeviceValue + NoUninit + AnyBitPattern> CreateGpuBufferReadable<T> for T {
    fn create_gpu_buffer_readable(&self, backend: &GpuBackend) -> GpuResult<GpuBufferReadable<T>> {
        let buffer = backend.init_buffer(
            &[*self],
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )?;
        let readback = backend.uninit_buffer(
            buffer.len(),
            BufferUsages::COPY_DST | BufferUsages::MAP_READ
        )?;
        Ok(
            GpuBufferReadable { buffer, readback }
        )
    }
}