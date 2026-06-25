pub mod prelude;
pub mod gpu_util;

use khal::backend::{Backend, Encoder, GpuBackend, GpuBuffer, GpuTimestamps};
use khal::re_exports::include_dir::{include_dir, Dir};
use khal::Shader;
use crate::gpu_util::{CreateGpuBuffer, CreateGpuBufferReadable, GpuBufferReadable};
use crate::prelude::GpuResult;

pub static SPIRV_DIR: Dir<'static> = include_dir!("$OUT_DIR/shaders-spirv");

#[derive(Shader)]
pub struct AddAssign {
    pub add_assign: taser_em_shaders::AddAssign
}

pub struct AddAssignRunner {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub buffers: AddAssignBuffers
}

impl AddAssignRunner {
    pub fn new(a: Vec<f32>, b: Vec<f32>, backend: &GpuBackend) -> GpuResult<Self>{
        let buffers = AddAssignBuffers {
            a: a.create_gpu_buffer_readable(backend)?,
            b: b.create_gpu_buffer(backend)?
        };

        Ok(Self { a, b, buffers })
    }

    pub fn submit(
        &mut self,
        kernel: &taser_em_shaders::AddAssign,
        backend: &GpuBackend
    ) -> GpuResult<GpuTimestamps>{
        let mut encoder = backend.begin_encoding();

        let mut timestamps = GpuTimestamps::new(backend, 1);
        let mut pass = encoder.begin_pass("add_assign", Some(&mut timestamps));

        let AddAssignBuffers {
            a, b
        } = &mut self.buffers;
        kernel.call(
            &mut pass,
            self.a.len(),
            &mut a.buffer,
            b
        )?;
        drop(pass);
        a.encode_copy_cmd(&mut encoder)?;

        backend.submit(encoder)?;

        Ok(timestamps)
    }
}

pub struct AddAssignBuffers {
    pub a: GpuBufferReadable<f32>,
    pub b: GpuBuffer<f32>,
}