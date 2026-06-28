pub mod prelude;
pub mod gpu_util;

pub use taser_em_shaders as shaders;

use khal::backend::{Backend, Encoder, GpuBackend, GpuBuffer, GpuTimestamps};
use khal::re_exports::include_dir::{include_dir, Dir};
use khal::Shader;
use taser_em_shaders::fdtd1::{GridParameters, PmlCoefficients};
use crate::gpu_util::{CreateGpuBuffer, CreateGpuBufferReadable, GpuBufferReadable};
use crate::prelude::GpuResult;

pub static SPIRV_DIR: Dir<'static> = include_dir!("$OUT_DIR/shaders-spirv");

macro_rules! shader_struct {
    ($name:ident, $inner:ty) => {
        #[derive(Shader)]
        pub struct $name {
            pub kernel: $inner
        }

        impl AsRef<$inner> for $name {
            fn as_ref(&self) -> &$inner {
                &self.kernel
            }
        }

        impl std::ops::Deref for $name {
            type Target = $inner;

            fn deref(&self) -> &Self::Target {
                &self.kernel
            }
        }
    };
}

shader_struct!(Fdtd1, taser_em_shaders::fdtd1::Fdtd1DnY);

pub struct Fdtd1Runner {
    pub h_x: Vec<f32>,
    pub dn_y: Vec<f32>,
    pub en_y: Vec<f32>,
    pub int_en_y: Vec<f32>,
    pub buffers: Fdtd1Buffers,
}

impl Fdtd1Runner {
    // TODO: make Fdtd1Runner with things other than just `num_cells`
    pub fn new(
        num_cells: usize,
        grid_params: GridParameters,
        backend: &GpuBackend
    ) -> GpuResult<Self> {
        let h_x = vec![Default::default(); num_cells];
        let dn_y = vec![Default::default(); num_cells];
        let en_y = vec![Default::default(); num_cells];
        let int_en_y = vec![Default::default(); num_cells];
        let coeffs = vec![PmlCoefficients::default(); num_cells];

        let buffers = Fdtd1Buffers {
            h_x: h_x.create_gpu_buffer_readable(backend)?,
            dn_y: dn_y.create_gpu_buffer_readable(backend)?,
            en_y: en_y.create_gpu_buffer_readable(backend)?,
            int_en_y: int_en_y.create_gpu_buffer(backend)?,
            coeffs: coeffs.create_gpu_buffer(backend)?,
            grid: grid_params.create_gpu_uniform(backend)?,
        };

        Ok(Self {
            h_x,
            dn_y,
            en_y,
            int_en_y,
            buffers,
        })
    }

    pub fn submit(
        &mut self,
        kernel: &taser_em_shaders::fdtd1::Fdtd1DnY,
        timestamps: Option<&mut GpuTimestamps>,
        backend: &GpuBackend
    ) -> GpuResult<()> {
        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("fdtd1_dn_y", timestamps);
        kernel.call(
            &mut pass,
            self.dn_y.len(),
            &mut self.buffers.h_x.buffer,
            &mut self.buffers.dn_y.buffer,
            &mut self.buffers.en_y.buffer,
            &mut self.buffers.int_en_y,
            &self.buffers.coeffs,
            &self.buffers.grid,
        )?;
        drop(pass);
        self.buffers.h_x.encode_copy_cmd(&mut encoder)?;
        self.buffers.dn_y.encode_copy_cmd(&mut encoder)?;
        self.buffers.en_y.encode_copy_cmd(&mut encoder)?;

        backend.submit(encoder)?;

        Ok(())
    }
}

pub struct Fdtd1Buffers {
    pub h_x: GpuBufferReadable<f32>,
    pub dn_y: GpuBufferReadable<f32>,
    pub en_y: GpuBufferReadable<f32>,
    pub int_en_y: GpuBuffer<f32>,
    pub coeffs: GpuBuffer<PmlCoefficients>,
    pub grid: GpuBuffer<GridParameters>,
}

shader_struct!(AddAssign, taser_em_shaders::AddAssign);

pub struct AddAssignRunner {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub buffers: AddAssignBuffers
}

impl AddAssignRunner {
    pub fn new(a: Vec<f32>, b: Vec<f32>, backend: &GpuBackend) -> GpuResult<Self> {
        let buffers = AddAssignBuffers {
            a: a.create_gpu_buffer_readable(backend)?,
            b: b.create_gpu_buffer(backend)?
        };

        Ok(Self { a, b, buffers })
    }

    pub fn submit(
        &mut self,
        kernel: &taser_em_shaders::AddAssign,
        timestamps: Option<&mut GpuTimestamps>,
        backend: &GpuBackend
    ) -> GpuResult<()> {
        let mut encoder = backend.begin_encoding();

        let mut pass = encoder.begin_pass("add_assign", timestamps);

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

        Ok(())
    }
}

pub struct AddAssignBuffers {
    pub a: GpuBufferReadable<f32>,
    pub b: GpuBuffer<f32>,
}