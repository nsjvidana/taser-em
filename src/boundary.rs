use khal::backend::{DispatchGrid, GpuBuffer, GpuPass};
use khal::Shader;
use taser_em_shaders::boundary::*;
use taser_em_shaders::fdtd::GridParameters;
use crate::prelude::*;

// TODO: might need different parameters for anisotropy...
pub trait BoundaryCondition<Axis: BoundaryAxis> {
    /// Runs before updating H field in each step.
    fn pre_update(
        &mut self,
        pass: &mut GpuPass,
        grid: &GpuBuffer<GridParameters>,
        h: &mut GpuBuffer<Vec4>,
        dn: &mut GpuBuffer<Vec4>,
        en: &mut GpuBuffer<Vec4>,
        thread_count: [u32; 3],
    ) -> TaserResult<()>;

    /// Runs before update the Dn and En fields in each step (runs immediately after H field update).
    fn before_de_update(
        &mut self,
        pass: &mut GpuPass,
        grid: &GpuBuffer<GridParameters>,
        h: &mut GpuBuffer<Vec4>,
        dn: &mut GpuBuffer<Vec4>,
        en: &mut GpuBuffer<Vec4>,
        thread_count: [u32; 3],
    ) -> TaserResult<()>;
}

pub trait BoundaryAxis {}

pub struct X;

impl BoundaryAxis for X {}

pub struct Y;

impl BoundaryAxis for Y {}

pub struct Z;

impl BoundaryAxis for Z {}

#[derive(Shader)]
pub struct PECBoundary {
    kernel: GpuPecBoundary
}

impl BoundaryCondition<X> for PECBoundary {
    fn pre_update(
        &mut self,
        pass: &mut GpuPass,
        grid: &GpuBuffer<GridParameters>,
        h: &mut GpuBuffer<Vec4>,
        dn: &mut GpuBuffer<Vec4>,
        en: &mut GpuBuffer<Vec4>,
        thread_count: [u32; 3]
    ) -> TaserResult<()>
    {
        self.kernel.call(
            pass,
            DispatchGrid::ThreadCount(thread_count),
            grid,
            h,
            dn,
            en
        )?;
        Ok(())
    }

    fn before_de_update(
        &mut self,
        _pass: &mut GpuPass,
        _grid: &GpuBuffer<GridParameters>,
        _h: &mut GpuBuffer<Vec4>,
        _dn: &mut GpuBuffer<Vec4>,
        _en: &mut GpuBuffer<Vec4>,
        _thread_count: [u32; 3],
    ) -> TaserResult<()> { Ok(()) }
}

macro_rules! impl_pec_boundary {
    ($name:ident, $axis:ident, $shader:ident) => {
        #[derive(Shader)]
        pub struct $name {
            kernel: $shader,
        }

        impl BoundaryCondition<$axis> for $name {
            fn pre_update(
                &mut self,
                pass: &mut GpuPass,
                grid: &GpuBuffer<GridParameters>,
                h: &mut GpuBuffer<Vec4>,
                dn: &mut GpuBuffer<Vec4>,
                en: &mut GpuBuffer<Vec4>,
                thread_count: [u32; 3]
            ) -> TaserResult<()>
            {
                self.kernel.call(
                    pass,
                    DispatchGrid::ThreadCount(thread_count),
                    grid,
                    h,
                    dn,
                    en
                )?;
                Ok(())
            }

            fn before_de_update(
                &mut self,
                _pass: &mut GpuPass,
                _grid: &GpuBuffer<GridParameters>,
                _h: &mut GpuBuffer<Vec4>,
                _dn: &mut GpuBuffer<Vec4>,
                _en: &mut GpuBuffer<Vec4>,
                _thread_count: [u32; 3],
            ) -> TaserResult<()> { Ok(()) }
        }
    };
}

#[cfg(not(feature = "dim1"))]
impl_pec_boundary!(PECBoundaryX, X, GpuPecBoundaryX);
#[cfg(not(feature = "dim1"))]
impl_pec_boundary!(PECBoundaryY, Y, GpuPecBoundaryY);
#[cfg(not(feature = "dim2"))]
impl_pec_boundary!(PECBoundaryZ, Z, GpuPecBoundaryZ);