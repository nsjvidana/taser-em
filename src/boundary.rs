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

pub struct BoundaryConditions<X, Y, Z>
where
    X: BoundaryCondition<self::X>,
    Y: BoundaryCondition<self::Y>,
    Z: BoundaryCondition<self::Z>,
{
    pub(crate) x_boundary: X,
    pub(crate) y_boundary: Y,
    pub(crate) z_boundary: Z,
}

impl<X, Y, Z> BoundaryConditions<X, Y, Z>
where
    X: BoundaryCondition<self::X>,
    Y: BoundaryCondition<self::Y>,
    Z: BoundaryCondition<self::Z>,
{
    pub fn pre_update(
        &mut self,
        pass: &mut GpuPass,
        grid: &GpuBuffer<GridParameters>,
        h: &mut GpuBuffer<Vec4>,
        dn: &mut GpuBuffer<Vec4>,
        en: &mut GpuBuffer<Vec4>,
        thread_count: [u32; 3],
    ) -> TaserResult<()> {
        self.x_boundary.pre_update(pass, grid, h, dn, en, thread_count)?;
        self.y_boundary.pre_update(pass, grid, h, dn, en, thread_count)?;
        self.z_boundary.pre_update(pass, grid, h, dn, en, thread_count)
    }

    pub fn before_de_update(
        &mut self,
        pass: &mut GpuPass,
        grid: &GpuBuffer<GridParameters>,
        h: &mut GpuBuffer<Vec4>,
        dn: &mut GpuBuffer<Vec4>,
        en: &mut GpuBuffer<Vec4>,
        thread_count: [u32; 3],
    ) -> TaserResult<()> {
        self.x_boundary.before_de_update(pass, grid, h, dn, en, thread_count)?;
        self.y_boundary.before_de_update(pass, grid, h, dn, en, thread_count)?;
        self.z_boundary.before_de_update(pass, grid, h, dn, en, thread_count)
    }
}

cfg_select! {
    feature = "dim1" => {
        impl<Z> BoundaryConditions<(), (), Z>
        where
            Z: BoundaryCondition<self::Z>,
        {
            pub fn new(z_boundary: Z) -> Self {
                Self { x_boundary: (), y_boundary: (), z_boundary }
            }
        }
    }
    feature = "dim2" => {
        impl<X, Y> BoundaryConditions<X, Y, ()>
        where
            X: BoundaryCondition<self::X>,
            Y: BoundaryCondition<self::Y>,
        {
            pub fn new(x_boundary: X, y_boundary: Y) -> Self {
                Self { x_boundary, y_boundary, z_boundary: () }
            }
        }
    }
    feature = "dim3" => {
        impl<X, Y, Z> BoundaryConditions<X, Y, Z>
        where
            X: BoundaryCondition<self::X>,
            Y: BoundaryCondition<self::Y>,
            Z: BoundaryCondition<self::Z>,
        {
            pub fn new(x_boundary: X, y_boundary: Y, z_boundary: Z) -> Self {
                Self { x_boundary, y_boundary, z_boundary }
            }
        }
    }
}

#[cfg(not(feature = "dim3"))]
macro_rules! unit_type_boundary {
    ($axis:ident) => {
        impl BoundaryCondition<$axis> for () {
            fn pre_update(
                &mut self,
                _: &mut GpuPass,
                _: &GpuBuffer<GridParameters>,
                _: &mut GpuBuffer<Vec4>,
                _: &mut GpuBuffer<Vec4>,
                _: &mut GpuBuffer<Vec4>,
                _: [u32; 3],
            ) -> TaserResult<()> {
                Ok(())
            }

            fn before_de_update(
                &mut self,
                _: &mut GpuPass,
                _: &GpuBuffer<GridParameters>,
                _: &mut GpuBuffer<Vec4>,
                _: &mut GpuBuffer<Vec4>,
                _: &mut GpuBuffer<Vec4>,
                _: [u32; 3],
            ) -> TaserResult<()> {
                Ok(())
            }
        }
    };
}

#[cfg(feature = "dim1")]
unit_type_boundary!(X);
#[cfg(feature = "dim1")]
unit_type_boundary!(Y);
#[cfg(feature = "dim2")]
unit_type_boundary!(Z);

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

macro_rules! periodic_boundary {
    ($name:ident, $en_kernel:ident, $h_kernel:ident, $boundary_axis:ident) => {
        #[derive(Shader)]
        pub struct $name {
            en_kernel: $en_kernel,
            h_kernel: $h_kernel,
        }
        
        impl BoundaryCondition<$boundary_axis> for $name {
            fn pre_update(
                &mut self,
                pass: &mut GpuPass,
                grid: &GpuBuffer<GridParameters>,
                _: &mut GpuBuffer<Vec4>,
                _: &mut GpuBuffer<Vec4>,
                en: &mut GpuBuffer<Vec4>,
                thread_count: [u32; 3],
            ) -> TaserResult<()> {
                Ok(self.en_kernel.call(
                    pass,
                    DispatchGrid::ThreadCount(thread_count),
                    grid,
                    en,
                )?)
            }
        
            fn before_de_update(
                &mut self,
                pass: &mut GpuPass,
                grid: &GpuBuffer<GridParameters>,
                h: &mut GpuBuffer<Vec4>,
                _: &mut GpuBuffer<Vec4>,
                _: &mut GpuBuffer<Vec4>,
                thread_count: [u32; 3],
            ) -> TaserResult<()> {
                Ok(self.h_kernel.call(
                    pass,
                    DispatchGrid::ThreadCount(thread_count),
                    grid,
                    h,
                )?)
            }
        }
    };
}

#[cfg(not(feature = "dim1"))]
periodic_boundary!(PeriodicBoundaryX, GpuPeriodicXEn, GpuPeriodicXH, X);
#[cfg(not(feature = "dim1"))]
periodic_boundary!(PeriodicBoundaryY, GpuPeriodicYEn, GpuPeriodicYH, Y);
#[cfg(not(feature = "dim2"))]
periodic_boundary!(PeriodicBoundaryZ, GpuPeriodicZEn, GpuPeriodicZH, Z);