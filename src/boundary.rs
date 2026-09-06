use khal::backend::{DispatchGrid, GpuPass};
use khal::Shader;
use taser_em_shaders::boundary::*;
use crate::prelude::*;

pub trait BoundaryCondition<Axis: BoundaryAxis> {
    /// Runs in the simulation pipeline initialize stage (before simulation is run)
    fn initialize(
        &mut self,
        _pass: &mut GpuPass,
        _state: &mut FdtdLossyState
    ) -> TaserResult<()> {
        Ok(())
    }

    /// Runs before updating H field in each step.
    fn pre_update(
        &mut self,
        pass: &mut GpuPass,
        state: &mut FdtdLossyState
    ) -> TaserResult<()>;

    /// Runs before update the Dn and En fields in each step (runs immediately after H field update).
    fn before_de_update(
        &mut self,
        pass: &mut GpuPass,
        state: &mut FdtdLossyState
    ) -> TaserResult<()>;
}

pub trait BoundaryAxis {}

macro_rules! boundary_axis {
    ($name:ident) => {
        pub struct $name;
        impl BoundaryAxis for $name {}
    };
}

boundary_axis!(X);
boundary_axis!(Y);
boundary_axis!(Z);

/// A struct containing all boundary conditions for the simulation.
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
    pub fn initialize(
        &mut self,
        pass: &mut GpuPass,
        state: &mut FdtdLossyState
    ) -> TaserResult<()> {
        self.x_boundary.initialize(pass, state)?;
        self.y_boundary.initialize(pass, state)?;
        self.z_boundary.initialize(pass, state)
    }

    pub fn pre_update(
        &mut self,
        pass: &mut GpuPass,
        state: &mut FdtdLossyState
    ) -> TaserResult<()> {
        self.x_boundary.pre_update(pass, state)?;
        self.y_boundary.pre_update(pass, state)?;
        self.z_boundary.pre_update(pass, state)
    }

    pub fn before_de_update(
        &mut self,
        pass: &mut GpuPass,
        state: &mut FdtdLossyState
    ) -> TaserResult<()> {
        self.x_boundary.before_de_update(pass, state)?;
        self.y_boundary.before_de_update(pass, state)?;
        self.z_boundary.before_de_update(pass, state)
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
                _: &mut FdtdLossyState
            ) -> TaserResult<()> {
                Ok(())
            }

            fn before_de_update(
                &mut self,
                _: &mut GpuPass,
                _: &mut FdtdLossyState
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
            init_kernel: $shader,
        }

        impl BoundaryCondition<$axis> for $name {
            fn initialize(&mut self, pass: &mut GpuPass, state: &mut FdtdLossyState) -> TaserResult<()> {
                self.init_kernel.call(
                    pass,
                    DispatchGrid::ThreadCount(state.thread_count),
                    &state.grid_params,
                    &mut state.grid_coeffs
                )?;
                Ok(())
            }

            fn pre_update(&mut self, _pass: &mut GpuPass, _state: &mut FdtdLossyState) -> TaserResult<()> { Ok(()) }

            fn before_de_update(&mut self, _pass: &mut GpuPass, _state: &mut FdtdLossyState) -> TaserResult<()> { Ok(()) }
        }
    };
}

#[cfg(not(feature = "dim1"))]
impl_pec_boundary!(PECBoundaryX, X, GpuPecBoundaryXInit);
#[cfg(not(feature = "dim1"))]
impl_pec_boundary!(PECBoundaryY, Y, GpuPecBoundaryYInit);
#[cfg(not(feature = "dim2"))]
impl_pec_boundary!(PECBoundaryZ, Z, GpuPecBoundaryZInit);

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
                state: &mut FdtdLossyState
            ) -> TaserResult<()> {
                Ok(self.en_kernel.call(
                    pass,
                    DispatchGrid::ThreadCount(state.thread_count),
                    &state.grid_params,
                    &mut state.en,
                )?)
            }
        
            fn before_de_update(
                &mut self,
                pass: &mut GpuPass,
                state: &mut FdtdLossyState
            ) -> TaserResult<()> {
                Ok(self.h_kernel.call(
                    pass,
                    DispatchGrid::ThreadCount(state.thread_count),
                    &state.grid_params,
                    &mut state.h,
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