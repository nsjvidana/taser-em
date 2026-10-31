use khal::backend::{DispatchGrid, GpuPass};
use khal::Shader;
use taser_em_macros::replace_idents;
use taser_em_shaders::boundary::*;
use crate::prelude::*;

macro_rules! make_bc_trait {
    ($name:ident) => {
        pub trait $name {
            /// The state type of this boundary condition, containing all the  buffers it needs.
            ///
            /// If no state is needed, just make this a unit type.
            type StateType;

            /// Construct this boundary condition's state type when creating simulation buffers.
            ///
            /// These buffers can be accessed through [`FdtdLossyState::bc_states`].
            fn construct_state(&self, simulation: &FdtdLossySimulation) -> TaserResult<Self::StateType>;

            /// Runs in the simulation pipeline initialize stage (before simulation is run)
            fn initialize<BC: BoundaryConditionTypes>(
                &self,
                _pass: &mut GpuPass,
                _state: &mut FdtdLossyState<BC>
            ) -> TaserResult<()> {
                Ok(())
            }

            /// Runs before updating H field in each step.
            fn pre_update<BC: BoundaryConditionTypes>(
                &self,
                pass: &mut GpuPass,
                state: &mut FdtdLossyState<BC>
            ) -> TaserResult<()>;

            /// Runs before update the Dn and En fields in each step (runs immediately after H field update).
            fn before_de_update<BC: BoundaryConditionTypes>(
                &self,
                pass: &mut GpuPass,
                state: &mut FdtdLossyState<BC>
            ) -> TaserResult<()>;
        }
    };
}

make_bc_trait!(BoundaryConditionX);
make_bc_trait!(BoundaryConditionY);
make_bc_trait!(BoundaryConditionZ);

/// This trait exists to prevent having to type generic trait bounds for each boundary condition on
/// each axis for every struct that stores boundary conditions. It gets very verbose without this
/// trait.
pub trait BoundaryConditionTypes {
    #[cfg(not(feature = "dim1"))]
    type BoundaryTypeX: BoundaryConditionX;
    #[cfg(not(feature = "dim1"))]
    type BoundaryTypeY: BoundaryConditionY;
    #[cfg(not(feature = "dim2"))]
    type BoundaryTypeZ: BoundaryConditionZ;
}

cfg_select! {
    feature = "dim1" => {
        impl<Z> BoundaryConditionTypes for (Z,)
        where
            Z: BoundaryConditionZ,
        {
            type BoundaryTypeZ = Z;
        }
    }
    feature = "dim2" => {
        impl<X, Y> BoundaryConditionTypes for (X, Y)
        where
            X: BoundaryConditionX,
            Y: BoundaryConditionY,
        {
            type BoundaryTypeX = X;
            type BoundaryTypeY = Y;
        }
    }
    feature = "dim3" => {
        impl<X, Y, Z> BoundaryConditionTypes for (X, Y, Z)
        where
            X: BoundaryConditionX,
            Y: BoundaryConditionY,
            Z: BoundaryConditionZ,
        {
            type BoundaryTypeX = X;
            type BoundaryTypeY = Y;
            type BoundaryTypeZ = Z;
        }
    }
}

/// A struct containing all boundary condition kernels for the simulation.
pub struct BoundaryConditions<BC: BoundaryConditionTypes>
{
    #[cfg(not(feature = "dim1"))]
    pub(crate) x_boundary: BC::BoundaryTypeX,
    #[cfg(not(feature = "dim1"))]
    pub(crate) y_boundary: BC::BoundaryTypeY,
    #[cfg(not(feature = "dim2"))]
    pub(crate) z_boundary: BC::BoundaryTypeZ,
}

impl<BC: BoundaryConditionTypes> BoundaryConditions<BC> {
    pub fn initialize(
        &self,
        pass: &mut GpuPass,
        state: &mut FdtdLossyState<BC>
    ) -> TaserResult<()> {
        #[cfg(not(feature = "dim1"))]
        self.x_boundary.initialize(pass, state)?;
        #[cfg(not(feature = "dim1"))]
        self.y_boundary.initialize(pass, state)?;
        #[cfg(not(feature = "dim2"))]
        self.z_boundary.initialize(pass, state)?;
        Ok(())
    }

    pub fn pre_update(
        &self,
        pass: &mut GpuPass,
        state: &mut FdtdLossyState<BC>
    ) -> TaserResult<()> {
        #[cfg(not(feature = "dim1"))]
        self.x_boundary.pre_update(pass, state)?;
        #[cfg(not(feature = "dim1"))]
        self.y_boundary.pre_update(pass, state)?;
        #[cfg(not(feature = "dim2"))]
        self.z_boundary.pre_update(pass, state)?;
        Ok(())
    }

    pub fn before_de_update(
        &self,
        pass: &mut GpuPass,
        state: &mut FdtdLossyState<BC>
    ) -> TaserResult<()> {
        #[cfg(not(feature = "dim1"))]
        self.x_boundary.before_de_update(pass, state)?;
        #[cfg(not(feature = "dim1"))]
        self.y_boundary.before_de_update(pass, state)?;
        #[cfg(not(feature = "dim2"))]
        self.z_boundary.before_de_update(pass, state)?;
        Ok(())
    }
}

cfg_select! {
    feature = "dim1" => {
        impl<Z> BoundaryConditions<(Z,)>
        where
            Z: BoundaryConditionZ,
        {
            pub fn new(z_boundary: Z) -> Self {
                Self { z_boundary }
            }
        }
    }
    feature = "dim2" => {
        impl<X, Y> BoundaryConditions<(X, Y)>
        where
            X: BoundaryConditionX,
            Y: BoundaryConditionY,
        {
            pub fn new(x_boundary: X, y_boundary: Y) -> Self {
                Self { x_boundary, y_boundary }
            }
        }
    }
    feature = "dim3" => {
        impl<X, Y, Z> BoundaryConditions<(X, Y, Z)>
        where
            X: BoundaryConditionX,
            Y: BoundaryConditionY,
            Z: BoundaryConditionZ,
        {
            pub fn new(x_boundary: X, y_boundary: Y, z_boundary: Z) -> Self {
                Self { x_boundary, y_boundary, z_boundary }
            }
        }
    }
}

pub struct BoundaryConditionStates<BC: BoundaryConditionTypes>
{
    #[cfg(not(feature = "dim1"))]
    pub x_state: <BC::BoundaryTypeX as BoundaryConditionX>::StateType,
    #[cfg(not(feature = "dim1"))]
    pub y_state: <BC::BoundaryTypeY as BoundaryConditionY>::StateType,
    #[cfg(not(feature = "dim2"))]
    pub z_state: <BC::BoundaryTypeZ as BoundaryConditionZ>::StateType,
}

impl<BC: BoundaryConditionTypes> BoundaryConditionStates<BC> {
    pub fn new(boundary_conditions: &BoundaryConditions<BC>, simulation: &FdtdLossySimulation) -> TaserResult<Self> {
        Ok(Self {
            #[cfg(not(feature = "dim1"))]
            x_state: boundary_conditions.x_boundary.construct_state(simulation)?,
            #[cfg(not(feature = "dim1"))]
            y_state: boundary_conditions.y_boundary.construct_state(simulation)?,
            #[cfg(not(feature = "dim2"))]
            z_state: boundary_conditions.z_boundary.construct_state(simulation)?,
        })
    }
}

#[cfg(not(feature = "dim3"))]
macro_rules! unit_type_boundary {
    ($axis:ident) => {
        impl $axis for () {
            type StateType = ();

            fn construct_state(&self, _simulation: &FdtdLossySimulation) -> TaserResult<Self::StateType> { Ok(()) }

            fn pre_update<BC: BoundaryConditionTypes>(
                &self,
                _: &mut GpuPass,
                _: &mut FdtdLossyState<BC>
            ) -> TaserResult<()> {
                Ok(())
            }

            fn before_de_update<BC: BoundaryConditionTypes>(
                &self,
                _: &mut GpuPass,
                _: &mut FdtdLossyState<BC>
            ) -> TaserResult<()> {
                Ok(())
            }
        }
    };
}

#[cfg(feature = "dim1")]
unit_type_boundary!(BoundaryConditionX);
#[cfg(feature = "dim1")]
unit_type_boundary!(BoundaryConditionY);
#[cfg(feature = "dim2")]
unit_type_boundary!(BoundaryConditionZ);

#[cfg_attr(not(feature = "dim1"), replace_idents(ShaderName = PECBoundaryX, InitShader = GpuPecBoundaryXInit))]
#[cfg_attr(not(feature = "dim1"), replace_idents(ShaderName = PECBoundaryY, InitShader = GpuPecBoundaryYInit))]
#[cfg_attr(not(feature = "dim2"), replace_idents(ShaderName = PECBoundaryZ, InitShader = GpuPecBoundaryZInit))]
#[derive(Shader)]
pub struct ShaderName {
    init_kernel: InitShader,
}

#[cfg_attr(not(feature = "dim1"), replace_idents(PECBoundaryZ = PECBoundaryX, BoundaryConditionZ = BoundaryConditionX))]
#[cfg_attr(not(feature = "dim1"), replace_idents(PECBoundaryZ = PECBoundaryY, BoundaryConditionZ = BoundaryConditionY))]
#[cfg_attr(not(feature = "dim2"), replace_idents(PECBoundaryZ = PECBoundaryZ))]
impl BoundaryConditionZ for PECBoundaryZ {
    type StateType = ();

    fn construct_state(&self, _: &FdtdLossySimulation) -> TaserResult<Self::StateType> { Ok(()) }

    fn initialize<BC: BoundaryConditionTypes>(
        &self,
        pass: &mut GpuPass,
        state: &mut FdtdLossyState<BC>
    ) -> TaserResult<()> {
        self.init_kernel.call(
            pass,
            DispatchGrid::Grid(state.workgroup_count),
            &state.grid,
            &mut state.pml_coeffs
        )?;
        Ok(())
    }

    fn pre_update<BC: BoundaryConditionTypes>(
        &self,
        _pass: &mut GpuPass,
        _state: &mut FdtdLossyState<BC>
    ) -> TaserResult<()> { Ok(()) }

    fn before_de_update<BC: BoundaryConditionTypes>(
        &self,
        _pass: &mut GpuPass,
        _state: &mut FdtdLossyState<BC>
    ) -> TaserResult<()> { Ok(()) }
}

#[cfg_attr(not(feature = "dim1"), replace_idents(ShaderName = PeriodicBoundaryX, EnKernel = GpuPeriodicXEn, HKernel = GpuPeriodicXH))]
#[cfg_attr(not(feature = "dim1"), replace_idents(ShaderName = PeriodicBoundaryY, EnKernel = GpuPeriodicYEn, HKernel = GpuPeriodicYH))]
#[cfg_attr(not(feature = "dim2"), replace_idents(ShaderName = PeriodicBoundaryZ, EnKernel = GpuPeriodicZEn, HKernel = GpuPeriodicZH))]
#[derive(Shader)]
pub struct ShaderName {
    en_kernel: EnKernel,
    h_kernel: HKernel,
}

#[cfg_attr(not(feature = "dim1"), replace_idents(PeriodicBoundaryZ = PeriodicBoundaryX, BoundaryConditionZ = BoundaryConditionX))]
#[cfg_attr(not(feature = "dim1"), replace_idents(PeriodicBoundaryZ = PeriodicBoundaryY, BoundaryConditionZ = BoundaryConditionY))]
#[cfg_attr(not(feature = "dim2"), replace_idents(PeriodicBoundaryZ = PeriodicBoundaryZ, BoundaryConditionZ = BoundaryConditionZ))]
impl BoundaryConditionZ for PeriodicBoundaryZ {
    type StateType = ();

    fn construct_state(&self, _: &FdtdLossySimulation) -> TaserResult<Self::StateType> { Ok(()) }

    fn pre_update<BC: BoundaryConditionTypes>(
        &self,
        pass: &mut GpuPass,
        state: &mut FdtdLossyState<BC>
    ) -> TaserResult<()> {
        Ok(self.en_kernel.call(
            pass,
            DispatchGrid::Grid(state.workgroup_count),
            &state.grid,
            &mut state.en,
        )?)
    }

    fn before_de_update<BC: BoundaryConditionTypes>(
        &self,
        pass: &mut GpuPass,
        state: &mut FdtdLossyState<BC>
    ) -> TaserResult<()> {
        Ok(self.h_kernel.call(
            pass,
            DispatchGrid::Grid(state.workgroup_count),
            &state.grid,
            &mut state.h,
        )?)
    }
}