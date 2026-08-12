pub use khal_std::glamx::*;

use bytemuck::{Pod, Zeroable};
pub use dim_types::*;

pub type Real = f32;
pub type Index = u32;

pub const MAX_DIM: usize = 3;

#[cfg(feature = "dim1")]
mod dim_types {
    use crate::math::{Index, Real};

    pub const DIM: usize = 1;
    /// A "vector" type used as a spatial position or a vector of staggered vector components.
    pub type Vect = Real;
    pub type GridIndex = Index;
    pub type BoolVect = bool;
}

#[cfg(feature = "dim2")]
mod dim_types {
    use khal_std::glamx::{BVec2, UVec2, Vec2};

    pub const DIM: usize = 2;
    /// A shader's vector field element
    pub type Vect = Vec2;
    pub type GridIndex = UVec2;
    pub type BoolVect = BVec2;
}

#[cfg(feature = "dim3")]
mod dim_types {
    use khal_std::glamx::{BVec3, UVec3, Vec3};
    pub const DIM: usize = super::MAX_DIM;
    /// A shader's vector field element
    pub type Vect = Vec3;
    pub type GridIndex = UVec3;
    pub type BoolVect = BVec3;
}

/// A trait specifically for vector field vectors.
pub trait VectExt: VectorValueExt {
    fn to_3d(self, mask: Vec3) -> Vec3;
    fn into_array(self) -> [Real; DIM];
    fn from_array(arr: [Real; DIM]) -> Vect;
    fn from_vec3(vec3: Vec3) -> Vect;
    fn from_vec4(v4: Vec4) -> Vect;
    fn as_grid_index(&self) -> GridIndex;
    fn magnitude(&self) -> Real;
}

/// A trait with functions for general vector values
pub trait VectorValueExt: Sized {
    type Element;
    type Boolean;
    const ZERO: Self;
    const ONE: Self;
    fn splat(value: Self::Element) -> Self;
    fn max_element(self) -> Self::Element;
    fn min_element(self) -> Self::Element;
    fn cmplt(self, rhs: Self) -> Self::Boolean;
    fn cmpgt(self, rhs: Self) -> Self::Boolean;
    fn cmpge(self, rhs: Self) -> Self::Boolean;
    fn cmpeq(self, rhs: Self) -> Self::Boolean;
    fn element_sum(self) -> Self::Element;
    fn element_product(self) -> Self::Element;
    fn map<F>(self, f: F) -> Self
    where
        F: FnMut(Self::Element) -> Self::Element;
    /// Glam's `with_x()` function, but for all dimensions of this crate.
    fn with_x(self, x: Self::Element) -> Self;
    /// Glam's `with_y()` function, but for all dimensions of this crate.
    fn with_y(self, y: Self::Element) -> Self;
    /// Glam's `with_z()` function, but for all dimensions of this crate.
    fn with_z(self, z: Self::Element) -> Self;
}

macro_rules! dim1_or_else {
    ($dim1:expr, $or_else:expr) => {
        cfg_select! {
            feature = "dim1" => { $dim1 }
            _ => { $or_else }
        }
    };
}

macro_rules! impl_vector_value_ext {
    ($v:ident, $elem:ty, $zero:expr, $one:expr) => {
        impl VectorValueExt for $v {
            type Element = $elem;
            type Boolean = BoolVect;
            const ZERO: Self = $zero;
            const ONE: Self = $one;

            #[inline]
            fn splat(value: Self::Element) -> Self {
                dim1_or_else!(value, Self::splat(value))
            }

            #[inline]
            fn max_element(self) -> Self::Element {
                dim1_or_else!(self, self.max_element())
            }

            #[inline]
            fn min_element(self) -> Self::Element {
                dim1_or_else!(self, self.min_element())
            }

            #[inline]
            fn cmplt(self, rhs: Self) -> Self::Boolean {
                dim1_or_else!(self < rhs, self.cmplt(rhs))
            }

            #[inline]
            fn cmpgt(self, rhs: Self) -> Self::Boolean {
                dim1_or_else!(self > rhs, self.cmpgt(rhs))
            }

            #[inline]
            fn cmpge(self, rhs: Self) -> Self::Boolean {
                dim1_or_else!(self >= rhs, self.cmpge(rhs))
            }

            #[inline]
            fn cmpeq(self, rhs: Self) -> Self::Boolean {
                dim1_or_else!(self == rhs, self.cmpeq(rhs))
            }

            #[inline]
            fn element_sum(self) -> Self::Element {
                dim1_or_else!(self, self.element_sum())
            }

            #[inline]
            fn element_product(self) -> Self::Element {
                dim1_or_else!(self, self.element_product())
            }

            #[inline]
            #[allow(unused_mut)]
            fn map<F>(self, mut f: F) -> Self
                where F: FnMut(Self::Element) -> Self::Element
            {
                dim1_or_else!(f(self), self.map(f))
            }

            #[inline]
            #[cfg_attr(feature = "dim1", allow(unused_variables))]
            fn with_x(self, x: Self::Element) -> Self {
                dim1_or_else!(self, self.with_x(x))
            }

            #[inline]
            #[cfg_attr(feature = "dim1", allow(unused_variables))]
            fn with_y(self, y: Self::Element) -> Self {
                dim1_or_else!(self, self.with_y(y))
            }

            #[inline]
            #[cfg_attr(not(feature = "dim3"), allow(unused_variables))]
            fn with_z(self, z: Self::Element) -> Self {
                dim1_or_else!(
                    z,
                    cfg_select! {
                        feature = "dim2" => self,
                        feature = "dim3" => self.with_z(z)
                    }
                )
            }
        }
    };
}

impl VectExt for Vect {
    /// Converts a [`Vect`] to [`Vec3`]
    /// `mask` sets the components that `self` don't already have.
    ///
    /// e.g. `to_3d(z, Vec3::ONE)` would return `Vec3::new(1., 1., z)` in 1D.
    #[inline]
    #[allow(unused_variables)]
    fn to_3d(self, mask: Vec3) -> Vec3 {
        #[cfg(feature = "dim1")]
        return mask.with_z(self);
        #[cfg(feature = "dim2")]
        return mask.with_xy(self);
        #[cfg(feature = "dim3")]
        self
    }

    #[inline]
    fn into_array(self) -> [Real; DIM] {
        #[cfg(feature = "dim1")]
        return [self];
        #[cfg(not(feature = "dim1"))]
        self.to_array()
    }

    #[inline]
    fn from_array(arr: [Real; DIM]) -> Vect {
        #[cfg(feature = "dim1")]
        return arr[0];
        #[cfg(not(feature = "dim1"))]
        Vect::from_array(arr)
    }

    #[inline]
    fn from_vec3(vec3: Vec3) -> Vect {
        #[cfg(feature = "dim1")]
        return vec3.z;
        #[cfg(feature = "dim2")]
        return vec3.xy();
        #[cfg(feature = "dim3")]
        vec3
    }

    #[inline]
    fn from_vec4(v4: Vec4) -> Vect {
        #[cfg(feature = "dim1")]
        { v4.z }
        #[cfg(feature = "dim2")] {
            use khal_std::glamx::Vec4Swizzles;
            v4.xy()
        }
        #[cfg(feature = "dim3")] {
            use khal_std::glamx::Vec4Swizzles;
            v4.xyz()
        }
    }

    /// Convert [`Vect`] to [`GridIndex`] using `as` keyword. So this effectively floors all components
    /// of `self`.
    #[inline]
    fn as_grid_index(&self) -> GridIndex {
        #[cfg(feature = "dim1")]
        return *self as GridIndex;
        #[cfg(feature = "dim2")]
        return self.as_uvec2();
        #[cfg(feature = "dim3")]
        self.as_uvec3()
    }

    #[inline]
    fn magnitude(&self) -> Real {
        cfg_select! {
            feature = "dim1" => *self,
            _ => self.length()
        }
    }
}

cfg_select! {
    feature = "dim1" => { impl_vector_value_ext!(Vect, Real, 0., 1.); }
    _ => { impl_vector_value_ext!(Vect, Real, Vect::ZERO, Vect::ONE); }
}

pub trait GridIndexExt: VectorValueExt {
    fn div_ceil(self, rhs: GridIndex) -> GridIndex;
    fn n_cells_to_3d(self) -> UVec3;
    fn cell_idx_to_3d(self) -> UVec3;
    fn as_vect(&self) -> Vect;
    fn to_3d(self, mask: UVec3) -> UVec3;
    fn into_array(self) -> [u32; DIM];
    fn from_index_array(arr: [u32; DIM]) -> GridIndex;
    fn from_uvec3(uvec3: UVec3) -> GridIndex;
    fn from_flat_idx(idx: u32, n_cells: GridIndex) -> GridIndex;
    fn to_flat_idx(self, n_cells: GridIndex) -> u32;
}

impl GridIndexExt for GridIndex {
    #[inline]
    fn div_ceil(self, rhs: GridIndex) -> GridIndex {
        #[cfg(feature = "dim1")]
        { self.div_ceil(rhs) }
        #[cfg(feature = "dim2")]
        { GridIndex::new(self.x.div_ceil(rhs.x), self.y.div_ceil(rhs.y)) }
        #[cfg(feature = "dim3")]
        GridIndex::new(self.x.div_ceil(rhs.x), self.y.div_ceil(rhs.y), self.z.div_ceil(rhs.z))
    }

    /// Helper function for converting the dimensions of a grid into a 3D vector
    #[inline]
    fn n_cells_to_3d(self) -> UVec3 {
        self.to_3d(UVec3::ONE)
    }

    /// Helper function for converting the index of a cell in a grid into a 3D vector index
    #[inline]
    fn cell_idx_to_3d(self) -> UVec3 {
        self.to_3d(UVec3::ZERO)
    }

    /// Convert [`GridIndex`] into [`Vect`] using `as` keyword.
    #[inline]
    fn as_vect(&self) -> Vect {
        #[cfg(feature = "dim1")]
        return *self as Vect;
        #[cfg(feature = "dim2")]
        return self.as_vec2();
        #[cfg(feature = "dim3")]
        self.as_vec3()
    }

    /// Converts a [`GridIndex`] to [`UVec3`]
    /// `mask` sets the components that `self` don't already have.
    ///
    /// e.g. `GridIndex::to_3d(k, UVec3::ONE)` would return `UVec3::new(1, 1, k)` in 1D.
    #[inline]
    #[allow(unused_variables)]
    fn to_3d(self, mask: UVec3) -> UVec3 {
        #[cfg(feature = "dim1")]
        return mask.with_z(self);
        #[cfg(feature = "dim2")]
        return mask.with_xy(self);
        #[cfg(feature = "dim3")]
        self
    }

    #[inline]
    fn into_array(self) -> [u32; DIM] {
        #[cfg(feature = "dim1")]
        return [self];
        #[cfg(not(feature = "dim1"))]
        self.to_array()
    }

    #[inline]
    fn from_index_array(arr: [u32; DIM]) -> GridIndex {
        #[cfg(feature = "dim1")]
        return arr[0];
        #[cfg(not(feature = "dim1"))]
        GridIndex::from_array(arr)
    }

    #[inline]
    fn from_uvec3(uvec3: UVec3) -> GridIndex {
        #[cfg(feature = "dim1")]
        return uvec3.z;
        #[cfg(feature = "dim2")]
        return uvec3.xy();
        #[cfg(feature = "dim3")]
        uvec3
    }

    #[inline]
    #[allow(unused_variables)]
    fn from_flat_idx(idx: Index, n_cells: GridIndex) -> GridIndex {
        #[cfg(feature = "dim1")]
        return idx;
        #[cfg(feature = "dim2")]
        return GridIndex::new(
            idx % n_cells.x,
            idx / n_cells.x,
        );
        #[cfg(feature = "dim3")]
        GridIndex::new(
            idx % n_cells.x,
            (idx / n_cells.x) % n_cells.y,
            idx / (n_cells.x * n_cells.y),
        )
    }

    #[inline]
    #[allow(unused_variables)]
    fn to_flat_idx(self, n_cells: GridIndex) -> u32 {
        #[cfg(feature = "dim1")]
        return self;
        #[cfg(feature = "dim2")]
        return self.y * n_cells.x + self.x;
        #[cfg(feature = "dim3")]
        {
            self.z * n_cells.x * n_cells.y +
                self.y * n_cells.x +
                self.x
        }
    }
}

cfg_select! {
    feature = "dim1" => { impl_vector_value_ext!(GridIndex, Index, 0, 1); }
    _ => { impl_vector_value_ext!(GridIndex, Index, GridIndex::ZERO, GridIndex::ONE); }
}

pub trait BoolVectExt {
    fn any(self) -> bool;
    fn all(self) -> bool;
}

impl BoolVectExt for BoolVect {
    fn any(self) -> bool {
        #[cfg(feature = "dim1")]
        { self }
        #[cfg(not(feature = "dim1"))]
        self.any()
    }

    fn all(self) -> bool {
        #[cfg(feature = "dim1")]
        { self }
        #[cfg(not(feature = "dim1"))]
        self.all()
    }
}

/// A helper enum for indexing into various things (e.g. indexing into components of [`Vect`] and [`GridIndex`])
///
/// NOT designed for passing between CPU and GPU (as denoted by no "repr" attribute)
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(u32)]
pub enum Axis {
    X = 0,
    Y = 1,
    Z = 2,
}

impl Axis {
    pub const ALL_AXES: [Self; MAX_DIM] = [Axis::X, Axis::Y, Axis::Z];
    pub const PERMUTATION: [Self; MAX_DIM] = [Axis::Y, Axis::Z, Axis::X];
    pub const BACK_PERMUTATION: [Self; MAX_DIM] = [Axis::Z, Axis::X, Axis::Y];

    /// Circular permutation of `self` in the following sequence:
    ///
    /// [`Axis::X`] -> [`Axis::Y`] -> [`Axis::Z`] -> [`Axis::X`] -> ...
    #[inline]
    pub const fn permute(&self) -> Self {
        Self::PERMUTATION[*self as usize]
    }


    /// Circular permutation in the reversed direction of [`Axis::permute()`]
    ///
    /// [`Axis::X`] -> [`Axis::Z`] -> [`Axis::Y`] -> [`Axis::X`] -> ...
    #[inline]
    pub const fn backwards_permute(&self) -> Self {
        Self::BACK_PERMUTATION[*self as usize]
    }
}

impl TryFrom<u32> for Axis {
    type Error = ();
    #[inline]
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Axis::X),
            1 => Ok(Axis::Y),
            2 => Ok(Axis::Z),
            _ => Err(()),
        }
    }
}

impl From<SpatialAxis> for Axis {
    #[allow(unused_variables)]
    #[inline]
    fn from(value: SpatialAxis) -> Self {
        match value {
            #[cfg(not(feature = "dim1"))]
            SpatialAxis::X => Axis::X,
            #[cfg(not(feature = "dim1"))]
            SpatialAxis::Y => Axis::Y,
            #[cfg(not(feature = "dim2"))]
            SpatialAxis::Z => Axis::Z,
        }
    }
}

// SAFETY: Axis has a zero variant.
unsafe impl Zeroable for Axis {}
// SAFETY: Axis has u32 representation, and u32 is also POD.
unsafe impl Pod for Axis {}

/// The axes that are in the computational domain. EM waves only propagate in spaces that these axes
/// form (Z axis in 1D; X-Y plane in 2D; X-Y-Z space in 3D).
#[derive(Copy, Clone, Debug)]
#[repr(u32)]
pub enum SpatialAxis {
    #[cfg(any(feature = "dim2", feature = "dim3"))]
    X = 0,
    #[cfg(any(feature = "dim2", feature = "dim3"))]
    Y = 1,
    #[cfg(any(feature = "dim1", feature = "dim3"))]
    Z = if cfg!(feature = "dim1") { 0 } else { 2 },
}

impl SpatialAxis {
    /// All spatial axes in an array of [`DIM`] elements, depending on how many dimensions there are:
    ///
    /// - 1D: Z axis
    /// - 2D: X and Y axes
    /// - 3D: X, Y, and Z axes
    pub const ALL_SPATIAL: [Self; DIM] = cfg_select! {
        feature = "dim1" => [Self::Z],
        feature = "dim2" => [Self::X, Self::Y],
        feature = "dim3" => [Self::X, Self::Y, Self::Z],
    };
    pub const ALL_AXES: [Axis; DIM] = cfg_select! {
        feature = "dim1" => [Axis::Z],
        feature = "dim2" => [Axis::X, Axis::Y],
        feature = "dim3" => [Axis::X, Axis::Y, Axis::Z],
    };

    /// Efficiently check if an [`Axis`] is a spatial axis.
    #[inline]
    pub const fn is_spatial_axis(axis: Axis) -> bool {
        #[cfg(feature = "dim1")]
        { matches!(axis, Axis::Z) }
        #[cfg(not(feature = "dim1"))]
        { (axis as usize) < DIM }
    }
}

impl TryFrom<Axis> for SpatialAxis {
    type Error = ();
    #[inline]
    fn try_from(axis: Axis) -> Result<Self, Self::Error> {
        match axis {
            Axis::X => cfg_select! {
                feature = "dim1" => Err(()),
                _ => Ok(Self::X),
            },
            Axis::Y => cfg_select! {
                feature = "dim1" => Err(()),
                _ => Ok(Self::Y),
            },
            Axis::Z => cfg_select! {
                feature = "dim2" => Err(()),
                _ => Ok(Self::Z),
            }
        }
    }
}

impl TryFrom<u32> for SpatialAxis {
    type Error = ();
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => cfg_select! {
                feature = "dim1" => Ok(Self::Z),
                _ => Ok(Self::X),
            },
            1 => cfg_select! {
                feature = "dim1" => Err(()),
                _ => Ok(Self::Y),
            },
            2 => cfg_select! {
                feature = "dim3" => Ok(Self::Z),
                _ => Err(()),
            },
            _ => Err(())
        }
    }
}

impl Default for SpatialAxis {
    fn default() -> Self {
        Self::try_from(0).unwrap()
    }
}

// SAFETY: SpatialAxis has a zero variant for every dim feature.
unsafe impl Zeroable for SpatialAxis {}
// SAFETY: SpatialAxis has u32 representation, and u32 is also POD.
unsafe impl Pod for SpatialAxis {}

#[derive(Copy, Clone, Debug, PartialEq, Default)]
#[repr(i32)]
pub enum WaveDirection {
    #[default]
    Positive = 1,
    None = 0,
    Negative = -1,
}

// SAFETY: WaveDirection has a zero variant.
unsafe impl Zeroable for WaveDirection {}
// SAFETY: WaveDirection has i32 representation, and i32 is also POD.
unsafe impl Pod for WaveDirection {}

macro_rules! impl_vector_indexing {
    ($v:ident, $elem_ty:ty, $axis_ty:ty, $dims: expr) => {
        impl core::ops::Index<$axis_ty> for $v {
            type Output = $elem_ty;
            #[inline]
            fn index(&self, index: $axis_ty) -> &Self::Output {
                &(unsafe { &*(self as *const $v as *const [$elem_ty; $dims]) } [index as usize])
            }
        }

        impl core::ops::IndexMut<$axis_ty> for $v {
            #[inline]
            fn index_mut(&mut self, index: $axis_ty) -> &mut Self::Output {
                &mut (unsafe { &mut *(self as *mut $v as *mut [$elem_ty; $dims]) } [index as usize])
            }
        }
    };
}

impl_vector_indexing!(Index, Index, Axis, 1);
impl_vector_indexing!(Real, Real, Axis, 1);
impl_vector_indexing!(UVec2, Index, Axis, UVec2::AXES.len());
impl_vector_indexing!(Vec2, Real, Axis, Vec2::AXES.len());
impl_vector_indexing!(UVec3, Index, Axis, UVec3::AXES.len());
impl_vector_indexing!(Vec3, Real, Axis, Vec3::AXES.len());
impl_vector_indexing!(Vec4, Real, Axis, Vec4::AXES.len());

impl_vector_indexing!(Vect, Real, SpatialAxis, DIM);
impl_vector_indexing!(GridIndex, Index, SpatialAxis, DIM);

/// A trait for indexing w/ a dynamic index.
///
///
/// Since SPIR-V's "OpVectorExtractDynamic" / "OpVectorInsertDynamic" isn't available with khal,
/// we have this trait.
pub trait GpuDynamicIndex<Idx: ?Sized> {
    type Output: ?Sized;

    fn dyn_idx(&self, index: Idx) -> Self::Output;
    fn dyn_insert(&mut self, index: Idx, val: Self::Output);
}

macro_rules! impl_vector_gpu_indexing_1d {
    ($v:ident, $elem_ty:ty, $axis_ty:ty) => {
        impl GpuDynamicIndex<$axis_ty> for $v {
            type Output = $elem_ty;
            
            #[inline]
            fn dyn_idx(&self, index: $axis_ty) -> Self::Output {
                cfg_select! {
                    target_arch = "spirv" => {
                        *self
                    }
                    _ => {
                        (&unsafe { &*(self as *const $v as *const [Self::Output; 1]) }) [index as usize]
                    }
                }
            }
            
            #[inline]
            fn dyn_insert(&mut self, index: $axis_ty, val: Self::Output) {
                cfg_select! {
                    target_arch = "spirv" => {
                        *self = val;
                    }
                    _ => {
                        (&mut unsafe { &mut *(self as *mut $v as *mut [Self::Output; 1]) }) [index as usize] = val;
                    }
                }
            }
        }
    };
}

impl_vector_gpu_indexing_1d!(Index, Index, Axis);
impl_vector_gpu_indexing_1d!(Real, Real, Axis);

macro_rules! impl_vector_gpu_indexing {
    ($v:ident, $elem_ty:ty, $axis_ty:ty, $dims: expr) => {
        impl GpuDynamicIndex<$axis_ty> for $v {
            type Output = $elem_ty;

            #[inline]
            fn dyn_idx(&self, index: $axis_ty) -> Self::Output {
                cfg_select! {
                    target_arch = "spirv" => {
                        unsafe { spirv_std::arch::vector_extract_dynamic(*self, index as usize) }
                    }
                    _ => {
                        (&unsafe { &*(self as *const $v as *const [Self::Output; $dims]) }) [index as usize]
                    }
                }
            }

            #[inline]
            fn dyn_insert(&mut self, index: $axis_ty, val: Self::Output) {
                cfg_select! {
                    target_arch = "spirv" => {
                        *self = unsafe { spirv_std::arch::vector_insert_dynamic(*self, index as usize, val) };
                    }
                    _ => {
                        (&mut unsafe { &mut *(self as *mut $v as *mut [Self::Output; $dims]) }) [index as usize] = val;
                    }
                }
            }
        }
    };
}

impl_vector_gpu_indexing!(UVec2, Index, Axis, 2);
impl_vector_gpu_indexing!(Vec2, Real, Axis, 2);
impl_vector_gpu_indexing!(UVec3, Index, Axis, 3);
impl_vector_gpu_indexing!(Vec3, Real, Axis, 3);
impl_vector_gpu_indexing!(Vec4, Real, Axis, 4);

cfg_select! {
    feature = "dim1" => {
        impl_vector_gpu_indexing_1d!(Vect, Real, SpatialAxis);
        impl_vector_gpu_indexing_1d!(GridIndex, Index, SpatialAxis);
    }
    _ => {
        impl_vector_gpu_indexing!(Vect, Real, SpatialAxis, DIM);
        impl_vector_gpu_indexing!(GridIndex, Index, SpatialAxis, DIM);
    }
}

/// A trait that computes `a.saturating_sub(b)` on the GPU.
/// Rust-GPU doesn't have the core library's `saturating_sub()` implemented yet, so we have this trait.
pub trait GpuSaturatingSub {
    fn gpu_saturating_sub(self, rhs: Self) -> Self;
}

impl GpuSaturatingSub for u32 {
    #[inline]
    fn gpu_saturating_sub(self, rhs: Self) -> Self {
        if self > rhs { self.wrapping_sub(rhs) } else { 0 }
    }
}

impl GpuSaturatingSub for usize {
    #[inline]
    fn gpu_saturating_sub(self, rhs: Self) -> Self {
        if self > rhs { self.wrapping_sub(rhs) } else { 0 }
    }
}