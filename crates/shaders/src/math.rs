use khal_std::glamx::{UVec2, UVec3, Vec2, Vec3, Vec4};
pub use dim_types::*;

pub type Real = f32;
pub type Index = u32;

pub const MAX_DIM: usize = 3;

#[cfg(feature = "dim1")]
mod dim_types {
    use crate::math::{Index, Real};

    pub const DIM: usize = 1;
    /// A shader's vector field element (a Z component in 1 dimension)
    pub type Vect = Real;
    pub type GridIndex = Index;
}

#[cfg(feature = "dim2")]
mod dim_types {
    use khal_std::glamx::{UVec2, Vec2};

    pub const DIM: usize = 2;
    /// A shader's vector field element
    pub type Vect = Vec2;
    pub type GridIndex = UVec2;
}

#[cfg(feature = "dim3")]
mod dim_types {
    use khal_std::glamx::{UVec3, Vec3};
    pub const DIM: usize = super::MAX_DIM;
    /// A shader's vector field element
    pub type Vect = Vec3;
    pub type GridIndex = UVec3;
}

/// A helper enum for indexing into various things (e.g. indexing into components of [`Vect`] and [`GridIndex`])
///
/// NOT designed for passing between CPU and GPU (as denoted by no "repr" attribute)
#[derive(Copy, Clone, Debug)]
#[repr(u32)]
pub enum Axis {
    X = 0,
    Y = 1,
    Z = 2,
}

impl Axis {
    pub const ALL_AXES: [Self; MAX_DIM] = [Axis::X, Axis::Y, Axis::Z];
    /// Circular permutation of `self` in the following sequence:
    ///
    /// [`Axis::X`] -> [`Axis::Y`] -> [`Axis::Z`] -> [`Axis::X`] -> ...
    #[inline]
    pub fn permute(&self) -> Self {
        match self {
            Axis::X => Axis::Y,
            Axis::Y => Axis::Z,
            Axis::Z => Axis::X,
        }
    }

    /// Transmutes `idx` into an [`Axis`]
    /// 
    /// # Safety
    /// Will cause undefined behavior if `idx` isn't `0`, `1`, or `2`.
    /// [`Axis::try_from()`] is a safer alternative to this function.
    #[inline]
    pub unsafe fn from_index_unchecked(idx: u32) -> Self {
        unsafe { core::mem::transmute::<u32, Self>(idx) }
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
        #[cfg(feature = "dim1")]
        { Axis::Z }
        #[cfg(not(feature = "dim1"))]
        unsafe { core::mem::transmute::<SpatialAxis, Axis>(value) }
    }
}

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
    /// All spatial axes in an array of [`DIM`] elements.
    pub const ALL_AXES: [Self; DIM] = cfg_select! {
        feature = "dim1" => [Self::Z],
        feature = "dim2" => [Self::X, Self::Y],
        feature = "dim3" => [Self::X, Self::Y, Self::Z],
    };

    /// Efficiently check if an [`Axis`] is a spatial axis.
    #[inline]
    pub fn is_spatial_axis(axis: Axis) -> bool {
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
        #[cfg(feature = "dim1")]
        match axis {
            Axis::Z => Ok(SpatialAxis::Z),
            _ => Err(())
        }
        #[cfg(not(feature = "dim1"))]
        match axis {
            Axis::Z => cfg_select! {
                feature = "dim2" => Err(()),
                feature = "dim3" => Ok(SpatialAxis::Z),
            },
            _ => unsafe { Ok(core::mem::transmute::<Axis, SpatialAxis>(axis)) }
        }
    }
}

impl TryFrom<u32> for SpatialAxis {
    type Error = ();
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            v if v < DIM as u32 => unsafe { Ok(core::mem::transmute::<u32, SpatialAxis>(value)) },
            _ => Err(())
        }
    }
}

macro_rules! impl_vector_indexing {
    ($v:ident, $elem_ty:ty, $axis_ty: ty, $n_dims: expr) => {
        impl core::ops::Index<$axis_ty> for $v {
            type Output = $elem_ty;
            #[inline]
            fn index(&self, index: $axis_ty) -> &Self::Output {
                &(unsafe { &*(self as *const $v as *const [$elem_ty; $n_dims]) } [index as usize])
            }
        }

        impl core::ops::IndexMut<$axis_ty> for $v {
            #[inline]
            fn index_mut(&mut self, index: $axis_ty) -> &mut Self::Output {
                &mut (unsafe { &mut *(self as *mut $v as *mut [$elem_ty; $n_dims]) } [index as usize])
            }
        }
    };
}

impl_vector_indexing!(Index, Index, Axis, 1);
impl_vector_indexing!(Real, Real, Axis, 1);
impl_vector_indexing!(UVec2, Index, Axis, 2);
impl_vector_indexing!(Vec2, Real, Axis, 2);
impl_vector_indexing!(UVec3, Index, Axis, 3);
impl_vector_indexing!(Vec3, Real, Axis, 3);
impl_vector_indexing!(Vec4, Real, Axis, 4);

impl_vector_indexing!(Vect, Real, SpatialAxis, DIM);
impl_vector_indexing!(GridIndex, Index, SpatialAxis, DIM);

#[allow(unused_imports)]
use khal_std::glamx::Vec3Swizzles;

/// Converts a [`Vect`] to [`Vec3`]
/// `mask` sets the components that `self` don't already have.
///
/// e.g. `to_3d(z, Vec3::ONE)` would return `Vec3::new(1., 1., z)` in 1D.
#[inline]
#[allow(unused_variables)]
pub fn vect_to_3d(v: Vect, mask: Vec3) -> Vec3 {
    #[cfg(feature = "dim1")]
    return mask.with_x(v);
    #[cfg(feature = "dim2")]
    return mask.with_xy(v);
    #[cfg(feature = "dim3")]
    v
}

#[inline]
pub fn vect_to_array(v: Vect) -> [Real; DIM] {
    #[cfg(feature = "dim1")]
    return [v];
    #[cfg(not(feature = "dim1"))]
    v.to_array()
}

#[inline]
pub fn vect_from_array(arr: [Real; DIM]) -> Vect {
    #[cfg(feature = "dim1")]
    return arr[0];
    #[cfg(not(feature = "dim1"))]
    Vect::from_array(arr)
}

#[inline]
pub fn vec3_to_vect(vec3: Vec3) -> Vect {
    #[cfg(feature = "dim1")]
    return vec3.z;
    #[cfg(feature = "dim2")]
    return vec3.xy();
    #[cfg(feature = "dim3")]
    vec3
}

#[inline]
pub fn vec4_to_vect(v: Vec4) -> Vect {
    #[cfg(feature = "dim1")]
    { v.z }
    #[cfg(feature = "dim2")] {
        use khal_std::glamx::Vec4Swizzles;
        v.xy()
    }
    #[cfg(feature = "dim3")] {
        use khal_std::glamx::Vec4Swizzles;
        v.xyz()
    }
}

/// Convert [`Vect`] to [`GridIndex`] using `as` keyword
#[inline]
pub fn to_grid_index(v: Vect) -> GridIndex {
    #[cfg(feature = "dim1")]
    return v as GridIndex;
    #[cfg(feature = "dim2")]
    return v.as_uvec2();
    #[cfg(feature = "dim3")]
    v.as_uvec3()
}

#[inline]
pub fn grid_index_as_vect(idx: GridIndex) -> Vect {
    #[cfg(feature = "dim1")]
    return idx as Vect;
    #[cfg(feature = "dim2")]
    return idx.as_vec2();
    #[cfg(feature = "dim3")]
    idx.as_vec3()
}

/// Converts a [`GridIndex`] to [`UVec3`]
/// `mask` sets the components that `self` don't already have.
///
/// e.g. `grid_index_to_3d(k, UVec3::ONE)` would return `UVec3::new(1, 1, k)` in 1D.
#[inline]
#[allow(unused_variables)]
pub fn grid_index_to_3d(idx: GridIndex, mask: UVec3) -> UVec3 {
    #[cfg(feature = "dim1")]
    return mask.with_x(idx);
    #[cfg(feature = "dim2")]
    return mask.with_xy(idx);
    #[cfg(feature = "dim3")]
    idx
}

#[inline]
pub fn grid_index_to_array(idx: GridIndex) -> [u32; DIM] {
    #[cfg(feature = "dim1")]
    return [idx];
    #[cfg(not(feature = "dim1"))]
    idx.to_array()
}

#[inline]
pub fn grid_index_from_array(arr: [u32; DIM]) -> GridIndex {
    #[cfg(feature = "dim1")]
    return arr[0];
    #[cfg(not(feature = "dim1"))]
    GridIndex::from_array(arr)
}

#[inline]
pub fn uvec3_to_grid_index(uvec3: UVec3) -> GridIndex {
    #[cfg(feature = "dim1")]
    return uvec3.z;
    #[cfg(feature = "dim2")]
    return uvec3.xy();
    #[cfg(feature = "dim3")]
    uvec3
}

#[inline]
#[allow(unused_variables)]
pub fn flat_idx_to_grid_index(idx: u32, grid_dim: GridIndex) -> GridIndex {
    #[cfg(feature = "dim1")]
    return idx;
    #[cfg(feature = "dim2")]
    return GridIndex::new(
        idx % grid_dim.x,
        (idx / grid_dim.x) % grid_dim.y,
    );
    #[cfg(feature = "dim3")]
    GridIndex::new(
        idx % grid_dim.x,
        (idx / grid_dim.x) % grid_dim.y,
        idx / (grid_dim.x * grid_dim.y),
    )
}

#[inline]
#[allow(unused_variables)]
pub fn grid_index_to_flat_idx(grid_idx: GridIndex, grid_dim: GridIndex) -> u32 {
    #[cfg(feature = "dim1")]
    return grid_idx;
    #[cfg(feature = "dim2")]
    return grid_idx.y * grid_dim.x + grid_idx.x;
    #[cfg(feature = "dim3")]
    {
        grid_idx.z * grid_dim.x * grid_dim.y +
            grid_idx.y * grid_dim.x +
            grid_idx.x
    }
}

/// A saturating_sub implementation that computes `a - b`.
/// Because Rust-GPU doesn't have the core library's saturating_sub() implemented yet, we have this function.
#[inline]
pub fn saturating_sub(a: usize, b: usize) -> usize {
    if a > b { a.wrapping_sub(b) } else { 0 }
}