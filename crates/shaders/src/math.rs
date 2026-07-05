use khal_std::glamx::{UVec3, Vec3, Vec4};
pub use dim_types::*;

pub type Real = f32;
pub type Index = u32;

pub const MAX_DIM: usize = 3;

#[cfg(feature = "dim1")]
mod dim_types {
    pub const DIM: usize = 1;
    /// A shader's vector field element (a Z component in 1 dimension)
    pub type Vect = f32;
    pub type GridIndex = u32;
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

#[allow(unused_imports)]
use khal_std::glamx::Vec3Swizzles;

/// Converts a [`Vect`] to [`Vec3`]
/// `mask` sets the components that `self` don't already have.
///
/// e.g. `to_3d(z, Vec3::ONE)` would return `Vec3::new(1., 1., z)` in 1D.
#[inline]
#[allow(unused_variables)]
pub fn to_3d(v: Vect, mask: Vec3) -> Vec3 {
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
pub fn vec4_to_vect(v: Vec4) -> Vect {
    #[cfg(feature = "dim1")]
    return v.z;
    #[cfg(feature = "dim2")]
    {
        use khal_std::glamx::Vec4Swizzles;
        return v.xy();
    }
    #[cfg(feature = "dim3")]
    {
        use khal_std::glamx::Vec4Swizzles;
        return v.xyz();
    }
}

/// Converts a [`GridIndex`] to [`UVec3`]
/// `mask` sets the components that `self` don't already have.
///
/// e.g. `grid_index_to_3d(k, UVec3::ONE)` would return `UVec3::new(1, 1, k)` in 1D.
#[inline]
pub fn grid_index_to_3d(idx: GridIndex, mask: UVec3) -> UVec3 {
    #[cfg(feature = "dim1")]
    return mask.with_x(idx);
    #[cfg(feature = "dim2")]
    return mask.with_xy(idx);
    #[cfg(feature = "dim3")]
    idx
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
pub fn vec3_to_vect(vec3: Vec3) -> Vect {
    #[cfg(feature = "dim1")]
    return vec3.z;
    #[cfg(feature = "dim2")]
    return vec3.xy();
    #[cfg(feature = "dim3")]
    vec3
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