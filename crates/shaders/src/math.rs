use khal_std::glamx::Vec3;
pub use dim_types::*;

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
    pub const DIM: usize = 3;
    /// A shader's vector field element
    pub type Vect = Vec3;
    pub type GridIndex = UVec3;
}

#[allow(unused_imports)]
use khal_std::glamx::Vec3Swizzles;

pub trait VectExt {
    fn to_3d(self, mask: Vec3) -> Vec3;
    fn to_grid_index(self) -> GridIndex;
}

impl VectExt for Vect {
    /// Converts a [`Vect`] to [`Vec3`]
    /// `mask` sets the components that `self` don't already have.
    ///
    /// e.g. `MathW<f32>::to_3d()` would return `Vec3::new(mask.x, mask.y, self.0)`
    #[inline]
    #[allow(unused_variables)]
    fn to_3d(self, mask: Vec3) -> Vec3 {
        #[cfg(feature = "dim1")]
        return mask.with_x(self);
        #[cfg(feature = "dim2")]
        return mask.with_xy(self);
        #[cfg(feature = "dim3")]
        self
    }

    /// Convert [`Vect`] to [`GridIndex`] using `as`
    #[inline]
    fn to_grid_index(self) -> GridIndex {
        #[cfg(feature = "dim1")]
        return self as GridIndex;
        #[cfg(feature = "dim2")]
        return self.as_uvec2();
        #[cfg(feature = "dim3")]
        self.as_uvec3()
    }
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