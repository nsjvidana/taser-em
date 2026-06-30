use khal_std::glamx::Vec3;
pub use dim_types::*;

/// Wrapper type for math utilities
pub struct MathW<T>(pub T);

impl<T> core::ops::Deref for MathW<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<T> core::ops::DerefMut for MathW<T> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

/// Trait for converting `Vect` to `Vec3`
pub trait To3D {
    /// `mask` sets the components that `self` don't already have.
    ///
    /// e.g. `MathW<f32>::to_3d()` would return `Vec3::new(mask.x, mask.y, self.0)`
    #[inline]
    fn to_3d(self, mask: Vec3) -> Vec3;
}

#[cfg(feature = "dim1")]
mod dim_types {
    use khal_std::glamx::Vec3;
    use crate::math::{MathW, To3D};

    pub const DIM: usize = 1;
    /// A shader's vector field element (a Z component in 1 dimension)
    pub type Vect = f32;
    pub type GridIndex = u32;

    impl To3D for MathW<Vect> {
        #[inline]
        fn to_3d(self, mask: Vec3) -> Vec3 { mask.with_x(self.0) }
    }
}

#[cfg(feature = "dim2")]
mod dim_types {
    use khal_std::glamx::{UVec2, Vec2, Vec3, Vec3Swizzles};

    pub const DIM: usize = 2;
    /// A shader's vector field element
    pub type Vect = Vec2;
    pub type GridIndex = UVec2;


    impl crate::math::To3D for crate::math::MathW<Vect> {
        #[inline]
        fn to_3d(self, mask: Vec3) -> Vec3 { mask.with_xy(self.0) }
    }
}

#[cfg(feature = "dim3")]
mod dim_types {
    use khal_std::glamx::{UVec3, Vec3};
    pub const DIM: usize = 3;
    /// A shader's vector field element
    pub type Vect = Vec3;
    pub type GridIndex = UVec3;

    impl crate::math::To3D for crate::math::MathW<Vect> {
        #[inline]
        fn to_3d(self, _mask: Vec3) -> Vec3 { self.0 }
    }
}