pub use dim_math::*;

#[cfg(feature = "dim1")]
mod dim_math {
    pub const DIM: usize = 1;
    /// Vector of a vector field (a Z component in 1 dimension)
    pub type Vect = f32;
    pub type GridIndex = u32;
}

#[cfg(feature = "dim2")]
mod dim_math {
    use khal_std::glamx::{UVec2, Vec2};

    pub const DIM: usize = 2;
    /// Vector of a vector field
    pub type Vect = Vec2;
    pub type GridIndex = UVec2;
}

#[cfg(feature = "dim3")]
mod dim_math {
    use khal_std::glamx::{UVec3, Vec3};
    pub const DIM: usize = 3;
    /// Vector of a vector field
    pub type Vect = Vec3;
    pub type GridIndex = UVec3;
}