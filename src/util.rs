use taser_em_shaders::math::{GridIndex, Index, Real, Vect};

pub fn vect_to_iter(v: Vect) -> impl Iterator<Item = Real> {
    #[cfg(feature = "dim1")]
    return std::iter::once(v);
    #[cfg(not(feature = "dim1"))]
    v.to_array().into_iter()
}

fn grid_idx_to_iter(idx: GridIndex) -> impl Iterator<Item = Index> {
    #[cfg(feature = "dim1")]
    return std::iter::once(idx);
    #[cfg(not(feature = "dim1"))]
    idx.to_array().into_iter()
}