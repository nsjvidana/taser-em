use crate::ElectricMaterial;
use glamx::Vec3;
use parry3d::bounding_volume::{Aabb, BoundingVolume};
use parry3d::math::Pose;
use parry3d::shape::{Cuboid, SharedShape};
use taser_em_shaders::fdtd1::PmlCoefficients;
use taser_em_shaders::math::{vec3_to_vect, GridIndex, Vect, VectExt, DIM};

pub struct MaterialRegions {
    pub regions: Vec<(SharedShape, Pose, ElectricMaterial)>,
}

impl MaterialRegions {
    pub fn fill_region(
        &mut self,
        start: Vect,
        end: Vect,
        material: ElectricMaterial
    ) -> &mut Self {
        let half_extents = (end - start).to_3d(Vec3::ONE);
        let shape = SharedShape::new(Cuboid::new(half_extents));
        let middle = ((start + end) / 2.).to_3d(Vec3::ZERO);
        let pose = Pose::from_translation(middle);

        self.regions.push((shape, pose, material));
        self
    }

    pub fn compute_bounding_box(&self) -> Aabb {
        let aabbs = self.regions
            .iter()
            .map(|(s, pose, _)| s.compute_aabb(pose))
            .collect::<Vec<_>>();

        let mut full_bb = Aabb::new_invalid();
        for bb in aabbs.iter() {
            full_bb.merge(bb);
        }
        full_bb
    }
}

pub struct YeeGrid {
    /// Relative Permeability (located at H field components)
    pub mu_r: Vec<[Vec3; DIM]>,
    /// Relative Permittivity (located at E field components)
    pub eps_r: Vec<[Vec3; DIM]>,
    /// Number of grid cells (in each principal direction)
    pub n_cells: GridIndex,
    /// Size of each cell (in meters)
    pub cell_size: Vect,
}

impl YeeGrid {
    pub fn empty_from_regions(regions: &MaterialRegions, cell_size: Vect) -> Self {
        let bb = regions.compute_bounding_box();
        let n_cells = vec3_to_vect((bb.extents() / cell_size.to_3d(Vec3::ONE)).ceil())
            .to_grid_index();

        Self {
            mu_r: vec![],
            eps_r: vec![],
            n_cells,
            cell_size,
        }
    }

    pub fn update_coeffs_pml(&self, _pml_dims: LayerDimensions, _n_cells: GridIndex) -> Vec<PmlCoefficients> {
        todo!()
    }

    /// Clears all stored material properties
    pub fn clear(&mut self) {
        self.mu_r.clear();
        self.eps_r.clear();
    }

    // pub fn shape_in_grid(shape: SharedShape, pose: Pose, material: ElectricMaterial)

    // normal map creation?
}

/// The widths of layers of cells at the boundary of the simulation. (e.g. PMLs)
///
/// Stores the "low" and "high" widths on each axis.
#[derive(Copy, Clone)]
pub struct LayerDimensions {
    pub widths: [[u32; 2]; DIM],
}

impl LayerDimensions {
    #[inline]
    #[cfg(feature = "dim1")]
    pub fn new(z_lo: u32, z_hi: u32) -> Self {
        Self {
            widths: [[z_lo, z_hi]]
        }
    }

    #[inline]
    #[cfg(feature = "dim2")]
    pub fn new(x_lo: u32, x_hi: u32, y_lo: u32, y_hi: u32) -> Self {
        Self {
            widths: [[x_lo, x_hi], [y_lo, y_hi]]
        }
    }

    #[inline]
    #[cfg(feature = "dim3")]
    pub fn new(x_lo: u32, x_hi: u32, y_lo: u32, y_hi: u32, z_lo: u32, z_hi: u32) -> Self {
        Self {
            widths: [[x_lo, x_hi], [y_lo, y_hi], [z_lo, z_hi]]
        }
    }

    #[inline]
    pub fn iter_axes(&self) -> impl Iterator<Item=(AxisIndex, &[u32; 2])> {
        self.widths.iter()
            .enumerate()
            .map(|(axis, lo_hi)| (AxisIndex::try_from(axis).unwrap(), lo_hi))
    }

    #[inline]
    pub fn iter_axes_mut(&mut self) -> impl Iterator<Item=(AxisIndex, &mut [u32; 2])> {
        self.widths.iter_mut()
            .enumerate()
            .map(|(axis, lo_hi)| (AxisIndex::try_from(axis).unwrap(), lo_hi))
    }
}

impl core::ops::Add for LayerDimensions {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self::Output {
        for ([lo1, hi1], [lo2, hi2]) in self.widths.iter_mut()
            .zip(rhs.widths.iter())
        {
            *lo1 += lo2;
            *hi1 += hi2;
        }
        self
    }
}

impl core::ops::Index<AxisIndex> for LayerDimensions {
    type Output = [u32; 2];
    fn index(&self, index: AxisIndex) -> &Self::Output {
        &self.widths[index as usize]
    }
}

impl core::ops::IndexMut<AxisIndex> for LayerDimensions {
    fn index_mut(&mut self, index: AxisIndex) -> &mut Self::Output {
        &mut self.widths[index as usize]
    }
}

#[repr(usize)]
#[cfg(any(feature = "dim2", feature = "dim3"))]
pub enum AxisIndex {
    X = 0,
    Y = 1,
    Z = 2,
}
#[cfg(feature = "dim1")]
pub enum AxisIndex {
    Z = 0,
}

impl TryFrom<usize> for AxisIndex {
    type Error = ();
    fn try_from(value: usize) -> Result<Self, Self::Error> {
        #[cfg(any(feature = "dim2", feature = "dim3"))]
        match value {
            0 => Ok(AxisIndex::X),
            1 => Ok(AxisIndex::Y),
            2 => Ok(AxisIndex::Z),
            _ => Err(()),
        }

        #[cfg(feature = "dim1")]
        if value == 0 {
            Ok(AxisIndex::Z)
        }
        else {
            Err(())
        }
    }
}