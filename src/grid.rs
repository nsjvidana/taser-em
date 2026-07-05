use crate::{ElectricMaterial, EPS_0};
use glamx::Vec3;
use parry3d::bounding_volume::{Aabb, BoundingVolume};
use parry3d::math::Pose;
use parry3d::shape::{Cuboid, SharedShape};
use khal::re_exports::bytemuck::{Pod, Zeroable};
use taser_em_shaders::fdtd1::PmlCoefficients;
use taser_em_shaders::math::{flat_idx_to_grid_index, grid_index_from_array, grid_index_to_array, to_3d, to_grid_index, vec3_to_vect, GridIndex, Real, Vect, DIM};

/// Information describing a 1-, 2-, or 3-D Yee Grid with E and H fields staggered by half a cell.
pub struct YeeGrid {
    /// Number of grid cells, in each principal direction, excluding spacer regions
    ///
    /// For the full dimensions of the entire computational domain, use [`YeeGrid::n_cells()`]
    pub inner_n_cells: GridIndex,
    /// Adds optional regions at the edges of the simulation. Good for preventing
    /// boundary conditions from affecting things like evanescent fields.
    ///
    /// The material at these regions are `background_material` no matter what.
    pub spacer_region_widths: LayerWidths,
    /// The "default" material in grid cells whose material isn't
    /// explicitly set by the user. (e.g. free space)
    pub background_material: ElectricMaterial,
    /// Size of each cell (in meters)
    pub cell_size: Vect,
}

impl YeeGrid {
    pub fn empty_from_regions(regions: &MaterialRegions, cell_size: Vect) -> Self {
        let bb = regions.compute_bounding_box();
        let n_cells_vec3 = (bb.extents() / to_3d(cell_size, Vec3::ONE)).ceil();
        let n_cells = to_grid_index(vec3_to_vect(n_cells_vec3));

        Self {
            inner_n_cells: n_cells,
            background_material: ElectricMaterial::FREE_SPACE,
            spacer_region_widths: LayerWidths::default(),
            cell_size,
        }
    }

    #[inline]
    pub fn add_spacer_region_offset(&mut self, offset: LayerWidths) -> &mut Self {
        self.spacer_region_widths += offset;
        self
    }

    /// Computes the total number of cells, in each principal direction,
    /// accounting for spacer regions as well.
    pub fn n_cells(&self) -> GridIndex {
        let mut n_cells = grid_index_to_array(self.inner_n_cells);
        for (n_cells_i, (_axis, lo_hi_widths)) in n_cells.iter_mut()
            .zip(self.spacer_region_widths.iter_axes())
        {
            *n_cells_i += lo_hi_widths.iter().sum::<u32>();
        }
        grid_index_from_array(n_cells)
    }

    pub fn update_coeffs_pml(&mut self, pml_dims: LayerWidths, dt: Real) -> Vec<PmlCoefficients> {
        todo!()
    }

    /// Clears all stored material properties
    pub fn reset(&mut self) {
        todo!()
    }

    // pub fn shape_in_grid(shape: SharedShape, pose: Pose, material: ElectricMaterial)

    // normal map creation?
}

#[derive(Default)]
pub struct MaterialRegions {
    pub regions: Vec<(SharedShape, Pose, ElectricMaterial)>,
}

impl MaterialRegions {
    pub fn new() -> Self { Self::default() }

    pub fn fill_region(
        &mut self,
        start: Vect,
        end: Vect,
        material: ElectricMaterial
    ) -> &mut Self {
        let half_extents = to_3d(end - start, Vec3::ONE);
        let shape = SharedShape::new(Cuboid::new(half_extents));
        let middle = to_3d((start + end) / 2., Vec3::ZERO);
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

/// The widths of layers of cells at the boundary of the simulation. (e.g. PMLs)
///
/// Stores the "low" and "high" widths on each axis.
#[derive(Copy, Clone, Default)]
pub struct LayerWidths {
    pub widths: [[u32; 2]; DIM],
}

impl LayerWidths {
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
    pub fn splat(width: u32) -> Self {
        #[cfg(feature = "dim1")]
        return Self::new(width, width);
        #[cfg(feature = "dim2")]
        return Self::new(width, width, width, width);
        #[cfg(feature = "dim3")]
        Self::new(width, width, width, width, width, width)
    }

    #[inline]
    pub fn iter_axes(&self) -> impl Iterator<Item = (AxisIndex, &[u32; 2])> {
        self.widths.iter()
            .enumerate()
            .map(|(axis, lo_hi)| (AxisIndex::try_from(axis).unwrap(), lo_hi))
    }

    #[inline]
    pub fn iter_axes_mut(&mut self) -> impl Iterator<Item = (AxisIndex, &mut [u32; 2])> {
        self.widths.iter_mut()
            .enumerate()
            .map(|(axis, lo_hi)| (AxisIndex::try_from(axis).unwrap(), lo_hi))
    }
}

impl core::ops::AddAssign for LayerWidths {
    fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}

impl core::ops::Add for LayerWidths {
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

impl core::ops::SubAssign for LayerWidths {
    fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; }
}

impl core::ops::Sub for LayerWidths {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self::Output {
        for ([lo1, hi1], [lo2, hi2]) in self.widths.iter_mut()
            .zip(rhs.widths.iter())
        {
            *lo1 -= lo2;
            *hi1 -= hi2;
        }
        self
    }
}

impl core::ops::Index<AxisIndex> for LayerWidths {
    type Output = [u32; 2];
    fn index(&self, index: AxisIndex) -> &Self::Output {
        &self.widths[index.into_dim_idx()]
    }
}

impl core::ops::IndexMut<AxisIndex> for LayerWidths {
    fn index_mut(&mut self, index: AxisIndex) -> &mut Self::Output {
        &mut self.widths[index.into_dim_idx()]
    }
}

#[derive(Copy, Clone)]
#[repr(usize)]
pub enum AxisIndex {
    X = 0,
    Y = 1,
    Z = 2,
}

impl AxisIndex {
    /// Convert to dimension index. [`AxisIndex::Z`] is `0` in 1 dimension.
    ///
    /// Returns an invalid index that can cause out-of-bounds access errors (e.g. [`usize::MAX`])
    /// if the dimension doesn't exist.
    pub fn into_dim_idx(self) -> usize {
        #[cfg(feature = "dim1")]
        match self {
            AxisIndex::Z => 0,
            _ => usize::MAX // an invalid dimension index
        }
        #[cfg(not(feature = "dim1"))]
        {
            self as usize
        }
    }
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
        if value == 0 { Ok(AxisIndex::Z) }
        else { Err(()) }
    }
}