use crate::{grid_cells_iter, ElectricMaterial, PmlParameters, C_0, EPS_0};
use glamx::{Pose3, UVec3, Vec3};
use parry3d::bounding_volume::{Aabb, BoundingVolume};
use parry3d::math::Pose;
use parry3d::shape::{Cuboid, SharedShape};
use std::num::NonZeroU32;
use taser_em_shaders::fdtd::PmlCoefficients2;
use taser_em_shaders::math::{flat_idx_to_grid_index, grid_index_as_vect, grid_index_from_array, grid_index_to_3d, grid_index_to_array, grid_index_to_flat_idx, to_grid_index, vec3_to_vect, vect_to_3d, Axis, GridIndex, Index, Real, SpatialAxis, Vect, DIM, MAX_DIM};

/// Information describing a 1-, 2-, or 3-D Yee Grid with E and H fields staggered by half a cell.
pub struct YeeGrid {
    /// Size of each cell (in meters)
    pub cell_size: Vect,
    /// Which vector components will exist in the computational domain. See [`PolarizationMode`] for
    /// more info.
    pub polarization_mode: PolarizationMode,
    /// Objects/devices within the simulation, stored as raw shapes.
    /// These shapes get pixelized/voxelized before running the simulation.
    pub material_regions: MaterialRegions,
    /// Resolution at which material regions will be smoothed
    pub material_resolution: NonZeroU32,
    /// The "default" material in grid cells whose material isn't
    /// explicitly set by the user. (e.g. free space)
    pub background_material: ElectricMaterial,
    /// Adds optional regions at the edges of the simulation. Good for preventing
    /// boundary conditions from affecting things like evanescent fields.
    ///
    /// The material at these regions are `background_material` no matter what.
    pub spacer_region_widths: LayerWidths,
}

impl YeeGrid {
    pub fn new(
        cell_size: Vect,
        polarization_mode: PolarizationMode,
        material_regions: MaterialRegions,
        material_resolution: NonZeroU32,
    ) -> Self {
        Self {
            cell_size,
            polarization_mode,
            material_regions,
            material_resolution,
            background_material: ElectricMaterial::FREE_SPACE,
            spacer_region_widths: LayerWidths::default(),
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
        let bb = self.material_regions.compute_bounding_box();
        let n_cells_vec3 = (bb.extents() / vect_to_3d(self.cell_size, Vec3::ONE)).ceil();
        let materials_n_cells = to_grid_index(vec3_to_vect(n_cells_vec3));

        self.spacer_region_widths
            .sum_with_n_cells(materials_n_cells)
    }

    /// Voxelize material regions and compute PML update coefficients.
    pub fn update_coeffs_pml(
        &self,
        pml_parameters: PmlParameters,
        scene_transform: Pose3,
        dt: Real
    ) -> (GridIndex, Vec<PmlCoefficients2>) {
        todo!()
    }

    /// Resets all stored material data
    pub fn reset(&mut self) {
        self.material_regions.regions.clear();
        self.background_material = ElectricMaterial::FREE_SPACE;
    }

    // normal map creation?
}

/// Polarization mode affects which field components are computed in the simulation, depending on how many spatial dimensions there are.
/// In 3D, all axes are computed for all fields.
/// See the docs of this enum's variants to know which axes are enabled for each mode.
///
/// A little pedantic note: The word "polarization" here actually refers to how a specific vector field has to be transverse to the
/// simulation domain in 2D FDTD, so the word makes less sense in 1D and 3D contexts, but it's used here anyway for simplicity.
#[derive(Copy, Clone, Debug)]
#[repr(u32)]
pub enum PolarizationMode {
    /// 1D: Ey, Hx
    /// 2D: Ex, Ey, Hz
    /// 3D: All axes
    TransverseMagnetic = 0,
    /// 1D: Ex, Hy
    /// 2D: Ez, Hx, Hy
    /// 3D: All axes
    TransverseElectric = 1,
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
        let half_extents = vect_to_3d(end - start, Vec3::ONE);
        let shape = SharedShape::new(Cuboid::new(half_extents));
        let middle = vect_to_3d((start + end) / 2., Vec3::ZERO);
        let pose = Pose::from_translation(middle);

        self.regions.push((shape, pose, material));
        self
    }

    pub fn import_shape_from_file(&mut self) -> &mut Self {
        todo!()
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

    /// Compute a material grid with the dimensions of `n_cells` where all material regions
    /// are centered at the middle of the grid.
    ///
    /// `obj_offset` gets multiplied to each region individually before being voxelized.
    /// Voxels whose origin don't intersect with any region are assigned `default_mat`
    pub fn compute_material_grid(
        &self,
        n_cells: GridIndex,
        cell_size: Vect,
        default_mat: ElectricMaterial,
        obj_offset: Pose,
    ) -> Vec<ElectricMaterial> {
        let cell_count = grid_index_to_array(n_cells).iter().product::<Index>()
            as usize;
        let mut mats = vec![ElectricMaterial::FREE_SPACE; cell_count];

        let grid_center = vect_to_3d(
            grid_index_as_vect(n_cells) * cell_size / 2., Vec3::ZERO
        );
        let regions_center = self.compute_bounding_box().center();
        let offset = obj_offset.append_translation(grid_center - regions_center);

        // TODO: par_iter
        for (i, mat) in mats.iter_mut().enumerate() {
            let grid_idx = flat_idx_to_grid_index(i as u32, n_cells);
            let pos = vect_to_3d(
                grid_index_as_vect(grid_idx) * cell_size, Vec3::ZERO
            );
            *mat = self.regions.iter()
                .find_map(|(s, pose, m)| {
                    let p = pose * offset;
                    s.contains_point(&p, pos).then_some(*m)
                })
                .unwrap_or(default_mat);
        }

        mats
    }

    /// Downscale a material grid using a box filter. Out-of-bounds material cells are ignored.
    ///
    /// Returns the dimensions of the new grid, and the new grid's data
    pub fn downscale_material_grid(
        grid: &[ElectricMaterial],
        n_cells: GridIndex,
        downscale_factor: NonZeroU32,
    ) -> (GridIndex, Vec<ElectricMaterial>) {
        let downscale_factor = downscale_factor.get();
        let n_cells_new = grid_index_from_array(
            grid_index_to_array(n_cells)
                .map(|dim| dim.div_ceil(downscale_factor))
        );
        let cell_count_new = grid_index_to_array(n_cells_new).iter()
            .product::<Index>();
        let mut grid_new = vec![ElectricMaterial::FREE_SPACE; cell_count_new as usize];

        let kernel = grid_cells_iter!(grid_index_from_array([downscale_factor; DIM]))
            .map(|t| grid_index_from_array(t.into()))
            .collect::<Vec<_>>();
        let n_cells3 = grid_index_to_3d(n_cells, UVec3::ONE);
        for i in 0..cell_count_new {
            let idx = flat_idx_to_grid_index(i, n_cells_new) * downscale_factor;
            let mut mat_sum = ElectricMaterial::ZERO;
            let mut n_sums = 0;
            for k in kernel.iter() {
                let k_idx = idx + k;
                let k_idx3 = grid_index_to_3d(k_idx, UVec3::ONE);
                if k_idx3.cmplt(n_cells3).all() {
                    let k_i = grid_index_to_flat_idx(k_idx, n_cells) as usize;
                    mat_sum.mu_r += grid[k_i].mu_r;
                    mat_sum.eps_r += grid[k_i].eps_r;
                    mat_sum.sig += grid[k_i].sig;
                    n_sums += 1;
                }
            }
            let n_sums = n_sums as f32;
            grid_new[i as usize] = ElectricMaterial {
                mu_r: mat_sum.mu_r / n_sums,
                eps_r: mat_sum.eps_r / n_sums,
                sig: mat_sum.sig / n_sums,
            };
        }

        (n_cells_new, grid_new)
    }
}

/// The widths of layers of cells at the boundary of the simulation. (e.g. PMLs)
///
/// Stores the "low" and "high" widths on each axis.
#[derive(Copy, Clone, Default)]
pub struct LayerWidths {
    pub widths: [[Index; 2]; DIM],
}

impl LayerWidths {
    #[inline]
    #[cfg(feature = "dim1")]
    pub fn new(z_lo: Index, z_hi: Index) -> Self {
        Self {
            widths: [[z_lo, z_hi]]
        }
    }

    #[inline]
    #[cfg(feature = "dim2")]
    pub fn new(x_lo: Index, x_hi: Index, y_lo: Index, y_hi: Index) -> Self {
        Self {
            widths: [[x_lo, x_hi], [y_lo, y_hi]]
        }
    }

    #[inline]
    #[cfg(feature = "dim3")]
    pub fn new(x_lo: Index, x_hi: Index, y_lo: Index, y_hi: Index, z_lo: Index, z_hi: Index) -> Self {
        Self {
            widths: [[x_lo, x_hi], [y_lo, y_hi], [z_lo, z_hi]]
        }
    }

    /// [`LayerWidths`] with widths along all axes set to `width`.
    #[inline]
    pub fn splat(width: Index) -> Self {
        #[cfg(feature = "dim1")]
        return Self::new(width, width);
        #[cfg(feature = "dim2")]
        return Self::new(width, width, width, width);
        #[cfg(feature = "dim3")]
        Self::new(width, width, width, width, width, width)
    }

    /// Adds `self` with `n_cells`, where `n_cells` is the dimensions of a [`YeeGrid`] in grid cells.
    pub fn sum_with_n_cells(&self, n_cells: GridIndex) -> GridIndex {
        let mut n_cells = grid_index_to_array(n_cells);
        for (n_cells_i, (_axis, lo_hi_widths)) in n_cells.iter_mut()
            .zip(self.iter_axes())
        {
            *n_cells_i += lo_hi_widths.iter().sum::<Index>();
        }
        grid_index_from_array(n_cells)
    }

    #[inline]
    pub fn iter_axes(&self) -> impl Iterator<Item = (SpatialAxis, &[Index; 2])> {
        SpatialAxis::ALL_AXES.into_iter()
            .zip(self.widths.iter())
    }

    #[inline]
    pub fn iter_axes_mut(&mut self) -> impl Iterator<Item = (SpatialAxis, &mut [Index; 2])> {
        SpatialAxis::ALL_AXES.into_iter()
            .zip(self.widths.iter_mut())
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

impl core::ops::Index<SpatialAxis> for LayerWidths {
    type Output = [Index; 2];
    #[inline]
    fn index(&self, index: SpatialAxis) -> &Self::Output {
        &self.widths[index as usize]
    }
}

impl core::ops::IndexMut<SpatialAxis> for LayerWidths {
    #[inline]
    fn index_mut(&mut self, index: SpatialAxis) -> &mut Self::Output {
        &mut self.widths[index as usize]
    }
}