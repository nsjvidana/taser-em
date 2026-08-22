use crate::par_iter_mut;
use crate::prelude::*;
use glamx::{Pose3, Vec3, Vec4};
use parry3d::bounding_volume::{Aabb, BoundingVolume};
use parry3d::math::Vector;
use parry3d::shape::{Cuboid, Shape, SharedShape};
use std::num::NonZeroU32;
use std::path::Path;
use taser_em_shaders::math::*;

use crate::consts::{C_0, EPS_0};
use crate::fdtd::{ElectricMaterial, PmlParameters};
use crate::mesh_loading::MeshConverter;
use crate::prelude::load_from_path;
use crate::util::grid_cells_iter;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use taser_em_shaders::fdtd::{PmlCoefficients, PolarizationModeIndex};

/// Polarization mode affects which field components are computed in the simulation, depending on how many spatial dimensions there are.
/// In 3D, all axes are computed for all fields.
/// See the docs of this enum's variants to know which axes are enabled for each mode.
///
/// A little pedantic note: The word "polarization" here actually refers to how a specific vector field has to be transverse to the
/// simulation domain in 2D FDTD, so the word makes less sense in 1D and 3D contexts, but it's used here anyway for simplicity.
#[derive(Copy, Clone, Debug)]
pub enum PolarizationMode {
    /// 1D: Ex, Hy
    /// 2D: Ez, Hx, Hy
    /// 3D: All axes
    TransverseElectric,
    /// 1D: Ey, Hx
    /// 2D: Ex, Ey, Hz
    /// 3D: All axes
    TransverseMagnetic,
}

impl PolarizationMode {
    // TODO: make these into "get/extract source" functions
    pub fn get_e_magnitude(&self, e: &Vec4) -> Real {
        match self {
            PolarizationMode::TransverseMagnetic => cfg_select! {
                feature = "dim1" => e.y.abs(),
                feature = "dim2" => e.z.abs(),
                feature = "dim3" => e.length(),
            },
            PolarizationMode::TransverseElectric => cfg_select! {
                feature = "dim1" => e.x.abs(),
                any(feature = "dim2", feature = "dim3") => e.length(),
            },
        }
    }

    pub fn extract_e_vector(&self, e: &Vec4) -> Vec3 {
        match self {
            PolarizationMode::TransverseMagnetic => cfg_select! {
                feature = "dim1" => Vec3::new(0., e.y, 0.),
                feature = "dim2" => Vec3::Z * e.z,
                feature = "dim3" => (*e).xyz(),
            },
            PolarizationMode::TransverseElectric => cfg_select! {
                feature = "dim1" => Vec3::new(e.x, 0., 0.),
                feature = "dim2" => Vec3::from(((*e).xy(), 0.)),
                feature = "dim3" => (*e).xyz(),
            },
        }
    }


    pub fn e_magnitude(&self, e: &Vec4) -> Real {
        match self {
            PolarizationMode::TransverseMagnetic => cfg_select! {
                feature = "dim1" => e.y,
                feature = "dim2" => e.z,
                feature = "dim3" => (*e).xyz().length(),
            },
            PolarizationMode::TransverseElectric => cfg_select! {
                feature = "dim1" => e.x,
                feature = "dim2" => (*e).xy().length(),
                feature = "dim3" => (*e).xyz().length(),
            },
        }
    }
}

impl From<PolarizationMode> for PolarizationModeIndex {
    fn from(value: PolarizationMode) -> Self {
        match value {
            PolarizationMode::TransverseMagnetic => PolarizationModeIndex::TM,
            PolarizationMode::TransverseElectric => PolarizationModeIndex::TE,
        }
    }
}

#[derive(Clone, Debug, Default)]
/// Regions where a certain [`ElectricMaterial`] is present in a grid, stored as generic shapes.
pub struct MaterialRegions {
    pub regions: Vec<MaterialRegion>,
    /// A transformation applied to all regions as an entire scene.
    pub scene_pose: Pose3,
}

impl MaterialRegions {
    pub fn new() -> Self { Self::default() }

    /// Fill a box-shaped region from `start` to `end`
    pub fn fill_region(
        &mut self,
        start: Vect,
        end: Vect,
        material: ElectricMaterial
    ) -> &mut Self {
        let region_dims = end - start;
        let half_extents = region_dims.to_3d(Vec3::splat(region_dims.max_element()));

        let shape = Cuboid::new(half_extents);
        let middle = ((start + end) / 2.).to_3d(Vec3::ZERO);
        let pose = Pose3::from_translation(middle);

        self.regions.push(MaterialRegion::new(SharedShape::new(shape), pose, material));
        self
    }

    /// Loads a scene of meshes/objects from `path` and appends it to `self` as a
    /// collection of [`MaterialRegion`].
    /// Returns mutable references to the newly-added [`MaterialRegion`]s.
    ///
    /// The poses of objects in scene at `path` are assumed to be world-space, thus the `pose` field
    /// of the new [`MaterialRegion`]s will be the poses of the objects as described in the file
    /// at `path`.
    pub fn load_trimesh_region(&mut self, material: ElectricMaterial, path: impl AsRef<Path>) -> Result<&mut [MaterialRegion], Error> {
        self.load_path_as_region(material, path, &MeshConverter::TriMesh, Vec3::ONE)
    }

    /// A version of [`MaterialRegions::load_trimesh_region`] with more parameters for the user to control.
    pub fn load_path_as_region(
        &mut self,
        material: ElectricMaterial,
        path: impl AsRef<Path>,
        converter: &MeshConverter,
        scale: Vec3,
    ) -> Result<&mut [MaterialRegion], Error> {
        let shapes = load_from_path(path, converter, scale)?
            .into_iter()
            .map(|v| v.map_err(Error::from))
            .collect::<Result<Vec<_>, _>>()?;
        let new_regions_start = self.regions.len();
        self.regions
            .extend(
                shapes.into_iter()
                    .map(|loaded_shape| {
                        MaterialRegion {
                            shape: loaded_shape.shape,
                            pose: loaded_shape.pose,
                            material,
                            #[cfg(feature = "render")]
                            mesh: Some(RegionMesh {
                                vertices: loaded_shape.raw_mesh.vertices
                                    .into_iter()
                                    .map(|arr| Vec3::from_array(arr))
                                    .collect(),
                                indices: loaded_shape.raw_mesh.faces,
                            })
                        }
                    })
            );

        Ok(&mut self.regions[new_regions_start..])
    }

    pub fn compute_bounding_box(&self) -> Aabb {
        let aabbs = self.regions
            .iter()
            .map(|r | r.shape.compute_aabb(&(self.scene_pose * r.pose)))
            .collect::<Vec<_>>();

        let mut full_bb = Aabb::new_invalid();
        for bb in aabbs.iter() {
            full_bb.merge(bb);
        }
        full_bb
    }
}

#[derive(Clone, Debug)]
pub struct MaterialRegion {
    pub shape: SharedShape,
    pub pose: Pose3,
    pub material: ElectricMaterial,
    #[cfg(feature = "render")]
    pub mesh: Option<RegionMesh>,
}

impl MaterialRegion {
    pub fn new(shape: SharedShape, pose: Pose3, material: ElectricMaterial) -> Self {
        #[cfg(feature = "render")]
        let mesh = crate::util::generate_mesh(&*shape)
            .map(|(vertices, indices)| RegionMesh { vertices, indices });
        Self {
            shape,
            pose,
            material,
            #[cfg(feature = "render")]
            mesh,
        }
    }
}

#[cfg(feature = "render")]
#[derive(Clone, Debug)]
pub struct RegionMesh {
    pub vertices: Vec<Vec3>,
    pub indices: Vec<[u32; 3]>,
}

/// The materials in a Yee Grid. It's the output of functions like [`MaterialRegions::material_yee_grid`].
#[derive(Clone, Debug)]
pub struct YeeGridMaterials {
    pub n_cells: GridIndex,
    pub cell_size: Vect,
    pub materials: Vec<ElectricMaterial>,
    /// The material applied to cells that don't intersect with any material regions.
    pub default_mat: ElectricMaterial,
    /// The translation applied to all objects in a [`MaterialRegions`] to center them on this grid.
    pub sim_offset: Vec3,
}

impl YeeGridMaterials {
    /// Spatial offset of each component in the Dn field at each cell
    pub const DN_OFFSETS: [Vec3; MAX_DIM] = [Vec3::X, Vec3::Y, Vec3::Z];
    /// Spatial offset of each component in the H field at each cell
    pub const H_OFFSETS: [Vec3; MAX_DIM] = [
        Vec3::new(0., 1., 1.),
        Vec3::new(1., 0., 1.),
        Vec3::new(1., 1., 0.),
    ];

    /// Compute a material grid with the dimensions of `n_cells` by sampling material regions.
    ///
    /// # How it Works
    /// First, the grid is moved **only along the spatial axes** (see [`SpatialAxis::ALL_SPATIAL`]) in a way
    /// that puts its center at the middle of all regions.
    /// Then point-intersection tests are done for each vector component's physical position using [`SharedShape::contains_point`]
    /// on each region. Components that don't intersect with any regions are assigned `default_mat`.
    ///
    /// Since grid centering only happens along spatial axes, whether a region is in the grid will
    /// depend on the dimension:
    /// - 1D: Only when intersecting w/ Z axis
    /// - 2D: Only when intersecting w/ X-Y plane
    /// - 3D: Grid of size `n_cells` is centered at the middle of a bounding box encapsulating all regions.
    ///   Whatever parts of regions that are inside the grid at this location are included.
    pub fn new_material_grid(
        n_cells: GridIndex,
        cell_size: Vect,
        simulation_bb: &Aabb,
        regions: &MaterialRegions,
        default_mat: ElectricMaterial,
    ) -> Self {
        let cell_count = n_cells.n_cells_to_3d().element_product() as usize;

        let sim_offset = Self::compute_simulation_offset(simulation_bb, n_cells, cell_size);
        let centered_scene_pose = regions.scene_pose.append_translation(sim_offset);

        let half_cell_size3 = (cell_size / 2.).to_3d(Vec3::ZERO);
        let dn_offsets = Self::DN_OFFSETS.map(|v| v * half_cell_size3);
        let h_offsets = Self::H_OFFSETS.map(|v| v * half_cell_size3);
        let regions_transformed = regions.regions.iter()
            .map(|r| (&r.shape, centered_scene_pose * r.pose, r.material))
            .collect::<Vec<_>>();
        let mat_at_pt = |pt| {
            regions_transformed.iter()
                .find_map(|(s, p, m)|
                    s.contains_point(p, pt).then_some(m)
                )
                .unwrap_or(&default_mat)
        };

        let mut mats = vec![ElectricMaterial::FREE_SPACE; cell_count];
        par_iter_mut!(mats)
            .enumerate()
            .for_each(|(i, mat)| {
                let grid_idx = GridIndex::from_flat_idx(i as u32, n_cells);
                let pos = (grid_idx.as_vect() * cell_size).to_3d(Vec3::ZERO);
                let dn_mat = dn_offsets.map(|off| mat_at_pt(pos + off));
                let h_mat = h_offsets.map(|off| mat_at_pt(pos + off));
                *mat = ElectricMaterial {
                    eps_r: Vec3::from_array(std::array::from_fn(|i| { dn_mat[i].eps_r[i] })),
                    sig: Vec3::from_array(std::array::from_fn(|i| { dn_mat[i].sig[i] })),
                    mu_r: Vec3::from_array(std::array::from_fn(|i| { h_mat[i].mu_r[i] })),
                };
            });

        Self {
            n_cells,
            cell_size,
            materials: mats,
            default_mat,
            sim_offset,
        }
    }

    /// Constructs a lower-resolution version of this grid. Resolution is reduced by a factor of `downscale_factor`
    /// using a box filter to smooth out values.
    pub fn downscaled(
        &self,
        downscale_factor: NonZeroU32,
    ) -> YeeGridMaterials {
        let downscale_factor = downscale_factor.get();

        let n_cells = self.n_cells.div_ceil(GridIndex::splat(downscale_factor));
        let cell_size = self.cell_size * downscale_factor as f32;
        let cell_count = n_cells.element_product() as usize;
        let mut materials = vec![ElectricMaterial::FREE_SPACE; cell_count];

        let kernel_cells = grid_cells_iter(GridIndex::from_index_array([downscale_factor; DIM]))
            .map(|t| GridIndex::from_index_array(t.into()))
            .collect::<Vec<_>>();
        let old_n_cells3 = self.n_cells.n_cells_to_3d();
        for (i, mat) in materials.iter_mut().enumerate() {
            let idx = GridIndex::from_flat_idx(i as u32, n_cells) * downscale_factor;
            let mut mat_sum = ElectricMaterial::ZERO;
            let mut n_sums = 0;
            for k in kernel_cells.iter() {
                let k_idx = idx + k;
                let k_idx3 = k_idx.cell_idx_to_3d();
                if k_idx3.cmplt(old_n_cells3).all() {
                    let k_i = k_idx.to_flat_idx(self.n_cells) as usize;
                    mat_sum.mu_r += self.materials[k_i].mu_r;
                    mat_sum.eps_r += self.materials[k_i].eps_r;
                    mat_sum.sig += self.materials[k_i].sig;
                    n_sums += 1;
                }
            }
            let n_sums = n_sums as f32;
            *mat = ElectricMaterial {
                mu_r: mat_sum.mu_r / n_sums,
                eps_r: mat_sum.eps_r / n_sums,
                sig: mat_sum.sig / n_sums,
            };
        }

        YeeGridMaterials {
            n_cells,
            cell_size,
            materials,
            default_mat: self.default_mat,
            sim_offset: self.sim_offset,
        }
    }

    pub fn compute_simulation_offset(
        simulation_bb: &Aabb,
        n_cells: GridIndex,
        cell_size: Vect
    ) -> Vec3 {
        let grid_center = n_cells.as_vect() * cell_size / 2.;
        let sim_center = Vect::from_vec3(simulation_bb.center());
        (grid_center - sim_center).to_3d(Vec3::ZERO)
    }
}

/// Grid of PML update coefficients for FDTD using Yee Grid
#[derive(Clone, Debug)]
pub struct PmlCoefficientsGrid {
    /// Dimensions of grid
    pub n_cells: GridIndex,
    /// Grid's update coefficients in a flattened array.
    pub coeffs: Vec<PmlCoefficients>,
}

impl PmlCoefficientsGrid {
    /// Voxelize material regions and compute PML update coefficients.
    ///
    /// `n_cells_inner` is the dimensions of the sub-grid that is encapsulated by the PML.
    pub fn new(
        grid_mats: &YeeGridMaterials,
        pml_parameters: PmlParameters,
        dt: Real
    ) -> (Vec3, Self) {
        let PmlParameters {
            widths: pml_widths,
            sig_max: pml_sig_max,
            grading_order: pml_grading_order
        } = pml_parameters;
        let n_cells = grid_mats.n_cells;
        let n_cells3 = n_cells.n_cells_to_3d();

        let YeeGridMaterials {
            n_cells: _n_cells,
            materials: mats,
            sim_offset: regions_offset,
            ..
        } = grid_mats;

        // PML conductivity terms (1D slices along each axis)
        let sig: [Vec<Real>; MAX_DIM] = Axis::ALL_AXES.map(|axis| {
            const HALF_CELL: Index = 1;
            const ONE_CELL: Index = HALF_CELL*2;
            let n_axis2x = n_cells3[axis]*2;
            let hi_end = (n_axis2x - 1).saturating_sub(ONE_CELL);
            let lo_end = ONE_CELL;
            (0..n_axis2x)
                .map(|i| {
                    if i < lo_end || i > hi_end { return 0. }
                    let lo_dist = i.abs_diff(lo_end) as Real;
                    let hi_dist = i.abs_diff(hi_end) as Real;
                    let lo_t = (1. - lo_dist / (pml_widths[axis].lo*2) as Real)
                        .clamp(0., 1.);
                    let hi_t = (1. - hi_dist / (pml_widths[axis].hi*2) as Real)
                        .clamp(0., 1.);
                    let sig = pml_sig_max * (lo_t + hi_t)
                        .powi(pml_grading_order.get());
                    if sig.is_finite() { sig } else { 0. }
                })
                .collect()
        });

        let cell_count = n_cells3.element_product() as usize;
        let mut coeffs = vec![PmlCoefficients::default(); cell_count];
        let inv_dt = dt.recip();
        let c0_dt = C_0 * dt;
        par_iter_mut!(coeffs)
            .enumerate()
            .for_each(|(i, coeff)| {
                let cell_idx = GridIndex::from_flat_idx(i as u32, n_cells);

                // Stagger indexing the conductivities as per the Yee grid staggering
                let idx_2x = (cell_idx * 2).cell_idx_to_3d().as_usizevec3();
                let dn_sigs: [Vec3; MAX_DIM] = core::array::from_fn(|axis_i| {
                    let mut sig_idx = idx_2x;
                        sig_idx[axis_i] += 1;
                    Vec3::new(
                        sig[0][sig_idx.x],
                        sig[1][sig_idx.y],
                        sig[2][sig_idx.z],
                    )
                });
                let h_sigs: [Vec3; MAX_DIM] = core::array::from_fn(|axis_i| {
                    let mut sig_idx = idx_2x + 1;
                        sig_idx[axis_i] -= 1;
                    Vec3::new(
                        sig[0][sig_idx.x],
                        sig[1][sig_idx.y],
                        sig[2][sig_idx.z],
                    )
                });

                for axis in Axis::ALL_AXES {
                    let axis_i = axis as usize;
                    let axis1 = axis.permute();
                    let axis2 = axis1.permute();

                    let h_sigs_axis = h_sigs[axis_i];
                    let coeff_term0 = (
                        inv_dt + ((h_sigs_axis[axis1] + h_sigs_axis[axis2]) / (2. * EPS_0)) +
                            ((h_sigs_axis[axis1] * h_sigs_axis[axis2] * dt) / (4. * EPS_0 * EPS_0))
                    ).recip();
                    let inv_mu_r_axis = mats[i].mu_r[axis].recip();
                    coeff.h1[axis] = coeff_term0 * (
                        inv_dt - ((h_sigs_axis[axis1] + h_sigs_axis[axis2]) / (2. * EPS_0)) -
                            ((h_sigs_axis[axis1] * h_sigs_axis[axis2] * dt) / (4. * EPS_0 * EPS_0))
                    );
                    coeff.h2[axis] = -coeff_term0 * C_0 * inv_mu_r_axis;
                    coeff.h3[axis] = -coeff_term0 * c0_dt * h_sigs_axis[axis] / EPS_0 * inv_mu_r_axis;
                    #[cfg(any(feature = "dim2", feature = "dim3"))]
                    {
                        coeff.h4[axis] = -coeff_term0 * (dt / (EPS_0 * EPS_0)) * h_sigs_axis[axis1] * h_sigs_axis[axis2];
                    }

                    let dn_sigs_axis = dn_sigs[axis_i];
                    let coeff_term0 = (
                        inv_dt + ((dn_sigs_axis[axis1] + dn_sigs_axis[axis2]) / (2. * EPS_0)) +
                            ((dn_sigs_axis[axis1] * dn_sigs_axis[axis2] * dt) / (4. * EPS_0 * EPS_0))
                    ).recip();
                    let mat_sig_axis = mats[i].sig[axis];
                    coeff.dn1[axis] = coeff_term0 * (
                        inv_dt - ((dn_sigs_axis[axis1] + dn_sigs_axis[axis2]) / (2. * EPS_0)) -
                            ((dn_sigs_axis[axis1] * dn_sigs_axis[axis2] * dt) / (4. * EPS_0 * EPS_0))
                    );
                    coeff.dn2[axis] = coeff_term0 * C_0;
                    coeff.dn_loss1[axis] = -coeff_term0 * mat_sig_axis / EPS_0;
                    coeff.dn_loss2[axis] = -coeff_term0 * dn_sigs_axis[axis] * mat_sig_axis * dt / (EPS_0 * EPS_0);
                    coeff.dn3[axis] = coeff_term0 * c0_dt * dn_sigs_axis[axis] / EPS_0;
                    #[cfg(any(feature = "dim2", feature = "dim3"))]
                    {
                        coeff.dn4[axis] = -coeff_term0 * (dt / (EPS_0 * EPS_0)) * dn_sigs_axis[axis1] * dn_sigs_axis[axis2];
                    }
                }
                coeff.en1 = Vec4::from((mats[i].eps_r.recip(), 0.));
            });

        (*regions_offset, Self { n_cells, coeffs, })
    }
}

/// The widths of layers of cells at the boundary of the simulation. (e.g. PMLs)
///
/// Stores the widths of a layer at the "low-end" and "high-ends" on each axis.
#[derive(Copy, Clone, Debug, Default)]
pub struct LayerWidths {
    pub widths: [LoHiWidths; MAX_DIM]
}

impl LayerWidths {
    /// [`LayerWidths`] with `lo` and `hi` widths along all spatial axes set to `width`.
    #[inline]
    pub fn splat_spatial(width: Index) -> Self {
        let mut selff = Self {
            widths: core::array::repeat(LoHiWidths::default()),
        };
        for s_axis in SpatialAxis::ALL_SPATIAL
        {
            selff[s_axis] = LoHiWidths::splat(width);
        }
        selff
    }

    /// Adds `self` with `n_cells`, where `n_cells` is the dimensions of a [`YeeGrid`] in grid cells.
    pub fn sum_with_n_cells(&self, n_cells: GridIndex) -> GridIndex {
        let mut n_cells = n_cells.into_array();
        for (n_cells_cmp, (_, lo_hi_widths_cmp)) in n_cells.iter_mut()
            .zip(self.iter_spatial_axes())
        {
            *n_cells_cmp += lo_hi_widths_cmp.lo + lo_hi_widths_cmp.hi;
        }
        GridIndex::from_index_array(n_cells)
    }

    /// Returns `self` with widths along `spatial_axis` set to `widths`
    pub fn with_axis_widths(mut self, spatial_axis: SpatialAxis, widths: LoHiWidths) -> Self {
        self[spatial_axis] = widths;
        self
    }

    #[inline]
    pub fn iter_spatial_axes(&self) -> impl Iterator<Item = (SpatialAxis, &LoHiWidths)> {
        SpatialAxis::ALL_SPATIAL
            .map(|axis| (axis, &self.widths[Axis::from(axis) as usize]))
            .into_iter()
    }
}

impl core::ops::AddAssign for LayerWidths {
    fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}

impl core::ops::Add for LayerWidths {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            widths: core::array::from_fn(|i| {
                LoHiWidths {
                    lo: self.widths[i].lo + rhs.widths[i].lo,
                    hi: self.widths[i].hi + rhs.widths[i].hi,
                }
            })
        }
    }
}

impl core::ops::SubAssign for LayerWidths {
    fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; }
}

impl core::ops::Sub for LayerWidths {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            widths: core::array::from_fn(|i| {
                LoHiWidths {
                    lo: self.widths[i].lo - rhs.widths[i].lo,
                    hi: self.widths[i].hi - rhs.widths[i].hi,
                }
            })
        }
    }
}

impl core::ops::Index<Axis> for LayerWidths {
    type Output = LoHiWidths;
    #[inline]
    fn index(&self, index: Axis) -> &Self::Output {
        &self.widths[index as usize]
    }
}

impl core::ops::IndexMut<Axis> for LayerWidths {
    #[inline]
    fn index_mut(&mut self, index: Axis) -> &mut Self::Output {
        &mut self.widths[index as usize]
    }
}

impl core::ops::Index<SpatialAxis> for LayerWidths {
    type Output = LoHiWidths;
    #[inline]
    fn index(&self, index: SpatialAxis) -> &Self::Output {
        &self.widths[Axis::from(index) as usize]
    }
}

impl core::ops::IndexMut<SpatialAxis> for LayerWidths {
    #[inline]
    fn index_mut(&mut self, index: SpatialAxis) -> &mut Self::Output {
        &mut self.widths[Axis::from(index) as usize]
    }
}

/// Widths of a layer on the lo- and hi- ends along an axis.
///
/// Used by [`LayerWidths`]
#[derive(Copy, Clone, Debug, Default)]
pub struct LoHiWidths {
    pub lo: Index,
    pub hi: Index,
}

impl LoHiWidths {
    #[inline]
    pub fn splat(width: Index) -> Self {
        Self {
            lo: width,
            hi: width,
        }
    }
}