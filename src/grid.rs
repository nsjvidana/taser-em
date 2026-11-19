#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::consts::{C_0, EPS_0};
use crate::fdtd::{ElectricMaterial, PmlParameters};
use crate::mesh_loading::MeshConverter;
use crate::prelude::load_from_path;
use crate::prelude::*;
use crate::util::grid_cells_iter;
use crate::{into_par_iter, par_iter_mut};
use derive_more::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use glamx::{Pose3, Vec3};
use parry3d::bounding_volume::{Aabb, BoundingVolume};
use parry3d::partitioning::{Bvh, BvhBuildStrategy};
use parry3d::shape::{Cuboid, Shape, SharedShape};
use std::num::NonZeroU32;
use std::path::Path;
use taser_em_shaders::fdtd::PmlCoefficients;
use taser_em_shaders::math::*;

#[derive(Clone, Debug, Default)]
/// Regions where a certain [`ElectricMaterial`] is present in a grid, stored as generic shapes.
pub struct MaterialRegions {
    pub regions: Vec<MaterialRegion>,
    /// A transformation applied to all regions as an entire scene.
    pub scene_pose: Pose3,
}

impl MaterialRegions {
    pub fn new() -> Self { Self::default() }

    /// Utility function that uses [`MaterialRegions::load_path_as_regions`] to load a mesh at `path`
    /// as a trimesh region using helpful [`TriMeshFlags`].
    ///
    /// `mesh_path` must contain a mesh whose face indices are **oriented**.
    pub fn load_trimesh_regions(
        &mut self,
        material: ElectricMaterial,
        mesh_path: impl AsRef<Path>,
        scale: Vec3,
    ) -> Result<&mut [MaterialRegion], Error> {
        self.load_path_as_regions(
            material,
            mesh_path,
            &MeshConverter::TriMeshWithFlags(TriMeshFlags::ORIENTED | TriMeshFlags::FIX_INTERNAL_EDGES),
            scale
        )
    }


    /// Loads a scene of meshes/objects from `path` and appends it to `self` as a
    /// collection of [`MaterialRegion`].
    /// Returns mutable slice of the newly-added [`MaterialRegion`]s stored in `self`.
    ///
    /// The transforms of objects in scene at `path` are assumed to be world-space, thus the `pose` field
    /// of the new [`MaterialRegion`]s will be the poses of the objects as described in the file
    /// at `path`.
    pub fn load_path_as_regions(
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
                                    .map(|arr| Vec3::from_array(arr) * scale)
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
    pub materials: Vec<YeeCellMaterials>,
    /// The material applied to cells that don't intersect with any material regions.
    pub default_mat: ElectricMaterial,
    /// The translation applied to all objects in a [`MaterialRegions`] to center them on a Yee grid.
    pub world_to_grid: Vec3,
}

impl YeeGridMaterials {
    /// Spatial offset of each component in the Dn field at each cell
    pub const DN_OFFSETS: [Vec3; MAX_DIM] = [
        Vec3::new(0.5, 0. , 0. ),
        Vec3::new(0. , 0.5, 0. ),
        Vec3::new(0. , 0. , 0.5),
    ];
    /// Spatial offset of each component in the H field at each cell
    pub const H_OFFSETS: [Vec3; MAX_DIM] = [
        Vec3::new(0. , 0.5, 0.5),
        Vec3::new(0.5, 0. , 0.5),
        Vec3::new(0.5, 0.5, 0. ),
    ];

    /// Compute a material grid with the dimensions of `n_cells` by sampling material regions.
    ///
    /// `world_to_grid` gets applied to the translation of all `regions` internally before they get discretized.
    ///
    /// # How it Works
    /// The grid spans from the world space origin to the grid extents (`cell_size * custom_n_cells`), in world space,
    /// with the **out-most layer of cells being boundary cells**.
    ///
    /// With `world_to_grid` translation applied to each region, cube-intersection tests with each region are done with
    /// the physical positions of each vector component in the grid.
    /// Components that don't intersect with any regions are assigned `default_mat`.
    ///
    /// ## Region Intersections
    /// Whether a region is in the grid will depend on the dimension:
    /// - 1D: Only parts of `regions` intersecting with the Z axis after `world_to_grid` is applied.
    /// - 2D: Only parts of `regions` intersecting with the X-Y plane after `world_to_grid` is applied.
    /// - 3D: Only parts of `regions` that end up inside the 3D grid after `world_to_grid` is applied.
    pub fn new_material_grid(
        n_cells: GridIndex,
        cell_size: Vect,
        world_to_grid: Vec3,
        regions: &MaterialRegions,
        default_mat: ElectricMaterial,
    ) -> Self {
        let cell_count = n_cells.n_cells_to_3d().element_product() as usize;

        let centered_scene_pose = regions.scene_pose.append_translation(world_to_grid);

        let regions_transformed = regions.regions.iter()
            .map(|r| {
                let p = centered_scene_pose * r.pose;
                (r.shape.clone(), p, r.shape.compute_aabb(&p), r.material)
            })
            .collect::<Vec<_>>();

        let aabbs = regions_transformed.iter().map(|(_, _, aabb, _)| *aabb).collect::<Vec<_>>();
        let bvh = Bvh::from_leaves(BvhBuildStrategy::Binned, &aabbs); // TODO: try using BvhBuildStrategy::Ploc instead

        let query_cube = Cuboid::new(cell_size.to_3d(Vec3::splat(cell_size.max_element() * 2.)) / 2.);

        let pen_depth_thresh = cell_size.length() * 0.5;

        let mat_at_cmp = |pt, is_de: bool, axis_idx: usize| {
            let query_pose = Pose3::from_translation(pt);
            bvh
                .intersect_aabb(&query_cube.compute_aabb(&query_pose))
                .find_map(|i| {
                    let leaf_id = i as usize;
                    let (shape, pose, _, mat) = &regions_transformed[leaf_id];
                    let contact = parry3d::query::contact(&query_pose, &query_cube, pose, &**shape, 0.0);
                    let Ok(Some(contact)) = contact else { return None };

                    if contact.dist >= pen_depth_thresh { return Some((*mat, 1.)); }
                    if mat.sig[axis_idx].is_infinite() { // shallow penetration means this is component is at a surface
                        // set PEC for vector field components at pec interface
                        // PEC surface conditions (vector field conditions at a PEC surface):
                        // - En components tangential to PEC surfaces are `0.`
                        // - H components orthogonal to PEC surface are `0.`
                        let component_dir = Axis::try_from(axis_idx as u32)
                            .expect("axis_idx must be valid. This is a bug.")
                            .to_vec3();
                        let angle = component_dir.angle_between(contact.normal2);
                        let angle_to_normal = angle.min(core::f32::consts::PI - angle);
                        // as H component gets closer to being parallel to normal, it approaches zero
                        // as En component gets closer to being 90deg from normal, it approaches zero
                        let mut coeff_lerp_t = angle_to_normal / core::f32::consts::FRAC_PI_2;
                        if is_de { coeff_lerp_t = 1. - coeff_lerp_t; }
                        // overwrite inf sig to stop zeroing out update coefficients in PmlCoefficientsGrid
                        let mut sig = mat.sig;
                        sig[axis_idx] = default_mat.sig[axis_idx];
                        return Some((ElectricMaterial { sig, ..*mat }, coeff_lerp_t))
                    }
                    None
                })
                .unwrap_or((default_mat, 1.))
        };

        let cell_size3 = cell_size.to_3d(Vec3::ZERO);
        let h_offsets = Self::H_OFFSETS.map(|v| v * cell_size3);
        let dn_offsets = Self::DN_OFFSETS.map(|v| v * cell_size3);
        let mut mats = vec![YeeCellMaterials::ZERO; cell_count];
        // TODO: use ndarray::Array3 with parallel feature instead
        par_iter_mut!(mats)
            .enumerate()
            .for_each(|(i, mat)| {
                let grid_idx = GridIndex::from_flat_idx(i as u32, n_cells);
                let pos = (grid_idx.as_vect() * cell_size).to_3d(Vec3::ZERO);
                let h_mat = std::array::from_fn(|i| {
                    mat_at_cmp(pos + h_offsets[i], false, i)
                });
                let dn_mat = std::array::from_fn(|i| {
                    mat_at_cmp(pos + dn_offsets[i], true, i)
                });
                *mat = YeeCellMaterials {
                    dn: dn_mat.map(|(mat, _)| mat),
                    h: h_mat.map(|(mat, _)| mat),
                    h_coeffs_weight: dn_mat.map(|(_, factor)| factor),
                    dn_coeffs_weight: h_mat.map(|(_, factor)| factor),
                };
            });

        Self {
            n_cells,
            cell_size,
            materials: mats,
            default_mat,
            world_to_grid,
        }
    }

    /// Constructs a lower-resolution version of this grid. Resolution is reduced by a factor of `downscale_factor`
    /// using a box filter to smooth out values.
    pub fn downscaled(
        &self,
        downscale_factor: NonZeroU32,
    ) -> Self {
        let downscale_factor = downscale_factor.get();

        let n_cells = self.n_cells.div_ceil(GridIndex::splat(downscale_factor));
        let cell_size = self.cell_size * downscale_factor as f32;

        let kernel_cells = grid_cells_iter(GridIndex::splat(downscale_factor)).collect::<Vec<_>>();
        let fine_n_cells = self.n_cells;
        let cell_count = n_cells.element_product() as usize;
        let start = std::time::Instant::now();
        let materials = into_par_iter!(0..cell_count)
            .map(|coarse_idx| {
                let subcell_fine = GridIndex::from_flat_idx(coarse_idx as u32, n_cells) * downscale_factor;
                let mut mat_sum = YeeCellMaterials::ZERO;
                let mut n_pecs_h = [0; 3];
                let mut n_pecs_dn = [0; 3];
                let mut n_sums = 0;
                for k_cell_offset in kernel_cells.iter() {
                    let fine_cell = subcell_fine + k_cell_offset;
                    if fine_cell.cmpge(fine_n_cells).any() { continue; }
                    
                    let fine_mat_idx = fine_cell.to_flat_idx(fine_n_cells) as usize;
                    let mats = self.materials[fine_mat_idx];
                    mat_sum = mat_sum + mats.with_no_pec();
                    n_sums += 1;
                    for (i, (mat, count)) in mats.h.iter().zip(&mut n_pecs_h).enumerate() {
                        if mat.sig[i].is_infinite() { *count += 1 }
                    }
                    for (i, (mat, count)) in mats.dn.iter().zip(&mut n_pecs_dn).enumerate() {
                        if mat.sig[i].is_infinite() { *count += 1 }
                    }
                }

                let mut downscaled_mats = mat_sum / (n_sums as f32);
                // set PEC for components with at least half their kernel cells being PEC material
                let n_sums_half = n_sums / 2;
                downscaled_mats.h
                    .iter_mut()
                    .zip(n_pecs_h)
                    .enumerate()
                    .for_each(|(i, (mat, pec_count))| {
                        if pec_count >= n_sums_half { mat.sig[i] = Real::INFINITY; }
                    });
                downscaled_mats.dn
                    .iter_mut()
                    .zip(n_pecs_dn)
                    .enumerate()
                    .for_each(|(i, (mat, pec_count))| {
                        if pec_count >= n_sums_half { mat.sig[i] = Real::INFINITY; }
                    });
                downscaled_mats
            })
            .collect::<Vec<_>>();
        println!("downscaling: {:?}", start.elapsed());

        YeeGridMaterials {
            n_cells,
            cell_size,
            materials,
            default_mat: self.default_mat,
            world_to_grid: self.world_to_grid,
        }
    }
}

/// The material present at each vector component
#[derive(Copy, Clone, Debug)]
pub struct YeeCellMaterials {
    /// Material at H components
    pub h: [ElectricMaterial; MAX_DIM],
    /// Material at Dn and En components
    pub dn: [ElectricMaterial; MAX_DIM],
    /// Weight value from `0.` to `1.` multiplied to H update coefficients. Used for approximating PEC
    /// surface fields
    pub h_coeffs_weight: [f32; MAX_DIM],
    /// Weight value from `0.` to `1.` multiplied to Dn update coefficients. Used for approximating PEC
    /// surface fields
    pub dn_coeffs_weight: [f32; MAX_DIM],
}

impl YeeCellMaterials {
    pub const ZERO: Self = Self {
        h: [ElectricMaterial::ZERO; MAX_DIM],
        dn: [ElectricMaterial::ZERO; MAX_DIM],
        h_coeffs_weight: [1.; 3],
        dn_coeffs_weight: [1.; 3],
    };

    /// Returns `self` with all PEC components having their conductivity set to zero
    pub fn with_no_pec(mut self) -> Self {
        self.h
            .iter_mut()
            .for_each(|v| {
                v.sig = v.sig.map(|v| if v.is_infinite() { 0. } else { v });
            });
        self.dn
            .iter_mut()
            .for_each(|v| {
                v.sig = v.sig.map(|v| if v.is_infinite() { 0. } else { v });
            });
        self
    }
}

impl core::ops::Add for YeeCellMaterials {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            h: core::array::from_fn(|i| self.h[i] + rhs.h[i]),
            dn: core::array::from_fn(|i| self.dn[i] + rhs.dn[i]),
            h_coeffs_weight: core::array::from_fn(|i| self.h_coeffs_weight[i] + rhs.h_coeffs_weight[i]),
            dn_coeffs_weight: core::array::from_fn(|i| self.dn_coeffs_weight[i] + rhs.dn_coeffs_weight[i]),
        }
    }
}

impl core::ops::Div<Real> for YeeCellMaterials {
    type Output = Self;

    fn div(self, rhs: Real) -> Self::Output {
        Self {
            h: core::array::from_fn(|i| self.h[i] / rhs),
            dn: core::array::from_fn(|i| self.dn[i] / rhs),
            h_coeffs_weight: core::array::from_fn(|i| self.h_coeffs_weight[i] / rhs),
            dn_coeffs_weight: core::array::from_fn(|i| self.dn_coeffs_weight[i] / rhs),
        }
    }
}

/// How objects get discretized / voxelized on the Yee Grid.
#[derive(Copy, Clone, Debug)]
pub enum MaterialDiscretization {
    /// Assign materials to cells directly without any smoothing. This means the grid will resort to
    /// stair-casing materials, which can lead to inaccuracies.
    Rough,
    /// Smooth out material values with a box filter. `resolution` determines the size of the box
    /// filter, with higher `resolution` giving more accuracy at the cost of simulation initialization time.
    Smooth { resolution: NonZeroU32 }
}

impl MaterialDiscretization {
    pub fn smooth_from_stability(stability: &FdtdStability) -> Self {
        Self::Smooth {
            resolution: stability.material_resolution,
        }
    }
}

/// Grid of PML update coefficients for FDTD using Yee Grid
#[derive(Clone, Debug)]
pub struct PmlCoefficientsGrid {
    /// Dimensions of grid
    pub n_cells: GridIndex,
    /// Grid's update coefficients in a flattened array.
    pub coeffs: Vec<PmlCoefficients>,
    pub world_to_grid: Vec3,
}

impl PmlCoefficientsGrid {
    /// Voxelize material regions and compute PML update coefficients.
    ///
    /// `n_cells_inner` is the dimensions of the sub-grid that is encapsulated by the PML.
    pub fn new(
        grid_mats: &YeeGridMaterials,
        pml_parameters: PmlParameters,
        dt: Real
    ) -> Self {
        let PmlParameters {
            widths: pml_widths,
            sig_max: pml_sig_max,
            grading_order: pml_grading_order,
            ..
        } = pml_parameters;
        let n_cells = grid_mats.n_cells;
        let n_cells3 = n_cells.n_cells_to_3d();

        let YeeGridMaterials {
            n_cells: _n_cells,
            materials: mats,
            world_to_grid,
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
                    let lo_t = (1. - lo_dist / (pml_widths.get_or_zero(axis).lo*2) as Real)
                        .clamp(0., 1.);
                    let hi_t = (1. - hi_dist / (pml_widths.get_or_zero(axis).hi*2) as Real)
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
            .for_each(|(idx, coeff)| {
                let cell_idx = GridIndex::from_flat_idx(idx as u32, n_cells);
                
                let cell_materials = mats[idx];

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

                    // H update coefficients
                    let h_mat = &cell_materials.h[axis_i];
                    if h_mat.sig[axis].is_infinite() {
                        coeff.h1[axis] = PmlCoefficients::PEC.h1[axis];
                        coeff.h2[axis] = PmlCoefficients::PEC.h2[axis];
                        coeff.h3[axis] = PmlCoefficients::PEC.h3[axis];
                        #[cfg(any(feature = "dim2", feature = "dim3"))]
                        {
                            coeff.h4[axis] = PmlCoefficients::PEC.h4[axis];
                        }
                    } else {
                        let weight = cell_materials.h_coeffs_weight[axis_i];
                        let h_sigs_axis = h_sigs[axis_i];
                        let coeff_term0 = (
                            inv_dt + ((h_sigs_axis[axis1] + h_sigs_axis[axis2]) / (2. * EPS_0)) +
                                ((h_sigs_axis[axis1] * h_sigs_axis[axis2] * dt) / (4. * EPS_0 * EPS_0))
                        ).recip();
                        let inv_mu_r_axis = h_mat.mu_r[axis].recip();
                        coeff.h1[axis] = coeff_term0 * (
                            inv_dt - ((h_sigs_axis[axis1] + h_sigs_axis[axis2]) / (2. * EPS_0)) -
                                ((h_sigs_axis[axis1] * h_sigs_axis[axis2] * dt) / (4. * EPS_0 * EPS_0))
                        );
                        coeff.h2[axis] = -coeff_term0 * C_0 * inv_mu_r_axis;
                        coeff.h3[axis] = -coeff_term0 * c0_dt * h_sigs_axis[axis] / EPS_0 * inv_mu_r_axis;
                        #[cfg(any(feature = "dim2", feature = "dim3"))]
                        {
                            coeff.h4[axis] = -coeff_term0 * (dt / (EPS_0 * EPS_0)) * h_sigs_axis[axis1] * h_sigs_axis[axis2];
                            coeff.h4[axis] *= weight;
                        }
                        // weight for things like PEC surface approximation
                        coeff.h1[axis] *= weight;
                        coeff.h2[axis] *= weight;
                        coeff.h3[axis] *= weight;
                    }

                    // Dn/En update coefficients
                    let dn_mat = &cell_materials.dn[axis_i];
                    if dn_mat.sig[axis].is_infinite() {
                        coeff.dn1[axis] = PmlCoefficients::PEC.dn1[axis];
                        coeff.dn2[axis] = PmlCoefficients::PEC.dn2[axis];
                        coeff.dn_loss1[axis] = PmlCoefficients::PEC.dn_loss1[axis];
                        coeff.dn_loss2[axis] = PmlCoefficients::PEC.dn_loss2[axis];
                        coeff.dn3[axis] = PmlCoefficients::PEC.dn3[axis];
                        #[cfg(any(feature = "dim2", feature = "dim3"))]
                        {
                            coeff.dn4[axis] = PmlCoefficients::PEC.dn4[axis];
                        }
                        coeff.en1[axis] = PmlCoefficients::PEC.en1[axis];
                    } else {
                        let weight = cell_materials.dn_coeffs_weight[axis_i];
                        let dn_sigs_axis = dn_sigs[axis_i];
                        let coeff_term0 = (
                            inv_dt + ((dn_sigs_axis[axis1] + dn_sigs_axis[axis2]) / (2. * EPS_0)) +
                                ((dn_sigs_axis[axis1] * dn_sigs_axis[axis2] * dt) / (4. * EPS_0 * EPS_0))
                        ).recip();
                        let mat_sig_axis = dn_mat.sig[axis];
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
                            coeff.dn4[axis] *= weight;
                        }
                        coeff.en1[axis] = dn_mat.eps_r[axis].recip();
                        // weight for things like PEC surface approximation
                        coeff.dn1[axis] *= weight;
                        coeff.dn2[axis] *= weight;
                        coeff.dn_loss1[axis] *= weight;
                        coeff.dn_loss2[axis] *= weight;
                        coeff.dn3[axis] *= weight;
                        coeff.en1[axis] *= weight;
                    }
                }
            });

        Self { n_cells, coeffs, world_to_grid: *world_to_grid }
    }
}

/// The widths of layers of cells at the boundary of the simulation. (e.g. PMLs)
///
/// Stores the widths of a layer at the "low-end" and "high-ends" on each axis.
#[derive(Copy, Clone, Debug, Default)]
pub struct LayerWidths {
    pub widths: [LoHiWidths; DIM]
}

impl LayerWidths {
    pub const ZERO: Self = Self { widths: [LoHiWidths::ZERO; DIM] };

    /// [`LayerWidths`] with `lo` and `hi` widths along all spatial axes set to `width`.
    #[inline]
    pub const fn splat_spatial(width: Index) -> Self {
        Self { widths: [LoHiWidths::splat(width); DIM] }
    }

    /// Tries to get widths along an axis. If `axis` isn't spatial, return [`LoHiWidths::ZERO`].
    #[inline]
    pub fn get_or_zero(&self, axis: Axis) -> LoHiWidths {
        if let Ok(s_axis) = SpatialAxis::try_from(axis) { self[s_axis] }
        else { LoHiWidths::ZERO }
    }

    /// Use index operator to index with a [`SpatialAxis`] (no extra checks)
    #[inline]
    pub fn get(&self, axis: Axis) -> Option<&LoHiWidths> {
        Some(&self[SpatialAxis::try_from(axis).ok()?])
    }

    /// Use index operator to index with a [`SpatialAxis`] (no extra checks)
    #[inline]
    pub fn get_mut(&mut self, axis: Axis) -> Option<&mut LoHiWidths> {
        Some(&mut self[SpatialAxis::try_from(axis).ok()?])
    }

    /// Adds `self` with `n_cells`, where `n_cells` is the dimensions of a grid.
    pub fn sum_with_n_cells(&self, mut n_cells: GridIndex) -> GridIndex {
        self.iter_spatial_axes()
            .for_each(|(axis, widths)| n_cells[axis] += widths.lo + widths.hi);
        n_cells
    }

    /// Returns `self` with widths along `spatial_axis` set to `widths`
    pub fn with_axis_widths(mut self, spatial_axis: SpatialAxis, widths: LoHiWidths) -> Self {
        self[spatial_axis] = widths;
        self
    }

    #[inline]
    pub fn iter_spatial_axes(&self) -> impl Iterator<Item = (SpatialAxis, &LoHiWidths)> {
        SpatialAxis::ALL_SPATIAL.into_iter().zip(self.widths.iter())
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

impl core::ops::Index<SpatialAxis> for LayerWidths {
    type Output = LoHiWidths;
    #[inline]
    fn index(&self, index: SpatialAxis) -> &Self::Output { &self.widths[index as usize] }
}

impl core::ops::IndexMut<SpatialAxis> for LayerWidths {
    #[inline]
    fn index_mut(&mut self, index: SpatialAxis) -> &mut Self::Output { &mut self.widths[index as usize] }
}

/// Widths of a layer on the lo- and hi- ends along an axis.
///
/// Used by [`LayerWidths`]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[derive(Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign)]
pub struct LoHiWidths {
    pub lo: Index,
    pub hi: Index,
}

impl LoHiWidths {
    pub const ZERO: Self = Self { lo: 0, hi: 0 };

    #[inline]
    pub const fn splat(width: Index) -> Self {
        Self {
            lo: width,
            hi: width,
        }
    }

    #[inline]
    pub const fn with_lo(mut self, lo: Index) -> Self {
        self.lo = lo;
        self
    }

    #[inline]
    pub const fn with_hi(mut self, hi: Index) -> Self {
        self.hi = hi;
        self
    }
}

#[derive(Copy, Clone, Debug)]
pub struct GridSizing {
    /// Grid dimensions in cells
    pub n_cells: GridIndex,
    /// Translation that brings simulation objects (e.g. [`MaterialRegion`], [`Source`]) from world-space
    /// to grid-space.
    ///
    /// World-space typically includes all user inputs that are transformations / positions
    pub world_to_grid: Vec3,
}