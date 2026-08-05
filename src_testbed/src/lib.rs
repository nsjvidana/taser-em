mod util;

pub mod re_exports {
    pub use anyhow;
    pub use kiss3d;
}

use glamx::*;
use kiss3d::camera::Projection;
use kiss3d::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;
use taser_em::grid::{MaterialRegions, YeeGridMaterials};
use taser_em::shaders::math::*;
use taser_em::{grid_cells_iter, FdtdLossySimulation, FdtdStability, par_iter, par_iter_mut, into_par_iter};
use taser_em::prelude::{PolarizationMode, Real, VectExt, VectorValueExt};
use crate::util::lerp_colors;

#[cfg(not(feature = "dim1"))]
use kiss3d::prelude::InstanceData3d;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

pub struct FdtdTestbedViewer {
    pub window: Window,
    pub camera: OrbitCamera3d,
    pub scene: SceneNode3d,
    pub vector_field_color: Color,
    pub material_region_alpha: f32,
    pub n_cells: GridIndex,
    pub cell_size: Vect,
    pub polarization_mode: PolarizationMode,
    pub visualization_mode: VisualizationMode,
}

impl FdtdTestbedViewer {
    /// Create a new testbed viewer window, camera, and scene with the visualized vector field color
    /// set to [`RED`].
    ///
    /// Defaults to [`ColorMode::AutoScale`] for visualizing field magnitudes
    pub async fn new(
        simulation: &FdtdLossySimulation,
        stability: &FdtdStability,
        mut visualization_mode: VisualizationMode
    ) -> anyhow::Result<Self> {
        let title_dim = cfg_select! {
            feature = "dim1" => "1D",
            feature = "dim2" => "2D",
            feature = "dim3" => "3D",
        };
        let window = Window::new(&format!("{title_dim} FDTD Testbed Viewer")).await;
        let mut camera = OrbitCamera3d::default();

        #[cfg(not(feature = "dim1"))]
        camera.set_up_axis(Vec3::Z);
        #[cfg(not(feature = "dim3"))]
        camera.set_projection(Projection::Orthographic);
        #[cfg(feature = "dim3")]
        camera.set_projection(Projection::Perspective);
        let mut scene = SceneNode3d::default();

        let sim_bb = simulation.compute_bounding_box();
        let n_cells = simulation.compute_n_cells(&sim_bb, stability);
        let cell_size = simulation.fdtd_parameters.cell_size;
        visualization_mode.initialize(&mut scene, n_cells, cell_size);

        let mut selff = Self {
            window,
            camera,
            scene,
            vector_field_color: RED,
            material_region_alpha: 0.9,
            n_cells,
            cell_size,
            polarization_mode: simulation.fdtd_parameters.polarization_mode,
            visualization_mode
        };
        let regions_offset = YeeGridMaterials::compute_simulation_offset(
            &simulation.compute_bounding_box(),
            n_cells,
            simulation.fdtd_parameters.cell_size,
        );
        selff.add_region_meshes(&simulation.material_regions, regions_offset);

        Ok(selff)
    }

    /// Set clipping planes of camera. Conserves FOV, eye, at, and projection; the rest
    /// gets overwritten w/ default values.
    pub fn set_clipping_planes(&mut self, znear: f32, zfar: f32) -> &mut Self {
        let old_proj = self.camera.projection();
        let mut new_camera = OrbitCamera3d::new_with_frustum(
            self.camera.fov(),
            znear,
            zfar,
            self.camera.eye(),
            self.camera.at()
        );
        new_camera.set_projection(old_proj);
        self.camera = new_camera;
        self
    }

    /// Add material regions as meshes rendered in the scene.
    pub fn add_region_meshes(&mut self, mat_regions: &MaterialRegions, regions_offset: Vec3) {
        for (mesh, pose) in mat_regions.regions.iter()
            .filter_map(|r| r.mesh.as_ref().map(|mesh| (mesh, r.pose)))
        {
            let kiss3d_mesh = Rc::new(RefCell::new(GpuMesh3d::new(
                mesh.vertices.clone(), mesh.indices.clone(), None, None, false
            )));
            self.scene.add_mesh(kiss3d_mesh, Vec3::ONE)
                .set_pose((mat_regions.scene_pose * pose).append_translation(regions_offset))
                .set_color(GRAY.with_alpha(self.material_region_alpha))
                .enable_backface_culling(false);
        }
    }

    /// Renders one frame of the vector field `v_field`.
    ///
    /// Returns `false` if the viewer should stop rendering (e.g. when the window closes).
    pub async fn render_frame(
        &mut self,
        v_field: &[Vec4],
    ) -> bool {
        self.visualization_mode.visualize(
            v_field,
            &mut self.window,
            self.polarization_mode
        );
        self.window.render_3d(&mut self.scene, &mut self.camera).await
    }
}

/// The mode in which vector fields are visualized.
///
/// Includes modes for all dimensions.
pub enum VisualizationMode {
    /// A line graph of vector field magnitudes
    #[cfg(feature = "dim1")]
    LineGraph {
        color: Color,
        /// The maximum magnitude of vector field
        max_magnitude: Real,
        graph_max_magnitude: Real,
        positions: Vec<Vec3>,
    },
    #[cfg(feature = "dim2")]
    Quads {
        instanced_quad: SceneNode3d,
        instances: Vec<InstanceData3d>,
        color_mode: ColorMode,
    },
    #[cfg(feature = "dim3")]
    Cubes {
        instanced_cube: SceneNode3d,
        instances: Vec<InstanceData3d>,
        color_mode: ColorMode,
        alpha_mode: AlphaMode
    },
}

impl VisualizationMode {
    pub fn default_with_color_mode(color_mode: ColorMode) -> Self {
        cfg_select! {
            feature = "dim1" => Self::LineGraph {
                color: color_mode.compute_color(Real::MAX),
                max_magnitude: 1.,
                graph_max_magnitude: 0.,
                positions: vec![],
            },
            feature = "dim2" => Self::Quads {
                color_mode,
                instanced_quad: SceneNode3d::default(),
                instances: vec![],
            },
            feature = "dim3" => Self::Cubes {
                color_mode,
                instanced_cube: SceneNode3d::default(),
                instances: vec![],
                alpha_mode: AlphaMode::Mask(0.1),
            }
        }
    }

    pub fn initialize(
        &mut self,
        #[cfg_attr(feature = "dim1", allow(unused_variables))] scene: &mut SceneNode3d,
        n_cells: GridIndex,
        cell_size: Vect,
    ) {
        let mut cell_positions = vec![Vec3::ZERO; n_cells.mul_elements() as _];
        for idx_tuple in grid_cells_iter(n_cells) {
            let cell_idx = GridIndex::from_index_array(idx_tuple.into());
            let flat_idx = cell_idx.to_flat_idx(n_cells);
            cell_positions[flat_idx as usize] = (cell_idx.as_vect() * cell_size).to_3d(Vec3::ZERO);
        }

        #[cfg(feature = "dim1")]
        {
            let VisualizationMode::LineGraph {
                positions, graph_max_magnitude, ..
            } = self;
            *positions = cell_positions;
            let grid_extents = n_cells.as_vect() * cell_size;
            *graph_max_magnitude = grid_extents / 100.;
        }

        #[cfg(any(feature = "dim2", feature = "dim3"))]
        {
            let initialize_instances = |
                instanced_obj: &mut SceneNode3d,
                instances: &mut Vec<InstanceData3d>,
                color_mode: &mut ColorMode,
            | {
                let cell_size_half3 = (cell_size / 2.).to_3d(Vec3::ZERO);
                color_mode.prepare(&[]);
                let color = color_mode.compute_color(Real::MIN);

                *instances = par_iter!(cell_positions)
                    .map(|pos| {
                        InstanceData3d {
                            color,
                            position: pos + cell_size_half3,
                            ..Default::default()
                        }
                    })
                    .collect();
                instanced_obj.set_instances(instances);
                instanced_obj.enable_backface_culling(false);
            };
            match self {
                #[cfg(feature = "dim2")]
                VisualizationMode::Quads {
                    instanced_quad, instances, color_mode
                } => {
                    *instanced_quad = scene.add_quad(cell_size.x, cell_size.y, 1, 1);
                    initialize_instances(instanced_quad, instances, color_mode)
                },
                #[cfg(feature = "dim3")]
                VisualizationMode::Cubes {
                    instanced_cube, instances, color_mode, alpha_mode
                } => {
                    *instanced_cube = scene
                        .add_cube(cell_size.x, cell_size.y, cell_size.z)
                        .set_alpha_mode(*alpha_mode);
                    initialize_instances(instanced_cube, instances, color_mode)
                }
            }
        }
    }

    #[allow(unused_variables)]
    pub fn visualize(&mut self, v_field: &[Vec4], window: &mut Window, polarization_mode: PolarizationMode) {
        #[cfg(feature = "dim1")]
        {
            let Self::LineGraph {
                color,
                max_magnitude,
                graph_max_magnitude,
                positions
            } = self;

            let mut prev_pos = positions[0] + (v_field[0].xyz() / *max_magnitude) * *graph_max_magnitude;
            for (cell_pos, vector) in positions.iter()
                .zip(v_field)
                .skip(1)
            {
                let pos = cell_pos + (polarization_mode.extract_e_vector(vector) / *max_magnitude) * *graph_max_magnitude;
                window.draw_line(
                    prev_pos, pos,
                    *color, 2., false
                );
                prev_pos = pos;
            }
        }

        #[cfg(any(feature = "dim2", feature = "dim3"))]
        {
            let update_colors = |
                instanced_obj: &mut SceneNode3d,
                instances: &mut Vec<InstanceData3d>,
                color_mode: &mut ColorMode,
            | {
                let magnitudes = v_field.iter()
                    .map(|v| polarization_mode.get_e_magnitude(v))
                    .collect::<Vec<_>>();
                color_mode.prepare(&magnitudes);

                par_iter_mut!(instances)
                    .zip(into_par_iter!(magnitudes))
                    .for_each(|(inst, mag)| {
                        inst.color = color_mode.compute_color(mag);
                    });
                instanced_obj.set_instances(instances);
            };
            match self {
                #[cfg(feature = "dim2")]
                VisualizationMode::Quads {
                    instanced_quad, instances, color_mode
                } => update_colors(instanced_quad, instances, color_mode),
                #[cfg(feature = "dim3")]
                VisualizationMode::Cubes {
                    instanced_cube, instances, color_mode, ..
                } => update_colors(instanced_cube, instances, color_mode),
            }
        }
    }
}

impl Default for VisualizationMode {
    fn default() -> Self {
        Self::default_with_color_mode(ColorMode::default())
    }
}

/// How vector field magnitudes are colored
#[derive(Copy, Clone, Debug)]
pub enum ColorMode {
    /// Automatically scale the maximum magnitude as the simulation progresses.
    AutoScale {
        color_min: Color,
        color_max: Color,
        v_max: Real,
    },
    /// Set a fixed range of magnitudes for interpolating between colors
    FixedRange {
        color_min: Color,
        color_max: Color,
        /// The minimum magnitude
        v_min: Real,
        /// The maximum magnitude
        v_max: Real,
    }
}

impl ColorMode {
    pub const DEFAULT_ALPHA: f32 = 0.5;

    /// Updates the state of `self` before computing the colors of `magnitudes`.
    ///
    /// Call this before calling `compute_color` on the elements of `magnitudes`.
    pub fn prepare(&mut self, magnitudes: &[Real]) {
        match self {
            ColorMode::AutoScale { v_max, .. } => {
                let curr_max = magnitudes.iter()
                    .copied()
                    .max_by(|a, b| a.total_cmp(b));
                if let Some(max_mag) = curr_max {
                    *v_max = v_max.max(max_mag);
                }
            }
            ColorMode::FixedRange { .. } => {}
        }
    }

    pub fn compute_color(&self, val: Real) -> Color {
        match self {
            ColorMode::AutoScale { color_min, color_max, v_max } => {
                lerp_colors((val / *v_max).clamp(0., 1.), *color_min, *color_max)
            }
            ColorMode::FixedRange { color_min, color_max, v_min, v_max } => {
                lerp_colors(
                    ((val - *v_min) / (*v_max - *v_min)).clamp(0., 1.),
                    *color_min,
                    *color_max
                )
            }
        }
    }

    pub fn set_alpha(&mut self, alpha: f32) {
        match self {
            ColorMode::AutoScale { color_min, color_max, .. } => {
                *color_min = color_min.with_alpha(alpha);
                *color_max = color_max.with_alpha(alpha);
            }
            ColorMode::FixedRange { color_min, color_max, .. } => {
                *color_min = color_min.with_alpha(alpha);
                *color_max = color_max.with_alpha(alpha);
            }
        }
    }
}

impl Default for ColorMode {
    fn default() -> Self {
        #[cfg(not(feature = "dim3"))]
        {
            ColorMode::AutoScale {
                color_min: BLUE,
                color_max: RED,
                v_max: Real::MIN,
            }
        }
        #[cfg(feature = "dim3")]
        {
            ColorMode::AutoScale {
                color_min: TRANSPARENT,
                color_max: RED.with_alpha(Self::DEFAULT_ALPHA),
                v_max: Real::MIN
            }
        }
    }
}