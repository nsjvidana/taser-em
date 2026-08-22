mod util;

pub mod re_exports {
    pub use anyhow;
    pub use kiss3d;
}

use crate::util::lerp_colors;
use glamx::*;
use kiss3d::camera::Projection;
use kiss3d::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;
use taser_em::grid::{MaterialRegions, YeeGridMaterials};
use taser_em::prelude::*;
use taser_em::*;

#[cfg(not(feature = "dim1"))]
use kiss3d::prelude::InstanceData3d;

#[cfg(feature = "rayon")]
#[allow(unused_imports)]
use rayon::prelude::*;

pub struct FdtdTestbedViewer {
    pub window: Window,
    pub camera: OrbitCamera3d,
    pub scene: SceneNode3d,
    pub material_region_alpha: f32,
    pub n_cells: GridIndex,
    pub cell_size: Vect,
    pub polarization_mode: PolarizationMode,
    pub visualization_mode: VisualizationMode,
    /// A light source locked at the camera's location (optional, but is a point source by default)
    pub cam_light: Option<SceneNode3d>,
}

impl FdtdTestbedViewer {
    pub const DEFAULT_REGION_ALPHA: f32 = cfg_select! {
        feature = "dim3" => 1.,
        _ => 0.8
    };

    /// Create a new testbed viewer window, camera, and scene with the visualized vector field color
    /// set to [`RED`].
    ///
    /// Defaults to [`ColorMode::AutoScale`] for visualizing field magnitudes
    pub async fn new(
        simulation: &FdtdLossySimulation,
        stability: &FdtdStability,
        mut visualization_mode: VisualizationMode,
    ) -> anyhow::Result<Self> {
        let title_dim = cfg_select! {
            feature = "dim1" => "1D",
            feature = "dim2" => "2D",
            feature = "dim3" => "3D",
        };
        let window = Window::new(&format!("{title_dim} FDTD Testbed Viewer")).await;
        let camera = {
            let mut camera = OrbitCamera3d::default();
            #[cfg(not(feature = "dim1"))]
            camera.set_up_axis(Vec3::Z);
            #[cfg(not(feature = "dim3"))]
            camera.set_projection(Projection::Orthographic);
            #[cfg(feature = "dim3")]
            camera.set_projection(Projection::Perspective);
            camera
        };
        let mut scene = SceneNode3d::default();

        let sim_bb = simulation.compute_bounding_box();
        let n_cells = simulation.compute_n_cells(&sim_bb, stability);
        let cell_size = simulation.fdtd_parameters.cell_size;
        visualization_mode.initialize(&mut scene, n_cells, cell_size);

        // Default cam light
        let cam_light = Some(
            scene.add_point_light((n_cells.as_vect() * cell_size).length() * 2.)
        );

        let mut selff = Self {
            window,
            camera,
            scene,
            material_region_alpha: Self::DEFAULT_REGION_ALPHA,
            n_cells,
            cell_size,
            polarization_mode: simulation.fdtd_parameters.polarization_mode,
            visualization_mode,
            cam_light,
        };
        let regions_offset = YeeGridMaterials::compute_simulation_offset(
            &simulation.compute_bounding_box(),
            n_cells,
            simulation.fdtd_parameters.cell_size,
        );
        selff.update_cam_light();
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
        for MaterialRegion {
            shape, pose,
            mesh,
            ..
        } in mat_regions.regions.iter()
        {
            let mut node = shape.as_cuboid()
                .map(|cuboid| {
                    let ext = cuboid.half_extents * 2.;
                    self.scene
                        .add_cube(ext.x, ext.y, ext.z)
                })
                .or_else(||
                    shape
                        .as_capsule()
                        .map(|caps| self.scene.add_capsule(caps.radius, caps.height()))
                )
                .or_else(||
                    shape
                        .as_ball()
                        .map(|ball| self.scene.add_sphere(ball.radius))
                )
                .unwrap_or_else(|| {
                    let mesh = mesh
                        .clone()
                        .expect("Failed to visualize material region");
                    let kiss3d_mesh = Rc::new(RefCell::new(GpuMesh3d::new(
                        mesh.vertices, mesh.indices, None, None, false
                    )));
                    self.scene
                        .add_mesh(kiss3d_mesh, Vec3::ONE)
                });
            let region_pose_world = (mat_regions.scene_pose * pose).append_translation(regions_offset);
            node
                .set_pose(region_pose_world)
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
        let window_bool = self.window.render_3d(&mut self.scene, &mut self.camera).await;
        self.visualization_mode.visualize(
            v_field,
            &mut self.window,
            self.polarization_mode
        );
        self.update_cam_light();
        window_bool
    }

    pub fn update_cam_light(&mut self) {
        let Some(cam_light) = &mut self.cam_light else { return; };
        let eye = self.camera.eye();
        cam_light.set_position(eye);
    }
}

/// The mode in which vector fields are visualized.
///
/// Includes modes for all dimensions.
#[non_exhaustive]
pub enum VisualizationMode {
    /// A line graph of vector field magnitudes
    #[cfg(feature = "dim1")]
    LineGraph {
        color_mode: ColorMode,
        graph_max_magnitude: Real,
        data_point_magnitude: Real,
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
        alpha_mode: AlphaMode,
        alpha: f32
    },
}

impl VisualizationMode {
    #[cfg(feature = "dim3")]
    pub const DEFAULT_VISUAL_ALPHA: f32 = 0.2;

    /// Return a new [`Self`] with a new [`ColorMode`]
    pub fn with_color_mode(mut self, color_mode: ColorMode) -> Self {
        cfg_select! {
            feature = "dim1" => {
                let Self::LineGraph {
                    color_mode: self_color, ..
                } = &mut self;
                *self_color = color_mode;
                self
            }
            feature = "dim2" => {
                let Self::Quads {
                    color_mode: self_color_mode, ..
                } = &mut self;
                *self_color_mode = color_mode;
                self
            }
            feature = "dim3" => {
                let Self::Cubes {
                    color_mode: self_color_mode, ..
                } = &mut self;
                *self_color_mode = color_mode;
                self
            }
        }
    }

    /// Set the [`AlphaMode`] and alpha of the visual.
    ///
    /// `vis_alpha` edits the alpha applied to the entire visual.
    #[cfg(feature = "dim3")]
    pub fn with_alpha(mut self, alpha_mode: AlphaMode, vis_alpha: f32) -> Self {
        let Self::Cubes { alpha_mode: mode, alpha, .. } = &mut self;
        *mode = alpha_mode;
        *alpha = vis_alpha;
        self
    }

    pub fn initialize(
        &mut self,
        #[cfg_attr(feature = "dim1", allow(unused_variables))] scene: &mut SceneNode3d,
        n_cells: GridIndex,
        cell_size: Vect,
    ) {
        let mut cell_positions = vec![Vec3::ZERO; n_cells.element_product() as _];
        for idx_tuple in grid_cells_iter(n_cells) {
            let cell_idx = GridIndex::from_index_array(idx_tuple.into());
            let flat_idx = cell_idx.to_flat_idx(n_cells);
            cell_positions[flat_idx as usize] = (cell_idx.as_vect() * cell_size).to_3d(Vec3::ZERO);
        }

        #[cfg(feature = "dim1")]
        {
            let VisualizationMode::LineGraph {
                positions, data_point_magnitude, ..
            } = self;
            *positions = cell_positions;
            let grid_extents = n_cells.as_vect() * cell_size;
            *data_point_magnitude = grid_extents / 10.;
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
                    instanced_cube, instances, color_mode, alpha_mode, alpha
                } => {
                    *instanced_cube = scene
                        .add_cube(cell_size.x, cell_size.y, cell_size.z)
                        .set_alpha_mode(*alpha_mode)
                        .set_color(WHITE.with_alpha(*alpha));
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
                color_mode,
                graph_max_magnitude,
                data_point_magnitude: graph_max_val,
                positions
            } = self;

            let magnitudes = par_iter!(v_field)
                .map(|v| polarization_mode.get_e_magnitude(v))
                .collect::<Vec<_>>();

            color_mode.prepare(&magnitudes);
            #[allow(unreachable_patterns)]
            let curr_max_mag = match color_mode {
                ColorMode::AutoScale { v_max, .. } => *v_max,
                ColorMode::FixedRange { v_max, .. } => *v_max,
                _ =>
                    par_iter!(magnitudes)
                        .copied()
                        .max_by(|a, b| a.total_cmp(b))
                        .expect("Cannot have zero-sized vector field")
            };

            *graph_max_magnitude = graph_max_magnitude.max(curr_max_mag);

            let line_positions = par_iter!(positions)
                .zip(par_iter!(v_field))
                .map(|(cell_pos, vector)|
                    cell_pos + (polarization_mode.extract_e_vector(vector) / *graph_max_magnitude) * *graph_max_val
                )
                .collect::<Vec<_>>();
            let mut prev_pos = positions[0];
            for (pos, magnitude) in line_positions.into_iter()
                .zip(magnitudes)
                .skip(1)
            {
                window.draw_line(
                    prev_pos, pos,
                    color_mode.compute_color(magnitude), 2., false
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
                instanced_obj.apply_to_object_mut(&mut |o| {
                    let magnitudes = par_iter!(v_field)
                        .map(|v| polarization_mode.get_e_magnitude(v))
                        .collect::<Vec<_>>();
                    color_mode.prepare(&magnitudes);
                    let inst_count = o
                        .instances()
                        .borrow_mut()
                        .colors
                        .len();
                    let colors = into_par_iter!(magnitudes)
                        .take(inst_count)
                        .map(|mag| {
                            let c = color_mode.compute_color(mag);
                            [c.r, c.g, c.b, c.a]
                        })
                        .collect::<Vec<_>>();
                    *o.instances().borrow_mut().colors.data_mut() = Some(colors);
                });
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
        cfg_select! {
            feature = "dim1" => Self::LineGraph {
                color_mode: ColorMode::default(),
                graph_max_magnitude: 0.,
                data_point_magnitude: 0.0 ,
                positions: vec![],
            },
            feature = "dim2" => Self::Quads {
                color_mode: ColorMode::default(),
                instanced_quad: SceneNode3d::default(),
                instances: vec![],
            },
            feature = "dim3" => Self::Cubes {
                color_mode: ColorMode::default(),
                instanced_cube: SceneNode3d::default(),
                instances: vec![],
                alpha_mode: AlphaMode::Blend,
                alpha: Self::DEFAULT_VISUAL_ALPHA,
            }
        }
    }
}

/// How vector field magnitudes are colored
#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
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
    pub const DEFAULT_ALPHA: f32 = 1.;

    /// Updates the state of `self` before computing the colors of `magnitudes`.
    ///
    /// Call this before calling `compute_color` on the elements of `magnitudes`.
    pub fn prepare(&mut self, magnitudes: &[Real]) {
        match self {
            ColorMode::AutoScale { v_max, .. } => {
                let curr_max = par_iter!(magnitudes)
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
}

impl Default for ColorMode {
    fn default() -> Self {
        #[cfg(not(feature = "dim3"))]
        {
            ColorMode::AutoScale {
                color_min: cfg_select! {
                    feature = "dim1" => RED,
                    feature = "dim2" => BLUE,
                },
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