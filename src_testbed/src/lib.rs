mod util;

pub mod re_exports {
    pub use anyhow;
}

use glamx::{Vec3, Vec3Swizzles, Vec4, Vec4Swizzles};
use kiss3d::camera::Projection;
use kiss3d::color::{BLUE, GRAY, RED};
use kiss3d::prelude::{Camera3d, Color, GpuMesh3d, OrbitCamera3d, SceneNode3d, Window};
use std::cell::RefCell;
use std::rc::Rc;
use taser_em::grid::{MaterialRegions, YeeGridMaterials};
use taser_em::shaders::math::{GridIndex, GridIndexExt, Vect};
use taser_em::{grid_cells_iter, FdtdLossySimulation, FdtdStability};
use taser_em::prelude::{Real, VectExt, VectorValueExt};

pub struct FdtdTestbedViewer {
    pub window: Window,
    pub camera: OrbitCamera3d,
    pub scene: SceneNode3d,
    pub vector_field_color: Color,
    pub material_region_alpha: f32,
    pub n_cells: GridIndex,
    pub cell_size: Vect,
    pub visualization_mode: VisualizationMode,
}

impl FdtdTestbedViewer {
    /// Create a new testbed viewer window, camera, and scene with the visualized vector field color
    /// set to [`RED`].
    ///
    /// Defaults to [`ColorMode::AutoScale`] for visualizing field magnitudes
    pub async fn new(simulation: &FdtdLossySimulation, stability: &FdtdStability) -> anyhow::Result<Self> {
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
        let mut scene = SceneNode3d::default();

        let sim_bb = simulation.compute_bounding_box();
        let n_cells = simulation.compute_n_cells(&sim_bb, stability);
        let cell_size = simulation.fdtd_parameters.cell_size;
        let mut visualization_mode = VisualizationMode::default();
        visualization_mode.initialize(&mut scene, n_cells, cell_size);
        // TODO: make slab size proportional to grid size

        let mut selff = Self {
            window,
            camera,
            scene,
            vector_field_color: RED,
            material_region_alpha: 0.9,
            n_cells,
            cell_size,
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
                .set_color(GRAY.with_alpha(self.material_region_alpha));
        }
    }

    /// Renders one frame of the vector field `v_field`.
    ///
    /// Returns `false` if the viewer should stop rendering (e.g. when the window closes).
    pub async fn render_frame(
        &mut self,
        v_field: &[Vec4],
    ) -> bool {
        let continue_rendering = self.window.render_3d(&mut self.scene, &mut self.camera).await;
        self.visualization_mode.visualize(
            v_field,
            &mut self.window,
            self.cell_size
        );

        continue_rendering
    }
}

/// The mode in which vector fields are visualized.
///
/// Includes modes for all dimensions.
#[derive(Clone)]
pub enum VisualizationMode {
    /// A line graph of vector field magnitudes
    #[cfg(feature = "dim1")]
    LineGraph {
        color: Color,
        /// The maximum value that the line graph can have.
        max: Real,
        positions: Vec<Vec3>,
    },
    #[cfg(feature = "dim2")]
    Quads {
        color_mode: ColorMode,
        quads: Vec<SceneNode3d>
    },
    #[cfg(feature = "dim3")]
    Cubes {
        color_mode: ColorMode,
        cubes: Vec<SceneNode3d>
    },
    #[cfg(not(feature = "dim1"))]
    Arrows {
        color_mode: ColorMode,
        arrow_starts: Vec<Vec3>
    }
}


impl VisualizationMode {
    pub fn initialize(&mut self, scene: &mut SceneNode3d, n_cells: GridIndex, cell_size: Vect) {
        let cell_positions = grid_cells_iter(n_cells)
            .map(|i| (GridIndex::from_index_array(i.into()).as_vect() * cell_size)
                .to_3d(Vec3::ZERO)
            )
        .collect::<Vec<_>>();
        
        match self {
            #[cfg(feature = "dim1")]
            VisualizationMode::LineGraph { positions, .. } => {
                *positions = cell_positions;
            },
            #[cfg(feature = "dim2")]
            VisualizationMode::Quads { quads, .. } => {
                *quads = cell_positions.iter()
                    .map(|pos| {
                        scene
                            .add_quad(cell_size.x, cell_size.y, 1, 1)
                            .translate(*pos)
                    })
                    .collect();
            }
            #[cfg(not(feature = "dim1"))]
            VisualizationMode::Arrows { arrow_starts, .. } => {
                *arrow_starts = cell_positions;
            }
            #[cfg(feature = "dim3")]
            VisualizationMode::Cubes { .. } => {
                todo!()
            }
        }
    }

    #[allow(unused_variables)]
    pub fn visualize(&mut self, v_field: &[Vec4], window: &mut Window, cell_size: Vect) {
        #[cfg(feature = "dim1")]
        {
            let Self::LineGraph { color, max, positions } = self else { unreachable!() };
            let mut prev_pos = positions[0] + (v_field[0].xyz() / *max) * cell_size;
            for (cell_pos, vector) in positions.iter()
                .zip(v_field)
                .skip(1)
            {
                let pos = cell_pos + (vector.xyz() / *max) * cell_size;
                window.draw_line(
                    prev_pos, pos,
                    *color, 2., false
                );
                prev_pos = pos;
            }
        }
        #[cfg(feature = "dim2")]
        {
            todo!()
        }
        #[cfg(feature = "dim3")]
        {
            todo!()
        }
    }
}

impl Default for VisualizationMode {
    fn default() -> Self {
        #[cfg(feature = "dim1")]
        {
            Self::LineGraph {
                color: RED,
                max: 1.,
                positions: vec![],
            }
        }
        #[cfg(feature = "dim2")]
        {
            Self::Quads {
                color_mode: Default::default(),
                quads: Vec::new(),
            }
        }
        #[cfg(feature = "dim3")]
        {
            Self::Cubes {
                color_mode: Default::default(),
                cubes: Vec::new(),
            }
        }
    }
}

/// How vector field magnitudes are colored
#[derive(Copy, Clone, Debug)]
pub enum ColorMode {
    /// Automatically scale the maximum magnitude as the simulation progresses
    AutoScale {
        color_min: Color,
        color_max: Color,
        max: Real
    },
    /// Set a fixed range of magnitudes for interpolating between colors
    FixedRange {
        color_min: Color,
        color_max: Color,
        max: Real
    }
}

impl Default for ColorMode {
    fn default() -> Self {
        ColorMode::AutoScale {
            color_min: BLUE,
            color_max: RED,
            max: 0.
        }
    }
}