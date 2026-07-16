pub mod re_exports {
    pub use anyhow;
}

use glamx::{Vec3, Vec4, Vec4Swizzles};
use kiss3d::camera::Projection;
use kiss3d::color::{GRAY, RED};
use kiss3d::prelude::{Camera3d, Color, GpuMesh3d, OrbitCamera3d, SceneNode3d, Window};
use std::cell::RefCell;
use std::rc::Rc;
use taser_em::grid::MaterialRegions;
use taser_em::shaders::math::{GridIndex, GridIndexExt, Vect};
use taser_em::grid_cells_iter;

pub struct FdtdTestbedViewer {
    pub window: Window,
    pub camera: OrbitCamera3d,
    pub scene: SceneNode3d,
    pub vector_field_color: Color,
    pub material_region_alpha: f32
}

impl FdtdTestbedViewer {
    /// Create a new testbed viewer window, camera, and scene with Dn field color set to [`RED`].
    pub async fn new() -> anyhow::Result<Self> {
        let title_dim = cfg_select! {
            feature = "dim1" => "1D",
            feature = "dim2" => "2D",
            feature = "dim3" => "3D",
        };
        let window = Window::new(&format!("{title_dim} FDTD Testbed Viewer")).await;
        let mut camera = OrbitCamera3d::default();
        #[cfg(not(feature = "dim3"))]
        camera.set_projection(Projection::Orthographic);
        let scene = SceneNode3d::default();

        Ok(
            Self {
                window,
                camera,
                scene,
                vector_field_color: RED,
                material_region_alpha: 0.9,
            }
        )
    }

    pub fn set_clipping_planes(&mut self, znear: f32, zfar: f32) -> &mut Self {
        let mut new_camera = OrbitCamera3d::new_with_frustum(
            self.camera.fov(),
            znear,
            zfar,
            self.camera.eye(),
            self.camera.at()
        );
        #[cfg(not(feature = "dim3"))]
        new_camera.set_projection(Projection::Orthographic);
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
        n_cells: GridIndex,
        cell_size: Vect
    ) -> bool {
        let continue_rendering = self.window.render_3d(&mut self.scene, &mut self.camera).await;
        let cell_size_splat = Vec3::splat(cell_size);

        #[cfg(feature = "dim1")]
        {
            let mut prev_pos = v_field[0].xyz().with_z(0.0);
            for cell_idx in grid_cells_iter(n_cells)
                .map(|i| GridIndex::from_index_array(i.into()))
            {
                let vector = v_field[cell_idx as usize].xyz() * cell_size_splat;
                let pos = vector.with_z(cell_idx.as_vect() * cell_size);
                self.window.draw_line(
                    prev_pos, pos,
                    self.vector_field_color, 2., false
                );
                prev_pos = pos;
            }
        }
        #[cfg(not(feature = "dim1"))]
        {
            todo!("visualize 2d and 3d vector fields")
        }

        continue_rendering
    }
}