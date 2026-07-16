mod fdtd_1d;

use std::cell::RefCell;
use glamx::{Vec3, Vec4};
use khal::backend::{Backend, Buffer, Encoder, GpuBackend, WebGpu};
use std::num::NonZeroU32;
use std::rc::Rc;
use kiss3d::camera::Projection;
use kiss3d::color::{GRAY, RED};
use kiss3d::prelude::{Camera3d, Color, GpuMesh3d, OrbitCamera3d, SceneNode3d, Window};
use taser_em::grid::{LayerWidths, MaterialRegions, PolarizationMode, YeeGrid};
use taser_em::prelude::GpuResult;
use taser_em::shaders::math::{GridIndex, GridIndexExt, Real, Vect, VectorValueExt};
use taser_em::{grid_cells_iter, ElectricMaterial, FdtdSolver, FdtdStability, Source, C_0};

pub const MAT_REGION_ALPHA: f32 = 0.9;

pub struct FdtdTestbedViewer {
    pub window: Window,
    pub camera: OrbitCamera3d,
    pub scene: SceneNode3d,
    pub dn_field_color: Color
}

impl FdtdTestbedViewer {
    /// Create a new testbed viewer window, camera, and scene with Dn field color set to [`RED`].
    pub async fn new() -> anyhow::Result<Self> {
        #[cfg(feature = "dim1")]
        let title = "1D FDTD Testbed";
        #[cfg(feature = "dim2")]
        let title = "2D FDTD Testbed";
        #[cfg(feature = "dim3")]
        let title = "3D FDTD Testbed";
        let window = Window::new(title).await;
        let mut camera = OrbitCamera3d::default();
        camera.set_projection(Projection::Orthographic);
        let scene = SceneNode3d::default();

        Ok(
            Self {
                window,
                camera,
                scene,
                dn_field_color: RED,
            }
        )
    }

    pub fn set_clipping_planes(&mut self, znear: f32, zfar: f32) {
        let new_camera = OrbitCamera3d::new_with_frustum(
            self.camera.fov(),
            znear,
            zfar,
            self.camera.eye(),
            self.camera.at()
        );
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
                .set_color(GRAY.with_alpha(MAT_REGION_ALPHA));
        }
    }

    /// Renders one frame.
    ///
    /// Returns `false` if the viewer should stop rendering (e.g. when the window closes).
    pub async fn render_frame(
        &mut self,
        dn_field: Vec<Vec4>,
        n_cells: GridIndex,
        cell_size: Vect
    ) -> bool {
        todo!()
    }
}

const WARMUP_ITERS: usize = 10;
const BENCH_ITERS: usize = 1000;

async fn benchmark(backend: &GpuBackend) -> GpuResult<f32> {
    let stability = FdtdStability::default();
    let f_max = 2.4e9; // 2.4 GHz
    let cell_size = stability.cell_size_from_min_wavelength(f_max);
    let dt = stability.cfl_condition(cell_size);
    let wavelen = C_0 / f_max;

    let slab_extents = [Vect::splat(1.), Vect::splat(1. + wavelen * 2.)];
    let source = Source::Dipole {
        position: 1. - wavelen,
        t_start: 0.,
        vals: Source::gaussian_max_f(f_max, 1., dt)
    };

    let mut mat_regions = MaterialRegions::new();
    let mat = ElectricMaterial {
        eps_r: Vec3::splat(7.),
        mu_r: Vec3::splat(1.),
        sig: Vec3::splat(1e-15),
    };
    mat_regions.fill_region(slab_extents[0], slab_extents[1], mat);
    let grid = YeeGrid::new(
        cell_size,
        PolarizationMode::TransverseMagnetic,
        // PolarizationMode::TransverseElectric,
        mat_regions,
        NonZeroU32::new(3).unwrap(),
        LayerWidths::splat(10)
    );
    let mut solver = FdtdSolver::new(backend, grid, dt)?;
    solver.add_source(source);

    // Warmup
    let (regions_offset, coeffs) = solver.compute_pml_coeffs();
    let mut buffers = solver.create_shader_data(backend, &coeffs, regions_offset)?;
    for _ in 0..WARMUP_ITERS {
        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("fdtd warmup", None);
        solver.submit_step(&mut buffers, &mut pass)?;
        drop(pass);
        backend.submit(encoder)?;
        backend.synchronize()?;
    }

    // Timed iters
    let start = std::time::Instant::now();
    for _ in 0..BENCH_ITERS {
        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("fdtd bench", None);
        solver.submit_step(&mut buffers, &mut pass)?;
        drop(pass);
        backend.submit(encoder)?;
        backend.synchronize()?;
    }
    let time = start.elapsed().as_secs_f32() / BENCH_ITERS as f32;
    let mut out = vec![glamx::Vec4::ZERO; buffers.h.buffer.len()];
    backend.slow_read_buffer(&buffers.dn.buffer, &mut out).await?;
    println!("{:?}", out);
    Ok(time)
}
