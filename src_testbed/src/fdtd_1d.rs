use std::cell::RefCell;
use std::num::NonZeroU32;
use std::rc::Rc;
use khal::backend::{Backend, Buffer, Encoder, GpuBackend};
use glamx::{Vec3, Vec4};
use kiss3d::prelude::{GpuMesh3d, SceneNode3d, Window, GRAY};
use kiss3d::camera::{OrbitCamera3d, Projection};
use kiss3d::event::{Action, Key};
use kiss3d::color::RED;
use taser_em1d::{grid_cells_iter, ElectricMaterial, FdtdSolver, FdtdStability, Source, C_0};
use taser_em1d::grid::{LayerWidths, MaterialRegions, PolarizationMode, YeeGrid};
use taser_em1d::prelude::GpuResult;
use taser_em1d::shaders::math::{GridIndexExt, Vect, VectorValueExt};
use crate::MAT_REGION_ALPHA;

pub async fn visualize(backend: &GpuBackend) -> GpuResult<()> {
    let stability = FdtdStability {
        dt_safety_factor: 6.,
        ..Default::default()
    };
    let f_max = 2.4e9; // 2.4 GHz
    let cell_size = stability.cell_size_from_min_wavelength(f_max);
    let dt = stability.cfl_condition(cell_size)
        .min(stability.dt_from_gaussian_freq(f_max));
    let wavelen = C_0 / f_max;

    let slab_extents = [Vect::splat(-20.), Vect::splat(-20. + wavelen * 2.)];
    let source = Source::Dipole {
        position: slab_extents[0] - wavelen * 3.,
        t_start: 0.,
        vals: Source::gaussian_max_f(f_max, 1., dt)
    };

    let mut mat_regions = MaterialRegions::new();
    let mat = ElectricMaterial {
        eps_r: Vec3::splat(7.),
        mu_r: Vec3::splat(1.),
        // sig: Vec3::splat(1e-15),
        sig: Vec3::splat(0.3),
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
    let n_cells = solver.grid_n_cells();
    let (regions_offset, coeffs) = solver.compute_pml_coeffs();
    let mut buffers = solver.create_shader_data(backend, &coeffs, regions_offset)?;

    let grid_extents = n_cells.as_vect() * cell_size;
    let mut window = Window::new("Kiss3d: cube").await;
    let mut camera = OrbitCamera3d::new_with_frustum(
        core::f32::consts::PI / 4.0,
        cell_size / 3.,
        grid_extents * 3.,
        Vec3::new(-grid_extents, 0., grid_extents / 2.),
        Vec3::Z * grid_extents / 2.
    );
    camera.set_projection(Projection::Orthographic);

    let mut scene = SceneNode3d::empty();
    for (mesh, pose) in solver.grid.material_regions.regions.iter()
        .filter_map(|r| r.mesh.as_ref().map(|mesh| (mesh, r.pose)))
    {
        let kiss3d_mesh = Rc::new(RefCell::new(GpuMesh3d::new(
            mesh.vertices.clone(), mesh.indices.clone(), None, None, false
        )));
        scene.add_mesh(kiss3d_mesh, Vec3::ONE)
            .set_pose(pose.append_translation(regions_offset))
            .set_color(GRAY.with_alpha(MAT_REGION_ALPHA));
    }

    let mut dn = vec![Vec4::ZERO; buffers.dn.buffer.len()];
    let mut n_iters = 0;
    while window.render_3d(&mut scene, &mut camera).await {
        backend.synchronize()?;
        buffers.dn.read(backend, &mut dn).await?;
        if n_iters == 1 || window.get_key(Key::P) == Action::Press {
            println!("{:?}", dn);
            println!("===============");
        }
        // Submit simulation step
        {
            let mut encoder = backend.begin_encoding();
            let mut pass = encoder.begin_pass("fdtd vis", None);
                solver.submit_step(&mut buffers, &mut pass)?;
            drop(pass);
            buffers.dn.encode_copy_cmd(&mut encoder)?;
            backend.submit(encoder)?;
            n_iters += 1;
        }

        let mut prev_pos = 0.;
        let mut prev_val = dn[0].y * cell_size;
        for (i,) in grid_cells_iter(n_cells).skip(1) {
            let pos = i.as_vect() * cell_size;
            let val = dn[i as usize].y * cell_size;
            window.draw_line(
                Vec3::new(0., prev_val, prev_pos), Vec3::new(0., val, pos),
                RED, 2., false
            );
            prev_pos = pos;
            prev_val = val;
        }
    }

    Ok(())
}