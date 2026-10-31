use rustfft::FftPlanner;
use std::sync::Arc;
use taser_em2d::prelude::*;
use taser_em2d::re_exports::ndarray::{Array, IxDyn};
use taser_em2d::shaders::dft::DftFields;
use taser_em_testbed2d::plot::*;
use taser_em_testbed2d::re_exports::anyhow;
use taser_em_testbed2d::*;

/// Binary grating example, taken from EMP-5304.
///
/// # Expected Outputs @ 10 GHz
/// | Diffraction Mode | Reflected Power | Transmitted Power |
/// |:----------------:|:---------------:|:-----------------:|
/// | -1               | 0.00%           | 15.55%            |
/// | 0                | 18.41%          | 50.48%            |
/// | 0                | 0.00%           | 15.55%            |
/// This example is close enough to these values. It's off by about 2% at most.
pub async fn binary_grating() -> anyhow::Result<()> {
    let freq = 10e9; // 10 GHz
    let sim_speed = 5;
    let cpw = 30;
    let pml_width = 20;
    let wavelen_0 = C_0 / freq;

    // Grating parameters
    let period = 1.5e-2; // 1.5 cm spatial period of grating
    let groove_depth = 0.75e-2; // 0.75 cm
    let upper_mat = ElectricMaterial::FREE_SPACE;
    let core_mat1 = ElectricMaterial {
        eps_r: Vec3::splat(9.0),
        ..ElectricMaterial::FREE_SPACE
    };
    let core_mat2 = ElectricMaterial::FREE_SPACE;
    let substrate_mat = ElectricMaterial {
        eps_r: Vec3::splat(9.0),
        ..ElectricMaterial::FREE_SPACE
    };
    let core_width1 = 0.5 * period;
    let core_width2 = period - core_width1;
    let n_ref = upper_mat.refractive_index().x;
    let n_trans = substrate_mat.refractive_index().x;

    // Construct simulation builder
    let stability = FdtdStability {
        spacer_region_widths: LayerWidths::ZERO
            .with_axis_widths(SpatialAxis::Y, LoHiWidths::ZERO.with_hi(cpw)),
        ..FdtdStability::from_cpw(cpw)
    };
    let unsnapped = stability.cell_size_from_min_wavelength(freq);
    let cell_size = stability.snap_to_critical_dim(unsnapped, unsnapped.with_x(core_width1));
    let dt = stability.cfl_condition(cell_size).min(stability.dt_from_gaussian_freq(freq));
    let fdtd_params = FdtdParameters::new(
        cell_size,
        dt,
        MaterialDiscretization::smooth_from_stability(&stability)
    );
    let pml_params = PmlParameters {
        overlay_on_grid: true,
        widths: LayerWidths::default()
            .with_axis_widths(SpatialAxis::Y, LoHiWidths::splat(pml_width)),
        ..PmlParameters::new(dt)
    };
    let mut simulation = FdtdLossySimulation::new(fdtd_params, pml_params);
    simulation.background_material = upper_mat; // Set upper material to be background mat
    simulation.grid_sizing_mode = GridSizingMode::AutoFloor(None);

    // Build the grating core and substrate
    simulation
        .fill_region(Vect::ZERO, Vect::new(core_width1, groove_depth), core_mat1)
        .fill_region(Vect::X * core_width1, Vect::new(core_width1 + core_width2, groove_depth), core_mat2)
        .fill_region(Vect::ZERO, Vect::new(period, -(wavelen_0 + pml_width as Real * cell_size.y)), substrate_mat);

    let (problem_min, problem_max) = simulation.problem_space_world(&stability);

    // Plane wave source
    let src_data_pts = Source::gaussian_max_f(freq, 1., dt);
    let tfsf = Tfsf {
        spatial_axis: SpatialAxis::Y,
        direction: Direction::Negative,
        t_start: 0.0,
        polarization: Vec3::Z,
        tfsf_buffer_width: LayerWidths::default()
            .with_axis_widths(SpatialAxis::Y, LoHiWidths::splat(3)),
        injection_plane_only: true,
    };
    simulation.add_tfsf(tfsf, src_data_pts);

    // Reflectance and transmittance planes
    let frequencies = vec![freq];
    let sim_bb = simulation.compute_bounding_box();
    let refl_plane = {
        let start = Vect::from_vec3(sim_bb.mins)
            .with_y(problem_max.y - cell_size.y); // 1 cell between PML and reflectance plane
        let mut end = start + Vect::new(sim_bb.extents().x, cell_size.y);
        if ((end.x - start.x) / cell_size.x) as Index % 2 == 0 { // ensure symmetrical diffraction modes
            end.x -= cell_size.x;
        }
        Arc::new(DftRegion::new(frequencies.clone(), start, end, DftFields::SN))
    };
    let trans_plane = {
        let start = Vect::from_vec3(sim_bb.mins)
            .with_y(problem_min.y + cell_size.y); // 1 cell between PML and transmittance plane
        let mut end = start + Vect::new(sim_bb.extents().x, cell_size.y);
        if ((end.x - start.x) / cell_size.x) as Index % 2 == 0 { // ensure symmetrical diffraction modes
            end.x -= cell_size.x;
        }
        Arc::new(DftRegion::new(frequencies.clone(), start, end, DftFields::SN))
    };
    let all_regions = &[trans_plane.clone(), refl_plane.clone()];

    // TODO: don't make pipeline/hooks store states, but take them as inputs instead
    //       this will allow using the same pipeline for different simulation instances (e.g. src-only sim)
    //       maybe add a required "state type" to the hooks trait to impl this.
    //       also remove all &mut self from the trait if possible. ideally pipelines are immutable

    /*
     * Preparing simulation for running
     */
    let backend = create_backend().await?;

    // Main simulation
    let boundary_conditions = BoundaryConditions::new(
        PeriodicBoundaryX::from_backend(&backend)?,
        PECBoundaryY::from_backend(&backend)?,
    );
    let mut pipeline = FdtdLossyPipeline::new(&backend, boundary_conditions, sim_speed)?;
    let mut state = simulation.finalize(&backend, &stability, &mut pipeline)?;
    let mut readback = FdtdStateReadback::new(&backend, &state, FdtdSimulationMode::TransverseMagneticZ)?;
    // DFT regions
    let region_states = DftRegionStates::new(&backend, &state, all_regions)?;
    let mut dft_regions_read = DftRegionReadback::new(&backend, &region_states)?;
    let mut pipeline = apply_pipeline_hooks(
        pipeline,
        DftRegionsHook::new(&backend, region_states)?
    );

    // Source-only sim
    let src_sim = simulation.sources_only();
    let boundary_conditions = BoundaryConditions::new(
        PeriodicBoundaryX::from_backend(&backend)?,
        PECBoundaryY::from_backend(&backend)?,
    );
    let mut pipeline_src = FdtdLossyPipeline::new(&backend, boundary_conditions, sim_speed)?;
    let mut state_src = src_sim.finalize(&backend, &stability, &mut pipeline_src)?;
    // DFT region in the source simulation
    let region_states_src = DftRegionStates::new(&backend, &state_src, all_regions)?;
    let mut dft_regions_read_src = DftRegionReadback::new(&backend, &region_states_src)?;
    let mut pipeline_src = apply_pipeline_hooks(
        pipeline_src,
        DftRegionsHook::new(&backend, region_states_src)?,
    );

    /*
     * Running simulation
     */
    // Diffraction modes
    let modes = {
        let desc = dft_regions_read.get_region_desc(&trans_plane)?;
        let n = desc.n_cells3.x as i32;
        assert_ne!(n % 2, 0, "DFT region must have odd number of cells for symmetrical diffraction modes");
        ((-n/2)..=(n/2)).map(|m| m as Real).collect::<Vec<_>>()
    };
    let modes = Array::from_shape_vec(IxDyn(&[modes.len(), 1, 1]), modes)?;

    // Plots
    let to_percents = Some(100.);
    let mut plot_window = PlotWindow::new("DFTs", Some("Diffraction Mode"), Some("% Power"));
    let mut refl_normalized = ComplexPlot::new("Reflected Harmonics", modes.as_slice().unwrap(), None, to_percents)?;
    let mut trans_normalized = ComplexPlot::new("Transmitted Harmonics", modes.as_slice().unwrap(), None, to_percents)?;

    // Run testbed viewer
    let vis_mode = VisualizationMode::default();
    let mut viewer = FdtdTestbedViewer::new(&simulation, &stability, vis_mode, VectorFieldVisual::En).await?;
    let steady_time = 4.5612865e-9; // found this value experimentally through multiple simulations
    let mut plot_calculated = false;
    while viewer.render_frame(&backend, &state, &mut readback).await? {
        readback.read_back_t_idx(&backend)?;
        readback.request_copy_t_idx(&backend, &state)?;
        let t = readback.get_t_idx() as Real * state.dt;

        if t < steady_time { // wait for steady-state
            pipeline.simulate(&backend, &mut state)?;
            pipeline_src.simulate(&backend, &mut state_src)?;
            continue;
        }
        if plot_calculated {
            plot_window.show(
                &mut viewer,
                vec![refl_normalized.create_line(), trans_normalized.create_line()]
            )?;
            continue;
        }

        dft_regions_read.request_copy(&backend, pipeline.hooks.state())?;
        dft_regions_read_src.request_copy(&backend, pipeline_src.hooks.state())?;
        dft_regions_read.read_back(&backend)?;
        dft_regions_read_src.read_back(&backend)?;

        // Prepare FFT
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(modes.len());

        let refl_and_trans = frequencies
            .iter()
            .enumerate()
            .map(|(freq_idx, f)| {
                let wavelen0 = C_0 / f;
                let k_0 = core::f32::consts::TAU / wavelen0;
                let k_y_inc = k_0 * n_ref;
                let k_x = -core::f32::consts::TAU * &modes / period;
                let k_y_refl = ((k_0 * n_ref).powi(2) - k_x.pow2())
                    .mapv(|v| Complex32::from(v).sqrt());
                let k_y_trans = ((k_0 * n_trans).powi(2) - k_x.pow2())
                    .mapv(|v| Complex32::from(v).sqrt());

                let src_en = dft_regions_read_src.get_en(&trans_plane, freq_idx)?.map(|v| v.z);
                let en_refl = dft_regions_read.get_en(&refl_plane, freq_idx)?.map(|v| v.z);
                let en_trans = dft_regions_read.get_en(&trans_plane, freq_idx)?.map(|v| v.z);

                // Reflectance
                let mut refl = en_refl / &src_en;
                // spatial harmonics
                fft.process(refl.as_slice_mut().unwrap());
                let mid = (refl.len() + 1) / 2;
                refl.as_slice_mut().unwrap().rotate_left(mid);
                refl = &refl / refl.len() as Real;
                // diffraction efficiency
                let refl_de = (k_y_refl / k_y_inc).map(|v| v.re)
                    * refl.mapv(|v| v.norm_sqr());
                let refl = refl_de.sum();

                // Transmittance
                let mut trans = en_trans / &src_en;
                // spatial harmonics
                fft.process(trans.as_slice_mut().unwrap());
                let mid = (trans.len() + 1) / 2;
                trans.as_slice_mut().unwrap().rotate_left(mid);
                trans = &trans / trans.len() as Real;
                // diffraction efficiency
                let trans_de = (k_y_trans / k_y_inc).map(|v| v.re)
                    * trans.mapv(|v| v.norm_sqr());
                let trans = trans_de.sum();

                Ok((refl, refl_de, trans, trans_de))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let (refl, refl_de, trans, trans_de) = &refl_and_trans[0];
        let refl_de = refl_de.map(Complex32::from);
        let trans_de = trans_de.map(Complex32::from);

        let conservation = refl + trans;
        println!("Energy Conservation: {}%", conservation * 100.);

        refl_normalized.set_y(refl_de.as_slice().unwrap());
        trans_normalized.set_y(trans_de.as_slice().unwrap());
        plot_calculated = true;
    }

    Ok(())
}