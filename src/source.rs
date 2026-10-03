#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::gpu_util::CreateGpuBuffer;
use crate::into_par_iter;
use crate::prelude::*;
use khal::backend::GpuBuffer;
use std::num::{NonZeroI32, NonZeroU32};
use std::sync::Arc;
use taser_em_shaders::source::*;

/// Inject energy into the simulation in various ways.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Source {
    /// Dipole (magnetic or electric).
    Dipole {
        /// Choose between an electric and magnetic dipole source.
        dipole_type: DipoleType,
        /// The position in space where the source should be injected.
        position: Vect,
        /// The time (in the simulation, not real-time) when the source begins injection (in seconds).
        t_start: f32,
        /// Signal data points.
        vals: Vec<f32>,
        /// The axis on which the dipole moves. Must be a unit vector, unless
        /// you want to scale `vals` by the components of `moment`.
        moment: Vec3,
    },
    /// Total-Field / Scattered-Field source
    Tfsf {
        /// The spatial axis along which the plane wave will travel.
        spatial_axis: SpatialAxis,
        /// The direction along `spatial_axis` the wave will travel in.
        direction: Direction,
        /// The time (in the simulation, not real-time) when the source begins injection (in seconds).
        t_start: f32,
        /// Signal data points.
        vals: Vec<f32>,
        /// Polarization direction of the plane wave (unit vector)
        polarization: Vec3,
        /// The distances between the TF/SF boundary and the border/PML, in grid cells.
        ///
        /// If you want to record values behind the TF/SF boundary, `LayerWidths::splat_spatial(3)` works well.
        tfsf_buffer_width: LayerWidths,
    }
}

impl Source {
    /// Helper function that generates data points for a Gaussian curve with a maximum frequency of
    /// `f_max` (Hz).
    ///
    /// # Panics
    /// When `f_max <= 0.` or when `dt <= 0.`
    pub fn gaussian_max_f(f_max: Real, amplitude: Real, dt: Real) -> Vec<Real> {
        assert!(f_max > 0.0, "f_max must be > 0");
        let tau = core::f32::consts::FRAC_1_PI / f_max;
        let t_0 = 6. * tau;
        let approx_dur = 12. * tau;
        Self::function_data_points(dt, approx_dur, |t| {
            amplitude * core::f32::consts::E.powf(-((t - t_0) / tau).powi(2))
        })
    }

    /// Generates datapoints for one cycle of sine (amplitude of `1.`)
    pub fn sin_cycle(f: Real, dt: Real) -> Vec<Real> {
        let omega = (2. * core::f32::consts::PI) * f;
        Self::function_data_points(dt, 1. / f, |t| Real::sin(omega * t))
    }

    /// Samples data points from the function of time `f`
    ///
    /// # Panics
    /// When `dt <= 0.` or `duration <= 0.`
    pub fn function_data_points(dt: Real, duration: Real, mut f: impl FnMut(Real) -> Real) -> Vec<Real> {
        assert!(dt > 0.0, "dt must be > 0");
        assert!(duration > 0.0, "source duration must be > 0");
        let num_vals = (duration / dt) as usize;
        let mut vals = vec![0.; num_vals];

        let mut t = 0.;
        for val in vals.iter_mut() {
            *val = f(t);
            t += dt;
        }
        vals
    }
}

/// Parameters for an auxiliary grid (used for TF/SF sources)
#[derive(Clone, Debug)]
pub struct AuxGridParameters {
    pub pml_width: NonZeroU32,
    pub pml_sig_max: Real,
    pub pml_grading_order: NonZeroI32
}

pub struct SourceFunction {

}

impl ToDft for SourceFunction {
    type GpuStateType = SourceStates;

    fn to_dft(self, frequencies: Vec<Real>) -> TaserResult<(Dft<Self>, Arc<Self>)> {
        todo!()
    }

    fn get_value_position(&self) -> usize {
        todo!()
    }

    fn get_value_buffer(state: &Self::GpuStateType) -> &GpuBuffer<Real> {
        todo!()
    }
}

pub struct SourcePipeline {
    tfsf_pipeline: TfsfPipeline,
    update_source_terms: UpdateSourceTerms,
    gpu_compute_dipole_terms: GpuComputeDipoleTerms,
}

impl SourcePipeline {
    pub fn new(backend: &GpuBackend) -> TaserResult<Self> {
        Ok(Self {
            tfsf_pipeline: TfsfPipeline::new(backend)?,
            update_source_terms: UpdateSourceTerms::from_dir(backend, &crate::SPIRV_DIR)?,
            gpu_compute_dipole_terms: GpuComputeDipoleTerms::from_dir(backend, &crate::SPIRV_DIR)?,
        })
    }

    pub fn initialize(&self, pass: &mut GpuPass, sim_state: &mut FdtdLossyState) -> TaserResult<()> {
        self.tfsf_pipeline.initialize_masks(
            pass,
            sim_state,
        )?;
        Ok(())
    }

    pub fn dispatch_step(&self, pass: &mut GpuPass, sim_state: &mut FdtdLossyState) -> TaserResult<()> {
        {
            let source_states = &mut sim_state.source_states;
            self.gpu_compute_dipole_terms.call(
                pass,
                DispatchGrid::Grid(source_states.dipole_terms_workgroups),
                &mut source_states.src_h,
                &mut source_states.src_dn,
                &sim_state.t_idx,
                &source_states.source_vals,
                &source_states.dipoles,
            )?;
        }

        self.tfsf_pipeline.dispatch_step(pass, sim_state)?;

        let source_states = &mut sim_state.source_states;
        self.update_source_terms.call(
            pass,
            DispatchGrid::Grid(sim_state.update_source_terms_workgroups),
            &mut source_states.src_h,
            &mut source_states.src_dn,
            &mut sim_state.source_terms
        )?;
        
        Ok(())
    }
}

/// Buffers describing the states of different kinds of sources.
pub struct SourceStates {
    pub src_h: GpuBuffer<u32>,
    pub src_dn: GpuBuffer<u32>,
    pub dipoles: GpuBuffer<GpuDipole>,
    pub tfsf_states: Option<TfsfStates>,
    pub source_vals: GpuBuffer<f32>,
    pub dipole_terms_workgroups: [u32; 3]
}

impl SourceStates {
    // TODO: rename regions_offsets to sim_offset globally for clarity
    pub fn new(
        backend: &GpuBackend,
        sim: &FdtdLossySimulation,
        n_cells3: UVec3,
        sim_offset: Vect,
        problem_space_min: UVec3,
        problem_space_max: UVec3,
    ) -> TaserResult<Self> {
        let FdtdParameters {
            cell_size, dt, ..
        } = sim.fdtd_parameters;

        let n_cells = GridIndex::from_uvec3(n_cells3);
        
        let mut source_vals: Vec<Real> = vec![];
        
        let mut dipoles = sim.sources.iter()
            .filter_map(|source| {
                let Source::Dipole { dipole_type, position, t_start, vals, moment } = source else {
                    return None;
                };
                let pos = (sim_offset + position) / cell_size;
                let cell_grid_idx = pos.round().as_grid_index();
                debug_assert!(!pos.min_element().is_sign_negative(), "negative source position!");
                debug_assert!(!cell_grid_idx.cmpge(n_cells).any(), "Out of bounds source!");
                let start = source_vals.len();
                source_vals.extend_from_slice(vals);
                Some(GpuDipole {
                    cell_idx: cell_grid_idx.to_flat_idx(n_cells),
                    vals_start: start as u32,
                    vals_end: source_vals.len() as u32 - 1,
                    t_start: (t_start / dt) as u32,
                    moment: Vec4::from((*moment, 0.)),
                    dipole_type: *dipole_type,
                    _padding0: [0; 3],
                })
            })
            .collect::<Vec<_>>();

        if dipoles.is_empty() { dipoles.push(GpuDipole::default()) }

        let tfsf_dispatch_data = TfsfStates::new(
            backend,
            sim,
            &mut source_vals,
            n_cells3,
            problem_space_min,
            problem_space_max
        )?;

        if source_vals.is_empty() { source_vals.push(0.0); }

        let cell_count = n_cells3.element_product();
        let src_components_zero = vec![Real::to_bits(0.); cell_count as usize * 3];

        debug_assert!(!dipoles.is_empty());
        debug_assert!(!source_vals.is_empty());
        // TODO: make optional DipoleStates
        Ok(Self {
            src_h: src_components_zero.create_gpu_buffer(backend)?,
            src_dn: src_components_zero.create_gpu_buffer(backend)?,
            dipoles: dipoles.create_gpu_buffer(backend)?,
            tfsf_states: tfsf_dispatch_data,
            source_vals: source_vals.create_gpu_buffer(backend)?,
            dipole_terms_workgroups: dipole_terms_workgroups(dipoles.len() as u32),
        })
    }
}

pub struct TfsfPipeline {
    aux_grid_update: AuxGridUpdate,
    init_tfsf_masks: InitTfsfMasks,
    gpu_compute_tfsf_terms: GpuComputeTfsfTerms,
}

impl TfsfPipeline {
    pub fn new(backend: &GpuBackend) -> TaserResult<Self> {
        Ok(Self {
            aux_grid_update: AuxGridUpdate::from_dir(backend, &crate::SPIRV_DIR)?,
            init_tfsf_masks: InitTfsfMasks::from_dir(backend, &crate::SPIRV_DIR)?,
            gpu_compute_tfsf_terms: GpuComputeTfsfTerms::from_dir(backend, &crate::SPIRV_DIR)?,
        })
    }

    /// Initializes [`TfsfMask`]s of a [`TfsfStates`] before running a simulation.
    pub fn initialize_masks(
        &self,
        pass: &mut GpuPass,
        sim_state: &mut FdtdLossyState,
    ) -> TaserResult<()> {
        if let Some(tfsf_states) = &mut sim_state.source_states.tfsf_states {
            self.init_tfsf_masks.call(
                pass,
                DispatchGrid::Grid(tfsf_states.mask_init_workgroups),
                &sim_state.grid_params,
                &tfsf_states.tfsf_sources,
                &mut tfsf_states.tfsf_masks,
            )?;
        }
        Ok(())
    }

    pub fn dispatch_step(
        &self,
        pass: &mut GpuPass,
        sim_state: &mut FdtdLossyState,
    ) -> TaserResult<()> {
        let source_states = &mut sim_state.source_states;
        let Some(tfsf_states) = &mut source_states.tfsf_states else {
            return Ok(());
        };

        self.aux_grid_update.call(
            pass,
            DispatchGrid::Grid(tfsf_states.aux_grid_workgroups),
            &tfsf_states.tfsf_sources,
            &sim_state.t_idx,
            &mut tfsf_states.corrections,
            &source_states.source_vals,
            &tfsf_states.auxgr_coeffs,
            &mut tfsf_states.h,
            &mut tfsf_states.dn,
            &mut tfsf_states.en
        )?;

        self.gpu_compute_tfsf_terms.call(
            pass,
            DispatchGrid::ThreadCount(sim_state.thread_count),
            &sim_state.grid_params,
            &mut source_states.src_h,
            &mut source_states.src_dn,
            &tfsf_states.tfsf_sources,
            &tfsf_states.corrections,
            &tfsf_states.tfsf_masks,
            &sim_state.grid_coeffs,
        )?;

        Ok(())
    }
}

/// The states of TFSF sources and their auxiliary grids.
pub struct TfsfStates {
    pub tfsf_sources: GpuBuffer<GpuTfsf>,
    pub tfsf_masks: GpuBuffer<TfsfMask>,
    pub corrections: GpuBuffer<TfsfSourceValues>,
    pub auxgr_coeffs: GpuBuffer<AuxGridPmlCoeffs>,
    pub h: GpuBuffer<AuxVect>,
    pub dn: GpuBuffer<AuxVect>,
    pub en: GpuBuffer<AuxVect>,
    /// Thread count for simulating auxiliary grids for multiple TFSF sources.
    pub aux_grid_workgroups: [u32; 3],
    pub mask_init_workgroups: [u32; 3],
}

impl TfsfStates {
    pub fn new(
        backend: &GpuBackend,
        sim: &FdtdLossySimulation,
        source_vals: &mut Vec<Real>,
        n_cells3: UVec3,
        problem_space_min: UVec3,
        problem_space_max: UVec3,
    ) -> TaserResult<Option<Self>> {
        let AuxGridParameters {
            pml_width, pml_sig_max, pml_grading_order
        } = &sim.tfsf_parameters;
        let FdtdParameters {
            dt, cell_size, ..
        } = &sim.fdtd_parameters;
        let cell_count = n_cells3.element_product() as usize;

        let mut corrections = Vec::new();
        let mut coeffs = Vec::new();
        let mut zeroed_vector_fields = Vec::new();
        let mut aux_grid_n_cells_max = 0;

        let inv_d = cell_size.recip().to_3d(Vec3::ZERO);
        let tfsf_srcs = sim.sources.iter()
            .filter_map(|source_val| {
                let Source::Tfsf {
                    spatial_axis, direction, t_start, vals,
                    polarization, tfsf_buffer_width
                } = source_val else { return None };
                let a = Axis::from(*spatial_axis);
                let a1 = a.permute();
                let a2 = a1.permute();

                let inv_d_a = inv_d[a];

                let buf_width = *tfsf_buffer_width;
                let tf_min_a = problem_space_min[a] + buf_width[a].lo;
                let tf_min_a1 = problem_space_min[a1] + buf_width[a1].lo;
                let tf_min_a2 = problem_space_min[a2] + buf_width[a2].lo;

                let tf_max_a = problem_space_max[a] - buf_width[a].hi;
                let tf_max_a1 = problem_space_max[a1] - buf_width[a1].hi;
                let tf_max_a2 = problem_space_max[a2] - buf_width[a2].hi;

                let num_correction_cells = (tf_max_a - tf_min_a + 1) + 2;
                let source_cell = 1;
                let n_cells = num_correction_cells + source_cell + pml_width.get();
                aux_grid_n_cells_max = aux_grid_n_cells_max.max(n_cells);

                let corrections_start = corrections.len() as u32;
                corrections.resize(corrections.len() + num_correction_cells as usize, TfsfSourceValues::default());

                let vals_start = source_vals.len() as u32;
                source_vals.extend_from_slice(vals);

                let grid_coeffs = {
                    let sig = {
                        const HALF_CELL: Index = 1;
                        const ONE_CELL: Index = HALF_CELL*2;
                        let n_axis2x = n_cells * ONE_CELL;
                        let pml_end = match direction {
                            Direction::Positive => n_axis2x - HALF_CELL,
                            Direction::Negative => 0,
                            _ => panic!("Invalid wave direction")
                        };
                        let pml_width2x = (pml_width.get() * ONE_CELL) as Real;
                        let pml_sig_max = *pml_sig_max;
                        into_par_iter!((0..n_axis2x))
                            .map(|i| {
                                let end_dist = i.abs_diff(pml_end) as Real;
                                let pml_interp = (1. - end_dist / pml_width2x)
                                    .clamp(0., 1.);
                                pml_sig_max * pml_interp.powi(pml_grading_order.get())
                            })
                            .collect::<Vec<_>>()
                    };
                    let h_sig = sig.iter()
                        .copied()
                        .skip(1)
                        .step_by(2)
                        .collect::<Vec<_>>();
                    let dn_sig = sig.iter()
                        .copied()
                        .step_by(2)
                        .collect::<Vec<_>>();

                    let inv_dt = dt.recip();
                    let inv_mu_r_xy = Vec2::new(
                        sim.background_material.mu_r[a1].recip(),
                        sim.background_material.mu_r[a2].recip(),
                    );
                    let inv_eps_r_xy = Vec2::new(
                        sim.background_material.eps_r[a1].recip(),
                        sim.background_material.eps_r[a2].recip(),
                    );
                    // TODO: make LosslessElectricMaterial for sim.background_material.
                    into_par_iter!((0..n_cells))
                        .map(|cell_idx| {
                            let idx = cell_idx as usize;
                            let h_coeff_term0 = Vec2::splat((inv_dt + (h_sig[idx] / (2. * EPS_0))).recip());
                            let dn_coeff_term0 = Vec2::splat((inv_dt + (dn_sig[idx] / (2. * EPS_0))).recip());
                            AuxGridPmlCoeffs {
                                h1: h_coeff_term0 * (inv_dt - (h_sig[idx] / (2. * EPS_0))),
                                h2: -h_coeff_term0 * C_0 * inv_mu_r_xy,
                                dn1: dn_coeff_term0 * (inv_dt - (dn_sig[idx] / (2. * EPS_0))),
                                dn2: dn_coeff_term0 * C_0,
                                en1: inv_eps_r_xy,
                            }
                        })
                        .collect::<Vec<_>>()
                };
                let coeffs_start = coeffs.len() as u32;
                debug_assert_eq!(coeffs.len(), zeroed_vector_fields.len());
                coeffs.extend_from_slice(&grid_coeffs);
                zeroed_vector_fields.extend_from_slice(&vec![AuxVect::ZERO; n_cells as usize]);

                Some(GpuTfsf {
                    a, a1, a2,
                    direction: *direction,
                    tf_min_a, tf_min_a1, tf_min_a2,
                    tf_max_a, tf_max_a1, tf_max_a2,
                    grid_start: coeffs_start,
                    vals_start,
                    vals_end: source_vals.len() as u32 - 1,
                    t_start: (t_start / dt) as u32,
                    n_cells,
                    polarization_a1: (*polarization)[a1],
                    polarization_a2: (*polarization)[a2],
                    corrections_start,
                    num_correction_cells,
                    inv_d_a,
                    inv_d_a1: inv_d[a1],
                    inv_d_a2: inv_d[a2],
                })
            })
            .collect::<Vec<_>>();

        if tfsf_srcs.is_empty() {
            return Ok(None);
        }

        let tfsf_masks = vec![TfsfMask::default(); tfsf_srcs.len() * cell_count];

        let n_tfsf_srcs = tfsf_srcs.len() as u32;
        let aux_grid_workgroups = aux_grid_update_workgroups(n_tfsf_srcs, aux_grid_n_cells_max);
        let mask_init_workgroups = init_tfsf_masks_workgroups(n_tfsf_srcs, n_cells3);

        debug_assert!(!tfsf_srcs.is_empty());
        debug_assert!(!tfsf_masks.is_empty());
        debug_assert!(!corrections.is_empty());
        debug_assert!(!coeffs.is_empty());
        debug_assert!(!zeroed_vector_fields.is_empty());
        Ok(Some(TfsfStates {
            tfsf_sources: tfsf_srcs.create_gpu_buffer(backend)?,
            tfsf_masks: tfsf_masks.create_gpu_buffer(backend)?,
            corrections: corrections.create_gpu_buffer(backend)?,
            auxgr_coeffs: coeffs.create_gpu_buffer(backend)?,
            h: zeroed_vector_fields.create_gpu_buffer(backend)?,
            dn: zeroed_vector_fields.create_gpu_buffer(backend)?,
            en: zeroed_vector_fields.create_gpu_buffer(backend)?,
            aux_grid_workgroups,
            mask_init_workgroups,
        }))
    }
}