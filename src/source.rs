#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::gpu_util::CreateGpuBuffer;
use crate::into_par_iter;
use crate::prelude::*;
use khal::backend::GpuBuffer;
use std::num::{NonZeroI32, NonZeroU32};
use std::sync::Arc;
use taser_em_shaders::fdtd::{GridParameters, PmlCoefficients};
use taser_em_shaders::source::*;

pub struct Source<S: SourceType> {
    /// Reference to the struct describing this source
    pub src_ref: Arc<S>,
    /// Source data points
    pub data_points: Vec<Real>,
}

impl<S: SourceType> Source<S> {
    pub fn from_source(source: S, data_points: Vec<Real>) -> Self {
        Self {
            src_ref: Arc::new(source),
            data_points,
        }
    }
}

impl Source<()> {
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

pub struct SourceFunction<S: SourceType> {
    #[allow(dead_code)]
    source: Arc<S>,
    val_pos: usize
}

impl<S: SourceType> SourceFunction<S> {
    pub fn new(source: &Arc<S>, source_states: &SourceStates) -> TaserResult<Self> {
        Ok(Self {
            source: source.clone(),
            val_pos: S::get_value_position(source, source_states)
                .ok_or(SourceError::NoSourceInstance(std::any::type_name::<S>().to_string()))?,
        })
    }
}

impl<S: SourceType> ToDft for SourceFunction<S> {
    fn to_dft(self, frequencies: Vec<Real>) -> TaserResult<(Dft<Self>, Arc<Self>)> {
        let func = Arc::new(self);
        Ok((Dft::new(frequencies, func.clone())?, func))
    }

    fn get_value_position(&self) -> usize {
        self.val_pos
    }

    fn get_value_buffer(state: &FdtdLossyState) -> TaserResult<&GpuBuffer<Real>> {
        S::get_value_buffer(&state.source_states)
                .ok_or(SourceError::NoSourceInstance(std::any::type_name::<S>().to_string()).into())
    }
}

pub struct SourcePipeline {
    dipole_pipeline: DipolePipeline,
    tfsf_pipeline: TfsfPipeline,
    zeroed_src_terms_update: GpuZeroAndUpdateSourceTerms,
    src_terms_update: GpuUpdateSourceTerms,
}

impl SourcePipeline {
    pub fn new(backend: &GpuBackend) -> TaserResult<Self> {
        Ok(Self {
            dipole_pipeline: DipolePipeline::new(backend)?,
            tfsf_pipeline: TfsfPipeline::new(backend)?,
            zeroed_src_terms_update: GpuZeroAndUpdateSourceTerms::from_dir(backend, &crate::SPIRV_DIR)?,
            src_terms_update: GpuUpdateSourceTerms::from_dir(backend, &crate::SPIRV_DIR)?,
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
        let mut zero_src_terms = true;
        if let Some(dipole_states) = &mut sim_state.source_states.dipole_states {
            self.dipole_pipeline.dispatch_step(
                pass,
                dipole_states,
                &sim_state.t_idx
            )?;
            self.update_src_terms(
                pass,
                sim_state.update_source_terms_workgroups,
                &mut dipole_states.src_h,
                &mut dipole_states.src_dn,
                &mut sim_state.source_terms,
                &mut zero_src_terms
            )?;
        }

        if let Some(tfsf_states) = &mut sim_state.source_states.tfsf_states {
            self.tfsf_pipeline.dispatch_step(
                pass,
                tfsf_states,
                &sim_state.t_idx,
                &sim_state.grid,
                &sim_state.pml_coeffs,
            )?;
            self.update_src_terms(
                pass,
                sim_state.update_source_terms_workgroups,
                &mut tfsf_states.src_h,
                &mut tfsf_states.src_dn,
                &mut sim_state.source_terms,
                &mut zero_src_terms
            )?;
        }

        Ok(())
    }

    fn update_src_terms(
        &self,
        pass: &mut GpuPass,
        workgroups: [u32; 3],
        src_h: &mut GpuBuffer<Index>,
        src_dn: &mut GpuBuffer<Index>,
        source_terms: &mut GpuBuffer<SourceTerms>,
        zero_src_terms: &mut bool,
    ) -> TaserResult<()> {
        if *zero_src_terms {
            self.zeroed_src_terms_update.call(
                pass,
                DispatchGrid::Grid(workgroups),
                src_h,
                src_dn,
                source_terms
            )?;
            *zero_src_terms = false;
        } else {
            self.src_terms_update.call(
                pass,
                DispatchGrid::Grid(workgroups),
                src_h,
                src_dn,
                source_terms
            )?;
        }
        Ok(())
    }
}

/// Buffers describing the states of different kinds of sources.
pub struct SourceStates {
    /// The global source terms, applied directly to H field update equations. Stored as `u32`s for
    /// atomic support.
    pub src_h: GpuBuffer<u32>,
    /// The global source terms, applied directly to Dn field update equations. Stored as `u32`s for
    /// atomic support.
    pub src_dn: GpuBuffer<u32>,
    pub dipole_states: Option<DipoleStates>,
    pub tfsf_states: Option<TfsfStates>,
}

impl SourceStates {
    pub fn new(
        backend: &GpuBackend,
        sim: &FdtdLossySimulation,
        n_cells3: UVec3,
        sim_offset: Vect,
        problem_space_min: UVec3,
        problem_space_max: UVec3,
    ) -> TaserResult<Self> {
        let n_cells = GridIndex::from_uvec3(n_cells3);

        let tfsf_dispatch_data = TfsfStates::new(
            backend,
            sim,
            n_cells3,
            problem_space_min,
            problem_space_max
        )?;

        let cell_count = n_cells.element_product();
        let src_components_zero = vec![Real::to_bits(0.); cell_count as usize * 3];

        Ok(Self {
            src_h: src_components_zero.create_gpu_buffer(backend)?,
            src_dn: src_components_zero.create_gpu_buffer(backend)?,
            dipole_states: DipoleStates::new(backend, sim, sim_offset, n_cells)?,
            tfsf_states: tfsf_dispatch_data,
        })
    }
}

/// Dipole source (magnetic or electric).
#[derive(Clone, Debug)]
pub struct Dipole {
    /// Choose between an electric and magnetic dipole source.
    pub dipole_type: DipoleType,
    /// The position in space where the source should be injected.
    pub position: Vect,
    /// The time (in the simulation, not real-time) when the source begins injection (in seconds).
    pub t_start: f32,
    /// The axis on which the dipole moves. Must be a unit vector, unless
    /// you want to scale `vals` by the components of `moment`.
    pub moment: Vec3,
}

impl Dipole {
    /// Construct electric dipole
    pub fn electric(position: Vect, moment: Vec3) -> Self {
        Self {
            dipole_type: DipoleType::Electric,
            position,
            t_start: 0.0,
            moment,
        }
    }
}

impl SourceType for Dipole {
    fn get_value_position(ref_self: &Arc<Self>, state: &SourceStates) -> Option<usize> {
        state.dipole_states.as_ref().map(|st|
            st.dipole_refs.iter()
                .position(|r| Arc::ptr_eq(r, ref_self))
                .expect("The source at `ref_self` should be in `state`")
        )
    }

    fn get_value_buffer(state: &SourceStates) -> Option<&GpuBuffer<Real>> {
        state.dipole_states.as_ref().map(|st| &st.curr_src_vals)
    }
}

pub struct DipolePipeline {
    gpu_compute_dipole_terms: GpuComputeDipoleTerms,
}

impl DipolePipeline {
    pub fn new(backend: &GpuBackend) -> TaserResult<Self> {
        Ok(Self {
            gpu_compute_dipole_terms: GpuComputeDipoleTerms::from_dir(backend, &crate::SPIRV_DIR)?,
        })
    }
    
    pub fn dispatch_step(&self, pass: &mut GpuPass, dipole_states: &mut DipoleStates, t_idx: &GpuBuffer<Index>) -> TaserResult<()> {
        self.gpu_compute_dipole_terms.call(
            pass,
            DispatchGrid::Grid(dipole_states.dipole_terms_workgroups),
            &mut dipole_states.src_h,
            &mut dipole_states.src_dn,
            &mut dipole_states.curr_src_vals,
            t_idx,
            &dipole_states.source_vals,
            &dipole_states.dipoles,
        )?;
        Ok(())
    }
}

pub struct DipoleStates {
    pub dipole_refs: Vec<Arc<Dipole>>,
    pub src_h: GpuBuffer<u32>,
    pub src_dn: GpuBuffer<u32>,
    pub dipoles: GpuBuffer<GpuDipole>,
    pub curr_src_vals: GpuBuffer<Real>,
    pub source_vals: GpuBuffer<Real>,
    pub dipole_terms_workgroups: [u32; 3]
}

impl DipoleStates {
    pub fn new(
        backend: &GpuBackend,
        sim: &FdtdLossySimulation,
        sim_offset: Vect,
        n_cells: GridIndex
    ) -> TaserResult<Option<Self>> {
        let FdtdParameters {
            cell_size, dt, ..
        } = sim.fdtd_parameters;

        let mut source_vals = vec![];
        let (dipole_refs, dipoles) = sim.dipoles.iter()
            .filter_map(|source| {
                let Dipole { dipole_type, position, t_start, moment } = &*source.src_ref;
                let vals = &source.data_points;
                if vals.is_empty() { return None; }

                let pos = (sim_offset + position) / cell_size;
                let cell_grid_idx = pos.round().as_grid_index();
                if pos.cmplt(Vect::ZERO).any() || cell_grid_idx.cmpge(n_cells).any() {
                    return Some(Err(
                        SourceError::SourceOutOfBounds(std::any::type_name_of_val(source).to_string())
                            .into()
                    ))
                }
                let start = source_vals.len();
                source_vals.extend_from_slice(vals);
                let dipole = GpuDipole {
                    cell_idx: cell_grid_idx.to_flat_idx(n_cells),
                    vals_start: start as u32,
                    vals_end: source_vals.len() as u32 - 1,
                    t_start: (t_start / dt) as u32,
                    moment: Vec4::from((*moment, 0.)),
                    dipole_type: *dipole_type,
                    _padding0: [0; 3],
                };
                Some(Ok((source.src_ref.clone(), dipole)))
            })
            .collect::<TaserResult<(Vec<_>, Vec<_>)>>()?;

        if dipoles.is_empty() {
            return Ok(None);
        }

        let cell_count = n_cells.element_product();
        let src_components_zero = vec![Real::to_bits(0.); cell_count as usize * 3];

        Ok(Some(Self {
            dipole_refs,
            src_h: src_components_zero.create_gpu_buffer(backend)?,
            src_dn: src_components_zero.create_gpu_buffer(backend)?,
            dipoles: dipoles.create_gpu_buffer(backend)?,
            curr_src_vals: vec![0.; dipoles.len()].create_gpu_buffer(backend)?,
            source_vals: source_vals.create_gpu_buffer(backend)?,
            dipole_terms_workgroups: dipole_terms_workgroups(dipoles.len() as u32),
        }))
    }
}

/// Total-Field / Scattered-Field source
///
/// A type of plane wave source that only exists within a rectangular region.
#[derive(Clone, Debug)]
pub struct Tfsf {
    /// The spatial axis along which the plane wave will travel.
    pub spatial_axis: SpatialAxis,
    /// The direction along `spatial_axis` the wave will travel in.
    pub direction: Direction,
    /// The time (in the simulation, not real-time) when the source begins injection (in seconds).
    pub t_start: f32,
    /// Polarization direction of the plane wave (unit vector)
    pub polarization: Vec3,
    /// The distances between the TF/SF boundary and the border/PML, in grid cells.
    ///
    /// If you want to record values behind the TF/SF boundary, `LayerWidths::splat_spatial(3)` works well.
    pub tfsf_buffer_width: LayerWidths,
}

impl SourceType for Tfsf {
    fn get_value_position(ref_self: &Arc<Self>, state: &SourceStates) -> Option<usize> {
        state.tfsf_states.as_ref().map(|st|
            st.tfsf_refs.iter()
                .position(|r| Arc::ptr_eq(r, ref_self))
                .expect("The source at `ref_self` should be in `state`")
        )
    }

    fn get_value_buffer(state: &SourceStates) -> Option<&GpuBuffer<Real>> {
        state.tfsf_states.as_ref().map(|st| &st.curr_src_vals)
    }
}

/// Parameters for an auxiliary grid (used for TF/SF sources)
#[derive(Clone, Debug)]
pub struct AuxGridParameters {
    pub pml_width: NonZeroU32,
    pub pml_sig_max: Real,
    pub pml_grading_order: NonZeroI32
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
                &sim_state.grid,
                &tfsf_states.tfsf_sources,
                &mut tfsf_states.tfsf_masks,
            )?;
        }
        Ok(())
    }

    pub fn dispatch_step(
        &self,
        pass: &mut GpuPass,
        tfsf_states: &mut TfsfStates,
        t_idx: &GpuBuffer<u32>,
        grid: &GpuBuffer<GridParameters>,
        pml_coeffs: &GpuBuffer<PmlCoefficients>,
    ) -> TaserResult<()> {
        self.aux_grid_update.call(
            pass,
            DispatchGrid::Grid(tfsf_states.aux_grid_workgroups),
            &mut tfsf_states.tfsf_sources,
            t_idx,
            &mut tfsf_states.corrections,
            &tfsf_states.source_vals,
            &tfsf_states.auxgr_coeffs,
            &mut tfsf_states.h,
            &mut tfsf_states.dn,
            &mut tfsf_states.en
        )?;

        self.gpu_compute_tfsf_terms.call(
            pass,
            DispatchGrid::Grid(tfsf_states.tfsf_terms_workgroups),
            grid,
            &mut tfsf_states.src_h,
            &mut tfsf_states.src_dn,
            &mut tfsf_states.curr_src_vals,
            &tfsf_states.tfsf_sources,
            &tfsf_states.corrections,
            &tfsf_states.tfsf_masks,
            pml_coeffs,
        )?;

        Ok(())
    }
}

/// The states of TFSF sources and their auxiliary grids.
pub struct TfsfStates {
    pub tfsf_refs: Vec<Arc<Tfsf>>,
    pub src_h: GpuBuffer<u32>,
    pub src_dn: GpuBuffer<u32>,
    pub tfsf_sources: GpuBuffer<GpuTfsf>,
    pub source_vals: GpuBuffer<Real>,
    pub curr_src_vals: GpuBuffer<Real>,
    pub tfsf_masks: GpuBuffer<TfsfMask>,
    pub corrections: GpuBuffer<TfsfSourceValues>,
    pub auxgr_coeffs: GpuBuffer<AuxGridPmlCoeffs>,
    pub h: GpuBuffer<AuxVect>,
    pub dn: GpuBuffer<AuxVect>,
    pub en: GpuBuffer<AuxVect>,
    /// Thread count for simulating auxiliary grids for multiple TFSF sources.
    pub aux_grid_workgroups: [u32; 3],
    pub mask_init_workgroups: [u32; 3],
    /// Workgroup count for [`GpuComputeTfsfTerms`].
    pub tfsf_terms_workgroups: [u32; 3],
}

impl TfsfStates {
    pub fn new(
        backend: &GpuBackend,
        sim: &FdtdLossySimulation,
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

        let mut source_vals = vec![];
        let mut corrections = Vec::new();
        let mut coeffs = Vec::new();
        let mut zeroed_vector_fields = Vec::new();
        let mut aux_grid_n_cells_max = 0;

        let inv_d = cell_size.recip().to_3d(Vec3::ZERO);
        let (tfsf_refs, tfsf_srcs) = sim.tfsf_sources.iter()
            .filter_map(|source| {
                let Tfsf {
                    spatial_axis, direction, t_start,
                    polarization, tfsf_buffer_width
                } = &*source.src_ref;
                let vals = &source.data_points;
                if vals.is_empty() { return None; }

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

                let pml_coeffs = {
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
                coeffs.extend_from_slice(&pml_coeffs);
                zeroed_vector_fields.extend_from_slice(&vec![AuxVect::ZERO; n_cells as usize]);
                let tfsf = GpuTfsf {
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
                    curr_src_val: 0.0,
                };

                Some((source.src_ref.clone(), tfsf))
            })
            .collect::<(Vec<_>, Vec<_>)>();

        if tfsf_srcs.is_empty() {
            return Ok(None);
        }

        let tfsf_masks = vec![TfsfMask::default(); tfsf_srcs.len() * cell_count];

        let n_tfsf_srcs = tfsf_srcs.len() as u32;
        let aux_grid_workgroups = aux_grid_update_workgroups(n_tfsf_srcs, aux_grid_n_cells_max);
        let mask_init_workgroups = init_tfsf_masks_workgroups(n_tfsf_srcs, n_cells3);

        let cell_count = n_cells3.element_product();
        let src_components_zero = vec![Real::to_bits(0.); cell_count as usize * 3];

        debug_assert!(!tfsf_srcs.is_empty());
        debug_assert!(!tfsf_masks.is_empty());
        debug_assert!(!corrections.is_empty());
        debug_assert!(!coeffs.is_empty());
        debug_assert!(!zeroed_vector_fields.is_empty());
        Ok(Some(TfsfStates {
            tfsf_refs,
            src_h: src_components_zero.create_gpu_buffer(backend)?,
            src_dn: src_components_zero.create_gpu_buffer(backend)?,
            tfsf_sources: tfsf_srcs.create_gpu_buffer(backend)?,
            source_vals: source_vals.create_gpu_buffer(backend)?,
            curr_src_vals: vec![0.; tfsf_srcs.len()].create_gpu_buffer(backend)?,
            tfsf_masks: tfsf_masks.create_gpu_buffer(backend)?,
            corrections: corrections.create_gpu_buffer(backend)?,
            auxgr_coeffs: coeffs.create_gpu_buffer(backend)?,
            h: zeroed_vector_fields.create_gpu_buffer(backend)?,
            dn: zeroed_vector_fields.create_gpu_buffer(backend)?,
            en: zeroed_vector_fields.create_gpu_buffer(backend)?,
            aux_grid_workgroups,
            mask_init_workgroups,
            tfsf_terms_workgroups: tfsf_terms_workgroups(n_cells3),
        }))
    }
}

/// Trait for sources that inject energy into the simulation in various ways.
///
/// Also has functions for
pub trait SourceType {
    /// Identical to [`ToDft::get_value_position`]
    fn get_value_position(ref_self: &Arc<Self>, state: &SourceStates) -> Option<usize>;
    /// Identical to [`ToDft::get_value_buffer`]
    fn get_value_buffer(state: &SourceStates) -> Option<&GpuBuffer<Real>>;
}

impl SourceType for () {
    fn get_value_position(_ref_self: &Arc<Self>, _state: &SourceStates) -> Option<usize> {
        None
    }

    fn get_value_buffer(_state: &SourceStates) -> Option<&GpuBuffer<Real>> {
        None
    }
}

#[derive(thiserror::Error, Debug)]
pub enum SourceError {
    #[error("No instances of source type {0} is in the simulation")]
    NoSourceInstance(String),
    #[error("A source of type {0} was found outside the simulation grid")]
    SourceOutOfBounds(String)
}