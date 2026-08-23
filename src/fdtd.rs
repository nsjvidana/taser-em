pub use taser_em_shaders::fdtd::DipoleType;

use crate::prelude::*;
use crate::gpu_util::{CreateGpuBuffer, CreateGpuBufferReadable, GpuBufferReadable};
use derivative::Derivative;
use khal::Shader;
use parry3d::bounding_volume::Aabb;
use std::num::{NonZeroI32, NonZeroU32};
use taser_em_shaders::fdtd::*;
use crate::*;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

// TODO: Docs.
pub struct FdtdLossySimulation {
    pub material_regions: MaterialRegions,
    pub background_material: ElectricMaterial,
    pub sources: Vec<Source>,
    pub fdtd_parameters: FdtdParameters,
    pub pml_parameters: PmlParameters,
    pub tfsf_parameters: TfsfParameters
}

impl FdtdLossySimulation {
    pub fn new(fdtd_parameters: FdtdParameters, pml_parameters: PmlParameters) -> Self {
        Self {
            material_regions: MaterialRegions::new(),
            background_material: ElectricMaterial::FREE_SPACE,
            sources: vec![],
            fdtd_parameters,
            pml_parameters,
            tfsf_parameters: TfsfParameters {
                pml_width: NonZeroU32::new(12).unwrap(),
                pml_sig_max: pml_parameters.sig_max,
                pml_grading_order: pml_parameters.grading_order,
            },
        }
    }

    pub fn add_source(&mut self, source: Source) -> &mut Self {
        self.sources.push(source);
        self
    }

    pub fn finalize(
        &self,
        backend: &GpuBackend,
        stability: &FdtdStability,
    ) -> TaserResult<FdtdLossyState> {
        let FdtdParameters {
            cell_size, dt, polarization_mode, ..
        } = self.fdtd_parameters;

        let sim_bb = self.compute_bounding_box();
        let n_cells = self.compute_n_cells(&sim_bb, stability);
        let n_cells3 = n_cells.n_cells_to_3d();

        let grid_mats = self.create_material_grid(&sim_bb, n_cells);
        let (regions_offset, grid_coeffs) = PmlCoefficientsGrid::new(&grid_mats, self.pml_parameters, dt);

        let cell_count = n_cells.element_product();
        let mut problem_space_min = GridIndex::ONE;
        self.pml_parameters.widths
            .iter_spatial_axes()
            .for_each(|(s_axis, w)| problem_space_min[s_axis] += w.lo);
        let mut problem_space_max = n_cells - 2;
        self.pml_parameters.widths
            .iter_spatial_axes()
            .for_each(|(s_axis, w)| problem_space_max[s_axis] -= w.hi);

        let mut source_vals: Vec<Real> = vec![];
        let regions_offset = Vect::from_vec3(regions_offset);
        let mut dipoles = self.sources.iter()
            .filter_map(|source| {
                let Source::Dipole { dipole_type, position, t_start, vals, moment } = source else {
                    return None;
                };
                let pos = (regions_offset + position) / cell_size;
                let cell_grid_idx = pos.as_grid_index();
                debug_assert!(!pos.min_element().is_sign_negative(), "negative source position!");
                debug_assert!(!pos.as_grid_index().cmpge(n_cells).any(), "Out of bounds source!");
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
        let tfsf_dispatch_data = self.create_tfsf_sources(
            backend,
            &mut source_vals,
            n_cells3,
            problem_space_min.cell_idx_to_3d(),
            problem_space_max.cell_idx_to_3d()
        )?;
        if dipoles.is_empty() { dipoles.push(GpuDipole::default()) }
        if source_vals.is_empty() { source_vals.push(0.0); }

        let flat_idx_incrs = {
            let mut incrs = UVec3::ZERO;
            for (spatial_axis, axis) in SpatialAxis::ALL_SPATIAL.into_iter()
                .zip(SpatialAxis::ALL_AXES)
            {
                let mut grid_incr = GridIndex::ZERO;
                grid_incr[spatial_axis] = 1;
                incrs[axis] = grid_incr.to_flat_idx(n_cells);
            }
            incrs
        };

        let cell_size3 = cell_size.to_3d(Vec3::ZERO);
        let grid_params = GridParameters {
            flat_idx_incrs,
            n_cells3: n_cells.n_cells_to_3d(),
            dt,
            d: cell_size3,
            inv_dt: dt.recip(),
            polarization_mode_index: polarization_mode.into(),
            cell_count,
            inv_d: cell_size3.recip(),
            problem_space_min: problem_space_min.to_3d(UVec3::ONE),
            _padding1: 0,
            problem_space_max: problem_space_max.to_3d(UVec3::ONE),
            _padding2: 0,
        };

        let cell_count = n_cells.element_product() as usize;
        let zeroed_vector_field = vec![Vec4::ZERO; cell_count];

        let buffers = FdtdLossyState {
            // Uniforms / thread-independent vars
            grid_params: grid_params.create_gpu_uniform(backend)?,
            t_idx: 0.create_gpu_buffer_readable(backend)?,
            // Vector fields
            h: zeroed_vector_field.create_gpu_buffer_readable(backend)?,
            dn: zeroed_vector_field.create_gpu_buffer_readable(backend)?,
            en: zeroed_vector_field.create_gpu_buffer_readable(backend)?,
            // For computing source terms
            dipoles: dipoles.create_gpu_buffer(backend)?,
            source_vals: source_vals.create_gpu_buffer(backend)?,
            // For update equation terms
            source_terms: vec![SourceTerms::default(); cell_count].create_gpu_buffer(backend)?,
            int_terms: vec![PmlIntegrals::default(); cell_count].create_gpu_buffer(backend)?,
            grid_coeffs: grid_coeffs.coeffs.create_gpu_buffer(backend)?,
            // Misc data
            thread_count: n_cells.n_cells_to_3d().to_array(),
            n_cells,
            tfsf_dispatch_data
        };

        Ok(buffers)
    }

    pub fn create_material_grid(&self, simulation_bb: &Aabb, n_cells: GridIndex) -> YeeGridMaterials {
        let FdtdParameters { material_discretization, cell_size, .. } =
            &self.fdtd_parameters;
        match material_discretization {
            MaterialDiscretization::Rough =>
                YeeGridMaterials::new_material_grid(
                    n_cells,
                    *cell_size,
                    simulation_bb,
                    &self.material_regions,
                    self.background_material,
                ),
            MaterialDiscretization::Smooth { resolution } => {
                let res = resolution.get();
                YeeGridMaterials::new_material_grid(
                    n_cells * res,
                    cell_size / res as Real,
                    simulation_bb,
                    &self.material_regions,
                    self.background_material,
                ).downscaled(*resolution)
            }
        }
    }

    pub fn create_tfsf_sources(
        &self,
        backend: &GpuBackend,
        source_vals: &mut Vec<Real>,
        n_cells3: UVec3,
        problem_space_min: UVec3,
        problem_space_max: UVec3,
    ) -> TaserResult<TfsfDispatchData> {
        let TfsfParameters {
            pml_width, pml_sig_max, pml_grading_order
        } = &self.tfsf_parameters;
        let FdtdParameters {
            dt, cell_size, ..
        } = &self.fdtd_parameters;
        let cell_count = n_cells3.element_product() as usize;

        let mut corrections = Vec::new();
        let mut coeffs = Vec::new();
        let mut zeroed_vector_fields = Vec::new();
        let mut aux_grid_n_cells_max = 0;

        let inv_d = cell_size.recip().to_3d(Vec3::ZERO);
        let mut tfsf_srcs = self.sources.iter()
            .filter_map(|source_val| {
                let Source::TFSF {
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
                            WaveDirection::Positive => n_axis2x - HALF_CELL,
                            WaveDirection::Negative => 0,
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
                        self.background_material.mu_r[a1].recip(),
                        self.background_material.mu_r[a2].recip(),
                    );
                    let inv_eps_r_xy = Vec2::new(
                        self.background_material.eps_r[a1].recip(),
                        self.background_material.eps_r[a2].recip(),
                    );
                    // TODO: loss (there's probably a use to having lossy background material)
                    // let mat_sig = self.background_material.sig;
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
                    a,
                    a1,
                    a2,
                    direction: *direction,
                    tf_min_a,
                    tf_min_a1,
                    tf_min_a2,
                    tf_max_a,
                    tf_max_a1,
                    tf_max_a2,
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

        let mut tfsf_masks = vec![TfsfMask::default(); tfsf_srcs.len() * cell_count];

        let has_tfsf_sources = !tfsf_srcs.is_empty();

        if tfsf_srcs.is_empty() { tfsf_srcs.push(GpuTfsf::default()) }
        if tfsf_masks.is_empty() { tfsf_masks.push(TfsfMask::default()) }
        if corrections.is_empty() { corrections.push(TfsfSourceValues::default()) }
        if coeffs.is_empty() { coeffs.push(AuxGridPmlCoeffs::default()) }

        let aux_grid_thread_count = has_tfsf_sources
            .then_some([tfsf_srcs.len() as u32, 1, aux_grid_n_cells_max]);
        let mask_init_thread_count = has_tfsf_sources
            .then(|| {
                cfg_select! {
                    feature = "dim1" => n_cells3.with_x(tfsf_srcs.len() as Index).to_array(),
                    feature = "dim2" => n_cells3.with_z(tfsf_srcs.len() as Index).to_array(),
                    feature = "dim3" => n_cells3.with_z(tfsf_srcs.len() as Index * n_cells3.z).to_array(),
                }
            });

        Ok(TfsfDispatchData {
            tfsf_sources: tfsf_srcs.create_gpu_buffer(backend)?,
            tfsf_masks: tfsf_masks.create_gpu_buffer(backend)?,
            corrections: corrections.create_gpu_buffer_readable(backend)?,
            auxgr_coeffs: coeffs.create_gpu_buffer(backend)?,
            h: zeroed_vector_fields.create_gpu_buffer_readable(backend)?,
            dn: zeroed_vector_fields.create_gpu_buffer_readable(backend)?,
            en: zeroed_vector_fields.create_gpu_buffer_readable(backend)?,
            aux_grid_thread_count,
            mask_init_thread_count,
        })
    }

    /// Compute the dimensions of a grid that can encompass `simulation_bb`, then add spacer regions
    /// from `stability` and PML widths from `self`.
    pub fn compute_n_cells(&self, simulation_bb: &Aabb, stability: &FdtdStability) -> GridIndex {
        let cell_size = self.fdtd_parameters.cell_size;
        let n_cells_vec3 = (simulation_bb.extents() / cell_size.to_3d(Vec3::ONE)).ceil();
        let materials_n_cells = Vect::from_vec3(n_cells_vec3).as_grid_index();

        let mut n_cells = stability.spacer_region_widths
            .sum_with_n_cells(materials_n_cells);
        n_cells = self.pml_parameters.widths.sum_with_n_cells(n_cells);
        LayerWidths::splat_spatial(1).sum_with_n_cells(n_cells)
    }

    /// Compute the bounding box surrounding all objects and sources in the simulation.
    pub fn compute_bounding_box(&self) -> Aabb {
        let mut regions_bb = self.material_regions.compute_bounding_box();
        let regions_center = regions_bb.center();
        let source_pts = self.sources.iter()
            .map(|src| {
                match src {
                    Source::Dipole { position, .. } => position.to_3d(Vec3::ZERO),
                    Source::TFSF { .. } => { regions_center }
                }
            })
            .collect::<Vec<_>>();
        for pt in source_pts.iter() {
            regions_bb.mins = regions_bb.mins.min(*pt);
            regions_bb.maxs = regions_bb.maxs.max(*pt);
        }
        // ensure the simulation encompasses everything by adding machine eps
        regions_bb.add_half_extents(Vec3::splat(Real::EPSILON))
    }
}

/// The shader pipeline for running diagonal anisotropy simulation with UPML.
pub struct FdtdLossyPipeline<BC: BoundaryCondition> {
    init_tfsf_masks: InitTfsfMasks,
    boundary_condition: BC,
    aux_grid_update: AuxGridUpdate,
    compute_source_terms: GpuComputeSourceTerms,
    update: GpuLossyUpdate,
    pub num_steps_per_submission: usize,
}

impl<BC: BoundaryCondition> FdtdLossyPipeline<BC> {
    pub fn new(backend: &GpuBackend, boundary_condition: BC, num_steps_per_submission: usize) -> TaserResult<Self> {
        Ok(Self {
            boundary_condition,
            init_tfsf_masks: InitTfsfMasks::from_dir(backend, &crate::SPIRV_DIR)?,
            aux_grid_update: AuxGridUpdate::from_dir(backend, &crate::SPIRV_DIR)?,
            compute_source_terms: GpuComputeSourceTerms::from_dir(backend, &crate::SPIRV_DIR)?,
            update: GpuLossyUpdate::from_dir(backend, &crate::SPIRV_DIR)?,
            num_steps_per_submission,
        })
    }

    /// Create new pipeline and dispatch initialization shaders to the GPU at the same time (calls [`FdtdLossyPipeline::initialize`]).
    pub fn new_initialized(
        backend: &GpuBackend,
        boundary_condition: BC,
        num_steps_per_submission: usize,
        state: &mut FdtdLossyState
    ) -> TaserResult<Self> {
        let pipeline = Self::new(backend, boundary_condition, num_steps_per_submission)?;

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("2d fdtd example", None);
        pipeline.initialize(&mut pass, state)?;
        drop(pass);
        backend.submit(encoder)?;

        Ok(pipeline)
    }

    pub fn initialize(
        &self,
        pass: &mut GpuPass,
        state: &mut FdtdLossyState
    ) -> TaserResult<()> {
        if let Some(thread_count) = state.tfsf_dispatch_data.mask_init_thread_count {
            self.init_tfsf_masks.call(
                pass,
                DispatchGrid::ThreadCount(thread_count),
                &state.grid_params,
                &state.tfsf_dispatch_data.tfsf_sources,
                &mut state.tfsf_dispatch_data.tfsf_masks,
            )?;
        }
        Ok(())
    }

    pub fn dispatch_steps(
        &mut self,
        pass: &mut GpuPass,
        state: &mut FdtdLossyState,
    ) -> TaserResult<()> {
        for _ in 0..self.num_steps_per_submission {
            self.boundary_condition.call(
                pass,
                &state.grid_params,
                &mut state.h.buffer,
                &mut state.dn.buffer,
                &mut state.en.buffer,
                state.thread_count
            )?;

            if let Some(thread_count) = state.tfsf_dispatch_data.aux_grid_thread_count {
                let tfsf = &mut state.tfsf_dispatch_data;
                self.aux_grid_update.call(
                    pass,
                    DispatchGrid::ThreadCount(thread_count),
                    &tfsf.tfsf_sources,
                    &state.t_idx.buffer,
                    &mut tfsf.corrections.buffer,
                    &state.source_vals,
                    &tfsf.auxgr_coeffs,
                    &mut tfsf.h.buffer,
                    &mut tfsf.dn.buffer,
                    &mut tfsf.en.buffer
                )?;
            }

            self.compute_source_terms.call(
                pass,
                DispatchGrid::ThreadCount(state.thread_count),
                &state.grid_params,
                &state.t_idx.buffer,
                &mut state.source_terms,
                &state.source_vals,
                &state.dipoles,
                &state.tfsf_dispatch_data.tfsf_sources,
                &state.tfsf_dispatch_data.corrections.buffer,
                &state.tfsf_dispatch_data.tfsf_masks,
                &state.grid_coeffs,
            )?;
            self.update.call(
                pass,
                DispatchGrid::ThreadCount(state.thread_count),
                &state.grid_params,
                &mut state.t_idx.buffer,
                &mut state.h.buffer,
                &mut state.dn.buffer,
                &mut state.en.buffer,
                &mut state.int_terms,
                &state.grid_coeffs,
                &state.source_terms,
            )?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct FdtdParameters {
    pub cell_size: Vect,
    pub dt: Real,
    pub polarization_mode: PolarizationMode,
    pub material_discretization: MaterialDiscretization
}

/// Buffers and data needed for running the shader
pub struct FdtdLossyState {
    // Uniforms / thread-independent vars
    pub grid_params: GpuBuffer<GridParameters>,
    pub t_idx: GpuBufferReadable<u32>,
    // Vector fields
    pub h: GpuBufferReadable<Vec4>,
    pub dn: GpuBufferReadable<Vec4>,
    pub en: GpuBufferReadable<Vec4>,
    // For computing source terms
    pub dipoles: GpuBuffer<GpuDipole>,
    pub tfsf_dispatch_data: TfsfDispatchData,
    pub source_vals: GpuBuffer<f32>,
    // For update equation terms
    pub source_terms: GpuBuffer<SourceTerms>,
    pub int_terms: GpuBuffer<PmlIntegrals>,
    pub grid_coeffs: GpuBuffer<PmlCoefficients>,
    // Misc data
    pub thread_count: [u32; 3],
    pub n_cells: GridIndex,
}

#[derive(Clone, Debug)]
pub struct TfsfParameters {
    pub pml_width: NonZeroU32,
    pub pml_sig_max: Real,
    pub pml_grading_order: NonZeroI32
}

pub struct TfsfDispatchData {
    pub tfsf_sources: GpuBuffer<GpuTfsf>,
    pub tfsf_masks: GpuBuffer<TfsfMask>,
    pub corrections: GpuBufferReadable<TfsfSourceValues>,
    pub auxgr_coeffs: GpuBuffer<AuxGridPmlCoeffs>,
    pub h: GpuBufferReadable<AuxVect>,
    pub dn: GpuBufferReadable<AuxVect>,
    pub en: GpuBufferReadable<AuxVect>,
    /// Thread count for simulating auxiliary grids for ALL plane waves.
    ///
    /// Is [`None`] only when there are no TF/SF sources.
    pub aux_grid_thread_count: Option<[u32; 3]>,
    pub mask_init_thread_count: Option<[u32; 3]>,
}

/// Parameters judging how the PML will be constructed in the simulation
#[derive(Copy, Clone)]
pub struct PmlParameters {
    /// Widths of PML along each axis (widths for low and high end of each axis).
    pub widths: LayerWidths,
    /// Maximum conductivity of the PML
    pub sig_max: Real,
    /// The order of the monomial that ramps PML conductivity up to `sig_max`
    pub grading_order: NonZeroI32
}

impl PmlParameters {
    /// A convenient constructor for a [`PmlParameters`] with some generally stable values.
    pub fn new(dt: Real) -> Self {
        Self {
            widths: LayerWidths::splat_spatial(12),
            sig_max: FdtdStability::pml_sig_max(dt),
            grading_order: NonZeroI32::new(3).unwrap(),
        }
    }
}

/// Helper struct containing parameters and functions for ensuring simulation stability.
///
/// The default of this struct contains hardcoded values that are generally stable.
#[derive(Derivative, Clone)]
#[derivative(Default)]
pub struct FdtdStability {
    #[derivative(Default(value = "10"))]
    pub cells_per_wavelength: Index,
    /// Divides CFL condition upper bound by `dt_safety_factor`.
    ///
    /// `dt_safety_factor > 1.` to improve stability.
    #[derivative(Default(value = "2."))]
    pub dt_safety_factor: Real,
    #[derivative(Default(value = "10"))]
    pub source_resolution: Index,
    #[derivative(Default(value = "NonZeroU32::new(3).unwrap()"))]
    pub material_resolution: NonZeroU32,
    #[derivative(Default(value = "LayerWidths::splat_spatial(10)"))]
    pub spacer_region_widths: LayerWidths,
}

impl FdtdStability {
    pub fn cell_size_from_min_wavelength(&self, f_max: Real) -> Vect {
        let min_wavelen = C_0 / f_max;
        let cell_size = min_wavelen / self.cells_per_wavelength as Real;
        Vect::from_array([cell_size; DIM])
    }

    pub fn cfl_condition(&self, cell_size: Vect) -> Real {
        let cell_size_term = cell_size
            .map(|v| {
                v.powi(2).recip()
            })
            .element_sum()
            .sqrt();
        let safety_factor = self.dt_safety_factor.max(1.);
        1. / (C_0 * cell_size_term * safety_factor)
    }

    pub fn snap_to_critical_dim(&self, cell_size: Vect, critical_dim: Vect) -> Vect {
        let cells_per_crit_dim = (critical_dim / cell_size).ceil();
        critical_dim / cells_per_crit_dim
    }

    /// Computes a stable maximum conductivity for a PML
    #[inline]
    pub fn pml_sig_max(dt: Real) -> Real {
        EPS_0 / (2. * dt)
    }

    /// Compute a stable dt from a gaussian curve maximum frequency
    #[inline]
    pub fn dt_from_gaussian_freq(&self, f_max: Real) -> Real {
        let tau = core::f32::consts::FRAC_1_PI / f_max;
        tau / self.source_resolution as f32
    }
}

#[derive(Shader)]
pub struct PECBoundary {
    kernel: GpuPecBoundary
}

impl BoundaryCondition for PECBoundary {
    fn call(
        &mut self,
        pass: &mut GpuPass,
        grid: &GpuBuffer<GridParameters>,
        h: &mut GpuBuffer<Vec4>,
        dn: &mut GpuBuffer<Vec4>,
        en: &mut GpuBuffer<Vec4>,
        thread_count: [u32; 3]
    ) -> TaserResult<()>
    {
        self.kernel.call(
            pass,
            DispatchGrid::ThreadCount(thread_count),
            grid,
            h,
            dn,
            en
        )?;
        Ok(())
    }
}

pub trait BoundaryCondition {
    // TODO: might need different parameters for anisotropy...
    fn call(
        &mut self,
        pass: &mut GpuPass,
        grid: &GpuBuffer<GridParameters>,
        h: &mut GpuBuffer<Vec4>,
        dn: &mut GpuBuffer<Vec4>,
        en: &mut GpuBuffer<Vec4>,
        thread_count: [u32; 3],
    ) -> TaserResult<()>;
}

#[derive(Copy, Clone, Debug)]
pub enum MaterialDiscretization {
    Rough,
    Smooth { resolution: NonZeroU32 }
}

/// Relative material properties
#[derive(Copy, Clone, Debug)]
pub struct ElectricMaterial {
    /// Relative permittivity
    pub eps_r: Vec3,
    /// Relative permeability
    pub mu_r: Vec3,
    /// Conductivity of the material (S/m)
    pub sig: Vec3,
}

impl ElectricMaterial {
    /// A material representing free space
    pub const FREE_SPACE: Self = Self {
        eps_r: Vec3::ONE, mu_r: Vec3::ONE, sig: Vec3::ZERO,
    };
    /// An invalid electric material with all values set to zero
    pub const ZERO: Self = Self {
        eps_r: Vec3::ZERO, mu_r: Vec3::ZERO, sig: Vec3::ZERO,
    };

    /// Compute refractive index on all axes
    #[allow(unused_variables)]
    pub fn refractive_index(&self) -> Vec3 {
        (self.eps_r * self.mu_r).sqrt()
    }
}

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
        /// The axis on which the dipole moves. **Must** be a **unit vector** (unless
        /// you want to scale `vals` by the magnitude of `moment`).
        moment: Vec3,
    },
    /// Total-Field / Scattered-Field source
    TFSF {
        /// The spatial axis along which the plane wave will travel.
        spatial_axis: SpatialAxis,
        /// The direction along `spatial_axis` the wave will travel in.
        direction: WaveDirection,
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
    pub fn gaussian_max_f(f_max: f32, amplitude: f32, dt: f32) -> Vec<f32> {
        assert!(f_max > 0.0, "f_max must be > 0");
        let tau = core::f32::consts::FRAC_1_PI / f_max;
        let t_0 = 6. * tau;
        let approx_dur = 12. * tau;
        Self::function_data_points(dt, approx_dur, |t| {
            amplitude * core::f32::consts::E.powf(-((t - t_0) / tau).powi(2))
        })
    }

    /// Samples data points from the function of time `f`
    ///
    /// # Panics
    /// When `dt <= 0.` or `duration <= 0.`
    pub fn function_data_points(dt: f32, duration: f32, mut f: impl FnMut(f32) -> f32) -> Vec<f32> {
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

/// Utility struct for reading back vector field data to the host device (CPU):
///
/// Follow these steps to get data:
/// 1. Use the request functions (e.g. [`request_copy_dn`](Self::request_copy_dn), [`request_copy_fields`](Self::request_copy_fields))
///    to initiate readback.
/// 2. Use the read-back functions to copy data to the CPU (e.g. [`read_back_dn`](Self::read_back_dn), [`read_back_fields`](Self::read_back_fields))
/// 3. Get vector field data using the appropriate functions
///    (e.g. [`get_dn_field`](Self::get_dn_field), [`dn_magnitudes`](Self::dn_magnitudes), [`h_magnitudes`](Self::h_magnitudes))
pub struct FdtdStateReadback {
    h: Vec<Vec4>,
    dn: Vec<Vec4>,
    en: Vec<Vec4>,
    h_read: GpuReadback<Vec4>,
    dn_read: GpuReadback<Vec4>,
    en_read: GpuReadback<Vec4>,
    #[cfg(not(feature = "dim3"))]
    mode: FdtdSimulationMode
}

macro_rules! request_copy_fn {
    ($name:ident, $read:ident, $buf:ident) => {
        #[inline]
        pub fn $name(&mut self, backend: &GpuBackend, state: &FdtdLossyState) -> TaserResult<()> {
            if self.$read.is_idle() {
                self.$read.request_copy(backend, &state.$buf.buffer, 0)?
            }
            Ok(())
        }
    };
}

macro_rules! try_read_back_fn {
    ($name:ident, $read:ident, $vfield:ident) => {
        #[inline]
        pub fn $name(&mut self, backend: &GpuBackend) -> bool { self.$read.try_take(backend, &mut self.$vfield) }
    };
}

macro_rules! read_back_fn {
    ($name:ident, $try_read:ident) => {
        #[inline]
        pub fn $name(&mut self, backend: &GpuBackend) -> TaserResult<()> {
            backend.synchronize()?;
            self.$try_read(backend);
            Ok(())
        }
    };
}

macro_rules! get_vect_field_fn {
    ($name:ident, $field:ident) => {
        #[inline]
        pub fn $name(&self) -> &Vec<Vec4> {
            &self.$field
        }
    };
}

impl FdtdStateReadback {
    pub fn new(
        backend: &GpuBackend,
        state: &FdtdLossyState,
        #[cfg(not(feature = "dim3"))] mode: FdtdSimulationMode
    ) -> TaserResult<Self> {
        let zeroed_vector_field = vec![Vec4::ZERO; state.n_cells.element_product() as usize];
        let cell_count = zeroed_vector_field.len();
        Ok(Self {
            h: zeroed_vector_field.clone(),
            dn: zeroed_vector_field.clone(),
            en: zeroed_vector_field,
            h_read: GpuReadback::new(backend, cell_count)?,
            dn_read: GpuReadback::new(backend, cell_count)?,
            en_read: GpuReadback::new(backend, cell_count)?,
            #[cfg(not(feature = "dim3"))]
            mode,
        })
    }

    /// Submit a command for copying all vector field data from GPU to CPU
    pub fn request_copy_fields(&mut self, backend: &GpuBackend, state: &FdtdLossyState) -> TaserResult<()> {
        self.request_copy_h(backend, state)?;
        self.request_copy_dn(backend, state)?;
        self.request_copy_en(backend, state)
    }

    request_copy_fn!(request_copy_h, h_read, h);
    request_copy_fn!(request_copy_dn, dn_read, dn);
    request_copy_fn!(request_copy_en, en_read, en);

    /// Blocks the thread until all vector fields are read into `self`. Must be called after [`request_copy_fields`](Self::request_copy_fields).
    #[inline]
    pub fn read_back_fields(&mut self, backend: &GpuBackend) -> TaserResult<()> {
        backend.synchronize()?;
        self.try_read_back_fields(backend);
        Ok(())
    }

    read_back_fn!(read_back_h, try_read_back_h);
    read_back_fn!(read_back_dn, try_read_back_dn);
    read_back_fn!(read_back_en, try_read_back_en);

    /// Try reading back fields without blocking the thread. Must be called after [`request_copy_fields`](Self::request_copy_fields).
    pub fn try_read_back_fields(&mut self, backend: &GpuBackend) -> bool {
        self.try_read_back_h(backend) &&
            self.try_read_back_dn(backend) &&
            self.try_read_back_en(backend)
    }

    try_read_back_fn!(try_read_back_h, h_read, h);
    try_read_back_fn!(try_read_back_dn, dn_read, dn);
    try_read_back_fn!(try_read_back_en, en_read, en);

    get_vect_field_fn!(get_h_field, h);
    get_vect_field_fn!(get_dn_field, dn);
    get_vect_field_fn!(get_en_field, en);

    /// Get magnitudes of the H vector field.
    /// Must be called after [`request_copy_h`](Self::request_copy_h) for updated results.
    pub fn h_magnitudes(&self) -> Vec<Real> {
        cfg_select! {
            feature = "dim3" => self.h.iter().map(|v| v.length()).collect(),
            _ =>
                match self.mode {
                    #[cfg(feature = "dim1")]
                    FdtdSimulationMode::EyHx => self.h.iter().map(|v| v.x).collect(),
                    #[cfg(feature = "dim1")]
                    FdtdSimulationMode::ExHy => self.h.iter().map(|v| v.y).collect(),
                    #[cfg(feature = "dim2")]
                    FdtdSimulationMode::TransverseMagneticZ => self.h.iter().map(|v| v.xy().length()).collect(),
                    #[cfg(feature = "dim3")]
                    FdtdSimulationMode::TransverseElectricZ => self.h.iter().map(|v| v.z).collect(),
                },
        }
    }

    /// Get magnitudes of the Dn vector field.
    /// Must be called after [`request_copy_dn`](Self::request_copy_dn) for updated results.
    pub fn dn_magnitudes(&self) -> Vec<Real> {
        cfg_select! {
            feature = "dim3" => self.dn.iter().map(|v| v.length()).collect(),
            _ =>
                match self.mode {
                    #[cfg(feature = "dim1")]
                    FdtdSimulationMode::EyHx => self.dn.iter().map(|v| v.y).collect(),
                    #[cfg(feature = "dim1")]
                    FdtdSimulationMode::ExHy => self.dn.iter().map(|v| v.x).collect(),
                    #[cfg(feature = "dim2")]
                    FdtdSimulationMode::TransverseMagneticZ => self.dn.iter().map(|v| v.z).collect(),
                    #[cfg(feature = "dim3")]
                    FdtdSimulationMode::TransverseElectricZ => self.dn.iter().map(|v| v.xy().length()).collect(),
                },
        }
    }

    /// Get magnitudes of the En vector field.
    /// Must be called after [`request_copy_en`](Self::request_copy_en) for updated results.
    pub fn en_magnitudes(&self) -> Vec<Real> {
        cfg_select! {
            feature = "dim3" => self.en.iter().map(|v| v.length()).collect(),
            _ =>
                match self.mode {
                    #[cfg(feature = "dim1")]
                    FdtdSimulationMode::EyHx => self.en.iter().map(|v| v.y).collect(),
                    #[cfg(feature = "dim1")]
                    FdtdSimulationMode::ExHy => self.en.iter().map(|v| v.x).collect(),
                    #[cfg(feature = "dim2")]
                    FdtdSimulationMode::TransverseMagneticZ => self.en.iter().map(|v| v.z).collect(),
                    #[cfg(feature = "dim3")]
                    FdtdSimulationMode::TransverseElectricZ => self.en.iter().map(|v| v.xy().length()).collect(),
                }
        }
    }
}

#[cfg(not(feature = "dim3"))]
pub enum FdtdSimulationMode {
    #[cfg(feature = "dim1")]
    EyHx,
    #[cfg(feature = "dim1")]
    ExHy,
    #[cfg(feature = "dim2")]
    TransverseMagneticZ,
    #[cfg(feature = "dim3")]
    TransverseElectricZ,
}