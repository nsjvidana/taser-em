use crate::prelude::*;
use crate::gpu_util::{CreateGpuBuffer, CreateGpuBufferReadable, GpuBufferReadable};
use derivative::Derivative;
use khal::Shader;
use parry3d::bounding_volume::Aabb;
use std::num::{NonZeroI32, NonZeroU32};
use taser_em_shaders::fdtd::*;
use crate::*;

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
    ) -> GpuResult<FdtdLossyDispatchData> {
        let FdtdParameters {
            cell_size, dt, polarization_mode, ..
        } = self.fdtd_parameters;

        let sim_bb = self.compute_bounding_box();
        let n_cells = self.compute_n_cells(&sim_bb, stability);

        let grid_mats = self.create_material_grid(&sim_bb, n_cells);
        let (regions_offset, grid_coeffs) = PmlCoefficientsGrid::new(&grid_mats, self.pml_parameters, dt);

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
                let Source::Dipole { position, t_start, vals, moment } = source else {
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
                })
            })
            .collect::<Vec<_>>();
        let tfsf_dispatch_data = self.create_tfsf_sources(
            backend,
            &mut source_vals,
            regions_offset,
            problem_space_max
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
            _padding0: 0,
            inv_d: cell_size3.recip(),
            problem_space_min: problem_space_min.to_3d(UVec3::ZERO),
            _padding1: 0,
            problem_space_max: problem_space_max.to_3d(UVec3::ONE),
            _padding2: 0,
        };

        let cell_count = n_cells.element_product() as usize;
        let zeroed_vector_field = vec![Vec4::ZERO; cell_count];

        let buffers = FdtdLossyDispatchData {
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
        regions_offset: Vect,
        problem_space_max: GridIndex,
    ) -> GpuResult<TfsfDispatchData> {
        let TfsfParameters {
            pml_width, pml_sig_max, pml_grading_order
        } = &self.tfsf_parameters;
        let FdtdParameters {
            dt, cell_size, ..
        } = &self.fdtd_parameters;

        let mut corrections = Vec::new();
        let mut coeffs = Vec::new();
        let mut total_num_cells = 0;
        let mut n_cells_max = 0;

        let mut tfsf_srcs = self.sources.iter()
            .filter_map(|source_val| {
                let Source::TFSF {
                    spatial_axis, position, direction, t_start, vals,
                    polarization, tfsf_buffer_width
                } = source_val else { return None };
                let a = Axis::from(*spatial_axis);
                let a1 = a.permute();
                let a2 = a1.permute();

                let inv_d_a = (*cell_size)[*spatial_axis].recip();

                let pos = (regions_offset[*spatial_axis] + position) * inv_d_a;
                let boundary_min = pos as u32;
                let boundary_max = problem_space_max[a] - tfsf_buffer_width.get();

                let num_correction_cells = (boundary_max - boundary_min + 1) + 1;
                let source_cell = 1;
                let n_cells = num_correction_cells + source_cell + pml_width.get();
                total_num_cells += n_cells as usize;
                n_cells_max = n_cells_max.max(n_cells);

                let corrections_start = corrections.len() as u32;
                corrections.resize(corrections.len() + num_correction_cells, TfsfCorrections::default());

                let vals_start = source_vals.len() as u32;
                source_vals.extend_from_slice(vals);

                let coeffs_start = coeffs.len() as u32;
                let grid_coeffs = {
                    const HALF_CELL: Index = 1;
                    const ONE_CELL: Index = HALF_CELL*2;
                    let n_axis2x = n_cells * ONE_CELL;
                    let pml_end = match direction {
                        WaveDirection::Positive => n_axis2x - ONE_CELL,
                        WaveDirection::Negative => 0,
                        _ => panic!("Invalid wave direction")
                    };
                    let pml_width2x = (pml_width.get() * ONE_CELL) as Real;
                    let pml_sig_max = *pml_sig_max;
                    let mut sig = {
                        let mut sig = [
                            vec![0., 0.],
                            vec![0., 0.],
                            vec![0., 0.],
                        ];
                        sig[a as usize] = into_par_iter!((0..n_axis2x))
                            .map(|i| {
                                let end_dist = i.abs_diff(pml_end) as Real;
                                let pml_interp = (1. - end_dist / pml_width2x)
                                    .clamp(0., 1.);
                                pml_sig_max * pml_interp.powi(pml_grading_order.get())
                            })
                            .collect::<Vec<_>>();
                        sig
                    };

                    let inv_dt = dt.recip();
                    let inv_mu_r = self.background_material.mu_r.recip();
                    let inv_eps_r = self.background_material.eps_r.recip();
                    // let mat_sig = self.background_material.sig;
                    into_par_iter!((0..n_cells))
                        .map(|cell_idx| {
                            // Stagger indexing the conductivities as per the Yee grid staggering
                            let idx_2x = USizeVec3::ZERO.with_z(cell_idx as usize * 2);
                            let dn_sigs: [Vec3; MAX_DIM] = core::array::from_fn(|axis_i| {
                                let mut sig_idx = idx_2x;
                                sig_idx[axis_i] += 1;
                                Vec3::new(
                                    sig[0][sig_idx.x],
                                    sig[1][sig_idx.y],
                                    sig[2][sig_idx.z],
                                )
                            });
                            let h_sigs: [Vec3; MAX_DIM] = core::array::from_fn(|axis_i| {
                                let mut sig_idx = idx_2x + 1;
                                sig_idx[axis_i] -= 1;
                                Vec3::new(
                                    sig[0][sig_idx.x],
                                    sig[1][sig_idx.y],
                                    sig[2][sig_idx.z],
                                )
                            });
                            let mut coeff = AuxGridPmlCoeffs::default();
                            for axis in Axis::ALL_AXES {
                                let axis_i = axis as usize;
                                let axis1 = axis.permute();
                                let axis2 = axis1.permute();

                                let h_sigs_axis = h_sigs[axis_i];
                                let coeff_term0 = (
                                    inv_dt + ((h_sigs_axis[axis1] + h_sigs_axis[axis2]) / (2. * EPS_0)) +
                                        ((h_sigs_axis[axis1] * h_sigs_axis[axis2] * dt) / (4. * EPS_0 * EPS_0))
                                ).recip();
                                coeff.h1[axis] = coeff_term0 * (
                                    inv_dt - ((h_sigs_axis[axis1] + h_sigs_axis[axis2]) / (2. * EPS_0)) -
                                        ((h_sigs_axis[axis1] * h_sigs_axis[axis2] * dt) / (4. * EPS_0 * EPS_0))
                                );
                                coeff.h2[axis] = -coeff_term0 * C_0 * inv_mu_r[a];

                                let dn_sigs_axis = dn_sigs[axis_i];
                                let coeff_term0 = (
                                    inv_dt + ((dn_sigs_axis[axis1] + dn_sigs_axis[axis2]) / (2. * EPS_0)) +
                                        ((dn_sigs_axis[axis1] * dn_sigs_axis[axis2] * dt) / (4. * EPS_0 * EPS_0))
                                ).recip();
                                coeff.dn1[axis] = coeff_term0 * (
                                    inv_dt - ((dn_sigs_axis[axis1] + dn_sigs_axis[axis2]) / (2. * EPS_0)) -
                                        ((dn_sigs_axis[axis1] * dn_sigs_axis[axis2] * dt) / (4. * EPS_0 * EPS_0))
                                );
                                coeff.dn2[axis] = coeff_term0 * C_0;
                                // TODO: loss (there's probably a use to having lossy background material)
                                // let mat_sig_axis = mat_sig[axis];
                                // coeff.dn_loss1[axis] = -coeff_term0 * mat_sig_axis / EPS_0;
                                // coeff.dn_loss2[axis] = -coeff_term0 * dn_sigs_axis[axis] * mat_sig_axis * dt / (EPS_0 * EPS_0);
                            }
                            coeff.en1 = Vec4::from((inv_eps_r, 0.));
                            coeff
                        })
                        .collect::<Vec<_>>()
                };
                coeffs.extend_from_slice(&grid_coeffs);

                Some(GpuTfsf {
                    prop_axis: *spatial_axis,
                    direction: *direction,
                    boundary_min,
                    boundary_max,
                    vals_start,
                    vals_end: source_vals.len() as u32 - 1,
                    t_start: (t_start / dt) as u32,
                    n_cells,
                    polarization_a1: (*polarization)[a1],
                    polarization_a2: (*polarization)[a2],
                    inv_d_a,
                    corrections_start,
                    num_correction_cells,
                    grid_start: coeffs_start,
                })
            })
            .collect::<Vec<_>>();

        total_num_cells = total_num_cells.max(1);
        let zeroed_vector_fields = vec![[0.; 2]; total_num_cells];

        let no_tfsf_sources = tfsf_srcs.is_empty();

        if tfsf_srcs.is_empty() { tfsf_srcs.push(GpuTfsf::default()) }
        if corrections.is_empty() { corrections.push(TfsfCorrections::default()) }
        if coeffs.is_empty() { coeffs.push(AuxGridPmlCoeffs::default()) }

        let aux_grid_thread_count =
            if no_tfsf_sources { None }
            else { Some([tfsf_srcs.len() as u32, 1, n_cells_max]) };

        Ok(TfsfDispatchData {
            tfsf_sources: tfsf_srcs.create_gpu_buffer(backend)?,
            corrections: corrections.create_gpu_buffer(backend)?,
            auxgr_coeffs: coeffs.create_gpu_buffer(backend)?,
            h: zeroed_vector_fields.create_gpu_buffer(backend)?,
            dn: zeroed_vector_fields.create_gpu_buffer(backend)?,
            en: zeroed_vector_fields.create_gpu_buffer(backend)?,
            aux_grid_thread_count
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
                    Source::TFSF { spatial_axis, position, ..} => {
                        let mut pos = regions_center;
                            pos[Axis::from(*spatial_axis)] = *position;
                        pos
                    }
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
    boundary_condition: BC,
    compute_source_terms: GpuComputeSourceTerms,
    update: GpuLossyUpdate,
    pub num_steps_per_submission: usize,
}

impl<BC: BoundaryCondition> FdtdLossyPipeline<BC> {
    pub fn new(backend: &GpuBackend, boundary_condition: BC, num_steps_per_submission: usize) -> GpuResult<Self> {
        Ok(Self {
            boundary_condition,
            compute_source_terms: GpuComputeSourceTerms::from_dir(backend, &crate::SPIRV_DIR)?,
            update: GpuLossyUpdate::from_dir(backend, &crate::SPIRV_DIR)?,
            num_steps_per_submission,
        })
    }

    pub fn dispatch_steps(
        &mut self,
        pass: &mut GpuPass,
        gpu_data: &mut FdtdLossyDispatchData,
    ) -> GpuResult<()> {
        for _ in 0..self.num_steps_per_submission {
            self.boundary_condition.call(
                pass,
                &gpu_data.grid_params,
                &mut gpu_data.h.buffer,
                &mut gpu_data.dn.buffer,
                &mut gpu_data.en.buffer,
                gpu_data.thread_count
            )?;
            self.compute_source_terms.call(
                pass,
                DispatchGrid::ThreadCount(gpu_data.thread_count),
                &gpu_data.grid_params,
                &gpu_data.t_idx.buffer,
                &mut gpu_data.source_terms,
                &gpu_data.source_vals,
                &gpu_data.dipoles,
                &gpu_data.tfsf_dispatch_data.tfsf_sources,
                &gpu_data.grid_coeffs,
            )?;
            self.update.call(
                pass,
                DispatchGrid::ThreadCount(gpu_data.thread_count),
                &gpu_data.grid_params,
                &mut gpu_data.t_idx.buffer,
                &mut gpu_data.h.buffer,
                &mut gpu_data.dn.buffer,
                &mut gpu_data.en.buffer,
                &mut gpu_data.int_terms,
                &gpu_data.grid_coeffs,
                &gpu_data.source_terms,
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
pub struct FdtdLossyDispatchData {
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
    pub corrections: GpuBuffer<TfsfCorrections>,
    pub auxgr_coeffs: GpuBuffer<AuxGridPmlCoeffs>,
    pub h: GpuBuffer<[Real; 2]>,
    pub dn: GpuBuffer<[Real; 2]>,
    pub en: GpuBuffer<[Real; 2]>,
    /// Thread count for simulating auxiliary grids for ALL plane waves.
    ///
    /// Is [`None`] only when there are no TF/SF sources.
    pub aux_grid_thread_count: Option<[u32; 3]>,
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
    ) -> GpuResult<()>
    {
        self.kernel.call(
            pass,
            DispatchGrid::ThreadCount(thread_count),
            grid,
            h,
            dn,
            en
        )
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
    ) -> GpuResult<()>;
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
    /// Total-field/Scattered-field source.
    TFSF {
        /// The spatial axis along which the plane wave will travel.
        spatial_axis: SpatialAxis,
        /// Position of the plane wave, along `spatial_axis`, in world coordinates.
        position: Real,
        /// The direction along `spatial_axis` the wave will travel in.
        direction: WaveDirection,
        /// The time (in the simulation, not real-time) when the source begins injection (in seconds).
        t_start: f32,
        /// Signal data points.
        vals: Vec<f32>,
        /// Polarization direction of the plane wave (unit vector)
        polarization: Vec3,
        /// The distance between the TF/SF boundary and the border/PML. This does NOT control the distance
        /// between the boundary and border/PML along the propagation axis. Change spacer region widths
        /// to control that.
        ///
        /// A width of `3` will suffice, especially if you want to record values
        /// behind the TF/SF boundary.
        tfsf_buffer_width: NonZeroU32,
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