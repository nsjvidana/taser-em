#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::gpu_util::*;
use crate::prelude::*;
use crate::*;
use derivative::Derivative;
use khal::backend::*;
use parry3d::bounding_volume::Aabb;
use parry3d::shape::{Cuboid, SharedShape};
use std::num::{NonZeroI32, NonZeroU32};
use std::sync::Arc;
use taser_em_shaders::fdtd::*;
use taser_em_shaders::math::*;
use taser_em_shaders::source::*;

// TODO: Docs.
pub struct FdtdLossySimulation {
    pub material_regions: MaterialRegions,
    pub background_material: ElectricMaterial,
    pub dipoles: Vec<Source<Dipole>>,
    pub tfsf_sources: Vec<Source<Tfsf>>,
    pub power_flux_monitors: Vec<Arc<PowerFluxMonitor>>,
    pub fdtd_parameters: FdtdParameters,
    pub pml_parameters: PmlParameters,
    pub tfsf_parameters: AuxGridParameters
}

impl FdtdLossySimulation {
    pub fn new(fdtd_parameters: FdtdParameters, pml_parameters: PmlParameters) -> Self {
        Self {
            material_regions: MaterialRegions::new(),
            background_material: ElectricMaterial::FREE_SPACE,
            dipoles: vec![],
            tfsf_sources: vec![],
            power_flux_monitors: vec![],
            fdtd_parameters,
            pml_parameters,
            tfsf_parameters: AuxGridParameters {
                pml_width: NonZeroU32::new(12).unwrap(),
                pml_sig_max: pml_parameters.sig_max,
                pml_grading_order: pml_parameters.grading_order,
            },
        }
    }

    /// Adds a dipole source to the simulation with `data_points` being the data points of the
    /// function that the dipole injects.
    pub fn add_dipole(&mut self, dipole: Dipole, data_points: Vec<Real>) -> Arc<Dipole> {
        let src = Source::from_source(dipole, data_points);
        let ptr = src.src_ref.clone();
        self.dipoles.push(src);
        ptr
    }

    /// Adds a Total-Field / Scattered-Field source to the simulation with `data_points` being the
    /// data points of the function that the TF/SF source injects.
    pub fn add_tfsf(&mut self, tfsf: Tfsf, data_points: Vec<Real>) -> Arc<Tfsf> {
        let src = Source::from_source(tfsf, data_points);
        let ptr = src.src_ref.clone();
        self.tfsf_sources.push(src);
        ptr
    }

    pub fn add_flux_monitor(&mut self, power_flux_monitor: PowerFluxMonitor) -> Arc<PowerFluxMonitor> {
        let ptr = Arc::new(power_flux_monitor);
        self.power_flux_monitors.push(ptr.clone());
        ptr
    }

    /// Fill a box-shaped region from `start` to `end` with `material`
    pub fn fill_region(
        &mut self,
        start: Vect,
        end: Vect,
        material: ElectricMaterial
    ) -> &mut Self {
        let region_dims = end - start;

        let vec3_mask = Vec3::splat(self.fdtd_parameters.cell_size.max_element() * 3.);
        let half_extents = region_dims.to_3d(vec3_mask).abs() * 0.5;

        let shape = Cuboid::new(half_extents);
        let middle = ((start + end) / 2.).to_3d(Vec3::ZERO);
        let pose = Pose3::from_translation(middle);

        self.material_regions.regions.push(MaterialRegion::new(SharedShape::new(shape), pose, material));
        self
    }

    pub fn finalize(
        &self,
        backend: &GpuBackend,
        stability: &FdtdStability,
    ) -> TaserResult<FdtdLossyState> {
        let FdtdParameters {
            cell_size, dt, ..
        } = self.fdtd_parameters;

        let sim_bb = self.compute_bounding_box();
        let n_cells = self.compute_n_cells(&sim_bb, stability);
        let n_cells3 = n_cells.n_cells_to_3d();

        let grid_mats = self.create_material_grid(&sim_bb, n_cells);
        let (regions_offset3, grid_coeffs) = PmlCoefficientsGrid::new(&grid_mats, self.pml_parameters, dt);

        let cell_count = n_cells.element_product();

        let regions_offset = Vect::from_vec3(regions_offset3);
        let (problem_space_min, problem_space_max) = self.compute_problem_space(n_cells);

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
            _padding0: 0,
            n_cells3: n_cells.n_cells_to_3d(),
            dt,
            d: cell_size3,
            inv_dt: dt.recip(),
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
            t_idx: 0.create_gpu_buffer(backend)?,
            // Vector fields
            h_previous: zeroed_vector_field.create_gpu_buffer(backend)?,
            h: zeroed_vector_field.create_gpu_buffer(backend)?,
            dn: zeroed_vector_field.create_gpu_buffer(backend)?,
            en: zeroed_vector_field.create_gpu_buffer(backend)?,
            // For computing source terms
            source_states: SourceStates::new(
                backend,
                self,
                n_cells3,
                regions_offset,
                problem_space_min.cell_idx_to_3d(),
                problem_space_max.cell_idx_to_3d()
            )?,
            // For update equation terms
            source_terms: vec![SourceTerms::default(); cell_count].create_gpu_buffer(backend)?,
            update_source_terms_workgroups: update_source_terms_workgroups(n_cells),
            int_terms: vec![PmlIntegrals::default(); cell_count].create_gpu_buffer(backend)?,
            grid_coeffs: grid_coeffs.coeffs.create_gpu_buffer(backend)?,
            // Misc data
            power_flux_states: PowerFluxStates::new(backend, self, n_cells, &regions_offset3)?,
            thread_count: n_cells.n_cells_to_3d().to_array(),
            n_cells,
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
        let source_pts = self.dipoles.iter()
            .map(|src| src.src_ref.position.to_3d(Vec3::ZERO))
            .chain(self.tfsf_sources.iter().map(|_| regions_center))
            .collect::<Vec<_>>();
        for pt in source_pts.iter() {
            regions_bb.mins = regions_bb.mins.min(*pt);
            regions_bb.maxs = regions_bb.maxs.max(*pt);
        }
        // ensure the simulation encompasses everything by adding machine eps
        regions_bb.add_half_extents(Vec3::splat(Real::EPSILON))
    }

    /// Computes the (min, max) grid indices of a rectangular region that's considered the "problem space"
    ///
    /// This includes all space but PML and boundary cells
    pub fn compute_problem_space(&self, n_cells: GridIndex) -> (GridIndex, GridIndex) {
        let mut problem_space_min = GridIndex::ONE;
        self.pml_parameters.widths
            .iter_spatial_axes()
            .for_each(|(s_axis, w)| problem_space_min[s_axis] += w.lo);
        let mut problem_space_max = n_cells - 2;
        self.pml_parameters.widths
            .iter_spatial_axes()
            .for_each(|(s_axis, w)| problem_space_max[s_axis] -= w.hi);
        (problem_space_min, problem_space_max)
    }
}

/// The shader pipeline for running diagonal anisotropy simulation with UPML.
pub struct FdtdLossyPipeline<BCx, BCy, BCz>
where
    BCx: BoundaryCondition<X>,
    BCy: BoundaryCondition<Y>,
    BCz: BoundaryCondition<Z>,
{
    init_pec: InitPec,
    boundary_conditions: BoundaryConditions<BCx, BCy, BCz>,
    source_pipeline: SourcePipeline,
    h_update: GpuLossyHUpdate,
    dn_en_update: GpuLossyDnEnUpdate,
    power_flux_pipeline: PowerFluxPipeline,
    pub num_steps_per_submission: usize,
}

impl<BCx, BCy, BCz> FdtdLossyPipeline<BCx, BCy, BCz>
where
    BCx: BoundaryCondition<X>,
    BCy: BoundaryCondition<Y>,
    BCz: BoundaryCondition<Z>,
{
    pub fn new(
        backend: &GpuBackend,
        boundary_conditions: BoundaryConditions<BCx, BCy, BCz>,
        num_steps_per_submission: usize
    ) -> TaserResult<Self> {
        Ok(Self {
            init_pec: InitPec::from_dir(backend, &crate::SPIRV_DIR)?,
            boundary_conditions,
            source_pipeline: SourcePipeline::new(backend)?,
            h_update: GpuLossyHUpdate::from_dir(backend, &crate::SPIRV_DIR)?,
            dn_en_update: GpuLossyDnEnUpdate::from_dir(backend, &crate::SPIRV_DIR)?,
            power_flux_pipeline: PowerFluxPipeline::new(backend)?,
            num_steps_per_submission,
        })
    }

    /// Create new pipeline and dispatch initialization shaders to the GPU at the same time (calls [`FdtdLossyPipeline::initialize`]).
    pub fn new_initialized(
        backend: &GpuBackend,
        boundary_conditions: BoundaryConditions<BCx, BCy, BCz>,
        num_steps_per_submission: usize,
        state: &mut FdtdLossyState
    ) -> TaserResult<Self> {
        let mut pipeline = Self::new(backend, boundary_conditions, num_steps_per_submission)?;

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("2d fdtd example", None);
        pipeline.initialize(&mut pass, state)?;
        drop(pass);
        backend.submit(encoder)?;

        Ok(pipeline)
    }

    pub fn initialize(
        &mut self,
        pass: &mut GpuPass,
        state: &mut FdtdLossyState
    ) -> TaserResult<()> {
        self.boundary_conditions.initialize(pass, state)?;

        self.source_pipeline.initialize(pass, state)?;

        self.init_pec.call(
            pass,
            DispatchGrid::ThreadCount(state.thread_count),
            &state.grid_params,
            &mut state.h,
            &mut state.dn,
            &mut state.en,
            &state.grid_coeffs
        )?;
        Ok(())
    }

    /// Dispatches `num_steps_per_submission` FDTD steps.
    pub fn dispatch_steps(
        &mut self,
        pass: &mut GpuPass,
        state: &mut FdtdLossyState,
    ) -> TaserResult<()> {
        self.dispatch_steps_aux(pass, state, |_, _| Ok(()))
    }

    /// Dispatches `num_steps_per_submission` FDTD steps.
    ///
    /// `aux_f` allows the user to do additional dispatching work that gets called immediately after
    /// each FDTD step (so `aux_f` runs `num_steps_per_submission` times).
    pub fn dispatch_steps_aux(
        &mut self,
        pass: &mut GpuPass,
        state: &mut FdtdLossyState,
        mut aux_f: impl FnMut(&mut GpuPass, &mut FdtdLossyState) -> TaserResult<()>
    ) -> TaserResult<()> {
        for _ in 0..self.num_steps_per_submission {
            self.source_pipeline.dispatch_step(pass, state)?;

            self.boundary_conditions.pre_update(pass, state)?;

            self.h_update.call(
                pass,
                DispatchGrid::ThreadCount(state.thread_count),
                &state.grid_params,
                &mut state.h_previous,
                &mut state.h,
                &mut state.en,
                &mut state.int_terms,
                &state.grid_coeffs,
                &state.source_terms,
            )?;

            self.boundary_conditions.before_de_update(pass, state)?;

            self.dn_en_update.call(
                pass,
                DispatchGrid::ThreadCount(state.thread_count),
                &state.grid_params,
                &mut state.t_idx,
                &mut state.h,
                &mut state.dn,
                &mut state.en,
                &mut state.int_terms,
                &state.grid_coeffs,
                &state.source_terms,
            )?;

            self.power_flux_pipeline.dispatch_steps(pass, state)?;

            aux_f(pass, state)?;
        }
        Ok(())
    }
}

macro_rules! request_copy_fn {
    ($name:ident, $read:ident, $buf:ident) => {
        #[inline]
        pub fn $name(&mut self, backend: &GpuBackend, state: &FdtdLossyState) -> TaserResult<()> {
            if self.$read.is_idle() {
                self.$read.request_copy(backend, &state.$buf, 0)?
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
    t_idx: Vec<u32>,
    h_read: GpuReadback<Vec4>,
    dn_read: GpuReadback<Vec4>,
    en_read: GpuReadback<Vec4>,
    t_idx_read: GpuReadback<u32>,
    #[cfg(not(feature = "dim3"))]
    mode: FdtdSimulationMode
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
            t_idx: vec![0],
            h_read: GpuReadback::new(backend, cell_count)?,
            dn_read: GpuReadback::new(backend, cell_count)?,
            en_read: GpuReadback::new(backend, cell_count)?,
            t_idx_read: GpuReadback::new(backend, 1)?,
            #[cfg(not(feature = "dim3"))]
            mode,
        })
    }

    #[cfg(not(feature = "dim3"))]
    pub fn get_simulation_mode(&self) -> FdtdSimulationMode { self.mode }

    /// Submit a command for copying all vector field data from GPU to CPU
    pub fn request_copy_fields(&mut self, backend: &GpuBackend, state: &FdtdLossyState) -> TaserResult<()> {
        self.request_copy_h(backend, state)?;
        self.request_copy_dn(backend, state)?;
        self.request_copy_en(backend, state)
    }

    request_copy_fn!(request_copy_h, h_read, h);
    request_copy_fn!(request_copy_dn, dn_read, dn);
    request_copy_fn!(request_copy_en, en_read, en);
    request_copy_fn!(request_copy_t_idx, t_idx_read, t_idx);

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
    read_back_fn!(read_back_t_idx, try_read_back_t_idx);

    /// Try reading back fields without blocking the thread. Must be called after [`request_copy_fields`](Self::request_copy_fields).
    pub fn try_read_back_fields(&mut self, backend: &GpuBackend) -> bool {
        self.try_read_back_h(backend) &&
            self.try_read_back_dn(backend) &&
            self.try_read_back_en(backend)
    }

    try_read_back_fn!(try_read_back_h, h_read, h);
    try_read_back_fn!(try_read_back_dn, dn_read, dn);
    try_read_back_fn!(try_read_back_en, en_read, en);
    try_read_back_fn!(try_read_back_t_idx, t_idx_read, t_idx);

    get_vect_field_fn!(get_h_field, h);
    get_vect_field_fn!(get_dn_field, dn);
    get_vect_field_fn!(get_en_field, en);

    /// The time step index that the simulation is currently on (a.k.a. the number of time steps simulated).
    pub fn get_t_idx(&self) -> u32 { self.t_idx[0] }

    /// Get magnitudes of the H vector field.
    /// Must be called after [`request_copy_h`](Self::request_copy_h) for updated results.
    pub fn h_magnitudes(&self) -> Vec<Real> {
        cfg_select! {
            feature = "dim3" => self.h.iter().map(|v| v.length()).collect(),
            _ =>
                match self.mode {
                    #[cfg(feature = "dim1")]
                    FdtdSimulationMode::EyHx => par_iter!(self.h).map(|v| v.x.abs()).collect(),
                    #[cfg(feature = "dim1")]
                    FdtdSimulationMode::ExHy => par_iter!(self.h).map(|v| v.y.abs()).collect(),
                    #[cfg(feature = "dim2")]
                    FdtdSimulationMode::TransverseMagneticZ => par_iter!(self.h).map(|v| v.xy().length()).collect(),
                    #[cfg(feature = "dim2")]
                    FdtdSimulationMode::TransverseElectricZ => par_iter!(self.h).map(|v| v.z.abs()).collect(),
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
                    FdtdSimulationMode::EyHx => par_iter!(self.dn).map(|v| v.y.abs()).collect(),
                    #[cfg(feature = "dim1")]
                    FdtdSimulationMode::ExHy => par_iter!(self.dn).map(|v| v.x.abs()).collect(),
                    #[cfg(feature = "dim2")]
                    FdtdSimulationMode::TransverseMagneticZ => par_iter!(self.dn).map(|v| v.z.abs()).collect(),
                    #[cfg(feature = "dim2")]
                    FdtdSimulationMode::TransverseElectricZ => par_iter!(self.dn).map(|v| v.xy().length()).collect(),
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
                    FdtdSimulationMode::EyHx => par_iter!(self.en).map(|v| v.y.abs()).collect(),
                    #[cfg(feature = "dim1")]
                    FdtdSimulationMode::ExHy => par_iter!(self.en).map(|v| v.x.abs()).collect(),
                    #[cfg(feature = "dim2")]
                    FdtdSimulationMode::TransverseMagneticZ => par_iter!(self.en).map(|v| v.z.abs()).collect(),
                    #[cfg(feature = "dim2")]
                    FdtdSimulationMode::TransverseElectricZ => par_iter!(self.en).map(|v| v.xy().length()).collect(),
                }
        }
    }
}

#[derive(Clone, Debug)]
pub struct FdtdParameters {
    pub cell_size: Vect,
    pub dt: Real,
    pub material_discretization: MaterialDiscretization
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

/// Buffers and data needed for running the shader
pub struct FdtdLossyState {
    // Uniforms / thread-independent vars
    pub grid_params: GpuBuffer<GridParameters>,
    pub t_idx: GpuBuffer<u32>,
    // Vector fields
    pub h_previous: GpuBuffer<Vec4>,
    pub h: GpuBuffer<Vec4>,
    pub dn: GpuBuffer<Vec4>,
    pub en: GpuBuffer<Vec4>,
    // For computing source terms
    pub source_states: SourceStates,
    // For update equation terms
    pub source_terms: GpuBuffer<SourceTerms>,
    pub update_source_terms_workgroups: [u32; 3],
    pub int_terms: GpuBuffer<PmlIntegrals>,
    pub grid_coeffs: GpuBuffer<PmlCoefficients>,
    // Monitors & DFTs
    pub power_flux_states: Option<PowerFluxStates>,
    // Misc data
    pub thread_count: [u32; 3], // TODO: turn this into a workgroups count
    pub n_cells: GridIndex,
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

#[derive(Copy, Clone, Debug)]
pub enum MaterialDiscretization {
    Rough,
    Smooth { resolution: NonZeroU32 }
}

/// Material properties
///
/// For Perfect Electric Conductors, set any component of the `sig` field to [`Real::INFINITY`].
#[derive(Copy, Clone, Debug)]
pub struct ElectricMaterial {
    /// Relative permittivity
    pub eps_r: Vec3,
    /// Relative permeability
    pub mu_r: Vec3,
    /// Conductivity of the material (S/m)
    ///
    /// Set any component of this vector to [`Real::INFINITY`] to make this a Perfect Electric Conductor
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
    /// Perfect Electric Conductor
    pub const PEC: Self = Self {
        sig: Vec3::INFINITY,
        ..Self::FREE_SPACE
    };

    /// Compute refractive index on all axes
    #[allow(unused_variables)]
    pub fn refractive_index(&self) -> Vec3 {
        (self.eps_r * self.mu_r).sqrt()
    }
}

impl core::ops::Add for ElectricMaterial {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            eps_r: self.eps_r + rhs.eps_r,
            mu_r: self.mu_r + rhs.mu_r,
            sig: self.sig + rhs.sig,
        }
    }

}

impl core::ops::Div<Real> for ElectricMaterial {
    type Output = Self;

    fn div(self, rhs: Real) -> Self::Output {
        Self {
            eps_r: self.eps_r / rhs,
            mu_r: self.mu_r / rhs,
            sig: self.sig / rhs,
        }
    }
}

#[cfg(not(feature = "dim3"))]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum FdtdSimulationMode {
    #[cfg(feature = "dim1")]
    EyHx,
    #[cfg(feature = "dim1")]
    ExHy,
    #[cfg(feature = "dim2")]
    TransverseMagneticZ,
    #[cfg(feature = "dim2")]
    TransverseElectricZ,
}

#[cfg(not(feature = "dim3"))]
impl FdtdSimulationMode {
    pub fn extract_h_vector(&self, h: &Vec4) -> Vec3 {
        match self {
            #[cfg(feature = "dim1")]
            Self::EyHx => Vec3::new(h.x, 0., 0.),
            #[cfg(feature = "dim1")]
            Self::ExHy => Vec3::new(0., h.y, 0.),
            #[cfg(feature = "dim2")]
            Self::TransverseMagneticZ => Vec3::new(h.x, h.y, 0.),
            #[cfg(feature = "dim2")]
            Self::TransverseElectricZ => Vec3::new(0., 0., h.z),
        }
    }

    pub fn extract_e_vector(&self, e: &Vec4) -> Vec3 {
        match self {
            #[cfg(feature = "dim1")]
            Self::EyHx => Vec3::new(0., e.y, 0.),
            #[cfg(feature = "dim1")]
            Self::ExHy => Vec3::new(e.x, 0., 0.),
            #[cfg(feature = "dim2")]
            Self::TransverseMagneticZ => Vec3::new(0., 0., e.z),
            #[cfg(feature = "dim2")]
            Self::TransverseElectricZ => Vec3::new(e.x, e.y, 0.),
        }
    }

    pub fn get_h_magnitude(&self, h: &Vec4) -> Real {
        match self {
            #[cfg(feature = "dim1")]
            Self::EyHx => h.x.abs(),
            #[cfg(feature = "dim1")]
            Self::ExHy => h.y.abs(),
            #[cfg(feature = "dim2")]
            Self::TransverseMagneticZ => h.xy().length(),
            #[cfg(feature = "dim2")]
            Self::TransverseElectricZ => h.z.abs(),
        }
    }

    pub fn get_e_magnitude(&self, e: &Vec4) -> Real {
        match self {
            #[cfg(feature = "dim1")]
            Self::EyHx => e.y.abs(),
            #[cfg(feature = "dim1")]
            Self::ExHy => e.x.abs(),
            #[cfg(feature = "dim2")]
            Self::TransverseMagneticZ => e.z.abs(),
            #[cfg(feature = "dim2")]
            Self::TransverseElectricZ => e.xy().length(),
        }
    }
}