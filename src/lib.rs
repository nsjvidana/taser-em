pub mod prelude;
pub mod gpu_util;
pub mod grid;
pub mod util;
pub mod re_exports {
    pub use glamx;
    pub use khal;
    pub use anyhow;
}

pub use taser_em_shaders as shaders;

use std::num::{NonZeroI32, NonZeroU32};

use crate::gpu_util::{CreateGpuBuffer, CreateGpuBufferReadable, GpuBufferReadable};
use crate::grid::{LayerWidths, MaterialRegions, PlaneWaveCoefficientsGrid, PmlCoefficientsGrid, YeeGridMaterials};
use crate::prelude::{GpuResult, PolarizationMode};
use derivative::Derivative;
use glamx::{UVec3, Vec3, Vec4};
use khal::backend::*;
use khal::re_exports::include_dir::{include_dir, Dir};
use khal::Shader;
use parry3d::bounding_volume::Aabb;
use taser_em_shaders::fdtd::{GpuLossyUpdate, GpuComputeSourceTerms, GpuDipole, GpuPecBoundary, GridParameters, PmlCoefficients, PmlIntegrals, GpuPlaneWave, PlaneWaveCoeffs, SourceTerms};
use taser_em_shaders::math::*;

pub static SPIRV_DIR: Dir<'static> = include_dir!("$OUT_DIR/shaders-spirv");

/// Speed of EM wave in free space
pub const C_0: f32 = 299792458.0;
/// Free space permittivity
pub const EPS_0: f32 = 8.854188e-12;
/// Free space permeability
pub const MU_0: f32 = 1.256637e-6;
/// Free space wave impedance
pub const IMPEDANCE_0: f32 = 376.73032;

#[cfg(feature = "dim1")]
pub trait GridCellsIter: Iterator<Item = (Index,)> {}
#[cfg(feature = "dim2")]
pub trait GridCellsIter: Iterator<Item = (Index, Index,)> {}
#[cfg(feature = "dim3")]
pub trait GridCellsIter: Iterator<Item = (Index, Index, Index,)> {}
#[cfg(feature = "dim1")]
impl<T> GridCellsIter for T where T: Iterator<Item = (Index,)> {}
#[cfg(feature = "dim2")]
impl<T> GridCellsIter for T where T: Iterator<Item = (Index, Index,)> {}
#[cfg(feature = "dim3")]
impl<T> GridCellsIter for T where T: Iterator<Item = (Index, Index, Index,)> {}

// TODO: use some "FdtdSolver" generics here?

// TODO: Docs.
pub struct FdtdLossySimulation {
    pub material_regions: MaterialRegions,
    pub background_material: ElectricMaterial,
    pub sources: Vec<Source>,
    pub fdtd_parameters: FdtdParameters,
    pub pml_parameters: PmlParameters,
}

impl FdtdLossySimulation {
    pub fn new(fdtd_parameters: FdtdParameters, pml_parameters: PmlParameters) -> Self {
        Self {
            material_regions: MaterialRegions::new(),
            background_material: ElectricMaterial::FREE_SPACE,
            sources: vec![],
            fdtd_parameters,
            pml_parameters,
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
    ) -> GpuResult<FdtdLossyGpuData> {
        let FdtdParameters {
            cell_size, dt, polarization_mode, ..
        } = self.fdtd_parameters;

        let sim_bb = self.compute_bounding_box();
        let n_cells = self.compute_n_cells(&sim_bb, stability);
        
        let grid_mats = self.create_material_grid(&sim_bb, n_cells);
        let (regions_offset, grid_coeffs) = PmlCoefficientsGrid::new(&grid_mats, self.pml_parameters, dt);

        let mut source_vals: Vec<Real> = vec![];
        let regions_offset = Vect::from_vec3(regions_offset);
        let mut dipoles = self.sources.iter()
            .filter_map(|source| {
                let Source::Dipole { position, t_start, vals, moment } = source else {
                    return None;
                };
                let pos = (regions_offset + position) / cell_size;
                let cell_grid_idx = pos.as_grid_index();
                debug_assert!(
                    !pos.min_element().is_sign_negative() && !pos.as_grid_index().cmpge(n_cells).any(),
                    "negative source position!"
                );
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
        let mut plane_waves = self.sources.iter()
            .filter_map(|source| {
                let Source::PlaneWave {
                    spatial_axis, position, direction, t_start, vals
                } = source else { return None; };
                let axis = Axis::from(*spatial_axis);
                let pos = (regions_offset[axis] + position) / cell_size[*spatial_axis];
                let position_idx = pos as Index;
                debug_assert!(
                    !pos.is_sign_negative() && !GridIndex::splat(position_idx).cmpge(n_cells).any(),
                    "negative source position!"
                );

                let start = source_vals.len();
                source_vals.extend_from_slice(vals);
                Some(GpuPlaneWave {
                    spatial_axis: *spatial_axis,
                    direction: *direction as i32,
                    position_idx,
                    vals_start: start as u32,
                    vals_end: source_vals.len() as u32 - 1,
                    t_start: (t_start / dt) as u32,
                    _padding0: [0; 2]
                })
            })
            .collect::<Vec<_>>();
        if dipoles.is_empty() { dipoles.push(GpuDipole::default()) }
        if plane_waves.is_empty() { plane_waves.push(GpuPlaneWave::default()) }
        if source_vals.is_empty() { source_vals.push(0.0); }

        let plane_wave_coeffs_grid = PlaneWaveCoefficientsGrid::new(
            &grid_mats, &plane_waves, dt / 2.
        );

        let flat_idx_incrs = {
            let mut incrs = UVec3::ZERO;
            for (spatial_axis, axis) in SpatialAxis::ALL_SPATIAL.into_iter()
                .zip(SpatialAxis::ALL_AXES)
            {
                let mut grid_incr = GridIndex::default();
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
        };

        let cell_count = n_cells.element_product() as usize;
        let zeroed_vector_field = vec![Vec4::ZERO; cell_count];

        let buffers = FdtdLossyGpuData {
            // Uniforms / thread-independent vars
            grid_params: grid_params.create_gpu_uniform(backend)?,
            steps: 0.create_gpu_buffer_readable(backend)?,
            // Vector fields
            h: zeroed_vector_field.create_gpu_buffer_readable(backend)?,
            dn: zeroed_vector_field.create_gpu_buffer_readable(backend)?,
            en: zeroed_vector_field.create_gpu_buffer_readable(backend)?,
            // For computing source terms
            dipoles: dipoles.create_gpu_buffer(backend)?,
            plane_waves: plane_waves.create_gpu_buffer(backend)?,
            source_vals: source_vals.create_gpu_buffer(backend)?,
            plane_wave_coeffs: plane_wave_coeffs_grid.coeffs.create_gpu_buffer(backend)?,
            // For update equation terms
            source_terms: vec![SourceTerms::default(); cell_count].create_gpu_buffer(backend)?,
            int_terms: vec![PmlIntegrals::default(); cell_count].create_gpu_buffer(backend)?,
            grid_coeffs: grid_coeffs.coeffs.create_gpu_buffer(backend)?,
            // Misc data
            thread_count: n_cells.n_cells_to_3d().to_array(),
            n_cells
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
        let source_pts = self.sources.iter()
            .filter_map(|src| {
                match src {
                    Source::Dipole { position, .. } => Some(position.to_3d(Vec3::ZERO)),
                    _ => None
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
        gpu_data: &mut FdtdLossyGpuData,
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
                &gpu_data.steps.buffer,
                &mut gpu_data.source_terms,
                &gpu_data.source_vals,
                &gpu_data.dipoles,
                &gpu_data.plane_waves,
                &gpu_data.plane_wave_coeffs,
            )?;
            self.update.call(
                pass,
                DispatchGrid::ThreadCount(gpu_data.thread_count),
                &gpu_data.grid_params,
                &mut gpu_data.steps.buffer,
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

#[derive(Copy, Clone, Debug)]
pub enum MaterialDiscretization {
    Rough,
    Smooth { resolution: NonZeroU32 }
}

// TODO: make an FdtdSolver trait?

/// Buffers and data needed for running the shader
pub struct FdtdLossyGpuData {
    // Uniforms / thread-independent vars
    pub grid_params: GpuBuffer<GridParameters>,
    pub steps: GpuBufferReadable<u32>,
    // Vector fields
    pub h: GpuBufferReadable<Vec4>,
    pub dn: GpuBufferReadable<Vec4>,
    pub en: GpuBufferReadable<Vec4>,
    // For computing source terms
    pub dipoles: GpuBuffer<GpuDipole>,
    pub plane_waves: GpuBuffer<GpuPlaneWave>,
    pub plane_wave_coeffs: GpuBuffer<PlaneWaveCoeffs>,
    pub source_vals: GpuBuffer<f32>,
    // For update equation terms
    pub source_terms: GpuBuffer<SourceTerms>,
    pub int_terms: GpuBuffer<PmlIntegrals>,
    pub grid_coeffs: GpuBuffer<PmlCoefficients>,
    // Misc data
    pub thread_count: [u32; 3],
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
    /// A plane wave traveling along an axis (positive direction).
    PlaneWave { // TODO: Implement plane wave in shader
        /// The spatial axis along which the plane wave will travel.
        spatial_axis: SpatialAxis,
        /// Position of the plane wave, along `spatial_axis`, in world coordinates.
        position: Real,
        /// The direction along `spatial_axis` the wave will travel in.
        direction: WaveDirection,
        /// The time (in the simulation, not real-time) when the source begins injection (in seconds).
        t_start: f32,
        /// Signal data points.
        vals: Vec<f32>
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

#[derive(Copy, Clone, Debug)]
#[repr(i32)]
pub enum WaveDirection {
    Positive = 1,
    Negative = -1,
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

/// Constructs an iterator of all cell positions in a grid of dimensions `n_cells`.
/// The cell positions are given as tuples but AREN'T IN ORDER
pub fn grid_cells_iter(n_cells: GridIndex) -> impl GridCellsIter {
    cfg_select! {
        feature = "dim1" => itertools::iproduct!(0..n_cells),
        feature = "dim2" => itertools::iproduct!(0..n_cells[SpatialAxis::X], 0..n_cells[SpatialAxis::Y]),
        feature = "dim3" => itertools::iproduct!(0..n_cells[SpatialAxis::X], 0..n_cells[SpatialAxis::Y], 0..n_cells[SpatialAxis::Z]),
    }
}