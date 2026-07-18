pub mod prelude;
pub mod gpu_util;
pub mod grid;
pub mod util;
pub mod re_exports {
    pub use glamx;
    pub use khal;
}

pub use taser_em_shaders as shaders;

use std::num::{NonZeroI32, NonZeroU32, NonZeroUsize};

use crate::gpu_util::{CreateGpuBuffer, CreateGpuBufferReadable, GpuBufferReadable};
use crate::grid::{LayerWidths, MaterialRegions, PmlCoefficientsGrid, YeeGridMaterials};
use crate::prelude::{GpuResult, PolarizationMode};
use derivative::Derivative;
use glamx::{UVec3, Vec3, Vec4};
use khal::backend::{Backend, DispatchGrid, Encoder, GpuBackend, GpuBuffer, GpuEncoder, GpuTimestamps};
use khal::re_exports::include_dir::{include_dir, Dir};
use khal::Shader;
use parry3d::bounding_volume::Aabb;
use taser_em_shaders::fdtd::{GpuDipole, GridParameters, IntegrationTerms, PmlCoefficients};
use taser_em_shaders::fdtd_v2::{PmlCoefficients2, PmlIntegrals2};
use taser_em_shaders::math::{Axis, BoolVectExt, GridIndexExt, VectExt, VectorValueExt};
use taser_em_shaders::math::{GridIndex, Index, Real, SpatialAxis, Vect, DIM};

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

        let buffers = {
            let mut source_vals: Vec<Real> = vec![];
            let regions_offset = Vect::from_vec3(regions_offset);
            let mut dipoles = self.sources.iter()
                .filter_map(|source| {
                    match source {
                        Source::Dipole { position, t_start, vals } => {
                            let pos = (regions_offset + position) / cell_size;
                            let cell_grid_idx = pos.as_grid_index();
                            debug_assert!(
                                !pos.smallest_element().is_sign_negative() && !BoolVectExt::any(VectorValueExt::ge(pos.as_grid_index(), n_cells)),
                                "negative source position!"
                            );
                            let start = source_vals.len();
                            source_vals.extend_from_slice(vals);
                            Some(GpuDipole {
                                cell_idx: cell_grid_idx.to_flat_idx(n_cells),
                                vals_range: [start as u32, source_vals.len() as u32 - 1],
                                t_start: (t_start / dt) as u32,
                            })
                        },
                        _ => None
                    }
                })
                .collect::<Vec<_>>();
            if dipoles.is_empty() { dipoles.push(GpuDipole::default()) }
            if source_vals.is_empty() { source_vals.push(0.0); }

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

            let cell_count = n_cells.into_array()
                .iter()
                .product::<Index>() as usize;
            let zeroed_vector_field = vec![Vec4::ZERO; cell_count];
            FdtdLossyGpuData {
                h: zeroed_vector_field.create_gpu_buffer_readable(backend)?,
                dn: zeroed_vector_field.create_gpu_buffer_readable(backend)?,
                en: zeroed_vector_field.create_gpu_buffer_readable(backend)?,
                int_terms: vec![PmlIntegrals2::default(); cell_count].create_gpu_buffer(backend)?,
                grid_coeffs: grid_coeffs.coeffs.create_gpu_buffer(backend)?,
                dipoles: dipoles.create_gpu_buffer(backend)?,
                source_vals: source_vals.create_gpu_buffer(backend)?,
                steps: 0.create_gpu_buffer(backend)?,
                grid_params: GridParameters {
                    flat_idx_incrs,
                    polarization_mode: polarization_mode.into(),
                    n_cells3: n_cells.n_cells_to_3d(),
                    d: cell_size.to_3d(Vec3::ZERO),
                    ..Default::default()
                }.create_gpu_uniform(backend)?,
                thread_count: n_cells.n_cells_to_3d().to_array(),
                n_cells
            }
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

        let n_cells_spacer = stability.spacer_region_widths
            .sum_with_n_cells(materials_n_cells);
        self.pml_parameters.widths.sum_with_n_cells(n_cells_spacer)
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

pub struct FdtdLossyPipeline {
    kernel: FdtdWithLoss,
    num_steps_per_submission: usize,
}

impl FdtdLossyPipeline {
    pub fn new(backend: &GpuBackend, num_steps_per_submission: NonZeroUsize) -> GpuResult<Self> {
        Ok(Self {
            kernel: FdtdWithLoss::from_backend(backend)?,
            num_steps_per_submission: num_steps_per_submission.get(),
        })
    }

    pub fn submit_steps(
        &self,
        backend: &GpuBackend,
        gpu_data: &mut FdtdLossyGpuData,
        timestamps: Option<&mut GpuTimestamps>,
        encoding_fn: impl Fn(&mut GpuEncoder, &mut FdtdLossyGpuData) -> GpuResult<()>
    ) -> GpuResult<()> {
        let timestamps = timestamps.filter(|ts| ts.is_idle());
        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("fdtd_lossy", timestamps);
        for _ in 0..self.num_steps_per_submission {
            self.kernel.call(
                &mut pass,
                DispatchGrid::ThreadCount(gpu_data.thread_count),
                &mut gpu_data.h.buffer,
                &mut gpu_data.dn.buffer,
                &mut gpu_data.en.buffer,
                &mut gpu_data.int_terms,
                &gpu_data.grid_coeffs,
                &gpu_data.dipoles,
                &gpu_data.source_vals,
                &mut gpu_data.steps,
                &gpu_data.grid_params,
            )?;
        }
        drop(pass);
        encoding_fn(&mut encoder, gpu_data)?;
        backend.submit(encoder)?;
        Ok(())
    }
}

macro_rules! shader_struct {
    ($name:ident, $inner:ty) => {
        #[derive(Shader)]
        pub struct $name {
            kernel: $inner
        }

        impl AsRef<$inner> for $name {
            fn as_ref(&self) -> &$inner {
                &self.kernel
            }
        }

        impl std::ops::Deref for $name {
            type Target = $inner;

            fn deref(&self) -> &Self::Target {
                &self.kernel
            }
        }
    };
}

shader_struct!(FdtdWithLoss, taser_em_shaders::fdtd_v2::FdtdLossyV2);

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
    pub h: GpuBufferReadable<Vec4>,
    pub dn: GpuBufferReadable<Vec4>,
    pub en: GpuBufferReadable<Vec4>,
    pub int_terms: GpuBuffer<PmlIntegrals2>,
    pub grid_coeffs: GpuBuffer<PmlCoefficients2>,
    pub dipoles: GpuBuffer<GpuDipole>,
    pub source_vals: GpuBuffer<f32>,
    pub steps: GpuBuffer<u32>,
    pub grid_params: GpuBuffer<GridParameters>,
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
            widths: LayerWidths::splat(12),
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
    #[derivative(Default(value = "LayerWidths::splat(10)"))]
    pub spacer_region_widths: LayerWidths,
}

impl FdtdStability {
    pub fn cell_size_from_min_wavelength(&self, f_max: Real) -> Vect {
        let min_wavelen = C_0 / f_max;
        let cell_size = min_wavelen / self.cells_per_wavelength as Real;
        Vect::from_array([cell_size; DIM])
    }

    pub fn cfl_condition(&self, cell_size: Vect) -> Real {
        let cell_size_term = cell_size.into_array()
            .map(|v| {
                v.powi(2).recip()
            })
            .iter()
            .sum::<Real>()
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
    pub fn dt_from_gaussian_freq(&self, f_max: Real) -> f32 {
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

    /// Get refractive index in a specific axis direction.
    #[allow(unused_variables)]
    pub fn refractive_index(&self, axis: Axis) -> f32 {
        let i = axis as usize;
        (self.eps_r[i] * self.mu_r[i]).sqrt()
    }
}

/// Inject energy into the simulation in various ways.
#[derive(Clone, Debug)]
pub enum Source {
    /// Electric Dipole
    Dipole {
        /// The position in space where the source should be injected
        position: Vect,
        /// How long to wait until the source should enable (in seconds)
        t_start: f32,
        /// Signal data points
        vals: Vec<f32>
    },
    /// A plane wave traveling along an axis (positive direction)
    PlaneWave {
        /// The axis along which the plane wave will travel.
        axis: SpatialAxis,
        /// How long to wait until the source should enable (in seconds)
        t_start: f32,
        /// Signal data points
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

impl Default for Source {
    fn default() -> Self {
        Self::Dipole {
            position: Default::default(),
            t_start: Default::default(),
            vals: Default::default(),
        }
    }
}


/// Constructs an iterator of all cell positions in a grid of dimensions `n_cells`.
/// The cell positions are given as tuples.
pub fn grid_cells_iter(n_cells: GridIndex) -> impl GridCellsIter {
    cfg_select! {
        feature = "dim1" => itertools::iproduct!(0..n_cells),
        feature = "dim2" => itertools::iproduct!(0..n_cells[SpatialAxis::X], 0..n_cells[SpatialAxis::Y]),
        feature = "dim3" => itertools::iproduct!(0..n_cells[SpatialAxis::X], 0..n_cells[SpatialAxis::Y], 0..n_cells[SpatialAxis::Z]),
    }
}