pub mod prelude;
pub mod gpu_util;
pub mod grid;
pub mod util;
pub mod re_exports {
    pub use glamx;
    pub use khal;
}

use std::num::{NonZeroI32, NonZeroU32};
pub use taser_em_shaders as shaders;

use crate::gpu_util::{CreateGpuBuffer, CreateGpuBufferReadable, GpuBufferReadable};
use crate::grid::{LayerWidths, PmlCoefficientsGrid, YeeGrid};
use crate::prelude::GpuResult;
use derivative::Derivative;
use glamx::{UVec3, Vec3, Vec4};
use khal::backend::{DispatchGrid, GpuBackend, GpuBuffer, GpuPass};
use khal::re_exports::include_dir::{include_dir, Dir};
use khal::Shader;
use taser_em_shaders::fdtd::{GpuDipole, GridParameters2, IntegrationTerms, PmlCoefficients2};
use taser_em_shaders::math::{Axis, GridIndexExt, VectExt, VectorValueExt};
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


/// Constructs an iterator of all cell positions in a grid of dimensions `n_cells`.
/// The cell positions are given as tuples.
pub fn grid_cells_iter(n_cells: GridIndex) -> impl GridCellsIter {
    cfg_select! {
        feature = "dim1" => itertools::iproduct!(0..n_cells),
        feature = "dim2" => itertools::iproduct!(0..n_cells[SpatialAxis::X], 0..n_cells[SpatialAxis::Y]),
        feature = "dim3" => itertools::iproduct!(0..n_cells[SpatialAxis::X], 0..n_cells[SpatialAxis::Y], 0..n_cells[SpatialAxis::Z]),
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

shader_struct!(FdtdWithLoss, taser_em_shaders::fdtd::FdtdLossyV2);

pub struct FdtdSolver {
    pub kernel: FdtdWithLoss,
    pub grid: YeeGrid,
    pub pml_parameters: PmlParameters,
    pub dt: Real,
    pub sources: Vec<Source>
}

impl FdtdSolver {
    /// Construct a solver with generally stable PML parameters.
    pub fn new(backend: &GpuBackend, grid: YeeGrid, dt: Real) -> GpuResult<FdtdSolver> {
        Ok(Self {
            kernel: FdtdWithLoss::from_backend(backend)?,
            grid,
            pml_parameters: PmlParameters::new(dt),
            dt,
            sources: Vec::new()
        })
    }

    #[inline]
    pub fn add_source(&mut self, source: Source) -> &mut Self {
        self.sources.push(source);
        self
    }

    /// Computes dimensions of entire Yee grid, including PML
    pub fn grid_n_cells(&self) -> GridIndex {
        self.pml_parameters.widths.sum_with_n_cells(self.n_cells_inner())
    }

    /// Computes dimensions of the grid, excluding PML
    pub fn n_cells_inner(&self) -> GridIndex {
        let source_pts = self.sources.iter()
            .filter_map(|src| {
                match src {
                    Source::Dipole { position, .. } => Some(position.to_3d(Vec3::ZERO)),
                    _ => None
                }
            })
            .collect::<Vec<_>>();
        self.grid.n_cells(Some(&source_pts))
    }

    /// Calculates materials (see [YeeGrid::compute_materials_smoothed]) PML coefficients.
    ///
    /// Returns the coefficients and the offset applied to each [`grid::MaterialRegion`] to align them with
    /// the grid.
    pub fn compute_pml_coeffs(&self) -> (Vec3, PmlCoefficientsGrid) {
        let n_cells = self.pml_parameters.widths.sum_with_n_cells(
            self.n_cells_inner()
        );
        let grid_mats = self.grid.compute_materials_smoothed(n_cells);
        PmlCoefficientsGrid::new(&grid_mats, self.pml_parameters, self.dt)
    }

    /// Creates GPU buffers and dispatch parameters for simulating.
    pub fn create_shader_data(
        &self,
        backend: &GpuBackend,
        coeffs_grid: &PmlCoefficientsGrid,
        regions_offset: Vec3
    ) -> GpuResult<FdtdSolverGpuData> {
        let n_cells = coeffs_grid.n_cells;
        let grid_coeffs = &coeffs_grid.coeffs;

        let mut source_vals: Vec<Real> = vec![];
        let regions_offset = Vect::from_vec3(regions_offset);
        let mut dipoles = self.sources.iter()
            .filter_map(|source| {
                // todo!("convert dipoles for GPU use, and populate source_vals")
                match source {
                    Source::Dipole { position, t_start, vals } => {
                        let pos = (regions_offset + position) / self.grid.cell_size;
                        debug_assert!(!pos.smallest_element().is_sign_negative());
                        let start = source_vals.len();
                        source_vals.extend_from_slice(vals);
                        Some(GpuDipole {
                            cell_idx: pos.as_grid_index().to_flat_idx(n_cells),
                            vals_range: [start as u32, source_vals.len() as u32 - 1],
                            t_start: (t_start / self.dt) as u32,
                        })
                    }
                    _ => None
                }
            })
            .collect::<Vec<_>>();
        if dipoles.is_empty() { dipoles.push(GpuDipole::default()) }
        if source_vals.is_empty() { source_vals.push(0.0); }

        let cell_count = n_cells.into_array()
            .iter()
            .product::<Index>() as usize;
        let zeroed_vector_field = vec![Vec4::ZERO; cell_count];
        let flat_idx_incrs = {
            let mut incrs = UVec3::ZERO;
            for (spatial_axis, axis) in SpatialAxis::ALL_SPATIAL.into_iter().zip(SpatialAxis::ALL_AXES) {
                let mut grid_incr = GridIndex::default();
                    grid_incr[spatial_axis] = 1;
                incrs[axis] = grid_incr.to_flat_idx(n_cells);
            }
            incrs
        };
        Ok(
            FdtdSolverGpuData {
                h: zeroed_vector_field.create_gpu_buffer_readable(backend)?,
                dn: zeroed_vector_field.create_gpu_buffer_readable(backend)?,
                en: zeroed_vector_field.create_gpu_buffer_readable(backend)?,
                int_terms: vec![IntegrationTerms::default(); cell_count].create_gpu_buffer(backend)?,
                grid_coeffs: grid_coeffs.create_gpu_buffer(backend)?,
                dipoles: dipoles.create_gpu_buffer(backend)?,
                source_vals: source_vals.create_gpu_buffer(backend)?,
                steps: 0.create_gpu_buffer(backend)?,
                grid_params: GridParameters2 {
                    flat_idx_incrs,
                    polarization_mode: self.grid.polarization_mode.into(),
                    n_cells3: n_cells.n_cells_to_3d(),
                    d: self.grid.cell_size.to_3d(Vec3::ZERO),
                    ..Default::default()
                }.create_gpu_uniform(backend)?,
                thread_count: n_cells.n_cells_to_3d().to_array()
            }
        )
    }

    /// Submit a simulation step into `pass` using the GPU buffers `buffers`.
    pub fn submit_step(&self, buffers: &mut FdtdSolverGpuData, pass: &mut GpuPass) -> GpuResult<()> {
        let FdtdSolverGpuData {
            h,
            dn,
            en,
            int_terms,
            grid_coeffs,
            dipoles,
            source_vals,
            steps,
            grid_params,
            thread_count,
        } = buffers;
        self.kernel.call(
            pass,
            DispatchGrid::ThreadCount(*thread_count),
            &mut h.buffer,
            &mut dn.buffer,
            &mut en.buffer,
            int_terms,
            grid_coeffs,
            dipoles,
            source_vals,
            steps,
            grid_params,
        )
    }
}

/// Buffers and data needed for running the shader
pub struct FdtdSolverGpuData {
    pub h: GpuBufferReadable<Vec4>,
    pub dn: GpuBufferReadable<Vec4>,
    pub en: GpuBufferReadable<Vec4>,
    pub int_terms: GpuBuffer<IntegrationTerms>,
    pub grid_coeffs: GpuBuffer<PmlCoefficients2>,
    pub dipoles: GpuBuffer<GpuDipole>,
    pub source_vals: GpuBuffer<f32>,
    pub steps: GpuBuffer<u32>,
    pub grid_params: GpuBuffer<GridParameters2>,
    pub thread_count: [u32; 3]
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