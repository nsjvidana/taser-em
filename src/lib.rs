pub mod prelude;
pub mod gpu_util;
pub mod grid;

use std::num::NonZeroI32;
pub use taser_em_shaders as shaders;

use crate::gpu_util::{CreateGpuBuffer, CreateGpuBufferReadable, GpuBufferReadable};
use crate::prelude::GpuResult;
use derivative::Derivative;
use glamx::{Pose3, UVec3, Vec3, Vec4};
use khal::backend::{Backend, Buffer, Encoder, GpuBackend, GpuBuffer, GpuPass, GpuTimestamps};
use khal::re_exports::include_dir::{include_dir, Dir};
use khal::Shader;
use taser_em_shaders::fdtd1::{GridParameters, PmlCoefficients};
use taser_em_shaders::fdtd::{GridParameters2, IntegrationTerms, PmlCoefficients2};
use taser_em_shaders::math::{grid_index_to_3d, grid_index_to_array, grid_index_to_flat_idx, vect_to_3d, vect_from_array, vect_to_array, GridIndex, Index, Real, SpatialAxis, Vect, DIM};
use taser_em_shaders::math::Axis;
use crate::grid::{LayerWidths, YeeGrid};

pub static SPIRV_DIR: Dir<'static> = include_dir!("$OUT_DIR/shaders-spirv");

/// Speed of EM wave in free space
pub const C_0: f32 = 299792458.0;
/// Free space permittivity
pub const EPS_0: f32 = 8.854188e-12;
/// Free space permeability
pub const MU_0: f32 = 1.256637e-6;
/// Free space wave impedance
pub const IMPEDANCE_0: f32 = 376.73032;

/// Constructs an iterator of all cell positions in a grid of dimensions `$n_cells`.
/// The cell positions are given as tuples.
#[macro_export]
macro_rules! grid_cells_iter {
    ($n_cells:expr) => {
        cfg_select! {
            feature = "dim1" => itertools::iproduct!(0..$n_cells),
            feature = "dim2" => itertools::iproduct!(0..$n_cells[SpatialAxis::X], 0..$n_cells[SpatialAxis::Y]),
            feature = "dim3" => itertools::iproduct!(0..$n_cells[SpatialAxis::X], 0..$n_cells[SpatialAxis::Y], 0..$n_cells[SpatialAxis::Z]),
        }
    };
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

shader_struct!(FdtdWithLoss, taser_em_shaders::fdtd::FdtdLossy);

pub struct FdtdSolver {
    pub kernel: FdtdWithLoss,
    pub grid: YeeGrid,
    pub pml_parameters: PmlParameters,
    pub dt: Real,
}

impl FdtdSolver {
    /// Construct a solver with generally stable PML parameters.
    pub fn new(backend: &GpuBackend, grid: YeeGrid, dt: Real) -> GpuResult<FdtdSolver> {
        Ok(Self {
            kernel: FdtdWithLoss::from_backend(backend)?,
            grid,
            pml_parameters: PmlParameters::new(dt),
            dt,
        })
    }

    /// Creates GPU buffers for simulating. Does the following:
    /// - discretize shapes
    /// - calculate update coefficients
    /// - initialize buffers and return them
    pub fn compute_and_create_buffers(&self, backend: &GpuBackend) -> GpuResult<FdtdSolverBuffers> {
        let (n_cells, grid_coeffs) = self.grid
            .update_coeffs_pml(self.pml_parameters, Pose3::IDENTITY, self.dt);

        let cell_count = grid_index_to_array(n_cells)
            .iter()
            .product::<Index>() as usize;
        let zeroed_vector_field = vec![Vec4::ZERO; cell_count];
        let flat_idx_incrs = {
            let mut incrs = UVec3::ZERO;
            for (spatial_axis, axis) in SpatialAxis::ALL_SPATIAL.into_iter().zip(SpatialAxis::ALL_AXES) {
                let mut grid_incr = GridIndex::default();
                    grid_incr[spatial_axis] = 1;
                incrs[axis] = grid_index_to_flat_idx(grid_incr, n_cells);
            }
            incrs
        };
        Ok(
            FdtdSolverBuffers {
                h: zeroed_vector_field.create_gpu_buffer_readable(backend)?,
                dn: zeroed_vector_field.create_gpu_buffer_readable(backend)?,
                en: zeroed_vector_field.create_gpu_buffer_readable(backend)?,
                int_terms: vec![IntegrationTerms::default(); cell_count].create_gpu_buffer(backend)?,
                grid_coeffs: grid_coeffs.create_gpu_buffer(backend)?,
                grid_params: GridParameters2 {
                    flat_idx_incrs,
                    polarization_mode: self.grid.polarization_mode as u32,
                    n_cells: grid_index_to_3d(n_cells, UVec3::ONE),
                    d: vect_to_3d(self.grid.cell_size, Vec3::ZERO),
                    ..Default::default()
                }.create_gpu_uniform(backend)?,
            }
        )
    }

    /// Submit a simulation step into `pass` using the GPU buffers `buffers`.
    pub fn submit_step(&self, buffers: &mut FdtdSolverBuffers, pass: &mut GpuPass) -> GpuResult<()> {
        let FdtdSolverBuffers {
            h,
            dn,
            en,
            int_terms,
            grid_coeffs,
            grid_params,
        } = buffers;
        self.kernel.call(
            pass,
            h.buffer.len(),
            &mut h.buffer,
            &mut dn.buffer,
            &mut en.buffer,
            int_terms,
            grid_coeffs,
            grid_params,
        )
    }
}

pub struct FdtdSolverBuffers {
    pub h: GpuBufferReadable<Vec4>,
    pub dn: GpuBufferReadable<Vec4>,
    pub en: GpuBufferReadable<Vec4>,
    pub int_terms: GpuBuffer<IntegrationTerms>,
    pub grid_coeffs: GpuBuffer<PmlCoefficients2>,
    pub grid_params: GpuBuffer<GridParameters2>,
}

/// Parameters for how the PML will be constructed in the simulation
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
}

impl FdtdStability {
    pub fn cell_size_from_min_wavelength(&self, f_max: Real) -> Vect {
        let min_wavelen = C_0 / f_max;
        let cell_size = min_wavelen / self.cells_per_wavelength as Real;
        vect_from_array([cell_size; DIM])
    }

    pub fn cfl_condition(&self, cell_size: Vect) -> Real {
        let cell_size_term = vect_to_array(cell_size)
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

    // pub fn dt_from_source(&self, source: FdtdSource) -> f32 {}
}

// TODO: remove this shader after testing it against FdtdLossy
shader_struct!(Fdtd1, taser_em_shaders::fdtd1::Fdtd1DnY);

#[derive(Default)]
pub struct FdtdParameters1 {
    pub dt: f32,
    /// Number of grid cells (in each principal direction)
    pub n_cells: GridIndex,
    /// Size of each cell (in meters)
    pub cell_size: Vect,
    // TODO: E-field (or H field for current loops) Source enum
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

pub struct Fdtd1Runner {
    pub h_x: Vec<f32>,
    pub dn_y: Vec<f32>,
    pub en_y: Vec<f32>,
    pub int_en_y: Vec<f32>,
    pub buffers: Fdtd1Buffers,
}

impl Fdtd1Runner {
    // TODO: make Fdtd1Runner with things other than just `num_cells`
    pub fn new(
        backend: &GpuBackend,
        num_cells: usize,
        grid_params: GridParameters
    ) -> GpuResult<Self> {
        let h_x = vec![Default::default(); num_cells];
        let dn_y = vec![Default::default(); num_cells];
        let en_y = vec![Default::default(); num_cells];
        let int_en_y = vec![Default::default(); num_cells];
        let coeffs = vec![PmlCoefficients::default(); num_cells];

        let buffers = Fdtd1Buffers {
            h_x: h_x.create_gpu_buffer_readable(backend)?,
            dn_y: dn_y.create_gpu_buffer_readable(backend)?,
            en_y: en_y.create_gpu_buffer_readable(backend)?,
            int_en_y: int_en_y.create_gpu_buffer(backend)?,
            coeffs: coeffs.create_gpu_buffer(backend)?,
            grid: grid_params.create_gpu_uniform(backend)?,
        };

        Ok(Self {
            h_x,
            dn_y,
            en_y,
            int_en_y,
            buffers,
        })
    }

    pub fn submit(
        &mut self,
        backend: &GpuBackend,
        kernel: &taser_em_shaders::fdtd1::Fdtd1DnY,
        timestamps: Option<&mut GpuTimestamps>
    ) -> GpuResult<()> {
        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("fdtd1_dn_y", timestamps);
        kernel.call(
            &mut pass,
            self.dn_y.len(),
            &mut self.buffers.h_x.buffer,
            &mut self.buffers.dn_y.buffer,
            &mut self.buffers.en_y.buffer,
            &mut self.buffers.int_en_y,
            &self.buffers.coeffs,
            &self.buffers.grid,
        )?;
        drop(pass);
        self.buffers.h_x.encode_copy_cmd(&mut encoder)?;
        self.buffers.dn_y.encode_copy_cmd(&mut encoder)?;
        self.buffers.en_y.encode_copy_cmd(&mut encoder)?;

        backend.submit(encoder)?;

        Ok(())
    }
}

pub struct Fdtd1Buffers {
    pub h_x: GpuBufferReadable<f32>,
    pub dn_y: GpuBufferReadable<f32>,
    pub en_y: GpuBufferReadable<f32>,
    pub int_en_y: GpuBuffer<f32>,
    pub coeffs: GpuBuffer<PmlCoefficients>,
    pub grid: GpuBuffer<GridParameters>,
}

#[derive(Clone, Debug)]
pub enum Source {
    Dipole {
        position: Vect,
        t_offset: f32,
        vals: Vec<f32>
    },
    PlaneWave {
        axis: SpatialAxis,
        t_offset: f32,
        vals: Vec<f32>
    }
}

impl Source {
    /// Helper function that generates data points for a Gaussian curve with a maximum frequency of
    /// `f_max` (Hz).
    ///
    /// # Panics
    /// When `f_max <= 0.` or when `dt <= 0.`
    pub fn gaussian_max_f(&mut self, f_max: f32, amplitude: f32, dt: f32) -> Vec<f32> {
        assert!(f_max > 0.0, "f_max must be > 0");
        let tau = core::f32::consts::FRAC_1_PI / f_max;
        let t_0 = 6. * tau;
        let approx_dur = 12. * tau;
        self.function_data_points(dt, approx_dur, |t| {
            amplitude * core::f32::consts::E.powf(-((t - t_0) / tau).powi(2))
        })
    }

    /// Samples data points from the function of time `f`
    ///
    /// # Panics
    /// When `dt <= 0.` or `duration <= 0.`
    pub fn function_data_points(&mut self, dt: f32, duration: f32, mut f: impl FnMut(f32) -> f32) -> Vec<f32> {
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
            t_offset: Default::default(),
            vals: Default::default(),
        }
    }
}

shader_struct!(AddAssign, taser_em_shaders::AddAssign);

pub struct AddAssignRunner {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub buffers: AddAssignBuffers
}

impl AddAssignRunner {
    pub fn new(backend: &GpuBackend, a: Vec<f32>, b: Vec<f32>) -> GpuResult<Self> {
        let buffers = AddAssignBuffers {
            a: a.create_gpu_buffer_readable(backend)?,
            b: b.create_gpu_buffer(backend)?
        };

        Ok(Self { a, b, buffers })
    }

    pub fn submit(
        &mut self,
        backend: &GpuBackend,
        kernel: &taser_em_shaders::AddAssign,
        timestamps: Option<&mut GpuTimestamps>
    ) -> GpuResult<()> {
        let mut encoder = backend.begin_encoding();

        let mut pass = encoder.begin_pass("add_assign", timestamps);

        let AddAssignBuffers {
            a, b
        } = &mut self.buffers;
        kernel.call(
            &mut pass,
            self.a.len(),
            &mut a.buffer,
            b
        )?;
        drop(pass);
        a.encode_copy_cmd(&mut encoder)?;

        backend.submit(encoder)?;

        Ok(())
    }
}

pub struct AddAssignBuffers {
    pub a: GpuBufferReadable<f32>,
    pub b: GpuBuffer<f32>,
}