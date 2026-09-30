use std::num::{NonZeroI32, NonZeroU32};
use khal::backend::GpuBuffer;
use taser_em_shaders::math::{Direction, Real, SpatialAxis, Vect};
use taser_em_shaders::source::{AuxGridPmlCoeffs, AuxVect, DipoleType, GpuTfsf, TfsfMask, TfsfSourceValues};
use crate::grid::LayerWidths;
use crate::prelude::Vec3;

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

pub struct TfsfDispatchData {
    pub tfsf_sources: GpuBuffer<GpuTfsf>,
    pub tfsf_masks: GpuBuffer<TfsfMask>,
    pub corrections: GpuBuffer<TfsfSourceValues>,
    pub auxgr_coeffs: GpuBuffer<AuxGridPmlCoeffs>,
    pub h: GpuBuffer<AuxVect>,
    pub dn: GpuBuffer<AuxVect>,
    pub en: GpuBuffer<AuxVect>,
    /// Thread count for simulating auxiliary grids for ALL plane waves.
    ///
    /// Is [`None`] only when there are no TF/SF sources.
    pub aux_grid_thread_count: Option<[u32; 3]>,
    pub mask_init_thread_count: Option<[u32; 3]>,
}