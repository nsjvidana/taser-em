use taser_em_shaders::math::*;

/// Measure Poynting Flux flowing through an infinitely wide plane perpendicular to `axis`.
///
/// Cuts into spacer regions, meaning spacer regions won't wrap around [`PowerFluxMonitor`]s.
/// If the monitor ends up outside the grid, an error will be returned.
#[derive(Copy, Clone, Debug)]
pub struct PowerFluxMonitor {
    pub axis: SpatialAxis,
    /// Position of plane along `axis` in world-space.
    pub position: Real,
    /// The direction to measure flow in.
    ///
    /// This matters when measuring transmittance / reflection.
    pub direction: Direction,
    // TODO: fourier transform frequencies
}

// TODO: probe monitor (measures at one point)