use taser_em_shaders::math::*;

/// Measure Poynting Flux flowing through a plane perpendicular to `axis`, stretching to the edges
/// of the grid
///
/// Cuts into spacer regions, meaning spacer regions won't wrap around [`PowerFluxMonitor`]s.
/// If the monitor ends up outside the grid, an error will be returned.
#[derive(Clone, Debug)]
pub struct PowerFluxMonitor {
    pub axis: SpatialAxis,
    /// Position of plane along `axis` in world-space.
    pub position: Real,
    /// The direction to measure flow in.
    ///
    /// This matters when measuring transmittance / reflection.
    pub direction: Direction,
    /// All the frequencies of a DFT that will be run on this monitor's recorded values
    pub dft_frequencies: Option<Vec<Real>>,
}

// TODO: probe monitor (measures at one point)