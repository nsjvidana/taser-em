use taser_em_shaders::fdtd1::PmlCoefficients;
use crate::ElectricMaterial;
use glamx::Vec3;
use taser_em_shaders::math::{GridIndex, DIM};

pub struct YeeGrid {
    /// Relative Permeability (located at H field components)
    pub mu_r: Vec<[Vec3; DIM]>,
    /// Relative Permittivity (located at E field components)
    pub eps_r: Vec<[Vec3; DIM]>,
    // #[cfg(feature = "parry")]
    // pub shapes: Vec<(SharedShape, Pose, ElectricMaterial)>,
}

impl YeeGrid {
    pub fn update_coeffs_pml(&self, dims: PmlDimensions, n_cells: GridIndex) -> Vec<PmlCoefficients> {
        todo!()
    }

    pub fn fill_region(
        &mut self,
        start: GridIndex,
        end: GridIndex,
        material: ElectricMaterial
    ) -> Vec<PmlCoefficients> {
        todo!()
    }

    // #[cfg(feature = "parry")]
    // pub fn shape_in_grid(shape: SharedShape, pose: Pose, material: ElectricMaterial)

    // normal map creation?
}

pub struct PmlDimensions {
    pub widths: [[u32; 2]; DIM],
}

impl PmlDimensions {
    // TODO: a new() function for each dim feature?
    #[cfg(feature = "dim1")]
    pub fn new(z_lo: u32, z_hi: u32) -> Self {
        Self {
            widths: [[z_lo, z_hi]]
        }
    }
}