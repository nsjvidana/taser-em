use taser_em_shaders::fdtd1::PmlCoefficients;
use crate::ElectricMaterial;
use glamx::Vec3;
use parry3d::math::Pose;
use parry3d::shape::{Cuboid, SharedShape};
use taser_em_shaders::math::{GridIndex, MathW, To3D, Vect, DIM};

pub struct YeeGrid {
    /// Relative Permeability (located at H field components)
    pub mu_r: Vec<[Vec3; DIM]>,
    /// Relative Permittivity (located at E field components)
    pub eps_r: Vec<[Vec3; DIM]>,
    pub cell_size: Vect,
    pub mat_regions: Vec<(SharedShape, Pose, ElectricMaterial)>,
}

impl YeeGrid {
    pub fn update_coeffs_pml(&self, _dims: PmlDimensions, _n_cells: GridIndex) -> Vec<PmlCoefficients> {
        todo!()
    }

    pub fn fill_region(
        &mut self,
        start: Vect,
        end: Vect,
        material: ElectricMaterial
    ) {
        let half_extents = MathW((end - start) * self.cell_size / 2.)
            .to_3d(Vec3::ONE);
        let shape = SharedShape::new(Cuboid::new(half_extents));
        let middle = MathW((start + end) / 2.).to_3d(Vec3::ZERO);
        let pose = Pose::from_translation(middle);

        self.mat_regions.push((shape, pose, material));
    }

    /// Clears all stored material properties and material regions
    pub fn clear(&mut self) {
        self.mu_r.clear();
        self.eps_r.clear();
        self.mat_regions.clear();
    }

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