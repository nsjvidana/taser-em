use taser_em_shaders::fdtd1::PmlCoefficients;
use crate::ElectricMaterial;
use glamx::Vec3;
use parry3d::bounding_volume::{Aabb, BoundingVolume};
use parry3d::math::Pose;
use parry3d::shape::{Cuboid, SharedShape};
use taser_em_shaders::math::{GridIndex, MathW, To3D, Vect, DIM};

pub struct MaterialRegions {
    pub regions: Vec<(SharedShape, Pose, ElectricMaterial)>,
}

impl MaterialRegions {
    pub fn fill_region(
        &mut self,
        start: Vect,
        end: Vect,
        material: ElectricMaterial
    ) -> &mut Self {
        let half_extents = MathW(end - start).to_3d(Vec3::ONE);
        let shape = SharedShape::new(Cuboid::new(half_extents));
        let middle = MathW((start + end) / 2.).to_3d(Vec3::ZERO);
        let pose = Pose::from_translation(middle);

        self.regions.push((shape, pose, material));
        self
    }

    pub fn compute_bounding_box(&self) -> Aabb {
        let aabbs = self.regions
            .iter()
            .map(|(s, pose, _)| s.compute_aabb(pose))
            .collect::<Vec<_>>();

        let mut full_bb = Aabb::new_invalid();
        for bb in aabbs.iter() {
            full_bb.merge(bb);
        }
        full_bb
    }
}

pub struct YeeGrid {
    /// Relative Permeability (located at H field components)
    pub mu_r: Vec<[Vec3; DIM]>,
    /// Relative Permittivity (located at E field components)
    pub eps_r: Vec<[Vec3; DIM]>,
    pub cell_size: Vect,
}

impl YeeGrid {
    pub fn update_coeffs_pml(&self, _dims: PmlDimensions, _n_cells: GridIndex) -> Vec<PmlCoefficients> {
        todo!()
    }

    /// Clears all stored material properties and material regions
    pub fn clear(&mut self) {
        self.mu_r.clear();
        self.eps_r.clear();
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