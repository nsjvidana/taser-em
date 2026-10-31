use crate::gpu_util::*;
use crate::prelude::*;
use itertools::Itertools;
use khal::backend::{Backend, Buffer, DispatchGrid, GpuBackend, GpuBuffer, GpuPass, GpuReadback};
use ndarray::IxDyn;
use parry3d::bounding_volume::Aabb;
use std::sync::Arc;
use taser_em_shaders::dft::*;
use taser_em_shaders::math::*;

#[derive(Clone, Debug)]
pub struct DftRegion {
    pub frequencies: Vec<Real>,
    /// Minimum point of region bounding box in world coordinates
    pub bb_min: Vect,
    /// Maximum point of region bounding box in world coordinates
    pub bb_max: Vect,
    pub fields: DftFields,
}

impl DftRegion {
    pub fn new(frequencies: Vec<Real>, bb_min: Vect, bb_max: Vect, fields: DftFields) -> DftRegion {
        Self { frequencies, bb_min, bb_max, fields }
    }
}

pub struct DftRegionsHook {
    update: GpuDftRegionsUpdate,
    state: DftRegionStates, // TODO: make states into an input for hooks
}

impl DftRegionsHook {
    pub fn new(backend: &GpuBackend, state: DftRegionStates) -> TaserResult<Self> {
        Ok(Self {
            update: GpuDftRegionsUpdate::from_dir(backend, &crate::SPIRV_DIR)?,
            state,
        })
    }
    
    pub fn state(&self) -> &DftRegionStates { &self.state }
}

impl FdtdLossyHook for DftRegionsHook {
    fn dispatch_step(&mut self, pass: &mut GpuPass, sim_state: &mut FdtdLossyState) -> TaserResult<()> {
        let DftRegionStates {
            params, dft_regions: regions,
            h_dfts, en_dfts, sn_dfts,
            update_workgroups, ..
        } = &mut self.state;

        // TODO: defer this shader by cloning En & H into separate buffer
        self.update.call(
            pass,
            DispatchGrid::Grid(*update_workgroups),
            params,
            &sim_state.t_idx,
            &sim_state.h,
            &sim_state.en,
            regions,
            h_dfts,
            en_dfts,
            sn_dfts
        )?;
        Ok(())
    }
}

pub struct DftRegionReadback {
    refs: Vec<DftRegionDescription>,
    h_dfts: Vec<ComplexVec3>,
    en_dfts: Vec<ComplexVec3>,
    sn_dfts: Vec<ComplexVec3>,
    h_dfts_read: GpuReadback<ComplexVec3>,
    en_dfts_read: GpuReadback<ComplexVec3>,
    sn_dfts_read: GpuReadback<ComplexVec3>,
}

macro_rules! readback_get {
    ($name:ident, $field:ident) => {
        pub fn $name(&self, region: &Arc<DftRegion>, freq_idx: usize) -> TaserResult<ndarray::ArrayD<ComplexVec3>> {
            let desc = self.get_region_desc(region)?;
            let idx_range = {
                let start = desc.freq_buf_offset(freq_idx);
                start..(start + desc.cell_count as usize)
            };
            let field = self.$field.get(idx_range).expect("freq_idx must be a valid frequency index");
            let dims = desc.n_cells3.as_usizevec3().to_array();
            Ok(ndarray::ArrayD::from_shape_vec(IxDyn(&dims), field.to_vec())
                    .expect("regions were initialized properly"))
        }
    };
}

impl DftRegionReadback {
    pub fn new(backend: &GpuBackend, state: &DftRegionStates) -> TaserResult<Self> {
        Ok(Self {
            refs: state.region_descs.clone(),
            h_dfts: vec![ComplexVec3::ZERO; state.h_dfts.len()],
            en_dfts: vec![ComplexVec3::ZERO; state.en_dfts.len()],
            sn_dfts: vec![ComplexVec3::ZERO; state.sn_dfts.len()],
            h_dfts_read: GpuReadback::new(backend, state.h_dfts.len())?,
            en_dfts_read: GpuReadback::new(backend, state.en_dfts.len())?,
            sn_dfts_read: GpuReadback::new(backend, state.sn_dfts.len())?,
        })
    }

    readback_get!(get_h, h_dfts);
    readback_get!(get_en, en_dfts);
    readback_get!(get_sn, sn_dfts);

    pub fn get_region_desc(&self, region: &Arc<DftRegion>) -> TaserResult<&DftRegionDescription> {
        self.refs.iter()
            .find(|r| Arc::ptr_eq(&r.ptr, region))
            .ok_or(DftError::CannotFindRegion(region.clone()).into())
    }

    pub fn sync(&mut self, backend: &GpuBackend, state: &DftRegionStates) -> TaserResult<()> {
        self.read_back(backend)?;
        self.request_copy(backend, state)
    }

    pub fn read_back(&mut self, backend: &GpuBackend) -> TaserResult<()> {
        backend.synchronize()?;
        self.try_read_back(backend);
        Ok(())
    }

    pub fn try_read_back(&mut self, backend: &GpuBackend) -> bool {
        self.h_dfts_read.try_take(backend, &mut self.h_dfts) &&
            self.en_dfts_read.try_take(backend, &mut self.en_dfts) &&
            self.sn_dfts_read.try_take(backend, &mut self.sn_dfts)
    }

    pub fn request_copy(&mut self, backend: &GpuBackend, state: &DftRegionStates) -> TaserResult<()> {
        if self.h_dfts_read.is_idle() {
            self.h_dfts_read.request_copy(backend, &state.h_dfts, 0)?;
        }
        if self.en_dfts_read.is_idle() {
            self.en_dfts_read.request_copy(backend, &state.en_dfts, 0)?;
        }
        if self.sn_dfts_read.is_idle() {
            self.sn_dfts_read.request_copy(backend, &state.sn_dfts, 0)?;
        }
        Ok(())
    }
}

pub struct DftRegionStates {
    region_descs: Vec<DftRegionDescription>,
    gpu_dft_regions: Vec<GpuDftRegion>,
    params: GpuBuffer<GpuDftRegionsParams>,
    dft_regions: GpuBuffer<GpuDftRegion>,
    h_dfts: GpuBuffer<ComplexVec3>,
    en_dfts: GpuBuffer<ComplexVec3>,
    sn_dfts: GpuBuffer<ComplexVec3>,
    update_workgroups: [u32; 3],
}

impl DftRegionStates {
    pub fn new(backend: &GpuBackend, sim: &FdtdLossyState, region_objs: &[Arc<DftRegion>]) -> TaserResult<Self> {
        if region_objs.is_empty() { return Err(DftError::NoRegions.into()) }
        let sim_offset = Vect::from_vec3(sim.grid_sizing.sim_offset);
        let grid_extents = sim.cell_size * sim.n_cells.as_vect();

        let region_descs = region_objs
            .iter()
            .scan(0, |total_dfts, region| {
                let min = region.bb_min.min(region.bb_max) + sim_offset;
                let max = region.bb_min.max(region.bb_max) + sim_offset;
                if min.cmplt(Vect::ZERO).any() || max.cmpgt(grid_extents).any() {
                    return Some(Err(DftError::OutOfBoundsRegion {
                        region: region.clone(),
                        grid_extents: Aabb::new(Vec3::ZERO, grid_extents.to_3d(Vec3::ZERO))
                            .translated(sim_offset.to_3d(Vec3::ZERO)),
                    }));
                }
                let cell_idx_min = (min / sim.cell_size).round().as_grid_index().cell_idx_to_3d();
                let n_cells = ((max - min) / sim.cell_size).round().as_grid_index();
                let n_cells3 = n_cells.n_cells_to_3d();
                let cell_count = n_cells.element_product();
                if n_cells.min_element() == 0 {
                    return Some(Err(DftError::InvalidRegionSize(region.clone())))
                }

                let dft_start_buf_offset = *total_dfts;
                *total_dfts += cell_count as usize * region.frequencies.len();

                let flat_idx_incrs = {
                    let mut incrs = UVec3::ZERO;
                    for (spatial_axis, axis) in SpatialAxis::ALL_SPATIAL.into_iter()
                        .zip(SpatialAxis::ALL_AXES)
                    {
                        let mut grid_incr = GridIndex::ZERO;
                        grid_incr[spatial_axis] = 1;
                        incrs[axis] = grid_incr.to_flat_idx(n_cells);
                    }
                    incrs
                };
                Some(Ok(DftRegionDescription {
                    ptr: region.clone(),
                    cell_idx_min,
                    flat_idx_incrs,
                    dft_start_buf_offset,
                    n_cells3,
                    cell_count,
                }))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let dft_regions = region_descs
            .iter()
            .map(|desc| {
                let frequencies = &desc.ptr.frequencies;
                if frequencies.is_empty() { return Err(DftError::NoFrequencies.into()) }
                let uninit_region_gpu = GpuDftRegion { fields: desc.ptr.fields, ..desc.gpu_desc_uninit() };
                let gpu_regions = frequencies
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(i, freq)| {
                        let dfts_start_idx = desc.freq_buf_offset(i) as u32;
                        GpuDftRegion {
                            freq,
                            dfts_start_idx,
                            kernel: Complex32::from_polar(1., -core::f32::consts::TAU * freq * sim.dt),
                            h_finalize_k: Complex32::from_polar(1., -core::f32::consts::TAU * freq * 0.5 * sim.dt),
                            ..uninit_region_gpu
                        }
                    })
                    .collect::<Vec<_>>();
                Ok(gpu_regions)
            })
            .flatten_ok()
            .collect::<TaserResult<Vec<_>>>()?;
        let dft_count = region_descs
            .last()
            .map(|d| d.freq_buf_offset(d.ptr.frequencies.len()))
            .unwrap();
        let dfts_zeroed = vec![ComplexVec3::ZERO; dft_count];

        let params = GpuDftRegionsParams::new(sim.n_cells.n_cells_to_3d(), sim.dt, &dft_regions);
        let dft_regions_buf = dft_regions.create_gpu_buffer(backend)?;
        Ok(Self {
            region_descs,
            gpu_dft_regions: dft_regions,
            params: params.create_gpu_uniform(backend)?,
            dft_regions: dft_regions_buf,
            h_dfts: dfts_zeroed.create_gpu_buffer(backend)?,
            en_dfts: dfts_zeroed.create_gpu_buffer(backend)?,
            sn_dfts: dfts_zeroed.create_gpu_buffer(backend)?,
            update_workgroups: dft_regions_update_workgroups(&params),
        })
    }

    pub fn h_dfts(&self) -> &GpuBuffer<ComplexVec3> { &self.h_dfts }

    pub fn en_dfts(&self) -> &GpuBuffer<ComplexVec3> { &self.en_dfts }

    pub fn regions(&self) -> &GpuBuffer<GpuDftRegion> { &self.dft_regions }

    pub(crate) fn gpu_region_descs(&self) -> &Vec<GpuDftRegion> {
        &self.gpu_dft_regions
    }

    pub fn get_region_desc(&self, region: &Arc<DftRegion>) -> Option<&DftRegionDescription> {
        self.region_descs.iter()
            .find(|v| Arc::ptr_eq(&v.ptr, region))
    }
}

#[derive(Clone)]
pub struct DftRegionDescription {
    pub ptr: Arc<DftRegion>,
    pub cell_idx_min: UVec3,
    pub flat_idx_incrs: UVec3,
    pub n_cells3: UVec3,
    pub cell_count: u32,
    /// Index of the start of this region's DFTs
    pub dft_start_buf_offset: usize,
}

impl DftRegionDescription {
    pub fn freq_buf_offset(&self, freq_idx: usize) -> usize {
        self.dft_start_buf_offset + self.cell_count as usize * freq_idx
    }

    pub fn gpu_desc_uninit(&self) -> GpuDftRegion {
        GpuDftRegion {
            cell_idx_min: self.cell_idx_min,
            dfts_start_idx: u32::MAX,
            n_cells3: self.n_cells3,
            freq: Real::NAN,
            flat_idx_incrs: self.flat_idx_incrs,
            kernel: Complex32::new(Real::NAN, Real::NAN),
            cell_count: self.cell_count,
            h_finalize_k: Complex32::new(Real::NAN, Real::NAN),
            fields: DftFields::empty(),
            _padding0: Default::default(),
        }
    }
}

/// Helper function for creating a list of frequencies for a DFT.
///
/// # Arguments
/// - `range` - range of frequencies to resolve.
/// - `resolution` - splits `range` by this resolution such that the output [`Vec`] will have
///   `resolution + 1` elements.
pub fn frequencies_from_range(range: core::ops::RangeInclusive<Real>, resolution: usize) -> Vec<Real> {
    let mut frequencies = vec![0.; resolution + 1];
    let df = (range.end() - range.start()) / resolution as Real;
    for (i, f) in frequencies.iter_mut().enumerate() {
        *f = range.start() + df * i as Real;
    }
    frequencies
}

#[derive(thiserror::Error, Debug)]
pub enum DftError {
    #[error("Expected at least one frequency in DFT but none were provided")]
    NoFrequencies,
    #[error("Expected at least one DFT")]
    NoDfts,
    #[error("Couldn't find DFT function")]
    CannotFindFunction,
    #[error("At least one DFT region must be provided")]
    NoRegions,
    #[error("Couldn't find DFT region {0:?}")]
    CannotFindRegion(Arc<DftRegion>),
    #[error("Part of DFT region is outside simulation grid. Region: {region:?}, Grid extents: {grid_extents:?}")]
    OutOfBoundsRegion {
        region: Arc<DftRegion>,
        grid_extents: Aabb,
    },
    #[error("DFT region has zero thickness: {0:?}")]
    InvalidRegionSize(Arc<DftRegion>),
}