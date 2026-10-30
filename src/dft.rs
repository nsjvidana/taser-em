use std::ops::Range;
use crate::gpu_util::*;
use crate::prelude::*;
use khal::backend::{Backend, Buffer, DispatchGrid, Encoder, GpuBackend, GpuBuffer, GpuPass, GpuReadback};
use std::sync::Arc;
use itertools::Itertools;
use ndarray::IxDyn;
use parry3d::bounding_volume::Aabb;
use taser_em_shaders::dft::*;
use taser_em_shaders::fdtd::GridParameters;
use taser_em_shaders::math::*;

/// Describes the DFT of some function.
///
/// Includes all frequencies to be resolved in DFT
#[derive(Clone)]
pub struct Dft<Func: ToDft + ?Sized> {
    /// The frequencies of the DFT in Hz
    frequencies: Vec<Real>,
    function: Arc<Func>,
}

impl<Func: ToDft> Dft<Func> {
    pub fn new(frequencies: Vec<Real>, function: Arc<Func>) -> TaserResult<Self> {
        if frequencies.is_empty() { return Err(DftError::NoFrequencies.into()) };
        Ok(Self {
            frequencies,
            function,
        })
    }

    pub fn get_frequencies(&self) -> &Vec<Real> { &self.frequencies }

    pub fn get_function(&self) -> &Arc<Func> { &self.function }
}

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

/// [`FdtdLossyHook`] applied to [`FdtdLossyPipeline`] for running DFTs of a function in the simulation.
///
/// Use [`DftHook::states`] to fetch the [`DftStates`] of this DFT.
pub struct DftHook<Func: ToDft> {
    pipeline: DftPipeline,
    states: DftStates<Func>,
}

impl<Func: ToDft> DftHook<Func> {
    pub fn new(backend: &GpuBackend, dfts: impl Into<Vec<Dft<Func>>>, sim_state: &FdtdLossyState) -> TaserResult<Self> {
        let pipeline = DftPipeline::new(backend)?;
        let states = DftStates::new(backend, dfts, sim_state, &pipeline)?;
        Ok(Self { pipeline, states })
    }

    pub fn states(&self) -> &DftStates<Func> { &self.states }
}

impl<Func: ToDft> FdtdLossyHook for DftHook<Func> {
    fn initialize(&mut self, pass: &mut GpuPass, state: &mut FdtdLossyState) -> TaserResult<()> {
        self.pipeline.initialize_states(pass, &state.grid, &mut self.states)
    }

    fn dispatch_step(&mut self, pass: &mut GpuPass, state: &mut FdtdLossyState) -> TaserResult<()> {
        self.pipeline.dispatch_step(pass, state, &mut self.states)
    }
}

pub struct DftPipeline {
    init_kernels: GpuComputeDftKernels,
    dft_shader: GpuDftShader
}

impl DftPipeline {
    pub fn new(backend: &GpuBackend) -> TaserResult<Self> {
        Ok(Self {
            init_kernels: GpuComputeDftKernels::from_dir(backend, &crate::SPIRV_DIR)?,
            dft_shader: GpuDftShader::from_dir(backend, &crate::SPIRV_DIR)?,
        })
    }

    pub fn initialize_states<Func: ToDft>(
        &self,
        pass: &mut GpuPass,
        grid: &GpuBuffer<GridParameters>,
        dft_states: &mut DftStates<Func>
    ) -> TaserResult<()> {
        self.init_kernels.call(
            pass,
            DispatchGrid::Grid(dft_states.workgroups),
            grid,
            &dft_states.func_dfts,
            &dft_states.dft_frequencies,
            &mut dft_states.dft_kernels
        )?;
        Ok(())
    }

    /// Dispatch DFT step in time.
    ///
    /// # Arguments
    /// - `time_step` - buffer of current time step index.
    /// - `functions` - buffer of the instantaneous values (occurring at `time_step` time step) of
    ///   every function whose DFT is being computed (parallel w/ `func_dfts`).
    /// - `dft_states` - states of function DFTs.
    pub fn dispatch_step<Func: ToDft>(
        &self,
        pass: &mut GpuPass,
        sim_state: &FdtdLossyState,
        dft_states: &mut DftStates<Func>
    ) -> TaserResult<()> {
        self.dft_shader.call(
            pass,
            DispatchGrid::Grid(dft_states.workgroups),
            &sim_state.t_idx,
            &dft_states.function_value_positions,
            Func::get_value_buffer(sim_state)?,
            &dft_states.func_dfts,
            &dft_states.dft_kernels,
            &mut dft_states.dft_outputs
        )?;
        Ok(())
    }
}

/// The states of multiple DFTs that will be evaluated in one shader dispatch
pub struct DftStates<Func: ToDft> {
    /// The DFTs stored on CPU side.
    pub dfts: Vec<Dft<Func>>,

    // Buffers / GPU data
    pub function_value_positions: GpuBuffer<Index>,
    pub func_dfts: GpuBuffer<GpuFunctionDft>,
    pub dft_frequencies: GpuBuffer<Real>,
    pub dft_kernels: GpuBuffer<Complex32>,
    pub dft_outputs: GpuBuffer<Complex32>,
    pub workgroups: [u32; 3]
}

impl<Func: ToDft> DftStates<Func> {
    /// Creates new DFT buffers that are properly initialized
    pub fn new(
        backend: &GpuBackend,
        dfts: impl Into<Vec<Dft<Func>>>,
        sim_state: &FdtdLossyState,
        dft_pipeline: &DftPipeline,
    ) -> TaserResult<Self> {
        let mut selff = Self::new_zeroed(backend, dfts)?;
        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("__dft_init", None);
        dft_pipeline.initialize_states(
            &mut pass,
            &sim_state.grid,
            &mut selff
        )?;
        drop(pass);
        backend.submit(encoder)?;
        Ok(selff)
    }

    /// Creates new zeroed-out DFT buffers.
    ///
    /// Use [`DftPipeline::initialize_states`] to populate buffers with non-zero kernels, or use
    /// [`DftStates::new`] instead to submit an initialize command right away.
    pub fn new_zeroed(backend: &GpuBackend, dfts: impl Into<Vec<Dft<Func>>>) -> TaserResult<Self> {
        let dfts = dfts.into();
        if dfts.is_empty() { return Err(DftError::NoDfts.into()) };

        let function_value_positions = dfts.iter()
            .map(|dft| dft.function.get_value_position() as Index)
            .collect::<Vec<_>>()
            .create_gpu_buffer(backend)?;

        let mut kernels_start = 0;
        let func_dfts = dfts.iter()
            .map(|dft| {
                let kernels_end = kernels_start + dft.frequencies.len() as u32 - 1;
                let gpu_ver = GpuFunctionDft {
                    kernels_start,
                    kernels_end,
                };
                kernels_start = kernels_end + 1;
                gpu_ver
            })
            .collect::<Vec<_>>()
            .create_gpu_buffer(backend)?;
        let dft_frequencies = dfts.iter()
            .flat_map(|d| &d.frequencies)
            .copied()
            .collect::<Vec<_>>()
            .create_gpu_buffer(backend)?;
        let dft_kernels_zeroed = vec![Complex32::ZERO; dft_frequencies.len()]
            .create_gpu_buffer(backend)?;
        let dft_outputs_zeroed = vec![Complex32::ZERO; dft_frequencies.len()]
            .create_gpu_buffer(backend)?;

        let max_n_kernels = dfts.iter()
            .map(|d| d.frequencies.len())
            .max()
            .expect("Dft type can't be initialized with an empty frequencies vec") as u32;
        let n_functions = func_dfts.len() as u32;

        Ok(Self {
            dfts,
            function_value_positions,
            func_dfts,
            dft_frequencies,
            dft_kernels: dft_kernels_zeroed,
            dft_outputs: dft_outputs_zeroed,
            workgroups: dft_workgroups(max_n_kernels, n_functions),
        })
    }
}

pub trait ToDft {
    /// Convert this [`ToDft`] into a [`Dft`] with the specified `frequencies`.
    fn to_dft(self, frequencies: Vec<Real>) -> TaserResult<(Dft<Self>, Arc<Self>)>;

    /// Gets the index of this function's current value in the buffer obtained by [`ToDft::get_value_position`]
    fn get_value_position(&self) -> usize;

    /// Gets the buffer that this function's current value is stored in
    fn get_value_buffer(state: &FdtdLossyState) -> TaserResult<&GpuBuffer<Real>>;
}

pub struct DftReadback<Func: ToDft> {
    func_ranges: Vec<(Arc<Func>, Range<usize>)>,
    frequencies: Vec<Real>,
    dft_outputs: Vec<Complex32>,
    dft_outputs_read: GpuReadback<Complex32>
}

impl<Func: ToDft> DftReadback<Func> {
    pub async fn new(backend: &GpuBackend, hooks: &DftHook<Func>) -> TaserResult<Self> {
        Self::from_states(backend, &hooks.states).await
    }

    pub async fn from_states(backend: &GpuBackend, dft_states: &DftStates<Func>) -> TaserResult<Self> {
        let mut gpu_dfts = vec![GpuFunctionDft::default(); dft_states.func_dfts.len()];
        backend.slow_read_buffer(&dft_states.func_dfts, &mut gpu_dfts).await?;
        let func_ranges = gpu_dfts.into_iter()
            .zip(dft_states.dfts.iter())
            .map(|(gpu_dft, dft)| {
                (dft.function.clone(), (gpu_dft.kernels_start as usize)..(gpu_dft.kernels_end as usize))
            })
            .collect::<Vec<_>>();

        let mut frequencies = vec![0.; dft_states.dft_frequencies.len()];
        backend.slow_read_buffer(&dft_states.dft_frequencies, &mut frequencies).await?;

        Ok(Self {
            func_ranges,
            frequencies,
            dft_outputs: vec![Complex32::ZERO; dft_states.dft_outputs.len()],
            dft_outputs_read: GpuReadback::new(backend, dft_states.dft_outputs.len())?,
        })
    }

    pub fn request_copy(&mut self, backend: &GpuBackend, state: &DftStates<Func>) -> TaserResult<()> {
        if self.dft_outputs_read.is_idle() {
        self.dft_outputs_read.request_copy(backend, &state.dft_outputs, 0)?;
        }
        Ok(())
    }

    pub fn read_back(&mut self, backend: &GpuBackend) -> TaserResult<()> {
        backend.synchronize()?;
        self.try_read_back(backend);
        Ok(())
    }

    pub fn try_read_back(&mut self, backend: &GpuBackend) -> bool {
        self.dft_outputs_read.try_take(backend, &mut self.dft_outputs)
    }

    pub fn get_frequencies(&self, dft_function: &Arc<Func>) -> Option<Vec<Real>> {
        self.func_ranges.iter()
            .find_map(|(m, r)|
                Arc::ptr_eq(m, dft_function)
                    .then(||
                        self.frequencies[r.clone()].to_vec()
                    )
            )
    }

    pub fn get_dft(&self, dft_function: &Arc<Func>) -> Option<Vec<Complex32>> {
        self.func_ranges.iter()
            .find_map(|(m, r)|
                Arc::ptr_eq(m, dft_function)
                    .then(||
                        self.dft_outputs[r.clone()].to_vec()
                    )
            )
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DftDataPoint {
    pub frequency: Real,
    pub complex: Complex32,
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