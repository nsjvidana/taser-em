use crate::FdtdTestbedViewer;
use egui::{Color32, Ui, WidgetText};
use egui_plot::{AxisHints, Legend, Line, Plot, PlotPoint, PlotPoints, PlotUi};
use std::sync::Arc;
use khal::backend::GpuBackend;
use taser_em::dft::*;
use taser_em::prelude::{Complex32, Real, TaserResult};

pub struct PlotWindow {
    pub name: String,
    pub open: bool,
    pub x_axis_title: Option<WidgetText>,
    pub y_axis_title: Option<WidgetText>,
}

impl PlotWindow {
    pub fn new(
        name: &str,
        x_axis_title: Option<&str>,
        y_axis_title: Option<&str>,
    ) -> Self {
        Self {
            name: name.to_string(),
            open: true,
            x_axis_title: x_axis_title.map(|v| v.into()),
            y_axis_title: y_axis_title.map(|v| v.into()),
        }
    }

    /// Show plot window with `lines_f` generating the lines to plot.
    ///
    /// `additonal_ui` adds anything to the plot window ui.
    pub fn show(
        &mut self,
        testbed: &mut FdtdTestbedViewer,
        lines: Vec<Line>,
    ) -> TaserResult<()> {
        testbed.window.draw_ui(|ctx| {
            egui::Window::new(self.name.as_str())
                .open(&mut self.open)
                .show(ctx, |ui| {
                    let mut p = Plot::new(self.name.as_str())
                        .legend(Legend::default());
                    if let Some(x_title) = self.x_axis_title.as_ref() {
                        p = p.custom_x_axes(vec![
                            AxisHints::new_x().label(x_title.clone())
                        ]);
                    }
                    if let Some(y_title) = self.y_axis_title.as_ref() {
                        p = p.custom_y_axes(vec![
                            AxisHints::new_y().label(y_title.clone())
                        ]);
                    }
                    p.show(ui, |plot_ui| -> TaserResult<()> {
                        for line in lines {
                            plot_ui.line(line)
                        }
                        Ok(())
                    });
                });
        });
        Ok(())
    }
}

pub trait PlotLine {
    /// An extra UI to go with the plot line. Can allow the user to edit certain settings
    /// of the plot line.
    fn aux_ui(&mut self, _ui: &mut egui::Ui) {}

    /// Create a [`egui_plot::Line`] from `self` to plot.
    fn create_line(&self) -> Line<'_>;

    /// Draws the plot line in a [`PlotUi`].
    fn draw<'a>(&'a self, plot_ui: &mut PlotUi<'a>) {
        plot_ui.line(self.create_line());
    }
}

pub struct DftPlotLine<Func: ToDft> {
    pub name: String,
    pub mode: DftPlotMode,
    pub color: Option<egui::Color32>,
    func: Arc<Func>,
    frequencies: Vec<Real>,
    /// Values of the DFT read back from the simulation
    pub dft: Vec<Complex32>,
    /// The [`PlotPoint`]s of the `egui` plot.
    pub pts: Vec<PlotPoint>,
}

impl<Func: ToDft> DftPlotLine<Func> {
    pub fn new(
        name: &str,
        func: &Arc<Func>,
        readback: &DftReadback<Func>,
        frequency_scale: Option<Real>
    ) -> TaserResult<Self> {
        let mut frequencies = readback.get_frequencies(func).ok_or(DftError::CannotFindFunction)?;
        if let Some(prefix) = frequency_scale {
            frequencies.iter_mut().for_each(|f| *f *= prefix)
        }
        let n_freqs = frequencies.len();

        Ok(Self {
            name: name.to_string(),
            mode: DftPlotMode::Magnitude,
            color: None,
            func: func.clone(),
            frequencies,
            dft: vec![Complex32::ZERO; n_freqs],
            pts: vec![PlotPoint::new(0., 0.); n_freqs],
        })
    }

    pub fn with_color(mut self, color: impl Into<Color32>) -> Self {
        self.color = Some(color.into());
        self
    }

    pub fn with_mode(mut self, mode: DftPlotMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn frequencies(&self) -> &Vec<Real> { &self.frequencies }

    /// Read back and update DFT plot values.
    ///
    /// Also calls [`DftReadback::request_copy`].
    pub fn update_dft_and_points(
        &mut self,
        backend: &GpuBackend,
        dft_state: &DftStates<Func>,
        readback: &mut DftReadback<Func>,
    ) -> TaserResult<()> {
        self.update_dft(backend, dft_state, readback)?;
        self.update_points_no_read();
        Ok(())
    }

    /// Update the DFT stored in this plot line with readback.
    pub fn update_dft(
        &mut self,
        backend: &GpuBackend,
        dft_state: &DftStates<Func>,
        readback: &mut DftReadback<Func>,
    ) -> TaserResult<()> {
        readback.read_back(backend)?;
        readback.request_copy(backend, dft_state)?;
        self.dft = readback.get_dft(&self.func).ok_or(DftError::CannotFindFunction)?;
        Ok(())
    }

    /// Similar to [`DftPlotLine::update_dft_and_points`], but doesn't do readback.
    pub fn update_points_no_read(&mut self) {
        self.pts = self.dft.iter()
            .copied()
            .zip(self.frequencies.iter().copied())
            .map(|(c, f)| PlotPoint::new(f, self.mode.get_dft_val(c)))
            .collect::<Vec<_>>();
    }
}

impl<Func: ToDft> PlotLine for DftPlotLine<Func> {
    fn aux_ui(&mut self, ui: &mut Ui) {
        egui::CollapsingHeader::new(format!("{} (DFT)", self.name))
            .show(ui, |ui| {
                egui::ComboBox::from_label("Mode")
                    .selected_text(format!("{:?}", self.mode))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.mode, DftPlotMode::Magnitude, "Magnitude");
                        ui.selectable_value(&mut self.mode, DftPlotMode::Phase, "Phase");
                    });
            });
    }

    fn create_line(&self) -> Line<'_> {
        let mut l = Line::new(self.name.clone(), PlotPoints::Borrowed(&self.pts));
        if let Some(color) = self.color { l = l.color(color); }
        l
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum DftPlotMode {
    Magnitude,
    Phase
}

impl DftPlotMode {
    pub fn get_dft_val(&self, complex: Complex32) -> Real {
        match self {
            DftPlotMode::Magnitude => complex.norm(),
            DftPlotMode::Phase => Real::atan2(complex.re, complex.im),
        }
    }
}