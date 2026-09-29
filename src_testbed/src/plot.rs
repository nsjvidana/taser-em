use crate::FdtdTestbedViewer;
use egui::{Ui, WidgetText};
use egui_plot::{AxisHints, Legend, Line, Plot, PlotPoint, PlotPoints, PlotUi};
use std::sync::Arc;
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

    /// Update every [`PlotLine`] in `plot_lines` with `readback` and
    /// Show `self` with all `plot_lines` drawn.
    pub fn show<L: PlotLine>(
        &mut self,
        testbed: &mut FdtdTestbedViewer,
        plot_lines: &mut Vec<L>,
        readback: &L::Readback,
    ) -> TaserResult<()> {
        plot_lines.update_points(readback)?;
        testbed.window.draw_ui(|ctx| {
            self.show_no_update(ctx, plot_lines);
        });
        Ok(())
    }

    pub fn show_no_update<L: PlotLine>(
        &mut self, 
        ctx: &egui::Context, 
        lines: &mut Vec<L>,
    ) {
        egui::Window::new(self.name.as_str())
            .open(&mut self.open)
            .show(ctx, |ui| {
                lines.aux_ui(ui);

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
                            line.draw(plot_ui);
                        }
                        Ok(())
                    });
            });
    }
}

pub trait PlotLine {
    type Readback;

    /// An extra UI to go with the plot line. Can allow the user to edit certain settings
    /// of the plot line.
    fn aux_ui(&mut self, _ui: &mut egui::Ui) {}

    /// A function for updating the plot line's data points.
    fn update_points(&mut self, readback: &Self::Readback) -> TaserResult<()>;

    /// Draws the plot line in a [`PlotUi`].
    fn draw<'a>(&'a self, plot_ui: &mut PlotUi<'a>);
}

impl<L: PlotLine> PlotLine for Vec<L> {
    type Readback = L::Readback;

    fn aux_ui(&mut self, ui: &mut Ui) {
        for l in self.iter_mut() {
            l.aux_ui(ui);
        }
    }

    fn update_points(&mut self, readback: &Self::Readback) -> TaserResult<()> {
        for l in self.iter_mut() {
            l.update_points(readback)?;
        }
        Ok(())
    }

    fn draw<'a>(&'a self, plot_ui: &mut PlotUi<'a>) {
        for l in self.iter() {
            l.draw(plot_ui);
        }
    }
}

pub struct DftPlotLine<Func: ToDft> {
    pub name: String,
    pub mode: DftPlotMode,
    func: Arc<Func>,
    frequencies: Vec<Real>,
    pts: Vec<PlotPoint>,
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
            func: func.clone(),
            frequencies,
            pts: vec![PlotPoint::new(0., 0.); n_freqs],
        })
    }

    pub fn with_mode(mut self, mode: DftPlotMode) -> Self {
        self.mode = mode;
        self
    }
}

impl<Func: ToDft> PlotLine for DftPlotLine<Func> {
    type Readback = DftReadback<Func>;

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

    fn update_points(&mut self, readback: &Self::Readback) -> TaserResult<()> {
        self.pts = readback.get_dft(&self.func)
            .ok_or(DftError::CannotFindFunction)?
            .into_iter()
            .zip(self.frequencies.iter())
            .map(|(c, f)| PlotPoint::new(*f, self.mode.get_dft_val(c)))
            .collect::<Vec<_>>();
        Ok(())
    }

    fn draw<'a>(&'a self, plot_ui: &mut PlotUi<'a>) {
        plot_ui.line(Line::new(self.name.clone(), PlotPoints::Borrowed(&self.pts)));
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