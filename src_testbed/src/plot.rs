use crate::FdtdTestbedViewer;
use egui::{Color32, WidgetText};
use egui_plot::{AxisHints, Legend, Line, Plot, PlotPoint, PlotPoints, PlotUi};
use glamx::DVec2;
use taser_em::dft::*;
use taser_em::prelude::{Complex32, Real, TaserResult};

pub struct PlotWindow {
    pub name: String,
    pub open: bool,
    pub x_axis_title: Option<WidgetText>,
    pub y_axis_title: Option<WidgetText>,
    pub x_decimals: usize,
    pub y_decimals: usize,
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
            x_decimals: 3,
            y_decimals: 3,
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
                        .legend(Legend::default())
                        .label_formatter(|_, pt| {
                            format!("{:.2$}, {:.3$}", pt.x, pt.y, self.x_decimals, self.y_decimals)
                        });
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

/// Helper struct for plotting complex numbers
pub struct ComplexPlot {
    pub name: String,
    pub mode: ComplexPlotMode,
    /// Plot color. Leave it [`None`] for automatic color.
    pub color: Option<Color32>,
    pts: Vec<PlotPoint>,
    /// Numbers multiplied to `x` and `y` coordinates of all [`PlotPoint`]s. 
    pub xy_scale: DVec2,
}

impl ComplexPlot {
    pub fn new(name: &str, x_axis: &[Real], x_scale: Option<Real>, y_scale: Option<Real>) -> TaserResult<Self> {
        if x_axis.is_empty() { return Err(DftError::NoFrequencies.into()) }
        let mut selff = Self {
            name: name.to_string(),
            mode: ComplexPlotMode::Magnitude,
            color: None,
            pts: vec![PlotPoint::new(0., 0.); x_axis.len()],
            xy_scale: DVec2::new(x_scale.unwrap_or(1.) as f64, y_scale.unwrap_or(1.) as f64),
        };
        selff.resize_x(x_axis)?;
        Ok(selff)
    }

    /// Set the values of this plot's x-axis, adding or remove plot points to have
    /// `x_new.len()` plot points.
    ///
    /// Currently-stored y values are preserved, except for plot points that are removed.
    pub fn resize_x(&mut self, x_new: &[Real]) -> TaserResult<()> {
        self.pts.resize(x_new.len(), PlotPoint::new(0., 0.));
        self.pts
            .iter_mut()
            .zip(x_new.iter())
            .for_each(|(pt, x_new)| pt.x = (*x_new as f64) * self.xy_scale.x);
        Ok(())
    }

    /// Set the complex `y` coordinate of each plot point. Use this updating DFT plots on-the-fly.
    ///
    /// This does **not** add or remove plot points, so only the first `y_new.len()` plot points
    /// will have their `y` coordinate changed. If `y_new` is too large, its extra elements will be ignored.
    pub fn set_y(&mut self, y_new: &[Complex32]) {
        self.pts
            .iter_mut()
            .zip(y_new.iter().copied())
            .for_each(|(pt, y_new)| {
                pt.y = self.mode.get_val(y_new) as f64 * self.xy_scale.y;
            });
    }

    pub fn plot_points(&self) -> &[PlotPoint] { self.pts.as_slice() }
}

impl PlotLine for ComplexPlot {
    fn create_line(&self) -> Line<'_> {
        Line::new(&self.name, PlotPoints::Borrowed(&self.pts))
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum ComplexPlotMode {
    Magnitude,
    Phase,
    Re,
    Im,
}

impl ComplexPlotMode {
    pub fn get_val(&self, complex: Complex32) -> Real {
        match self {
            ComplexPlotMode::Magnitude => complex.norm(),
            ComplexPlotMode::Phase => complex.arg(),
            ComplexPlotMode::Re => complex.re,
            ComplexPlotMode::Im => complex.im,
        }
    }
}