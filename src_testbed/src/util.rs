use kiss3d::prelude::Color;

pub fn lerp_colors(t: f32, color1: Color, color2: Color) -> Color {
    Color::new(
        color1.r + (color2.r - color1.r) * t,
        color1.g + (color2.g - color1.g) * t,
        color1.b + (color2.b - color1.b) * t,
        color1.a + (color2.a - color1.a) * t,
    )
}