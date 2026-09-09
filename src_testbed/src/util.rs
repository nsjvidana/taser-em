use glamx::*;
use kiss3d::prelude::{Color, Window};

pub fn lerp_colors(t: f32, color1: Color, color2: Color) -> Color {
    Color::new(
        color1.r + (color2.r - color1.r) * t,
        color1.g + (color2.g - color1.g) * t,
        color1.b + (color2.b - color1.b) * t,
        color1.a + (color2.a - color1.a) * t,
    )
}

pub fn draw_bb(window: &mut Window, start: Vec3, end: Vec3, color: Color) {
    let verts = [
        start, start.with_x(end.x), end.with_z(start.z), start.with_y(end.y),
        start.with_z(end.z), end.with_y(start.y), end, end.with_x(start.x),
    ];
    window.draw_line(verts[0], verts[1], color, 2., false);
    window.draw_line(verts[1], verts[2], color, 2., false);
    window.draw_line(verts[2], verts[3], color, 2., false);
    window.draw_line(verts[3], verts[0], color, 2., false);
    window.draw_line(verts[4], verts[5], color, 2., false);
    window.draw_line(verts[5], verts[6], color, 2., false);
    window.draw_line(verts[6], verts[7], color, 2., false);
    window.draw_line(verts[7], verts[4], color, 2., false);
    window.draw_line(verts[0], verts[4], color, 2., false);
    window.draw_line(verts[1], verts[5], color, 2., false);
    window.draw_line(verts[2], verts[6], color, 2., false);
    window.draw_line(verts[3], verts[7], color, 2., false);
}