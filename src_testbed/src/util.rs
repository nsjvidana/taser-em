use glamx::{Quat, Vec3};
use kiss3d::prelude::{Color, Window};

/// Draw an arrow in 3D space
pub fn draw_arrow(window: &mut Window, start: Vec3, end: Vec3, color: Color) {
    window.draw_line(start, end, color, 2., false);

    let tip_ends_local = [
        Vec3::X,
        Vec3::Y,
        Vec3::NEG_X,
        Vec3::NEG_Y,
    ];
    let tip_local = Vec3::Z;

    let arrow_dir = end - start;
    let tip_len = arrow_dir.length() / 10.;
    let rotation = Quat::from_rotation_arc(tip_local, arrow_dir.normalize());
    let tip_ends_world = tip_ends_local.map(|pos| (rotation * pos * tip_len) + end);
    for pos in tip_ends_world {
        window.draw_line(end, pos, color, 2., true);
    }
}