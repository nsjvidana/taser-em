pub mod prelude;
pub mod gpu_util;

use khal::re_exports::include_dir::{include_dir, Dir};

pub static SPIRV_DIR: Dir<'static> = include_dir!("$OUT_DIR/shaders-spirv");