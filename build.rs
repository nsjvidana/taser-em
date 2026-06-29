use std::env;
use std::path::PathBuf;
use khal_builder::KhalBuilder;

fn main() {
    taser_em_build::dimensions_check();

    let output_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR environment variable not set"))
        .join("shaders-spirv");

    KhalBuilder::from_dependency("taser-em-shaders", true)
        .build(&output_dir);
}