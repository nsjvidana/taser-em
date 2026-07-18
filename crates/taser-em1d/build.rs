use std::path::PathBuf;
use khal_builder::KhalBuilder;

fn main() {
    let output_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR not set by cargo"))
        .join("shaders-spirv");

    KhalBuilder::from_dependency("taser-em-shaders1d", true)
        .feature("dim1")
        .build(&output_dir);
}