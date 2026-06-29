use std::env;
use std::path::PathBuf;
use khal_builder::KhalBuilder;

fn main() {
    taser_em_build::dimensions_check();

    let output_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR environment variable not set"))
        .join("shaders-spirv");

    let mut builder = KhalBuilder::from_dependency("taser-em-shaders", true);

    // Enable dim features in sahder compilation
    for dim in 1..=3 {
        if env::var(format!("CARGO_FEATURE_DIM{dim}")).is_ok() {
            builder = builder.feature(format!("dim{dim}"));
        }
    }

    builder.build(&output_dir);
}