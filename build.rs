use std::env;
use std::path::PathBuf;
use khal_builder::KhalBuilder;

fn main() {
    let dim_features = vec![
        "CARGO_FEATURE_DIM1",
        "CARGO_FEATURE_DIM2",
        "CARGO_FEATURE_DIM3",
    ];
    let enabled_dim_count = dim_features
        .iter()
        .filter(|&&f| env::var(f).is_ok())
        .count();
    if enabled_dim_count > 1 || enabled_dim_count == 0 {
        panic!("Only one dimension must be enabled.");
    }

    let output_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR environment variable not set"))
        .join("shaders-spirv");

    KhalBuilder::from_dependency("taser-em-shaders", true)
        .build(&output_dir);
}