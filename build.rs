use std::env;
use std::path::PathBuf;
use khal_builder::KhalBuilder;

fn main() {
    taser_em_build::dimensions_check();

    let output_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR environment variable not set"))
        .join("shaders-spirv");

    let shader_src = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("taser-em manifest dir doesn't exist")
    ).join("..").join("shaders").join("src");

    let links_name = (1..=3)
        .find(|dim| env::var(format!("CARGO_FEATURE_DIM{dim}")).is_ok())
        .map(|dim| format!("taser-em-shaders{dim}d"))
        .unwrap();
    let mut builder = KhalBuilder::from_dependency(links_name.as_str(), true)
        .shader_src(&shader_src);

    // Enable dim features in shader compilation
    for dim in 1..=3 {
        if env::var(format!("CARGO_FEATURE_DIM{dim}")).is_ok() {
            builder = builder.feature(format!("dim{dim}"));
        }
    }

    builder.build(&output_dir);
}