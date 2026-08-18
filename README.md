# Taser Electromagnetics
Taser Electromagnetics is a hobby project of mine that provides shaders and an API for
simulating  Maxwell's Equations using the Finite-Difference Time-Domain method on the GPU.
It features wide GPU backend support, and performant 1-, 2-, and 3-dimensional simulations written in Rust—all
in the same codebase.

> **Warning**
> This project is very new, so functionality and documentation are limited and
> won't actually reflect the description above right now.

# Features
- Support for various compute backends (Vulkan, CPU, NVIDIA) via [khal](https://github.com/dimforge/khal/).
- Diagonal Anisotropy
- Loss (conductivity)
- Material smoothing (permittivity, permeability, and conductivity)
- Uniaxial Perfectly Matched Layer
- Visualizer crates for all dimensions
### Unfinished Features
- 3D simulations are currently broken...
- NVIDIA support isn't tested yet
- Arbitrary anisotropy through spatial approximations (just on the roadmap at the moment)
- Python bindings (I might add this far into the future, since I don't know how yet...)

## Running Examples
### Clone the Repository
```bash
git clone https://github.com/nsjvidana/taser-em.git
cd taser-em
```
### Follow khal's Setup
See khal's [Development Setup](https://github.com/dimforge/khal/tree/main#development-setup) README section.
### Run the Examples!
```bash
cargo run --bin examples1d
cargo run --bin examples2d
cargo run --bin examples3d
```
To run the examples on different GPU backends, enable the respective features:
```bash
# Webgpu is the default backend. Compiles to SPIR-V.
cargo run --bin examples3d --features webgpu
# When running on the CPU, use cpu-parallel for better performance
cargo run --bin examples3d --features cpu
cargo run --bin examples3d --features cpu-parallel
# For CUDA / PTX targets
cargo run --bin examples3d --features cuda
```
# As a Rust dependency
Simply include the taser-em crate of your choice in your Cargo.toml (using the git repository):
```toml
[dependencies]
taser-em2d = { git = "https://github.com/nsjvidana/taser-em.git" }
```
Use the testbed crates to easily visualize your simulations:
```toml
taser-em-testbed2d = { git = "https://github.com/nsjvidana/taser-em.git" }
```
# References
The following resources were imperative to designing this simulation API:
- [EMPossible's FDTD course](https://empossible.net/academics/emp5304/) (EMP-5304) serves as the backbbone
  of this project, as all diagonal anisotropy update equations and stability improvements were derived from these lectures.
- [khal](https://github.com/dimforge/khal/) provides an amazing unified interface for compiling Rust shaders for multiple GPU backends.
- The rest of the [dimforge](https://github.com/dimforge) tools (e.g. parry3d's geometry, kiss3d's renderer)
- [Dr. John Brand Schneider's lecture notes](https://eecs.wsu.edu/~schneidj/ufdtd/chap8.pdf) on the 1D auxiliary grid implementation of 
  TF/SF plane wave source.