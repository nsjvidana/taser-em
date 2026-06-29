use std::env;

pub fn dimensions_check() {
    let dim_features = vec![
        "CARGO_FEATURE_DIM1",
        "CARGO_FEATURE_DIM2",
        "CARGO_FEATURE_DIM3",
    ];
    let enabled_dim_count = dim_features
        .iter()
        .filter(|&&f| env::var(f).is_ok())
        .count();
    if enabled_dim_count == 0 {
        panic!("One dimension feature must be enabled.")
    }
    else if enabled_dim_count > 1 {
        panic!("Too many dimension features enabled.");
    }
}