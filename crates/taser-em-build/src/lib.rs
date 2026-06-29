use std::env;

pub fn dimensions_check() {
    let dim_features = vec![
        "CARGO_FEATURE_DIM1",
        "CARGO_FEATURE_DIM2",
        "CARGO_FEATURE_DIM3",
    ];
    let enabled_dims = dim_features
        .iter()
        .filter(|&&f| env::var(f).is_ok())
        .collect::<Vec<_>>();
    let enabled_dim_count = enabled_dims.len();
    if enabled_dim_count == 0 {
        panic!("One dimension feature must be enabled, but none were.")
    }
    else if enabled_dim_count > 1 {
        panic!("Too many dimension features enabled: {enabled_dims:?}");
    }
}