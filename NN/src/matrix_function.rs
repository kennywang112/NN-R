use rand::prelude::*;
use nalgebra::DMatrix;
use ndarray::Array;

pub fn random_matrix(
    rows: usize, cols: usize, min: f64, max: f64
) -> DMatrix<f64> {

    let mut rng = rand::thread_rng();
    DMatrix::from_fn(rows, cols, |_, _| rng.gen_range(min..max))
}

pub fn random_array(
    dim: usize
) -> Array<f64, ndarray::IxDyn> {

    let mut rng = rand::thread_rng();
    Array::from_shape_fn(ndarray::IxDyn(&[dim]), |_| rng.gen_range(0.0..1.0))
}