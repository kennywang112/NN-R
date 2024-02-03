use rand::prelude::*;
use nalgebra::DMatrix;

pub fn random_matrix(
    rows: usize, cols: usize, min: f64, max: f64
) -> DMatrix<f64> {

    let mut rng = rand::thread_rng();
    DMatrix::from_fn(rows, cols, |_, _| rng.gen_range(min..max))

}