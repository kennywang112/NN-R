use rand::prelude::*;
use nalgebra::DMatrix;

pub fn FeedForward(sigmoid: fn(f64) -> f64) {

    // 初始化參數
    let input_size: usize = 2;
    let hidden_size: usize = 3;
    let output_size: usize = 1;

    // 生成並初始化權重矩陣
    // (2, 3)
    let weights_input_hidden = random_matrix(input_size, hidden_size, -1.0, 1.0);
    // (3, 1)
    let weights_hidden_output = random_matrix(hidden_size, output_size, -1.0, 1.0);

    // 生成並初始化偏差矩陣
    // (2, 3)
    let bias_hidden = random_matrix(2, hidden_size, -1.0, 1.0);
    // (2, 1)
    let bias_output = random_matrix(2, output_size, -1.0, 1.0);

    // 定義輸入數據
    let input_data = DMatrix::from_row_slice(2, input_size, &[0.5, 0.8, 0.1, 0.2]);

    println!("input_data: \n{}", input_data);

    // feedforward
    // (2, 2) * (2, 3) + (2, 3) = (2, 3)
    let hidden_layer_input = &input_data * &weights_input_hidden + &bias_hidden;
    let hidden_layer_output = hidden_layer_input.map(|x| sigmoid(x));
    println!("hidden_layer_output: \n{}", hidden_layer_output);
    // (2, 3) * (3, 1) + (2, 1) = (2, 1)
    let output_layer_input = &hidden_layer_output * &weights_hidden_output + &bias_output;
    let output_layer_output = output_layer_input.map(|x| sigmoid(x));
    println!("output_layer_output: \n{}", output_layer_output);

    println!("output_layer_output: \n{}", output_layer_output);
}

fn random_matrix(rows: usize, cols: usize, min: f64, max: f64) -> DMatrix<f64> {

    let mut rng = rand::thread_rng();
    DMatrix::from_fn(rows, cols, |_, _| rng.gen_range(min..max))

}