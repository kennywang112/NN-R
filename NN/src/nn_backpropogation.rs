use nalgebra::DMatrix;

use crate::activation_function::{sigmoid, sigmoid_derivative};
use crate::matrix_function::random_matrix;

pub fn backpropogation() {

    // 初始化參數
    let input_size = 2;
    let hidden_size = 3;
    let output_size = 1;
    let learning_rate = 0.1;
    let epochs = 10;

    // 初始化權重和偏差
    let mut weights_input_hidden = random_matrix(input_size, hidden_size, -1.0, 1.0);
    let mut weights_hidden_output = random_matrix(hidden_size, output_size, -1.0, 1.0);
    let mut bias_hidden = random_matrix(2, hidden_size, -1.0, 1.0);
    let mut bias_output = random_matrix(2, output_size, -1.0, 1.0);
    // 定義輸入數據和目標輸出
    let input_data = DMatrix::from_row_slice(2, input_size, &[0.5, 0.8, 0.1, 0.2]);
    let target_output = DMatrix::from_row_slice(2, output_size, &[0.6, 0.4]);

    for _epoch in 1 ..= epochs {
        // 前向傳播
        // (2, 2) * (2, 3) + (2, 3) = (2, 3)
        let hidden_layer_input = &input_data * &weights_input_hidden + &bias_hidden;
        let hidden_layer_output = hidden_layer_input.map(|x| sigmoid(x));
        // (2, 3) * (3, 1) + (2, 1) = (2, 1)
        let output_layer_input = &hidden_layer_output * &weights_hidden_output + &bias_output;
        let output_layer_output = output_layer_input.map(|x| sigmoid(x));
        // 計算誤差
        // (2, 1) - (2, 1) = (2, 1)
        let error = &target_output - &output_layer_output;
        // 計算梯度
        // (2, 1)
        let output_delta = error.iter()
            .zip(output_layer_output.iter().map(|x| sigmoid_derivative(*x)))
            .map(|(e, d)| e * d)
            .collect::<Vec<_>>();
        let output_delta_matrix = DMatrix::from_column_slice(output_delta.len(), 1, &output_delta);

        let hidden_error = &output_delta_matrix * weights_hidden_output.transpose();

        let hidden_delta = hidden_error.iter()
            .zip(hidden_layer_output.iter().map(|x| sigmoid_derivative(*x)))
            .map(|(e, d)| e * d)
            .collect::<Vec<_>>();
        let hidden_delta_matrix = DMatrix::from_column_slice(2, 3, &hidden_delta);

        // 更新權重和偏差
        let dot_products_hidden: Vec<_> = hidden_layer_output.iter()
            .map(|&x| x * learning_rate)
            .collect();
        let dot_products_hidden_matrix = DMatrix::from_column_slice(2, 3, &dot_products_hidden);
        // (3, 1) += (3, 2) * (2, 1)
        weights_hidden_output += &dot_products_hidden_matrix.transpose() * &output_delta_matrix;

        let dot_products_input: Vec<_> = input_data.iter()
            .map(|&x| x * learning_rate)
            .collect();
        let dot_products_input_matrix = DMatrix::from_column_slice(2, 2, &dot_products_input);
        // (2, 3) += (2, 2) * (2, 3)
        weights_input_hidden += &dot_products_input_matrix.transpose() * &hidden_delta_matrix;

        bias_output += learning_rate * output_delta_matrix;
        bias_hidden += learning_rate * hidden_delta_matrix;

        // println!("Final output after training:\n{}", hidden_error);
    }

}