use std::f64;

// sigmoid
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// softmax
pub fn _softmax(x: Vec<f64>) -> Vec<f64> {
    
    let max_x = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_x: Vec<f64> = x.iter().map(|&val| (val - max_x).exp()).collect();
    let sum_exp_x: f64 = exp_x.iter().sum();

    exp_x.iter().map(|&val| val / sum_exp_x).collect()
}

// tanh
pub fn tanh(x: f64) -> f64 {
    (f64::exp(x) - f64::exp(-x)) / (f64::exp(x) + f64::exp(-x))
}

// sigmoid 的導數
pub fn sigmoid_derivative(x: f64) -> f64 {
    let sigmoid_x = sigmoid(x);
    sigmoid_x * (1.0 - sigmoid_x)
}