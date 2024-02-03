mod activation_function;
mod matrix_function;

mod nn_feedforward;
use nn_feedforward::feedforward;

mod nn_backpropogation;
use nn_backpropogation::backpropogation;

fn main() {

    // feedforward();

    backpropogation();

}