mod activation_function;
mod matrix_function;

mod nn_feedforward;
use nn_feedforward::feedforward;

mod nn_backpropogation;
use nn_backpropogation::backpropogation;

mod rnn;
use rnn::recurrent;

fn main() {

    feedforward();
    backpropogation();
    recurrent();

}