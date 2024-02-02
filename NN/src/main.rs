mod nn_feedforward;
use nn_feedforward::FeedForward;  

mod activation_function;
use activation_function::sigmoid;

fn main() {

    FeedForward(sigmoid);
}