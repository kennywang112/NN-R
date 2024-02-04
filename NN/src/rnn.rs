use ndarray::Array;
use rand::Rng;
// use crate::activation_function::tanh;

fn random_array(dim: usize) -> Array<f64, ndarray::IxDyn> {
    let mut rng = rand::thread_rng();
    Array::from_shape_fn(ndarray::IxDyn(&[dim]), |_| rng.gen_range(0.0..1.0))
}

pub fn recurrent() {

    let timestamp = 10;
    let input_features = 3;
    let output_features = 4;

    let inputs = random_array(timestamp * input_features).into_shape((timestamp, input_features)).unwrap();
    // (4, 1)
    let mut state_t = Array::zeros((output_features, 1));
    println!("inputs:{:?} state_t:{:?}", inputs.shape(), state_t.shape());

    let w = random_array(output_features * input_features).into_shape((output_features, input_features)).unwrap();
    let u = random_array(output_features * output_features).into_shape((output_features, output_features)).unwrap();
    // (4, 1)
    let b = random_array(output_features).into_shape((output_features, 1)).unwrap();
    println!("W:{:?} U:{:?} b:{:?}", w.shape(), u.shape(), b.shape());
    // (10, 4)
    let mut output_sequence = Array::zeros((timestamp, output_features));
    println!("output_sequence: {:?}", output_sequence.shape());

    for i in 0..timestamp {

        // let input_t = inputs.row(i);
        let input_t = inputs.row(i).insert_axis(ndarray::Axis(1));

        // (4, 3) * (3, 1) + (4, 4) * (4, 1) = (4, 1)
        let output_t = w.dot(&input_t) + u.dot(&state_t) + &b;
        println!("{:?} * {:?} + {:?} * {:?} + {:?}", w.shape() , input_t.shape(), u.shape(), state_t.shape(), b.shape());
        let output_t = output_t.mapv(|x| x.tanh());
        // 獲取第i個時間的輸出並保存output_t到output_sequence中
        output_sequence.row_mut(i).assign(&output_t);
        // println!("output_sequence: {:?}", output_sequence.row_mut(i));
        state_t = output_t.clone();
        
        println!("state: {:?}", state_t);
    }
}
