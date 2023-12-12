# Neural-Network

To enhance my understanding of neural networks and their variants, I am constructing basic neural networks for this project without relying on external packages. I will continually update and list the lessons learned below.

## Model

1. Neural Network
2. Recurrent Neural Network
3. Convolutional Neural Network
4. Long Short-Term Memory
5. Gated Recurrent Unit

## Activation function

### Tanh
- Usually use in hidden layer
- Centered around 0
- Vanishing gradient
### ReLU
- Solve the vanishing gradient problem
- Fast computation speed
- Dead ReLU
- Not centered around 0
- Fast convergence
### Leaky ReLU
- Alleviated the dead ReLU problem
- A range from negative infinity to positive infinity
### Parametric ReLU
- If hyperparameter equals 0: ReLU
- If hyperparameter is small: Leaky ReLU
### Exponential Linear Unit (ELU)
- No Dead ReLU issue
- High computational cost
- Similar to Leaky ReLU, although theoretically superior to ReLU, there is currently insufficient empirical evidence to conclusively demonstrate that ELU is always better than ReLU in practice
### Scaled Exponential Linear Unit (SELU)
- It can induce self-normalizing properties, such as variance stabilization, thereby mitigating the issues of gradient explosion and vanishing
- The SELU (Scaled Exponential Linear Unit) function is obtained by multiplying the output of the ELU (Exponential Linear Unit) function by a scaling factor lambda.
- The multiplication by λ ensures that in certain regions, the slope is greater than 1. This implies that if you input a relatively small change, after passing through that region, it will be amplified by a factor of 1.0507700987. Therefore, the input can be magnified.
### Softmax
- Usually use in output layer
- Use it for multi-class classification
- Not differentiable at zero
- The gradient of negative input being zero implies that for the activation in that region, weights will not be updated during backpropagation, leading to the emergence of dead neurons that never get activated, similarity as Dead ReLU.
- We can consider it as a probabilistic version or 'soft' version of the argmax function.
### Sigmoid
- Usually use in output layer
- Use it for binary classification
- Not centered around 0
- Vanishing gradient
- High computational cost