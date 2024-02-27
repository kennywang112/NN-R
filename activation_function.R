# Tanh
# - (-1, 1)
# - Usually use in hidden layer
# - Centered around 0
# - Vanishing gradient
tanh <- function(x) {
  return((exp(x) - exp(-x)) / (exp(x) + exp(-x)))
}
# ReLU
# - (0, x)
# - Solve the vanishing gradient problem
# - Fast computation speed
# - Dead ReLU
# - Not centered around 0
# - Fast convergence
relu <- function(x) {
  return(ifelse(x > 0, x, 0))
}
# Leaky ReLU
# - Alleviated the dead ReLU problem
# - A range from negative infinity to positive infinity
leaky_relu <- function(x, alpha = 0.01) {
  return(ifelse(x > 0, x, alpha * x))
}
# Parametric ReLU
# - If hyperparameter equals 0: ReLU
# - If hyperparameter is small: Leaky ReLU
prelu <- function(x, alpha = 0.01) {
  return(ifelse(x > 0, x, alpha * x))
}
# Exponential Linear Unit (ELU)
# - No Dead ReLU issue
# - High computational cost
# - Similar to Leaky ReLU, although theoretically superior to ReLU, there is currently insufficient empirical evidence to conclusively demonstrate that ELU is always better than ReLU in practice
elu <- function(x, alpha = 1.0) {
  return(ifelse(x > 0, x, alpha * (exp(x) - 1)))
}
# Scaled Exponential Linear Unit (SELU)
# - It can induce self-normalizing properties, such as variance stabilization, thereby mitigating the issues of gradient explosion and vanishing
# - The SELU (Scaled Exponential Linear Unit) function is obtained by multiplying the output of the ELU (Exponential Linear Unit) function by a scaling factor lambda.
# - The multiplication by λ ensures that in certain regions, the slope is greater than 1. This implies that if you input a relatively small change, after passing through that region, it will be amplified by a factor of 1.0507700987. Therefore, the input can be magnified.
selu <- function(x, alpha = 1.67326, scale = 1.0507) {
  return(ifelse(x > 0, scale * x, scale * alpha * (exp(x) - 1)))
}
# Softmax
# - Usually use in output layer
# - Use it for multi-class classification
# - Not differentiable at zero
# - The gradient of negative input being zero implies that for the activation in that region, weights will not be updated during backpropagation, leading to the emergence of dead neurons that never get activated, similarity as Dead ReLU.
# - We can consider it as a probabilistic version or 'soft' version of the argmax function.
softmax <- function(x) {
  exp_x <- exp(x - max(x))  # 防止数值溢出
  return(exp_x / sum(exp_x))
}
# Sigmoid
# - (0,1)
# - Usually use in output layer
# - Use it for binary classification (logistic)
# - Not centered around 0
# - Vanishing gradient
# - High computational cost
sigmoid <- function(x) {
  return(1 / (1 + exp(-x)))
}