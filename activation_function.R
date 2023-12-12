# Tanh
tanh <- function(x) {
  return((exp(x) - exp(-x)) / (exp(x) + exp(-x)))
}

# ReLU
relu <- function(x) {
  return(ifelse(x > 0, x, 0))
}

# Leaky ReLU
leaky_relu <- function(x, alpha = 0.01) {
  return(ifelse(x > 0, x, alpha * x))
}

# Parametric ReLU
prelu <- function(x, alpha = 0.01) {
  return(ifelse(x > 0, x, alpha * x))
}

# Exponential Linear Unit (ELU)
elu <- function(x, alpha = 1.0) {
  return(ifelse(x > 0, x, alpha * (exp(x) - 1)))
}

# Scaled Exponential Linear Unit (SELU)
selu <- function(x, alpha = 1.67326, scale = 1.0507) {
  return(ifelse(x > 0, scale * x, scale * alpha * (exp(x) - 1)))
}

# Softmax
softmax <- function(x) {
  exp_x <- exp(x - max(x))  # 防止数值溢出
  return(exp_x / sum(exp_x))
}

# Sigmoid
sigmoid <- function(x) {
  return(1 / (1 + exp(-x)))
}