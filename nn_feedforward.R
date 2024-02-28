source("activation_function.R")
# 初始化參數
input_size <- 2
hidden_size <- 3
output_size <- 1
# Initialize weights
weights_input_hidden <- matrix(runif(input_size * hidden_size), nrow = input_size, ncol = hidden_size)
weights_hidden_output <- matrix(runif(hidden_size * output_size), nrow = hidden_size, ncol = output_size)
# Initialize bias
bias_hidden <- runif(1, -1, 1)
bias_output <- runif(1, -1, 1)

input_data <- matrix(c(0.5, 0.8), nrow = 1, ncol = input_size)
# Feedforward
hidden_layer_input <- input_data %*% weights_input_hidden + bias_hidden
hidden_layer_output <- sigmoid(hidden_layer_input)

output_layer_input <- hidden_layer_output %*% weights_hidden_output + bias_output
output_layer_output <- sigmoid(output_layer_input)

cat("Input: ", input_data, "\n")
cat("Output: ", output_layer_output, "\n")
