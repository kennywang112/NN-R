# Feedforward
source("activation_function.R")
# 初始化參數
input_size <- 2
hidden_size <- 3
output_size <- 1
# 初始化權重
weights_input_hidden <- matrix(runif(input_size * hidden_size), nrow = input_size, ncol = hidden_size)
weights_hidden_output <- matrix(runif(hidden_size * output_size), nrow = hidden_size, ncol = output_size)
# 初始化偏差
bias_hidden <- matrix(runif(1, -1, 1), nrow = 1, ncol = hidden_size)
bias_output <- matrix(runif(1, -1, 1), nrow = 1, ncol = output_size)
# 定義輸入
input_data <- matrix(c(0.5, 0.8), nrow = 1, ncol = input_size)
# 前向傳播
hidden_layer_input <- input_data %*% weights_input_hidden + bias_hidden
hidden_layer_output <- sigmoid(hidden_layer_input)

output_layer_input <- hidden_layer_output %*% weights_hidden_output + bias_output
output_layer_output <- sigmoid(output_layer_input)
# 顯示結果
cat("Input: ", input_data, "\n")
cat("Output: ", output_layer_output, "\n")
