source("activation_function.R")
sigmoid_derivative <- function(x) {
  return(x * (1 - x))
}
# 初始化參數
input_size <- 2
hidden_size <- 3
output_size <- 1
learning_rate <- 0.1
epochs <- 10000

# 初始化權重和偏差
weights_input_hidden <- matrix(runif(input_size * hidden_size), nrow = input_size, ncol = hidden_size)
weights_hidden_output <- matrix(runif(hidden_size * output_size), nrow = hidden_size, ncol = output_size)
bias_hidden <- matrix(runif(1, -1, 1), nrow = 1, ncol = hidden_size)
bias_output <- matrix(runif(1, -1, 1), nrow = 1, ncol = output_size)

# 輸入和目標輸出
input_data <- matrix(c(0.5, 0.8), nrow = 1, ncol = input_size)
target_output <- matrix(0.6, nrow = 1, ncol = output_size)

# 訓練神經網路
for (epoch in 1:epochs) {
  # 前向傳播
  hidden_layer_input <- input_data %*% weights_input_hidden + bias_hidden
  hidden_layer_output <- sigmoid(hidden_layer_input)
  
  output_layer_input <- hidden_layer_output %*% weights_hidden_output + bias_output
  output_layer_output <- sigmoid(output_layer_input)
  
  # 計算誤差
  error <- target_output - output_layer_output
  
  # 計算梯度
  output_delta <- error * sigmoid_derivative(output_layer_output)
  hidden_error <- output_delta %*% t(weights_hidden_output)
  hidden_delta <- hidden_error * sigmoid_derivative(hidden_layer_output)
  
  # 更新權重和偏差
  weights_hidden_output <- weights_hidden_output + learning_rate * t(hidden_layer_output) %*% output_delta
  weights_input_hidden <- weights_input_hidden + learning_rate * t(input_data) %*% hidden_delta
  bias_output <- bias_output + learning_rate * output_delta
  bias_hidden <- bias_hidden + learning_rate * hidden_delta
}

# 顯示結果
cat("Input: ", input_data, "\n")
cat("Target Output: ", target_output, "\n")
cat("Output after training: ", output_layer_output, "\n")
