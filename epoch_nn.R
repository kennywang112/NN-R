# 設定隨機種子以確保結果可重複
set.seed(123)

# 設定神經網絡參數
input_features <- 32
hidden_units <- 64
output_features <- 64
learning_rate <- 0.01
epochs <- 10

# 初始化權重和偏差
W_input_hidden <- matrix(runif(input_features * hidden_units, -1, 1), nrow = input_features, ncol = hidden_units)
b_hidden <- matrix(runif(hidden_units, -1, 1), nrow = hidden_units, ncol = 1)
W_hidden_output <- matrix(runif(hidden_units * output_features, -1, 1), nrow = hidden_units, ncol = output_features)
b_output <- matrix(runif(output_features, -1, 1), nrow = output_features, ncol = 1)

# 定義激活函數（這裡使用tanh）
tanh <- function(x) {
  (exp(x) - exp(-x)) / (exp(x) + exp(-x))
}

# 定義模型訓練函數
train_model <- function(inputs, targets, W_input_hidden, b_hidden, W_hidden_output, b_output, learning_rate, epochs) {
  for (epoch in 1:epochs) {
    # 前向傳播
    hidden_activations <- tanh(inputs %*% W_input_hidden + b_hidden)
    output_activations <- tanh(hidden_activations %*% W_hidden_output + b_output)
    
    # 計算損失（這裡使用均方誤差）
    loss <- mean((output_activations - targets)^2)
    cat("Epoch:", epoch, "  Loss:", loss, "\n")
    
    # 反向傳播
    output_error <- output_activations - targets
    hidden_error <- output_error %*% t(W_hidden_output) * (1 - hidden_activations^2)
    
    # 更新權重和偏差
    W_hidden_output <- W_hidden_output - learning_rate * t(hidden_activations) %*% output_error
    b_output <- b_output - learning_rate * colSums(output_error)
    W_input_hidden <- W_input_hidden - learning_rate * t(inputs) %*% hidden_error
    b_hidden <- b_hidden - learning_rate * colSums(hidden_error)
  }
  
  return(list(W_input_hidden = W_input_hidden, b_hidden = b_hidden, W_hidden_output = W_hidden_output, b_output = b_output))
}

# 創建隨機輸入和目標
inputs <- matrix(runif(100 * input_features, -1, 1), nrow = 100, ncol = input_features)
targets <- matrix(runif(100 * output_features, -1, 1), nrow = 100, ncol = output_features)

# 訓練模型
trained_model <- train_model(inputs, targets, W_input_hidden, b_hidden, W_hidden_output, b_output, learning_rate, epochs)

# 獲取訓練後的權重和偏差
W_input_hidden <- trained_model$W_input_hidden
b_hidden <- trained_model$b_hidden
W_hidden_output <- trained_model$W_hidden_output
b_output <- trained_model$b_output

# 使用訓練後的模型進行預測
new_data <- matrix(runif(input_features, -1, 1), nrow = 1, ncol = input_features)
hidden_activations <- tanh(new_data %*% W_input_hidden + b_hidden)
predicted_output <- tanh(hidden_activations %*% W_hidden_output + b_output)

cat('Predicted Output:', predicted_output, "\n")
