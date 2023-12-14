# 設定隨機種子以確保結果可重複
set.seed(123)

# 設定神經網絡參數
input_features <- 32 # N
hidden_units <- 64 # M
output_features <- 64
learning_rate <- 0.01
epochs <- 10

# 創建隨機輸入和目標
# (64, 32), 輸入為 inputs[1,] -> N-by-1 matrix
inputs <- matrix(runif(64 * input_features, -1, 1), nrow = 64, ncol = input_features)
inputs[1,] # 1st row = (32, 1)
# (64, 1)
#targets <- matrix(runif(64 * output_features, -1, 1), nrow = 64, ncol = output_features)
targets <- matrix(runif(64 * 1, -1, 1), nrow = 64, ncol = 1)

# 初始化權重和偏差
# runif創建均勻分布的隨機偏差，再做成對應矩陣
# N-by-M matrix (32, 64)
W_input_hidden <- matrix(runif(input_features * hidden_units, -1, 1), nrow = input_features, ncol = hidden_units)
# M-by-1 matrix (64, 1)
b_hidden <- matrix(runif(hidden_units, -1, 1), nrow = hidden_units, ncol = 1)
# (64, 64)
W_hidden_output <- matrix(runif(hidden_units * output_features, -1, 1), nrow = hidden_units, ncol = output_features)
# (64, 1)
b_output <- matrix(runif(output_features, -1, 1), nrow = output_features, ncol = 1)

# 定義激活函數（這裡使用tanh）
tanh <- function(x) {
  (exp(x) - exp(-x)) / (exp(x) + exp(-x))
}

# 定義模型訓練函數
train_model <- function(inputs, targets, W_input_hidden, b_hidden, W_hidden_output, b_output, learning_rate, epochs) {
  
  for (epoch in 1:epochs) {
    # 前向傳播
    # (64, 32)*(32, 1) + (64, 1) = (64, 1)
    hidden_activations <- tanh(t(W_input_hidden)%*%inputs + b_hidden)
    # (64, 64)*(64, 1) + (64, 1) = (64, 1)
    output_activations <- tanh(W_hidden_output%*%hidden_activations + b_output)

    # 計算損失（這裡使用均方誤差）
    # (64,1) - (64, 1) = (64, 1)
    loss <- mean((output_activations - targets)^2)
    cat("Epoch:", epoch, "  Loss:", loss, "\n")
    
    # 反向傳播
    # (64, 1) - (64, 1) = (64, 1)
    output_error <- output_activations - targets
    # 計算輸出層誤差對隱藏層輸出影響，再乘上tanh的導數
    # (64, 64)*(64, 1)x(64, 1) = (64, 1)
    hidden_error <- t(W_hidden_output)%*%output_error * (1 - hidden_activations^2)
    
    # 更新權重和偏差
    # (64, 64) - (64, 1)*(1, 64) = (64, 64)
    W_hidden_output <- W_hidden_output - learning_rate * hidden_activations %*% t(output_error)
    # (64, 1) - (64, 1) = (64, 1)
    b_output <- b_output - learning_rate * colSums(output_error)
    # (32, 64) - (32, 1)*(1, 64) = (32, 64)
    W_input_hidden <- W_input_hidden - learning_rate * inputs %*% t(hidden_error)
    # (64, 1) - (64, 1) = (64, 1)
    b_hidden <- b_hidden - learning_rate * colSums(hidden_error)
    b_hidden <- matrix(b_hidden, nrow = nrow(hidden_activations), ncol = 1)
    
  }

  return(list(W_input_hidden = W_input_hidden, b_hidden = b_hidden, W_hidden_output = W_hidden_output, b_output = b_output))
}

# 訓練模型
trained_model <- train_model(inputs[1,], targets, W_input_hidden, b_hidden, W_hidden_output, b_output, learning_rate, epochs)

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
