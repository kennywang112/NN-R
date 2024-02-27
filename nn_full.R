source("activation_function.R")
# 設定隨機種子以確保結果可重複
set.seed(123)

# 設定神經網絡參數
input_features <- 3 # N, variables
hidden_units <- 4 # M, neurons
output_features <- 4
learning_rate <- 0.01
epochs <- 5

# 創建隨機輸入和目標
# (4, 3), 輸入為 inputs[1,] -> N-by-1 matrix
inputs <- matrix(runif(4 * input_features, -1, 1), nrow = hidden_units, ncol = input_features)
inputs[1,] # 1st row = (3, 1)
# (4, 1)
targets <- matrix(runif(4 * 1, -1, 1), nrow = 4, ncol = 1)

# 初始化權重和偏差
# runif創建均勻分布的隨機偏差，再做成對應矩陣
# N-by-M matrix (3, 4)
W_input_hidden <- matrix(runif(input_features * hidden_units, -1, 1), nrow = input_features, ncol = hidden_units)
# M-by-1 matrix (4, 4)
b_hidden <- matrix(runif(hidden_units, -1, 1), nrow = hidden_units, ncol = hidden_units)
# (4, 1)
W_output_hidden <- matrix(runif(1 * output_features, -1, 1), nrow = output_features, ncol = 1)
# (4, 1)
b_output <- matrix(runif(output_features, -1, 1), nrow = output_features, ncol = 1)

# 定義模型訓練函數
train_model <- function(
    inputs, targets, W_input_hidden, b_hidden, 
    W_output_hidden, b_output, learning_rate, epochs, batch_size
    ) {
  
  for (epoch in 1 : epochs) {
    
    for (i in seq(1, nrow(inputs), by = batch_size)) {
      
      # 使用批次大小為 batch_size 的子集進行訓練
      # inputs_batch <- inputs[i:(i + batch_size - 1),, drop = FALSE]
      # targets_batch <- targets[i:(i + batch_size - 1),, drop = FALSE]
      inputs_batch <- inputs 
      targets_batch <- targets
      
      # 前向傳播
      # (4, 3)*(3, 4) + (4, 4) = (4, 4)
      hidden_layer_input <- inputs_batch %*% W_input_hidden + b_hidden
      hidden_layer_output <- tanh(hidden_layer_input)
      # (4, 4)*(4, 1) + (4, 1) = (4, 1)
      output_layer_input <- hidden_layer_output %*% W_output_hidden + b_output
      predicted_output <- tanh(output_layer_input)
  
      # 計算損失（這裡使用均方誤差）
      # (4,1) - (4, 1) = (4, 1)
      loss <- mean((predicted_output - targets_batch)^2)
      cat("Epoch:", epoch, "  Loss:", loss, "\n")
      
      # 反向傳播
      # (4, 1) - (4, 1) = (4, 1)
      output_error <- predicted_output - targets_batch
      # (4, 1) x (4, 1) x (4, 1) = (4, 1)
      # 計算輸出層誤差對隱藏層輸出影響，再乘上tanh的導數
      output_delta <- output_error * predicted_output * (1 - predicted_output)
      # (4, 1) * (1, 4) = (4, 4)
      hidden_error <- output_delta %*% t(W_output_hidden)
      # (4, 4) x (4, 4) x (4, 4) = (4, 4)
      hidden_layer_delta <- hidden_error * hidden_layer_output * (1 - hidden_layer_output)
      
      # 更新權重和偏差
      # (4, 1) + (4, 4) * (4, 1) = (4, 1)
      W_output_hidden <- W_output_hidden + learning_rate * t(hidden_layer_output) %*% output_delta
      # (4, 1) + (4, 1) = (4, 1)
      b_output <- b_output + learning_rate * rowSums(output_delta)
      
      # (3, 4) + (3, 4)*(4, 4) = (3, 4)
      W_input_hidden <- W_input_hidden + learning_rate * t(inputs_batch) %*% hidden_layer_delta
      # (4, 1) + (4, 1) = (4, 1)
      b_hidden <- b_hidden + learning_rate * rowSums(hidden_layer_delta)
    }
  }

  return(list(W_input_hidden = W_input_hidden, b_hidden = b_hidden, W_output_hidden = W_output_hidden, b_output = b_output))
}

# 訓練模型
batch_size <- 5
trained_model <- train_model(inputs, targets, W_input_hidden, b_hidden, 
                             W_output_hidden, b_output, learning_rate, epochs, batch_size)