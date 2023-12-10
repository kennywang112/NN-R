# 定義 LSTM 相關函數
sigmoid <- function(x) {
  1 / (1 + exp(-x))
}

# 定義 tanh 函數
tanh <- function(x) {
  (exp(x) - exp(-x)) / (exp(x) + exp(-x))
}

# 設定參數
timestamp <- 100
input_features <- 32
output_features <- 64

# 生成隨機輸入數據
random_array <- function(dim) {
  array(runif(prod(dim)), dim = dim)
}

# 初始化 LSTM 相關權重和偏置
Wf <- random_array(dim = c(output_features, input_features))
Uf <- random_array(dim = c(output_features, output_features))
bf <- random_array(dim = c(output_features, 1))

Wi <- random_array(dim = c(output_features, input_features))
Ui <- random_array(dim = c(output_features, output_features))
bi <- random_array(dim = c(output_features, 1))

Wo <- random_array(dim = c(output_features, input_features))
Uo <- random_array(dim = c(output_features, output_features))
bo <- random_array(dim = c(output_features, 1))

Wc <- random_array(dim = c(output_features, input_features))
Uc <- random_array(dim = c(output_features, output_features))
bc <- random_array(dim = c(output_features, 1))

# 初始化 LSTM 狀態
ct <- rep_len(0, length = c(output_features))
ht <- rep_len(0, length = c(output_features))

# 初始化存儲 LSTM 输出序列的矩陣
output_sequence <- array(0, dim = c(timestamp, output_features))

# 迭代多個時間步
for (i in 1:timestamp) {
  # 當前時間步的輸入
  input_t <- random_array(dim = c(1, input_features))
  
  # 遺忘門
  ft <- sigmoid(as.numeric((Wf %*% input_t) + (Uf %*% ht) + bf))
  
  # 輸入門
  it <- sigmoid(as.numeric((Wi %*% input_t) + (Ui %*% ht) + bi))
  
  # 輸出門
  ot <- sigmoid(as.numeric((Wo %*% input_t) + (Uo %*% ht) + bo))
  
  # 候選細胞狀態
  c_tilde <- tanh(as.numeric((Wc %*% input_t) + (Uc %*% ht) + bc))
  
  # 細胞狀態更新
  ct <- ft * ct + it * c_tilde
  
  # 隱藏狀態更新
  ht <- ot * tanh(ct)
  
  # 更新 LSTM 输出序列
  output_sequence[i,] <- as.numeric(ht)
  
  cat('output :', ht, "\n")
}

# 打印 LSTM 输出序列
print(output_sequence)
