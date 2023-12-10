# 定義 sigmoid 函數
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

# 初始化 GRU 相關權重和偏置
Wrz <- random_array(dim = c(output_features, input_features))
Urz <- random_array(dim = c(output_features, output_features))
brz <- random_array(dim = c(output_features, 1))

Wh <- random_array(dim = c(output_features, input_features))
Uh <- random_array(dim = c(output_features, output_features))
bh <- random_array(dim = c(output_features, 1))

# 初始化 GRU 狀態
ht <- rep_len(0, length = c(output_features))

# 初始化存儲 GRU 输出序列的矩陣
output_sequence <- array(0, dim = c(timestamp, output_features))

# 迭代多個時間步
for (i in 1:timestamp) {
  # 當前時間步的輸入
  input_t <- random_array(dim = c(1, input_features))
  
  # 重置門
  rz_t <- sigmoid(as.numeric((Wrz %*% input_t) + (Urz %*% ht) + brz))
  
  # 更新門
  h_tilde <- tanh(as.numeric((Wh %*% input_t) + rz_t * (Uh %*% ht) + bh))
  
  # 更新隱藏狀態
  ht <- (1 - rz_t) * ht + rz_t * h_tilde
  
  # 更新 GRU 输出序列
  output_sequence[i,] <- as.numeric(ht)
  
  cat('output :', ht, "\n")
}

# 打印 GRU 输出序列
print(output_sequence)
