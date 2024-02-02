timestamp <- 100
# 皆代表特徵空間維度
input_features <- 32
output_features <- 64

random_array <- function(dim) {
  # prod(dim) 是維度中所有元素的乘積，這裡為32*100，runif用來創建均勻分布的隨機偏差
  array(runif(prod(dim)), dim = dim)
  # 根據指定的維度（dim）轉換成相應維度的數組
}

inputs <- random_array(dim = c(timestamp, input_features)) # 輸入數據: 這個例子為隨機噪聲
state_t <- rep_len(0, length = c(output_features)) # 初始狀態: 全零向量

# 創建隨機權重矩陣
W <- random_array(dim = c(output_features, input_features))
U <- random_array(dim = c(output_features, output_features))
b <- random_array(dim = c(output_features, 1))

output_sequence <- array(0, dim = c(timestamp, output_features))

for (i in 1 : nrow(inputs)) {
  
  input_t <- inputs[i,]
  
  # 維度：(64, 32) * (32, 1) + (64, 64) * (64, 1) + (64,1) = (64, 1)
  output_t <- tanh(as.numeric((W %*% input_t) + (U %*% state_t) + b)) # 將輸入與當前狀態(前一個輸出)結合
  output_sequence[i,] <- as.numeric(output_t)
  # 1:64
  state_t <- output_t # 更新下一個時間步的狀態
  
  cat('state :', state_t, "\n")
}
