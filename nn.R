# 將循環神經網絡（RNN）改為一般的前饋神經網絡（Feedforward Neural Network）

# timestamp <- 100
# input_features <- 32
# output_features <- 64

# 去掉時間步
# inputs <- random_array(dim = c(timestamp, input_features))
inputs <- random_array(dim = c(1, input_features))  # 不再有時間步的概念

# 刪除 RNN 的初始狀態
# state_t <- rep_len(0, length = c(output_features))

# 創建隨機權重矩陣
W <- random_array(dim = c(output_features, input_features))
b <- random_array(dim = c(output_features, 1))

# 刪除存儲 RNN 输出序列的矩陣
# output_sequence <- array(0, dim = c(timestamp, output_features))

# 刪除 for 循環
# for (i in 1:nrow(inputs)) {

# 刪除時間步的概念
input_t <- inputs

# 刪除 RNN 的狀態更新
# output_t <- tanh(as.numeric((W %*% input_t) + (U %*% state_t) + b))
output_t <- tanh(as.numeric((W %*% input_t) + b))  # 不再使用狀態

# 刪除 RNN 输出序列的存儲
# output_sequence[i,] <- as.numeric(output_t)

cat('output :', output_t, "\n")

# }

# 刪除 RNN 输出序列的輸出
# print(output_sequence)
