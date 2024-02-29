library(tidyverse)
source("activation_function.R")
source("useful_function.R")
data(iris)
# Iris label to number, use case_when
iris$Species <- case_when (
  iris$Species == "setosa" ~ 1,
  iris$Species == "versicolor" ~ 2,
  iris$Species == "virginica" ~ 3
)
# Initialize parameters
input_size <- 4 # N, variables
hidden_size1 <- 10 # M, neurons
hidden_size2 <- 6
output_size <- 3
learning_rate <- 0.1
epochs <- 5
# Split data
index <- sample(1:nrow(iris), nrow(iris) * 0.7)
train_data <- iris[index, ]
test_data <- iris[-index, ]
x_train <- as.matrix(train_data[, 1:4])
x_train <-scale(x_train)
y_train <- as.matrix(train_data[, 5])
y_train <- scale(y_train)
x_test <- as.matrix(test_data[, 1:4])
y_test <- as.matrix(test_data[, 5])
# One-hot encode the target variable for multi-class classification
y_train <- to_categorical(y_train, num_classes = output_size)
# Initialize weights and biases
set.seed(860)
weights_input_hidden1 <- matrix(runif(input_size * hidden_size1), nrow = input_size, ncol = hidden_size1)
weights_hidden1_hidden2 <- matrix(runif(hidden_size1 * hidden_size2), nrow = hidden_size1, ncol = hidden_size2)
weights_hidden2_output <- matrix(runif(hidden_size2 * output_size), nrow = hidden_size2, ncol = output_size)
bias_hidden1 <- runif(hidden_size1)
bias_hidden2 <- runif(hidden_size2)
bias_output <- runif(output_size)
# Train the neural network
for (epoch in 1:epochs) {
  #  Forward propagation
  hidden_layer_input1 <- x_train %*% weights_input_hidden1 + bias_hidden1
  hidden_layer_output1 <- relu(hidden_layer_input1)
  
  hidden_layer_input2 <- hidden_layer_output1 %*% weights_hidden1_hidden2 + bias_hidden2
  hidden_layer_output2 <- relu(hidden_layer_input2)
  
  output_layer_input <- hidden_layer_output2 %*% weights_hidden2_output + bias_output
  output_layer_output <- softmax(output_layer_input)
  
  # loss <- -1 * sum(y_train * log(output_layer_output))
  loss <- mean((output_layer_output - y_train)^2)
  print(paste("Epoch:", epoch, "Loss:", loss))
  
  # Backpropagation
  output_error <- output_layer_output - y_train
  output_delta <- output_error * output_layer_output * (1 - output_layer_output)
  hidden2_error <- output_delta %*% t(weights_hidden2_output)
  hidden2_delta <- hidden2_error * hidden_layer_output2 * (1 - hidden_layer_output2)
  hidden1_error <- hidden2_delta %*% t(weights_hidden1_hidden2)
  hidden1_delta <- hidden1_error * hidden_layer_output1 * (1 - hidden_layer_output1)
  
  # Update weights and biases
  weights_hidden2_output <- weights_hidden2_output - learning_rate * t(hidden_layer_output2) %*% output_delta
  weights_hidden1_hidden2 <- weights_hidden1_hidden2 - learning_rate * t(hidden_layer_output1) %*% hidden2_delta
  weights_input_hidden1 <- weights_input_hidden1 - learning_rate * t(x_train) %*% hidden1_delta
  bias_output <- bias_output - learning_rate * colSums(output_delta)
  bias_hidden2 <- bias_hidden2 - learning_rate * colSums(hidden2_delta)
  bias_hidden1 <- bias_hidden1 - learning_rate * colSums(hidden1_delta)
}
get_pred <- function(x) {
  hidden_layer_input1 <- x %*% weights_input_hidden1 + bias_hidden1
  hidden_layer_output1 <- relu(hidden_layer_input1)
  hidden_layer_input2 <- hidden_layer_output1 %*% weights_hidden1_hidden2 + bias_hidden2
  hidden_layer_output2 <- relu(hidden_layer_input2)
  output_layer_input <- hidden_layer_output2 %*% weights_hidden2_output + bias_output
  output_layer_output <- softmax(output_layer_input)
  predicted_classes <- apply(output_layer_output, 1, which.max)
  accuracy <- sum(predicted_classes == y_test) / length(y_test)
  return(list(predicted_classes, accuracy))
}
get_pred(x_test)
# Use package
library(keras)
model <- keras_model_sequential() %>%
  layer_dense(units = 100, activation = 'relu', input_shape = c(4)) %>%
  layer_dense(units = 20, activation = 'relu') %>%
  # layer_dense(units = 300, activation = 'relu') %>%
  # layer_dropout(rate = 0.2) %>%
  layer_dense(units = 3, activation = 'softmax')

# 編譯模型
model %>% compile(
  # optimizer = 'adam',
  optimizer = optimizer_sgd(learning_rate = 0.01),
  loss = 'binary_crossentropy',
  metrics = c('accuracy')
)
history <- model %>% fit(
  x_train, y_train,
  epochs = 5,
  batch_size = 1,
  validation_split = 0.2
)


