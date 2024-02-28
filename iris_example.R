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
hidden_size <- 1000 # M, neurons
output_size <- 3
learning_rate <- 0.1
epochs <- 5
# Split data
index <- sample(1:nrow(iris), nrow(iris) * 0.7)
train_data <- iris[index, ]
test_data <- iris[-index, ]
x_train <- as.matrix(train_data[, 1:4])
y_train <- as.matrix(train_data[, 5])
x_test <- as.matrix(test_data[, 1:4])
y_test <- as.matrix(test_data[, 5])
# One-hot encode the target variable for multi-class classification
y_train <- to_categorical(y_train, num_classes = output_size)
# Initialize weights and biases
set.seed(8)
weights_input_hidden <- matrix(runif(input_size * hidden_size), nrow = input_size, ncol = hidden_size)
weights_hidden_output <- matrix(runif(hidden_size * output_size), nrow = hidden_size, ncol = output_size)
bias_hidden <- runif(hidden_size)
bias_output <- runif(output_size)
# Train the neural network
for (epoch in 1:epochs) {
  #  Forward propagation
  hidden_layer_input <- x_train %*% weights_input_hidden + bias_hidden
  hidden_layer_output <- relu(hidden_layer_input)
  output_layer_input <- hidden_layer_output %*% weights_hidden_output + bias_output
  output_layer_output <- softmax(output_layer_input)
  
  # Calculate loss
  # loss <- -1 * sum(y_train * log(output_layer_output))
  loss <- mean((output_layer_output - y_train)^2)
  print(paste("Epoch:", epoch, "Loss:", loss))
  
  # Backpropagation
  output_error <- output_layer_output - y_train
  output_delta <- output_error * output_layer_output * (1 - output_layer_output)
  hidden_error <- output_delta %*% t(weights_hidden_output)
  hidden_delta <- hidden_error * hidden_layer_output * (1 - hidden_layer_output)
  
  # Update weights and biases
  weights_hidden_output <- weights_hidden_output - learning_rate * t(hidden_layer_output) %*% output_delta
  weights_input_hidden <- weights_input_hidden - learning_rate * t(x_train) %*% hidden_delta
  bias_output <- bias_output - learning_rate * colSums(output_delta)
  bias_hidden <- bias_hidden - learning_rate * colSums(hidden_delta)
}
# Predictions
# (45,4)*(4,3) + (45, 3)
hidden_layer_input <- x_test %*% weights_input_hidden + bias_hidden
hidden_layer_output <- softmax(hidden_layer_input)
output_layer_input <- hidden_layer_output %*% weights_hidden_output + bias_output
output_layer_output <- softmax(output_layer_input)
output_layer_output
# Get the predicted classes
predicted_classes <- apply(output_layer_output, 1, which.max)
predicted_classes
# Accuracy calculation
accuracy <- sum(predicted_classes == y_test) / length(y_test)
accuracy

# Use package
library(keras)
model <- keras_model_sequential() %>%
  layer_dense(units = 1000, activation = 'relu', input_shape = c(4)) %>%
  layer_dense(units = 500, activation = 'relu') %>%
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
  batch_size = 32,
  validation_split = 0.2
)


