library(tidyverse)
source("activation_function.R")
source("useful_function.R")
data(iris)
# Iris label to number, use case_when
iris$Species <- case_when(
  iris$Species == "setosa" ~ 1,
  iris$Species == "versicolor" ~ 2,
  iris$Species == "virginica" ~ 3
)
# Initialize parameters
input_size <- 4
hidden_size <- 5
output_size <- 3
learning_rate <- 0.1
epochs <- 3
# Initialize weights and biases
set.seed(3)
weights_input_hidden <- matrix(runif(input_size * hidden_size), nrow = input_size, ncol = hidden_size)
weights_hidden_output <- matrix(runif(hidden_size * output_size), nrow = hidden_size, ncol = output_size)
bias_hidden <- runif(hidden_size)
bias_output <- runif(output_size)
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
# list for error
error_list <- list()
# Train the neural network
for (epoch in 1:epochs) {
  # Forward propagation
  # (105,4)*(4,5) + (105, 5)
  hidden_layer_input <- x_train %*% weights_input_hidden + bias_hidden
  hidden_layer_output <- softmax(hidden_layer_input)
  output_layer_input <- hidden_layer_output %*% weights_hidden_output + bias_output
  output_layer_output <- softmax(output_layer_input)
  # Calculate error
  error <- y_train - output_layer_output
  error_list <- append(error_list, mean(error))
  # Backpropagation
  output_delta <- error
  # (105,3)*(3,3)
  hidden_error <- output_delta %*% t(weights_hidden_output)
  hidden_delta <- hidden_error * sigmoid_derivative(hidden_layer_output)
  # Update weights and biases
  weights_hidden_output <- weights_hidden_output + learning_rate * t(hidden_layer_output) %*% output_delta
  weights_input_hidden <- weights_input_hidden + learning_rate * t(x_train) %*% hidden_delta
  bias_output <- bias_output + learning_rate * colSums(output_delta)
  bias_hidden <- bias_hidden + learning_rate * colSums(hidden_delta)
  cat(mean(bias_hidden))
}
# Predictions
# (45,4)*(4,3) + (45, 3)
hidden_layer_input <- x_test %*% weights_input_hidden + bias_hidden
hidden_layer_output <- sigmoid(hidden_layer_input)
output_layer_input <- hidden_layer_output %*% weights_hidden_output + bias_output
output_layer_output <- softmax(output_layer_input)
# Get the predicted classes
predicted_classes <- apply(output_layer_output, 1, which.max)
# Accuracy calculation
accuracy <- sum(predicted_classes == y_test) / length(y_test)
accuracy
# Plot the error

