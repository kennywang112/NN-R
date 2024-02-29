source("activation_function.R")
source("useful_function.R")
# Parameters
input_size <- 2
hidden_size <- 3
output_size <- 1
learning_rate <- 0.1
epochs <- 10
# Initialize weights and biases
weights_input_hidden <- matrix(runif(input_size * hidden_size), nrow = input_size, ncol = hidden_size)
weights_hidden_output <- matrix(runif(hidden_size * output_size), nrow = hidden_size, ncol = output_size)
bias_hidden <- runif(1, -1, 1)
bias_output <- runif(1, -1, 1)
input_data <- matrix(c(0.5, 0.8), nrow = 1, ncol = input_size)
target_output <- matrix(0.6, nrow = 1, ncol = output_size)
# Train the neural network
for (epoch in 1:epochs) {
  # Front propagation
  hidden_layer_input <- input_data %*% weights_input_hidden + bias_hidden
  hidden_layer_output <- sigmoid(hidden_layer_input)
  
  output_layer_input <- hidden_layer_output %*% weights_hidden_output + bias_output
  output_layer_output <- sigmoid(output_layer_input)
  
  # Calculate the error
  error <- target_output - output_layer_output
  print(paste("Epoch:", epoch, "Error:", error))
  
  # Calculate the gradient
  output_delta <- error * sigmoid_derivative(output_layer_output)
  hidden_error <- output_delta %*% t(weights_hidden_output)
  hidden_delta <- hidden_error * sigmoid_derivative(hidden_layer_output)
  
  # Update weights and biases
  weights_hidden_output <- weights_hidden_output + learning_rate * t(hidden_layer_output) %*% output_delta
  weights_input_hidden <- weights_input_hidden + learning_rate * t(input_data) %*% hidden_delta
  bias_output <- bias_output + learning_rate * output_delta
  bias_hidden <- bias_hidden + learning_rate * hidden_delta
}

# Result
cat("Input: ", input_data, "\n")
cat("Target Output: ", target_output, "\n")
cat("Output after training: ", output_layer_output, "\n")

