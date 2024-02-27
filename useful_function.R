to_categorical <- function(y, num_classes) {
  n <- length(y)
  categorical <- matrix(0, nrow = n, ncol = num_classes)
  for (i in 1:n) {
    categorical[i, y[i]] <- 1
  }
  return(categorical)
}
sigmoid_derivative <- function(x) {
  return(x * (1 - x))
}