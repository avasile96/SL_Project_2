library(keras)
library(tfdatasets)
library(tfautograph)
library(reticulate)
library(purrr)

mnist <- dataset_mnist()
mnist$train$x <- mnist$train$x/255
mnist$test$x <- mnist$test$x/255

dim(mnist$train$x) <- c(dim(mnist$train$x), 1)
dim(mnist$test$x) <- c(dim(mnist$test$x), 1)

train_ds <- mnist$train %>% 
  tensor_slices_dataset() %>%
  dataset_take(20000) %>% 
  dataset_map(~modify_at(.x, "x", tf$cast, dtype = tf$float32)) %>% 
  dataset_map(~modify_at(.x, "y", tf$cast, dtype = tf$int64)) %>% 
  dataset_shuffle(10000) %>% 
  dataset_batch(32)

test_ds <- mnist$test %>% 
  tensor_slices_dataset() %>% 
  dataset_take(2000) %>% 
  dataset_map(~modify_at(.x, "x", tf$cast, dtype = tf$float32)) %>%
  dataset_map(~modify_at(.x, "y", tf$cast, dtype = tf$int64)) %>% 
  dataset_batch(32)

simple_conv_nn <- function(filters, kernel_size) {
  keras_model_custom(name = "MyModel", function(self) {
    
    self$conv1 <- layer_conv_2d(
      filters = filters, 
      kernel_size = rep(kernel_size, 2),
      activation = "relu"
    )
    
    self$flatten <- layer_flatten()
    
    self$d1 <- layer_dense(units = 128, activation = "relu")
    self$d2 <- layer_dense(units = 10, activation = "softmax")
    
    
    function(inputs, mask = NULL) {
      inputs %>% 
        self$conv1() %>% 
        self$flatten() %>% 
        self$d1() %>% 
        self$d2()
    }
  })
}

model <- simple_conv_nn(filters = 32, kernel_size = 3)

loss <- loss_sparse_categorical_crossentropy
optimizer <- optimizer_adam()

train_loss <- tf$keras$metrics$Mean(name='train_loss')
train_accuracy <-  tf$keras$metrics$SparseCategoricalAccuracy(name='train_accuracy')

test_loss <- tf$keras$metrics$Mean(name='test_loss')
test_accuracy <- tf$keras$metrics$SparseCategoricalAccuracy(name='test_accuracy')

train_step <- function(images, labels) {
  
  with (tf$GradientTape() %as% tape, {
    predictions <- model(images)
    l <- loss(labels, predictions)
  })
  
  gradients <- tape$gradient(l, model$trainable_variables)
  optimizer$apply_gradients(purrr::transpose(list(
    gradients, model$trainable_variables
  )))
  
  train_loss(l)
  train_accuracy(labels, predictions)
  
}

test_step <- function(images, labels) {
  predictions <- model(images)
  l <- loss(labels, predictions)
  
  test_loss(l)
  test_accuracy(labels, predictions)
}


training_loop <- tf_function(autograph(function(train_ds, test_ds) {
  
  for (b1 in train_ds) {
    train_step(b1$x, b1$y)
  }
  
  for (b2 in test_ds) {
    test_step(b2$x, b2$y)
  }
  
  tf$print("Acc", train_accuracy$result(), "Test Acc", test_accuracy$result())
  
  train_loss$reset_states()
  train_accuracy$reset_states()
  test_loss$reset_states()
  test_accuracy$reset_states()
  
}))

for (epoch in 1:5) {
  cat("Epoch: ", epoch, " -----------\n")
  training_loop(train_ds, test_ds)  
}







