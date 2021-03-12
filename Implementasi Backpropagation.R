library(neuralnet)
library(ggplot2)

data_set <- read.csv("D:/Kuliah Smt 6/Sisdas/dataset_tugas3.csv")
head(data_set)
dim(data_set)

normalisasi <- function(x){
  return((2*((x-min(x)) / (max(x)-min(x))))-1)
}

data_set[c("age", "anaemia", "creatinine_phosphokinase", "diabetes",
           "ejection_fraction", "high_blood_pressure", "platelets",
           "serum_creatinine", "serum_sodium", "sex", "smoking", "time")] <- 
  lapply(data_set[c("age", "anaemia", "creatinine_phosphokinase", "diabetes",
                    "ejection_fraction", "high_blood_pressure", "platelets",
                    "serum_creatinine", "serum_sodium", "sex", "smoking", "time")], normalisasi)

y <- data_set$DEATH_EVENT
x <- cbind(c(data_set$age), c(data_set$anaemia), c(data_set$creatinine_phosphokinase),
           c(data_set$diabetes), c(data_set$ejection_fraction), c(data_set$high_blood_pressure),
           c(data_set$platelets), c(data_set$serum_creatinine), c(data_set$serum_sodium), 
           c(data_set$sex), c(data_set$smoking), c(data_set$time))
cbind(x, y)

random_vector <- runif(12, -0.5, 0.5)
random_matrix <- matrix(
  random_vector,
  nrow = 12,
  ncol = 4,
  byrow = TRUE
)


my_ann <- list(
  #predictor variabel
  input = x,
  
  #weight for layer 1
  weight1 = random_matrix,
  
  #weight for layer 2
  weight2 = matrix(runif(4), ncol = 1),
  
  #actual observed
  y = y,
  
  #store the predicted outcome
  output = matrix(
    rep(0, times = 1),
    ncol = 1
  )
)


fungsi_sigmoid <- function(x){
  1.0 / (1.0 + exp(-x))
}
#fungsi turunan sigmoid
sigmoid_turunan <- function(x){
  x * (1.0 - x)
}
loss_func <- function(nn){
  sum((nn$y - nn$output)^2)
}

loss_dataset <- data.frame(
  iteration = 1:3000,
  loss = vector("numeric", length = 3000)
)


# Feedforward
# Epoch = 3000
# Learning rate = 0.75
for (j in seq_len(3000)) {
  for (i in seq_len(299)) {
    my_ann$layer <- fungsi_sigmoid(my_ann$input[i,] %*% my_ann$weight1)
    my_ann$output[i] <- fungsi_sigmoid(my_ann$layer %*% my_ann$weight2)
    
    d_weight2 <- 0
    d_weight2[1] <- my_ann$layer[1] %*% (0.75*(my_ann$y[i] - my_ann$output[i]) + sigmoid_turunan(my_ann$output[i]))
    d_weight2[2] <- my_ann$layer[2] %*% (0.75*(my_ann$y[i] - my_ann$output[i]) + sigmoid_turunan(my_ann$output[i]))
    d_weight2[3] <- my_ann$layer[3] %*% (0.75*(my_ann$y[i] - my_ann$output[i]) + sigmoid_turunan(my_ann$output[i]))
    d_weight2[4] <- my_ann$layer[4] %*% (0.75*(my_ann$y[i] - my_ann$output[i]) + sigmoid_turunan(my_ann$output[i]))
    
    e_layer <- 0
    e_layer[1] <- ((my_ann$y[i] - my_ann$output[i]) * sigmoid_turunan(my_ann$output[i])) %*% my_ann$weight2[1] 
    e_layer[2] <- ((my_ann$y[i] - my_ann$output[i]) * sigmoid_turunan(my_ann$output[i])) %*% my_ann$weight2[2]
    e_layer[3] <- ((my_ann$y[i] - my_ann$output[i]) * sigmoid_turunan(my_ann$output[i])) %*% my_ann$weight2[3]
    e_layer[4] <- ((my_ann$y[i] - my_ann$output[i]) * sigmoid_turunan(my_ann$output[i])) %*% my_ann$weight2[4]
    
    e_layer <- e_layer * sigmoid_turunan(my_ann$layer)
    
    d_weight1 <- 0
    d_weight1 <- my_ann$input[i,] %*% e_layer * 0.75
    
    my_ann$weight1 <- my_ann$weight1 + d_weight1
    my_ann$weight2 <- my_ann$weight2 + d_weight2
    
    square_error <- 0
    square_error[i] <- ((my_ann$y[i] - my_ann$output[i])^2)
  }
}

#Visualisasi data dan menghitung error
ggplot(data = loss_dataset, aes(x = iteration, y = loss)) + geom_line()

err = 0
outputs = 0
for (i in seq_len(299)) {
  outputs[i] <- round(my_ann$output[i])
  if(outputs[i] == my_ann$y[i]){
    err = err + 1
  }
}

akurasi <- (err/299)*100
akurasi