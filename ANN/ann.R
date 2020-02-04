dataset = read.csv('customers_data.csv')
dataset = dataset[, 4:14]

dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                   levels = c('Female', 'Male'),
                                   labels = c(1, 2)))

library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-11] = scale(training_set[-11])
test_set[-11] = scale(test_set[-11])

# install.packages('h2o')
library(h2o)
h2o.init(nthreads = -1)
classifier = h2o.deeplearning(y = 'Exited', 
                              training_frame = as.h2o(training_set), 
                              activation = 'Rectifier', 
                              hidden = c(6, 6), 
                              epochs = 100,
                              train_samples_per_iteration = -2)




prob_pred = h2o.predict(classifier, newdata = as.h2o(test_set[-11]))

y_pred = prob_pred > 0.5
y_pred = as.vector(y_pred)

# Making the confusion matrix
cm = table(test_set[, 11], y_pred)

accuracy = (1547 + 180) / 2000

h2o.shutdown()
