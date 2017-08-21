# Siamese LSTM for Text Classification

## Goal
Create model for text classification problems (including intent detection and sentiment analysis) that only requires a small amount of labeled data.

## Architecture
Uses a regression Siamese Recurrent Network that serves as the distance function for a k-Nearest Neighbor model.

The Siamese network learns to generate distance values for each pair of sentences within the corpus. A pair with the same label comes with a desired value of 0, while a pair with different labels comes with a very large arbitrary value. The current loss function is mean squared error.

The learned network is then used by a k-NN model using the training data. Evaluation is done on the k-NN with test data.

#### Siamese Model Diagram

![Diagram](docs/siamese_model.png)
