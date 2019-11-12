## Titanic-Machine-Learning-from-Disaster

A simple 1-layer network to predict the survival rates of passengers aboard the ill-fated Titanic.

This implementation uses a simple 1-layer network with 6 parameters from the training set: 
- Age
- Sex
- Passenger class
- Number of siblings
- Number of parents/children
- The price paid by the passenger for the tickets

This implementation uses an Adam Optimizer with a learning rate of 0.01, along with a sigmoid cross entropy loss function. This network was trained using batch gradient descent, using the entire dataset to train the network during each epoch.

The training accuracy of this network after 50000 epochs was **76.43%**, while the test set accuracy was **69.377%** according to Kaggle's evaluation.

