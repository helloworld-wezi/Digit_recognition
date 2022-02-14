# Digit_recognition

This project used MNIST (Mixed National Institute of Standards and Technology) database as the training, validation, and test data. The MNIST database contains binary images of handwritten digits commonly used to train image processing systems. The digits were collected from among Census Bureau employees and high school students. The database contains 60,000 training digits and 10,000 testing digits, all of which have been size-normalized and centered in a fixed-size image of 28 × 28 pixels. 

## Part I

### Setup
The project setup (part I) is as follows:
1. The project used Python 3.6 with the following packages:
    1. NumPy (numerical toolbox)
    2. Matplotlib (plotting toolbox)
    3. Scikit-learn (predictive data analysis tool)
2. The methods used are as follows:
    1. Regression
    2. Principal component analysis (PCA) dimensionality reduction
    3. Kernels
3. The goals are to compare the performance of different methods by comparing the error on test data and various regularization weights.

The code used in Part I follows the scikit-learn package guidance [here](https://github.com/Varal7/ml-tutorial/blob/master/Part1.ipynb).

### Results and Conclusions
1. First, linear regression with closed-form solution is used. The feature vector has 28 x 28 = 784 dimensions. The errors are expected to be high since the closed form solution of linear regression is the solution of optimizing the mean squared error loss. This is not an appropriate loss function for a classification problem. The error on test data on varying regularization weights (lambda):
    1. Lambda = 1, Error = 76.97%
    2. Lambda = 0.1, Error = 76.98%
    3. Lambda = 0.01, Error = 77.02%
2. Second, Support Vector Machine (SVM) is implemented using the scikit-learn package. Both the one-vs-rest (binary) SVM and Multiclass SVM are implemented and the errors on test data are compared. In the one-vs-rest SVM, the labels of digits 1-9 are changed to 1 and label 0 is kept for digit 0. The value of parameter C used in the package are varied. (C represents the tolerance of error, similar to lambda, the weight of regularization term in perceptron. A larger C means we are punishing more on the classification error, thus being less tolerant to misclassifications. Therefore, we will get a smaller margin hyperplane). The results are as follows:
    1. Using C = 0.1, the error on test data of one-vs-rest SVM: 0.75%
    2. Using C = 0.1, the error on test data of multiclass SVM: 8.19%
3. Third, Multinomial (Softmax) Regression and Gradient Descent is implemented. The error on test data using softmax regression is 10.05% for temperature parameter value of 1. Temperature is a hyperparameter that affect the hyperplane separator resulted from the algorithm. Smaller temperature parameter favors larger thetas (the hyperplane separator), which lower the regularization effect and thus tend to overfit the data. Using temperature values of 0.5 and 2, the errors on test data are 8.39% and 12.61% respectively.
4. Fourth, PCA dimensionality reduction method is used. Principal Components Analysis (PCA) is the most popular method for linear dimension reduction of data and is widely used in data analysis. The in-depth exposition can be seen [here](https://online.stat.psu.edu/stat505/lesson/11). This method finds (orthogonal) directions of maximal variation in the data. By projecting an n x d dataset X onto k < d of these directions, we get a new dataset of lower dimension that reflects more variation in the original data than any other k-dimensional linear projection of X. The error on the test data using this method is 14.74%.
5. Finally, the kernel trick is implemented, in which case, a cubic feature mapping is used. This is method also highly reduces the dimension of feature vector in order for the algorithm to run faster. The error on the test data using this method is 8.49%.


## Part II

### Setup
The project setup (part II) is as follows:
1. In addition the packages used in part I, in part II PyTorch package is used for implementing the deep neural network. The code in part II follows the PyTorch package guidance [here](https://github.com/Varal7/ml-tutorial/blob/master/Part2.ipynb) and [here](https://pytorch.org/docs/stable/index.html).
2. The training, validation and test data are the same as those used in part I.
3. First, the baseline model use the fully-connected neural nets and the parameters used are as follows:
    1. batch size 32,
    2. hidden size 10,
    3. learning rate 0.1,
    4. momentum 0, and
    5. the ReLU activation function
4. The accuracy on the test data using the baseline parameters is 94.00%.
5. Second, the convolutional neural network (CNN) is used to improve the performance of the model. The baseline parameters used are as follows:
    1. A convolutional layer with 32 filters of size 3 x 3
    2. A ReLU nonlinearity
    3. A max pooling layer with size 2 x 2
    4. A convolutional layer with 64 filters of size 3 x 3
    5. A ReLU nonlinearity
    6. A max pooling layer with size 2 x 2
    7. A flatten layer
    8. A fully connected layer with 128 neurons
    9. A dropout layer with drop probability 0.5
    10. A fully-connected layer with 10 neurons
6. The accuracy on the test data using the baseline parameters is 99.13%.
7. Third, in this part, the model will be adjusted to predict two overlapping digits. The training, validation and test data are taken from the MNIST as well. The baseline parameters used are as follows:
    1. A convolutional layer with 2 filters of size 3 x 3
    2. A ReLU nonlinearity
    3. A convolutional layer with 32 filters of size 5 x 5
    4. A ReLU nonlinearity
    5. A convolutional layer with 32 filters of size 5 x 5
    6. A batch normalization layer 2d with 32 filters
    7. A ReLU nonlinearity
    8. A 2D max pooling layer with kernel size of 2 x 2
    9. A dropout layer with probability 0.25 (for regularization and preventing the co-adaptation of neuron)
    10. A convolutional layer with 64 filters of size 3 x 3
    11. A ReLU nonlinearity
    12. A convolutional layer with 64 filters of size 3 x 3
    13. A batch normalization layer 2d with 64 filters
    14. A ReLU nonlinearity
    15. A 2D max pooling layer with kernel size of 2 x 2
    16. A dropout layer with probability 0.25
    17. A flattening layer
    18. A ReLU nonlinearity
    19. A fully connected layer with 256 neurons
    20. A ReLU nonlinearity
    21. A fully connected layer with 64 neurons
    22. A ReLU nonlinearity
    23. A dropout layer with probability 0.25
    24. A fully connected layer with 10 neurons (final output layer).
8. The model also uses Stochastic Gradient Descent (SGD) optimizer. The test accuracy is 98.64%.
    

### Results and Conclusions
