import numpy as np
import random


def l2_loss(predictions,Y):
    '''
        Computes L2 loss (sum squared loss) between true values, Y, and predictions.
        :param Y: A 1D Numpy array with real values (float64)
        :param predictions: A 1D Numpy array of the same size of Y
        :return: L2 loss using predictions for Y.
    '''
    # TODO
    l2 = np.sum(np.power((Y-predictions),2))
    return l2

def sigmoid(x):
    '''
        Sigmoid function f(x) =  1/(1 + exp(-x))
        :param x: A scalar or Numpy array
        :return: Sigmoid function evaluated at x (applied element-wise if it is an array)
    '''
    return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (np.exp(x) + np.exp(0)))

def sigmoid_derivative(x):
    '''
        First derivative of the sigmoid function with respect to x.
        :param x: A scalar or Numpy array
        :return: Derivative of sigmoid evaluated at x (applied element-wise if it is an array)
    '''
    # TODO
    val = sigmoid(x)
    return val * (1 - val)

class OneLayerNN:
    '''
        One layer neural network trained with Stocastic Gradient Descent (SGD)
    '''
    def __init__(self):
        '''
        @attrs:
            weights: The weights of the neural network model.
            batch_size: The number of examples in each batch
            learning_rate: The learning rate to use for SGD
            epochs: The number of times to pass through the dataset
            v: The resulting predictions computed during the forward pass
        '''
        # initialize self.weights in train()
        self.weights = None
        self.learning_rate = 0.001
        self.epochs = 25

        # initialize self.v in forward_pass()
        self.v = None

    def train(self, X, Y, print_loss=True):
        '''
        Trains the OneLayerNN model using SGD.
        :param X: 2D Numpy array where each row contains an example
        :param Y: 1D Numpy array containing the corresponding values for each example
        :param print_loss: If True, print the loss after each epoch.
        :return: None
        '''
        # TODO: initialize weights
        self.weights = np.zeros(len(X[0]))
        # TODO: figure out shape for weights ^^

        for epoch in range(self.epochs):
            for x,y in zip(X, Y):
                self.forward_pass(x)
                self.backward_pass(x, y)
                # Print the loss after every epoch
            if print_loss:
                print('Epoch: {} | Loss: {}'.format(epoch, self.loss(X, Y)))

    def forward_pass(self, X):
        '''
        Computes the predictions for a single layer given examples X and
        stores them in self.v
        :param X: 2D Numpy array where each row contains an example.
        :return: None
        '''
        # TODO:
        self.v = np.matmul(X, self.weights)
        

    def backward_pass(self, X, Y):
        '''
        Computes the weights gradient and updates self.weights
        :param X: 2D Numpy array where each row contains an example
        :param Y: 1D Numpy array containing the corresponding values for each example
        :return: None
        '''
        # TODO: Compute the gradients for the model's weights using backprop
        gradients = self.backprop(X,Y)
        # TODO: Update the weights using gradient descent
        self.gradient_descent(gradients)
        

    def backprop(self, X, Y):
        '''
        Returns the average weights gradient for the given batch
        :param X: 2D Numpy array where each row contains an example.
        :param Y: 1D Numpy array containing the corresponding values for each example
        :return: A 1D Numpy array representing the weights gradient
        '''
        # TODO: Compute the average weights gradient
        # Refer to the SGD algorithm in slide 5 in Lecture 19: Backpropagation
         #gradient - > calculate, then gradient descent
        #find what gradient is, use it to update weivghts in gradient descent
        
        # difference between v and true labels
        res = 2*(self.v - Y)*X
        # find avg
        avg = res / len(X)
        return avg


    def gradient_descent(self, grad_W):
        '''
        Updates the weights using the given gradient
        :param grad_W: A 1D Numpy array representing the weights gradient
        :return: None
        '''
        # TODO: Update the weights using the given gradient and the learning rate
        # Refer to the SGD algorithm in slide 5 in Lecture 19: Backpropagation
        # converge = False
        # losses = []
        # while (not converge):
        #     epoch+=1
        #     # shuffle examples AND labels -> so the labels are the same (both the same way)
        #     shuffler = np.random.permutation(len(grad_W))
        #     W_shuffled = grad_W[shuffler]
        #     for epoch in range(np.ciel(len(grad_W)/b) -1): #do i calculate the cieling?
        #         xBatch = X_shuffled[i: i + self.batch_size]
        #         yBatch = Y_shuffled[i: i + self.batch_size]
        #         L = np.zeros_like(self.weights)
        #         for x,y in zip(xBatch, yBatch):
        #             for j in range(0, self.n_classes):  
        #                 if (y == j):
        #                     L[j] += (softmax(np.matmul(self.weights,x))[j] - 1) * x
        #                 else:
        #                     L[j] += (softmax(np.matmul(self.weights,x))[j]) * x
        #         self.weights -= self.alpha * L / len(xBatch)
        #     losses.append(self.loss(X_shuffled, Y_shuffled))
        #     if (len(losses)>1 and abs(losses[-1]-losses[-2]) <= self.conv_threshold):
        #         converge = True
        self.weights = self.weights - (self.learning_rate * grad_W) #scaling gradient vector
        

    def loss(self, X, Y):
        '''
        Returns the total squared error on some dataset (X, Y).
        :param X: 2D Numpy array where each row contains an example
        :param Y: 1D Numpy array containing the corresponding values for each example
        :return: A float which is the squared error of the model on the dataset
        '''
        # Perform the forward pass and compute the l2 loss
        self.forward_pass(X)
        return l2_loss(self.v, Y)

    def average_loss(self, X, Y):
        '''
        Returns the mean squared error on some dataset (X, Y).
        MSE = Total squared error/# of examples
        :param X: 2D Numpy array where each row contains an example
        :param Y: 1D Numpy array containing the corresponding values for each example
        :return: A float which is the mean squared error of the model on the dataset
        '''
        return self.loss(X, Y) / X.shape[0]

class TwoLayerNN:

    def __init__(self, hidden_size, activation=sigmoid, activation_derivative=sigmoid_derivative):
        '''
        @attrs:
            activation: the activation function applied after the first layer
            activation_derivative: the derivative of the activation function. Used for training.
            hidden_size: The hidden size of the network (an integer)
            batch_size: The number of examples in each batch
            learning_rate: The learning rate to use for SGD
            epochs: The number of times to pass through the dataset
            wh: The first (hidden) layer weights of the neural network model.
            bh: The first (hidden) layer bias of the neural network model.
            wout: The second (output) layer weights of the neural network model.
            bout: The second (output) layer bias of the neural network model.
            a1: The output of the first layer computed during the forward pass
            v1: The activated output of the first layer computed during the forward pass
            a2: The output of the second layer computed during the forward pass
            v2: The resulting predictions computed during the forward pass (layer 2 has the identity activation function)
            output_neurons: The number of outputs of the network
        '''
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.hidden_size = hidden_size
        self.learning_rate = 0.01
        self.epochs = 25

        # initialize the following weights and biases in the train() method
        self.wh = None
        self.bh = None
        self.wout = None
        self.bout = None

        # initialize the following values in the forward_pass() method
        # these values will be stored and used for the backward_pass()
        # note that you may not need to use them all in backward_pass()
        self.a1 = None
        self.v1 = None
        self.a2 = None
        self.v2 = None


        # In this assignment, we will only use output_neurons = 1.
        self.output_neurons = 1

    def _get_layer2_bias_gradient(self, x, y):
        '''
        Computes the gradient of the loss with respect to the output bias, bout.
        :param x: Numpy array for a single training example with dimension: input_size by 1
        :param y: Label for the training example
        :return: the partial derivates dL/dbout, a numpy array of dimension: output_neurons by 1
        '''
        # TODO:
        return 2*(self.v2 - y)

    def _get_layer2_weights_gradient(self, x, y):
        '''
        Computes the gradient of the loss with respect to the output weights, wout.
        :param x: Numpy array for a single training example with dimension: input_size by 1
        :param y: Label for the training example
        :return: the partial derivates dL/dwout, a numpy array of dimension: output_neurons by hidden_size
        '''
        # TODO:
        return 2*(self.v2 - y)*self.a1.T

    def _get_layer1_bias_gradient(self, x, y):
        '''
        Computes the gradient of the loss with respect to the hidden bias, bh.
        :param x: Numpy array for a single training example with dimension: input_size by 1
        :param y: Label for the training example
        :return: the partial derivates dL/dbh, a numpy array of dimension: hidden_size by 1
        '''
        # TODO:
        # redo?
        return (2*(self.v2 - y))*self.wout.T*(self.activation_derivative(self.v1))
       

    def _get_layer1_weights_gradient(self, x, y):
        '''
        Computes the gradient of the loss with respect to the hidden weights, wh.
        :param x: Numpy array for a single training example with dimension: input_size by 1
        :param y: Label for the training example
        :return: the partial derivates dL/dwh, a numpy array of dimension: hidden_size by input_size
        '''
        # TODO:
        #redo?
        print("debugging")
        print(self.activation_derivative(self.v1).shape)
        print(x.T.shape)
        # return (2*(self.v2 - y))*(self.activation_derivative(self.v1)@x.T)*self.wout
        return (2*(self.v2 - y))*self.wout.T*(np.matmul(self.activation_derivative(self.v1),x.T))

    def train(self, X, Y, print_loss=True):
        '''
        Trains the TwoLayerNN with SGD using Backpropagation.
        :param X: 2D Numpy array where each row contains an example
        :param Y: 1D Numpy array containing the corresponding values for each example
        :param learning_rate: The learning rate to use for SGD
        :param epochs: The number of times to pass through the dataset
        :param print_loss: If True, print the loss after each epoch.
        :return: None
        '''
        # NOTE:
        # Use numpy arrays of the following dimensions for your model's parameters.
        # layer 1 weights (wh): hidden_size x input_size
        # layer 1 bias (bh): hidden_size x 1
        # layer 2 weights (wout): output_neurons x hidden_size
        # layer 2 bias (bout): output_neurons x 1
        # HINT: for best performance initialize weights with np.random.normal or np.random.uniform

        # TODO: Weight and bias initialization
        self.wh = np.random.uniform(size=(self.hidden_size,len(X[0])))
        # self.wh.shape = (self.hidden_size,len(X))
        self.bh = np.zeros((self.hidden_size,1))
        # self.bh.shape = (self.hidden_size,1)
        self.wout = np.random.uniform(size=(self.output_neurons,self.hidden_size))
        # self.wout.shape = (self.output_neurons,self.hidden_size)
        self.bout = np.zeros((self.output_neurons,1))
        # self.bout.shape = (self.output_neurons,1)

        # TODO: Train network for certain number of epochs

        # TODO: Shuffle the examples (X) and labels (Y)

        # TODO: We need to iterate over each data point for each epoch
        # iterate through the examples in batch size increments

        # TODO: Perform the forward and backward pass on the current batch

        for epoch in range(self.epochs):
            # shuffle?
            index = np.arange(len(X))
            np.random.shuffle(index)
            x = X[index]
            y = Y[index]
            for x_new,y_new in zip(x, y):
                x_new = x_new.reshape((1,-1))
                self.forward_pass(x_new.T)
                self.backward_pass(x_new.T, y_new.T)
                # Print the loss after every epoch
            if print_loss:
                print('Epoch: {} | Loss: {}'.format(epoch, self.loss(X, Y)))


    def forward_pass(self, X):
        '''
        Computes the predictions for a 2 layer NN given examples X and
        stores them in self.v2.
        Stores intermediate values before the prediction task in self.v1 and
        self.a1
        :param X: 2D Numpy array where each row contains an example.
        :return: None
        '''
        # TODO:
        # pass data through neural network
        # First layer pre-activation
    
        self.v1 = np.matmul(self.wh,X.T) + self.bh

        # First layer activation
        self.a1 = self.activation(self.v1)

        # Second layer pre-activation
        # self.v2 = self.a1.dot(self.wout) + self.bout
        self.v2 = np.matmul(self.wout,self.a1) + self.bout

    def backward_pass(self, X, Y):
        '''
        Computes the weights gradient and updates all four weights and bias gradients
        :param X: 2D Numpy array where each row contains an example
        :param Y: 1D Numpy array containing the corresponding values for each example
        :return: None
        '''
        # TODO: Compute the gradients for the model's weights using backprop
        wh_gradient, bh_gradient, wout_gradient, bout_gradient = self.backprop(X,Y)
        # TODO: Update the weights using gradient descent
        self.gradient_descent(wh_gradient,bh_gradient,wout_gradient,bout_gradient)

    def backprop(self, X, Y):
        '''
        Computes the average weights and biases gradients for the given batch
        :param X: 2D Numpy array where each row contains an example.
        :param Y: 1D Numpy array containing the corresponding values for each example
        :return: 4 Numpy arrays representing the computed gradients for each weight and bias
        '''
        # TODO: Call the "get gradient" methods
        return self._get_layer1_bias_gradient(X,Y), self._get_layer1_weights_gradient(X,Y), self._get_layer2_bias_gradient(X,Y), self._get_layer2_weights_gradient(X,Y)
        

    def gradient_descent(self, grad_wh, grad_bh, grad_wout, grad_bout):
        '''
        Updates the weights using the given gradients
        :param grad_wh: Numpy array representing the hidden weights gradient
        :param grad_bh: Numpy array representing the hidden bias gradient
        :param grad_wout: Numpy array representing the output weights gradient
        :param grad_bout: Numpy array representing the output bias gradient
        :return: None
        '''
        # TODO: Update the weights using the given gradients and the learning rate
        # Refer to the SGD algorithm in slide 5 in Lecture 19: Backpropagation

        # OLD VERSION
        # self.weights = self.weights - (self.learning_rate * grad_wh * grad_wout * grad_bh * grad_bout) #scaling gradient vector
        
        # updated version ?
        self.wh = self.wh - (self.learning_rate * grad_wh)
        self.bh = self.bh - (self.learning_rate * grad_bh)
        self.wout = self.wout - (self.learning_rate * grad_wout)
        self.bout = self.bout - (self.learning_rate * grad_bout)


    def loss(self, X, Y):
        '''
        Returns the total squared error on some dataset (X, Y).
        :param X: 2D Numpy array where each row contains an example
        :param Y: 1D Numpy array containing the corresponding values for each example
        :return: A float which is the squared error of the model on the dataset
        '''
        # Perform the forward pass and compute the l2 loss
        self.forward_pass(X.T)
        return l2_loss(self.v2, Y)

    def average_loss(self, X, Y):
        '''
        Returns the mean squared error on some dataset (X, Y).
        MSE = Total squared error/# of examples
        :param X: 2D Numpy array where each row contains an example
        :param Y: 1D Numpy array containing the corresponding values for each example
        :return: A float which is the mean squared error of the model on the dataset
        '''
        return self.loss(X, Y) / X.shape[0]
