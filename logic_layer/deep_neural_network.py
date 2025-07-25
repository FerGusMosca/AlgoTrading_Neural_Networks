import copy
import csv

import numpy as np

from common.util.logging.light_logger import LightLogger


class DeepNeuralNetwork():

    #everythin to initiate the model
    def __init__(self):
        pass


    #region Private Methods

    def sigmoid(self,Z):
        A = 1 / (1 + np.exp(-Z))
        return (A,Z)

    def relu (self,Z):
        A = np.maximum(0, Z)
        return  (A,Z)

    def relu_backward(self,dA, activation_cache):
        Z=activation_cache
        dZ = np.array(dA, copy=True)  # Convertir dA a un array numpy

        # Establecer a 0 todos los elementos de dZ donde Z <= 0
        dZ[Z <= 0] = 0
        return  dZ

    def linear_backward(self,dZ, linear_cache,lambd=0):#lambd=regularization param
        """
            Implement the linear portion of backward propagation for a single layer (layer l)

            Arguments:
            dZ -- Gradient of the cost with respect to the linear output (of current layer l)
            cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

            Returns:
            dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
            dW -- Gradient of the cost with respect to W (current layer l), same shape as W
            db -- Gradient of the cost with respect to b (current layer l), same shape as b
            """
        A_prev, W, b = linear_cache
        m = A_prev.shape[1]

        ### START CODE HERE ### (≈ 3 lines of code)
        # dW = ...
        # db = ... sum by the rows of dZ with keepdims=True
        # dA_prev = ...

        reg_lamd= (lambd/m)*W #regularization parameter

        dW = (np.dot(dZ, A_prev.T) / m) + (reg_lamd)
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    def sigmoid_backward(self,dA, activation_cache):
        Z=activation_cache
        sig = 1 / (1 + np.exp(-Z))
        dZ = dA * sig * (1 - sig)
        return dZ

    def linear_forward(self,A, W, b):
        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter
        cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """

        # (≈ 1 line of code)
        # Z = ...
        Z = np.dot(W, A) + b

        cache = (A, W, b)

        return Z, cache

    def linear_activation_forward(self,A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function, also called the post-activation value
        cache -- a python tuple containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
        """

        if activation == "sigmoid":
            # (≈ 2 lines of code)
            # Z, linear_cache = ...
            # A, activation_cache = ...
            # YOUR CODE STARTS HERE
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.sigmoid(Z)
            # YOUR CODE ENDS HERE

        elif activation == "relu":
            # (≈ 2 lines of code)
            # Z, linear_cache = ...
            # A, activation_cache = ...
            # YOUR CODE STARTS HERE
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.relu(Z)
            # YOUR CODE ENDS HERE
        cache = (linear_cache, activation_cache)

        return A, cache

    def L_model_forward(self,X, parameters,activations):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()

        Returns:
        AL -- activation value from the output (last) layer
        caches -- list of caches containing:
                    every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
        """

        caches = []

        A = X
        L = len(parameters) // 2  # number of layers in the neural network

        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        # The for loop starts at 1 because layer 0 is the input
        for l in range(1, L):
            A_prev = A
            # (≈ 2 lines of code)
            # A, cache = ...
            # caches ...
            # (A_prev, W, b, activation)
            A, cache = self.linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activations[l-1])
            caches.append(cache)
            A_prev = A

        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        # (≈ 2 lines of code)
        # AL, cache = ...
        # caches ...
        AL, cache = self.linear_activation_forward(A_prev, parameters["W" + str(L)], parameters["b" + str(L)],  activations[L-1])

        caches.append(cache)

        return AL, caches

    def initialize_parameters_deep(self,layer_dims,use_He_init=False):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network

        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """

        np.random.seed(3)
        parameters = {}
        L = len(layer_dims)  # number of layers in the network

        for l in range(1, L):
            # (≈ 2 lines of code)
            # parameters['W' + str(l)] = ...
            # parameters['b' + str(l)] = ...

            if use_He_init:
                parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2./layer_dims[l-1])
            else:#Standaard Low numbers weights init
                parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01


            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

            assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
            assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

        return parameters

    def compute_cost(self,AL, Y):
        """
        Implement the cost function defined by equation (7).

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """

        m = Y.shape[1]

        # Compute loss from aL and y.
        # (≈ 1 lines of code)
        # cost = ...
        logpros = (np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), 1 - Y)) / m
        cost = -np.sum(logpros)

        cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

        return cost

    def linear_activation_backward(self,dA, cache, activation,lambd=0):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.

        Arguments:
        dA -- post-activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear_cache, activation_cache = cache

        if activation == "relu":
            # (≈ 2 lines of code)
            # dZ =  ...
            # dA_prev, dW, db =  ...
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache,lambd=lambd)

        elif activation == "sigmoid":
            # (≈ 2 lines of code)
            # dZ =  ...
            # dA_prev, dW, db =  ...
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache,lambd=lambd)


        return dA_prev, dW, db

    def L_model_backward(self,AL, Y,activations, caches,lambd=0):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

        Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ...
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ...
        """
        grads = {}
        L = len(caches)  # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

        # Initializing the backpropagation
        # (1 line of code)
        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))#also known as dZ

        # YOUR CODE ENDS HERE

        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
        # (approx. 5 lines)
        # current_cache = ...
        # dA_prev_temp, dW_temp, db_temp = ...
        # grads["dA" + str(L-1)] = ...
        # grads["dW" + str(L)] = ...
        # grads["db" + str(L)] = ...
        current_cache = caches[L - 1]
        dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dAL, current_cache,activations[L-1],lambd)
        grads["dA" + str(L - 1)] = dA_prev_temp
        grads["dW" + str(L)] = dW_temp
        grads["db" + str(L)] = db_temp

        # Loop from l=L-2 to l=0
        for l in reversed(range(L - 1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
            # (approx. 5 lines)
            # current_cache = ...
            # dA_prev_temp, dW_temp, db_temp = ...
            # grads["dA" + str(l)] = ...
            # grads["dW" + str(l + 1)] = ...
            # grads["db" + str(l + 1)] = ...
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activations[l],lambd)
            #print("dA:{}".format(dA_prev_temp))
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def update_parameters(self,params, grads, learning_rate):
        """
        Update parameters using gradient descent

        Arguments:
        params -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients, output of L_model_backward

        Returns:
        parameters -- python dictionary containing your updated parameters
                      parameters["W" + str(l)] = ...
                      parameters["b" + str(l)] = ...
        """
        parameters = copy.deepcopy(params)
        L = len(parameters) // 2  # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        # (≈ 2 lines of code)
        for l in range(L):
            # parameters["W" + str(l+1)] = ...
            # parameters["b" + str(l+1)] = ...
            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

        return parameters

    #endregion


    #region Public Methods

    def L_layer_model_train(self,X, Y, layers_dims,activations, learning_rate=0.0075, num_iterations=3000, print_cost=False,
                            parameters=None,loop=0,lambd=0,use_He_init=False):
        """
            Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

            Arguments:
            X -- input data, of shape (n_x, number of examples)
            Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
            layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
            learning_rate -- learning rate of the gradient descent update rule
            num_iterations -- number of iterations of the optimization loop
            lambd--> Regularization parameter
            print_cost -- if True, it prints the cost every 100 steps

            Returns:
            parameters -- parameters learnt by the model. They can then be used to predict.
            """

        np.random.seed(1)
        costs = []  # keep track of cost

        if parameters is None:
            # Parameters initialization.
            parameters = self.initialize_parameters_deep(layers_dims,use_He_init=use_He_init)

        # YOUR CODE ENDS HERE

        # Loop (gradient descent)
        for i in range(0, num_iterations):

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            # (≈ 1 line of code)
            # AL, caches = ...
            # YOUR CODE STARTS HERE

            AL, caches = self.L_model_forward(X, parameters,activations)

            # YOUR CODE ENDS HERE

            # Compute cost.
            cost = self.compute_cost(AL, Y)

            # Backward propagation.
            grads = self.L_model_backward(AL, Y,activations, caches,lambd=lambd)

            # YOUR CODE ENDS HERE
            parameters = self.update_parameters(parameters, grads, learning_rate)

            # YOUR CODE ENDS HERE
            LightLogger.do_log("Cost after iteration {} (loop {}): {}".format(i,loop, np.squeeze(cost)))
            # Print the cost every 100 iterations
            if print_cost and i % 100 == 0 or i == num_iterations - 1:
                LightLogger.do_log("Cost after iteration {}(loop {}): {}".format(i,loop, np.squeeze(cost)))
            if i % 100 == 0 or i == num_iterations:
                costs.append(cost)

        return parameters, costs


    def L_layer_model_test(self,X,Y,image_idx,parameters,activations):
        AL, caches = self.L_model_forward(X, parameters,activations)

        Yhat=(AL >= 0.5).astype(int)

        matches = np.sum(Y == Yhat)

        # Calcular la proporción de accuracy
        accuracy = matches / len(Y[0])
        return accuracy

    def build_layers_dims(self,num_features,arch_file):

        if arch_file is not None:
            output_arr=[num_features]

            with open(arch_file.replace('"',""), newline='') as file:
                csv_reader = csv.reader(file, delimiter=',')

                for row in csv_reader:
                    # Convert the values in the row to integers and append them to the list
                    output_arr.extend([int(value) for value in row])


            return  output_arr
        else:
            return [num_features, 20, 7, 5, 1]#just the default network

    def build_activations(self,activations_file):

        if activations_file is not None:
            output_arr=[]

            with open(activations_file.replace('"',""), newline='') as file:
                csv_reader = csv.reader(file, delimiter=',')

                for row in csv_reader:
                    # Convert the values in the row to integers and append them to the list
                    output_arr.extend([str(value) for value in row])


            return  output_arr
        else:
            return ["relu","relu","relu","sigmoid"]#just the default network

    def persist_parameters(self,parameters, activations,output_file):
        clean_output_file=output_file.replace('"','').replace('.npz','')
        parameters["activations"]=activations

        np.savez(clean_output_file, **parameters)

    def retrieve_parameters(self,output_file):
        clean_output_file = output_file.replace('"', '')
        file_paramters = np.load(clean_output_file)

        model_params={}
        for key in file_paramters.keys():
            model_params[key]=file_paramters[key]


        return  model_params





    #endregion