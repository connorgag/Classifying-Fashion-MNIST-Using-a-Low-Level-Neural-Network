import numpy as np
import util
from util import calculateCorrect, append_bias


class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    """

    def __init__(self, activation_type = "sigmoid"):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU", "output"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type

        # Placeholder for input. This can be used for computing gradients.
        self.x = None

    def __call__(self, z):
        """
        This method allows your instances to be callable.
        """
        return self.forward(z)

    def forward(self, z):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(z)

        elif self.activation_type == "tanh":
            return self.tanh(z)

        elif self.activation_type == "ReLU":
            return self.ReLU(z)

        elif self.activation_type == "output":
            return self.output(z)

    def backward(self, z):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            return self.grad_sigmoid(z)

        elif self.activation_type == "tanh":
            return self.grad_tanh(z)

        elif self.activation_type == "ReLU":
            return self.grad_ReLU(z)

        elif self.activation_type == "output":
            return self.grad_output(z)


    def sigmoid(self, x):
        self.x = x
        return 1/(1+np.exp(-self.x))

    def tanh(self, x):
        """
        TODO: Implement tanh here.
        """
        self.x = x
        return np.tanh(self.x, out = None)

    def ReLU(self, x):
        """
        TODO: Implement ReLU here.
        """

        self.x = x
        return np.maximum(0, self.x)


    def output(self, x):
        """
        TODO: Implement softmax function here.
        Remember to take care of the overflow condition (i.e. how to avoid denominator becoming zero).
        
        Input x: 2D list of size batch-size x number of classes, representing scores for each class
        Output: 2D list of size batch-size x number of classes, representing probability of each class for each batch
        Example (without using e^x): [[10, 6, 4], [2, 2, 1]] -> [[.5, .3, .2], [.4, .4, .2]]
        """

        self.x = x

        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  

        softmax_output = exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-8)
        
    
        return softmax_output


    def grad_sigmoid(self, x):
        """
        TODO: Compute the gradient for sigmoid here.
        """

        sigmoid_x = self.sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)
    

    def grad_tanh(self, x):
        """
        TODO: Compute the gradient for tanh here.
        """
        grad_tanh = (1.0 - np.square(np.tanh(x)))
        return grad_tanh
    

    def grad_ReLU(self, x):
        """
        TODO: Compute the gradient for ReLU here.
        """
        grad_relu = (x > 0).astype(float)
        return grad_relu
        # raise NotImplementedError("ReLU gradient not implemented")

    def grad_output(self, x):
        """
        Deliberately returning 1 for output layer case since we don't multiply by any activation for final layer's delta. Feel free to use/disregard it
        """
        return 1 


class Layer():
    """
    This class implements Fully Connected layers for your neural network.
    """

    def __init__(self, in_units, out_units, activation):
        """
        Define the architecture and create placeholders.
        """
        np.random.seed(42)

        # Randomly initialize weights
        self.w = 0.01 * np.random.random((in_units + 1, out_units))
        self.v = 0
        # self.v = np.zeros_like(self.w)
        self.x = None    # Save the input to forward in this
        self.a = None    #output without activation
        self.z = None    # Output After Activation
        self.activation=activation

        self.dw = 0  # Save the gradient w.r.t w in this. w already includes bias term

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        TODO: Compute the forward pass (activation of the weighted input) through the layer here and return it.
        """
        self.x = append_bias(x)
        self.a = np.dot(self.x, self.w)
        self.z = self.activation(self.a)

        return self.z
        # raise NotImplementedError("Forward propagation not implemented for Layer")

    def backward(self, deltaCur, learning_rate, momentum_gamma, momentum,regularization, gradReqd=True, gradient_check=False):
    #     """
    #     TODO
    #     Write the code for backward pass. This takes in gradient from its next layer as input and
    #     computes gradient for its weights and the delta to pass to its previous layers. gradReqd is used to specify whether to update the weights i.e. whether self.w should
    #     be updated after calculating self.dw

    #     The delta expression for any layer consists of delta and weights from the next layer and derivative of the activation function
    #     of weighted inputs i.e. g'(a) of that layer. Hence deltaCur (the input parameter) will have to be multiplied with the derivative of the activation function of the weighted
    #     input of the current layer to actually get the delta for the current layer. Remember, this is just one way of interpreting it and you are free to interpret it any other way.
    #     Feel free to change the function signature if you think of an alternative way to implement the delta calculation or the backward pass

    #     When implementing softmax regression part, just focus on implementing the single-layer case first.
    #     """
    #

     
        delta = deltaCur*self.activation.backward(self.a)
        batch_size = self.x.shape[0]

        # Calculate weight gradients
        self.dw = np.dot(self.x.T, delta) / batch_size

        if regularization:
            self.dw += float(regularization) * self.w

        if gradReqd:
            if momentum:
                self.v = momentum_gamma * self.v + learning_rate * self.dw
                self.w -= self.v
            else:
                self.w -= learning_rate * self.dw

        if gradient_check:
            return self.dw

        # Calculate delta for the previous layer (excluding bias column)
        delta_prev = np.dot(delta, self.w[:-1, :].T)

        return delta_prev

    
        # batch_size = self.x.shape[0]

        # # Apply activation gradient to the incoming delta
        # delta = deltaCur * self.activation.backward(self.a)

        # # Calculate weight gradients
        # self.dw = np.dot(self.x.T, delta) / batch_size  # Corrected delta usage

        # if regularization:
        #     self.dw += regularization * self.w  # Add L2 regularization

        # if gradReqd:
        #     self.w -= learning_rate * self.dw

        # # Calculate delta for the previous layer (excluding bias column)
        # delta_prev = np.dot(delta, self.w.T)  # Exclude the bias term

        # return delta_prev





class Neuralnetwork():
    """
    Create a Neural Network specified by the network configuration mentioned in the config yaml file.
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []  # Store all layers in this list.
        self.num_layers = len(config['layer_specs']) - 1  # Set num layers here
        self.x = None  # Save the input to forward in this
        self.y = None  # For saving the output vector of the model
        self.targets = None  # For saving the targets
        self.batch_size = config['batch_size']
        self.early_stop = config['early_stop']
        self.learning_rate = config['learning_rate']
        self.momentum_gamma = config['momentum_gamma']
        self.L2_pentalty = config['L2_penalty']
        self.momentum = config['momentum']

        # print(f"Learning rate: {self.learning_rate}")

        # Add layers specified by layer_specs.
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1], Activation(config['activation'])))
            elif i  == self.num_layers - 1:
                self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1], Activation("output")))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        # print(f"NN x size: {x.shape}")
        """
        TODO
        Compute forward pass through all the layers in the network and return the loss.
        If targets are provided, return loss and accuracy/number of correct predictions as well.
        
        Args:
            x: Input data.
            targets: Target labels (if provided).
        
        Returns:
            If targets are provided, returns accuracy and loss.
            If targets are not provided, returns the computed output.
        """
        self.x = x
        self.targets = targets
        out = self.x
        for layer in self.layers:
            out = layer.forward(out)
        self.y = out
        
        # for i in self.layers:
        #     self.y = i.forward(self.x)
        
        if targets is not None:
            return self.loss(self.y, self.targets), calculateCorrect(self.y, self.targets)
        else: 
            return self.y


    def loss(self, logits, targets):

        '''
        TODO
        Compute the categorical cross-entropy loss and return it.
        
        Args:
            logits: The predicted logits or probabilities.
            targets: The true target labels.
        
        Returns:
            The categorical cross-entropy loss.

        Input comes in batches for both the logits and the targets.
        We want to minimize the loss.
        '''
       
        epsilon = 1e-15  # Small value to avoid log(0)
        log_logits = np.log(logits + epsilon)  # Apply log with epsilon for stability
        sum_logits = np.sum(targets * log_logits, axis=1)  # Sum across classes
        result = -np.mean(sum_logits)  # Average over all examples
        return result
    


    def backward(self, gradReqd=True, return_grad=False, grad_return_layer=0):
        
        '''
        TODO
        Implement backpropagation here by calling the backward method of Layers class.
        Call backward methods of individual layers.
        
        Args:
            gradReqd: A boolean flag indicating whether to update the weights.
        '''
       
        delta = self.y - self.targets

        # For gradient check
        if return_grad and grad_return_layer is not None:
            print(f"(Gradient Check) Returning the gradient for layer: {grad_return_layer}. Available layers: {[i for i in range(self.num_layers-1,-1,-1)]}")
            for i in range(self.num_layers-1,-1,-1):
                if i == grad_return_layer:
                    model_weights = self.layers[i].backward(delta, self.learning_rate, self.momentum_gamma, self.L2_pentalty, self.momentum, gradReqd = True, gradient_check=True)
                    return model_weights
                else:
                    delta = self.layers[i].backward(delta, self.learning_rate, self.momentum_gamma, self.L2_pentalty, self.momentum, gradReqd = True, gradient_check=False)

            return "Error: Layer not found"

        # For normal backpropagation
        for layer in reversed(self.layers):
            delta = layer.backward(delta, self.learning_rate, self.momentum_gamma, self.momentum, self.L2_pentalty, gradReqd)


    def get_weights_from_layer(self, layer):
        """
        Get the weights of a specific layer.
        """
        return self.layers[layer].w
    

    def set_weights_for_layer(self, layer, weights):
        """
        Set the weights of a specific layer.
        """
        self.layers[layer].w = weights
