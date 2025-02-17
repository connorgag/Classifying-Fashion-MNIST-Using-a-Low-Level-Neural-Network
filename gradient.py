import numpy as np
from neuralnet import Neuralnetwork
import random
import copy

def check_grad(model, x_train, y_train, epsilon=1e-2, layer_idx=0, bias=False):

    """
    TODO
    Checks if gradients computed numerically are within O(epsilon**2)

    Args:
        model: The neural network model to check gradients for.
        x_train: Small subset of the original train dataset.
        y_train: Corresponding target labels of x_train.
        epsilon: Small constant for numerical approximation.

    Prints gradient difference of values calculated via numerical approximation and backprop implementation.
    """    
    # Grab a random example from the training set
    example_index = random.randint(0, len(x_train)-1)
    x_train = np.array([x_train[example_index]])
    y_train = np.array([y_train[example_index]])

    # Get the initial weights
    init_weights = model.get_weights_from_layer(layer_idx)

    # Set random weight to check
    if bias:
        x_index =(init_weights.shape[0] - 1)
    else:
        x_index = random.randint(0, init_weights.shape[0] - 2)
        
    y_index = random.randint(0, init_weights.shape[1]-1)


    # --Manually compute gradient for a random weight in the layer--

    # Increase one of the weights by a small amount, compute loss after forward pass
    plus_model = copy.deepcopy(model)
    weights_plus_epsilon = plus_model.get_weights_from_layer(layer_idx)
    weights_plus_epsilon[x_index][y_index] = weights_plus_epsilon[x_index][y_index] + epsilon
    plus_model.set_weights_for_layer(layer_idx, weights_plus_epsilon)
    plus_loss, _ = plus_model.forward(x_train, y_train)


    # Decrease one of the weights by a small amount, compute loss after forward pass
    minus_model = copy.deepcopy(model)
    weights_minus_epsilon = minus_model.get_weights_from_layer(layer_idx)
    weights_minus_epsilon[x_index][y_index] = weights_minus_epsilon[x_index][y_index] - epsilon
    minus_model.set_weights_for_layer(layer_idx, weights_minus_epsilon)
    minus_loss, _ = minus_model.forward(x_train, y_train)

    computed_gradient = (plus_loss - minus_loss) / (2.0 * epsilon)


    #-- Our model's backward pass gradient --
    backprop_model = copy.deepcopy(model)
    backprop_model.forward(x_train, y_train)
    layer_weights = backprop_model.backward(gradReqd=True, return_grad=True, grad_return_layer=layer_idx)

    print(f"(Gradient Check) Epsilon: {epsilon}")
    print(f"(Gradient Check) Testing example {example_index} for the gradient for weight at index ({x_index}, {y_index})")
    print(f"Bias weight: {bias}")
    print("(Gradient Check) Computed gradient: ", computed_gradient)
    print("(Gradient Check) Our backward pass gradient: ", layer_weights[x_index][y_index])
    print("(Gradient Check) Difference: ", abs(computed_gradient - layer_weights[x_index][y_index]))


def checkGradient(x_train, y_train, config):
    raise NotImplementedError("checkGradient not implemented in gradient.py")
