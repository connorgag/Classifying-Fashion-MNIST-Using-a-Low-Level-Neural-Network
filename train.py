import copy
from neuralnet import *
from util import plots
from util import calculateCorrect



def train(model, x_train, y_train, x_valid, y_valid, config, mute=False):

    """
    TODO: Train your model here.
    Implements mini-batch SGD to train the model.
    Implements Early Stopping.
    Uses config to set parameters for training like learning rate, momentum, etc.

    args:
        model - an object of the NeuralNetwork class
        x_train - the train set examples
        y_train - the test set targets/labels
        x_valid - the validation set examples
        y_valid - the validation set targets/labels

    returns:
        the trained model
    """
    batch_size = config['batch_size']
    epochs = config['epochs']
    # early_stop = False
    early_stop = config['early_stop']
    early_stop_epoch = config['early_stop_epoch']
    momentum = config['momentum']

    # Metrics and early stopping
    training_loss_list = []
    training_accuracy_list = []
    val_loss_list = []
    val_accuracy_list = []
    best_val_loss = float('inf')
    patience = 0
    best_model = None

    for i in range(epochs):
        # Shuffle the data
        indices = np.random.permutation(x_train.shape[0])
        x_shuffled = x_train[indices]
        y_shuffled = y_train[indices]

        epoch_loss = 0
        epoch_correct = 0
        epoch_samples = 0


        for j in range(0, x_train.shape[0], batch_size):

            model.layers[0].dw = np.zeros_like(model.layers[0].w)
            
            x_batch = x_shuffled[j:j + batch_size]
            y_batch = y_shuffled[j:j + batch_size]

            # Forward pass
            batch_loss, correct_predictions = model(x_batch, y_batch)

            # print(f"Epoch {i}, Batch {j//batch_size}: Loss: {batch_loss}, Correct: {correct_predictions}")
       
            epoch_loss += batch_loss * x_batch.shape[0]
            epoch_correct += correct_predictions
            epoch_samples += y_batch.shape[0]

            model.backward(gradReqd=True)
            # for idx, layer in enumerate(model.layers):
                # print(f"Layer {idx}, Max Gradient: {np.max(layer.dw)}, Mean Gradient: {np.mean(layer.dw)}")
                # print(f"Weight Norm (Layer {idx}): {np.linalg.norm(layer.w)}")

        # Average loss and accuracy for the epoch
        avg_epoch_loss = epoch_loss / epoch_samples
        train_accuracy = epoch_correct / epoch_samples

        # Evaluate on validation set
        val_loss, val_correct = evaluate(model, x_valid, y_valid)
        val_accuracy = val_correct / x_valid.shape[0]

        # Store metrics
        training_loss_list.append(avg_epoch_loss)
        training_accuracy_list.append(train_accuracy)
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_accuracy)

        # print(f"Epoch {epochs}, Layer {i}: Max Weight Change: {np.max(np.abs(layer.dw))}")
        # print(f"Layer {i}, Max Gradient: {np.max(layer.dw)}, Mean Gradient: {np.mean(layer.dw)}")


        print(f"Epoch {i+1}/{epochs}")
        print(f"Training Loss: {avg_epoch_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}\n")

        # Early stopping
        if early_stop:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                best_model = copy.deepcopy(model)
            else:
                patience += 1
                if patience >= early_stop_epoch:
                    print(f"Early stopping triggered at epoch {i+1}")
                    break
        else:
            best_model = model

    if (mute == False):
        # Plot training curves
        plots(training_loss_list, training_accuracy_list, val_loss_list, val_accuracy_list, 
            earlyStop=i if early_stop and patience >= early_stop_epoch else None)

    return best_model



def evaluate(model, x, y):
    """Evaluate the model on the given dataset."""
    loss, correct = model(x, y)
    # accuracy = correct
    return loss, correct





def modelTest(model, X_test, y_test):
    """
    TODO
    Calculates and returns the accuracy & loss on the test set.

    
    args:
        model - the trained model, an object of the NeuralNetwork class
        X_test - the test set examples
        y_test - the test set targets/labels

    returns:
        test accuracy
        test loss
    """

    

    y_pred = model(X_test)
    corrPred = calculateCorrect(y_pred, y_test)
    test_loss = model.loss(y_pred, y_test)
    test_acc = corrPred/len(y_test)

    return test_acc,test_loss 


