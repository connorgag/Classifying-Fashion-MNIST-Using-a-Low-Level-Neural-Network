import copy
import os, gzip
import yaml
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import constants
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt 
import os



def load_config(path):
    """
    Loads the config yaml from the specified path

    args:
        path - Complete path of the config yaml file to be loaded
    returns:
        yaml - yaml object containing the config file
    """
    return yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)



def normalize_data(inp):
    
    mean = np.mean(inp, axis=1, keepdims=True)
    std = np.std(inp, axis=1, keepdims=True) + 1e-9 
    return (inp - mean) / std


def one_hot_encoding(labels, num_classes=10):
    """
    TODO
    Encodes labels using one hot encoding.

    args:
        labels : N dimensional 1D array where N is the number of examples
        num_classes: Number of distinct labels that we have (10 for FashionMNIST)

    returns:
        oneHot : N X num_classes 2D array
    """ 
   
    ohe_labels = np.zeros((labels.size, num_classes))

    # Fill in the one-hot encoded values
    for i in range(len(labels)):
        index = int(labels[i]) 
        ohe_labels[i, index] = 1  

    return ohe_labels



def generate_minibatches(dataset, batch_size=64):
    """
        Generates minibatches of the dataset

        args:
            dataset : 2D Array N (examples) X d (dimensions)
            batch_size: mini batch size. Default value=64

        yields:
            (X,y) tuple of size=batch_size

        """

    X, y = dataset
    l_idx, r_idx = 0, batch_size
    while r_idx < len(X):
        yield X[l_idx:r_idx], y[l_idx:r_idx]
        l_idx, r_idx = r_idx, r_idx + batch_size

    yield X[l_idx:], y[l_idx:]


def calculateCorrect(y, t):  #Feel free to use this function to return accuracy instead of number of correct predictions
    """
    TODO
    Calculates the number of correct predictions

    args:
        y: Predicted Probabilities
        t: Labels in one hot encoding

    returns:
        the number of correct predictions
    """
    corrPred = np.sum(np.argmax(t, axis = 1) == np.argmax(y, axis = 1))

    return corrPred
    
    # raise NotImplementedError("calculateCorrect not implemented")



def append_bias(X):
    """
    TODO
    Appends bias to the input
    args:
        X (N X d 2D Array)
    returns:
        X_bias (N X (d+1)) 2D Array
    """
    bias = np.ones((X.shape[0], 1))
    
    X_bias = np.hstack((X, bias))

    return X_bias


def plots(trainEpochLoss, trainEpochAccuracy, valEpochLoss, valEpochAccuracy, earlyStop=None):

    """
    Helper function for creating the plots
    """
    if not os.path.exists(constants.saveLocation):
        os.makedirs(constants.saveLocation)

    fig1, ax1 = plt.subplots(figsize=((24, 12)))
    epochs = np.arange(1,len(trainEpochLoss)+1,1)
    ax1.plot(epochs, trainEpochLoss, 'r', label="Training Loss")
    ax1.plot(epochs, valEpochLoss, 'g', label="Validation Loss")
    if earlyStop != None:
        plt.scatter(epochs[earlyStop],valEpochLoss[earlyStop],marker='x', c='g',s=400,label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs),max(epochs)+1,10), fontsize=35 )
    plt.yticks(fontsize=35)
    ax1.set_title('Loss Plots', fontsize=35.0)
    ax1.set_xlabel('Epochs', fontsize=35.0)
    ax1.set_ylabel('Cross Entropy Loss', fontsize=35.0)
    ax1.legend(loc="upper right", fontsize=35.0)
    # plt.savefig(constants.saveLocation+"loss.eps")
    plt.show()

    fig2, ax2 = plt.subplots(figsize=((24, 12)))
    ax2.plot(epochs, trainEpochAccuracy, 'r', label="Training Accuracy")
    ax2.plot(epochs, valEpochAccuracy, 'g', label="Validation Accuracy")
    if earlyStop != None:
        plt.scatter(epochs[earlyStop], valEpochAccuracy[earlyStop], marker='x', c='g', s=400, label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs),max(epochs)+1,10), fontsize=35)
    plt.yticks(fontsize=35)
    ax2.set_title('Accuracy Plots', fontsize=35.0)
    ax2.set_xlabel('Epochs', fontsize=35.0)
    ax2.set_ylabel('Accuracy', fontsize=35.0)
    ax2.legend(loc="lower right", fontsize=35.0)
    # plt.savefig(constants.saveLocation+"accuarcy.eps")
    plt.show()


    

    #Saving the losses and accuracies for further offline use
    pd.DataFrame(trainEpochLoss).to_csv(constants.saveLocation+"trainEpochLoss.csv")
    pd.DataFrame(valEpochLoss).to_csv(constants.saveLocation+"valEpochLoss.csv")
    pd.DataFrame(trainEpochAccuracy).to_csv(constants.saveLocation+"trainEpochAccuracy.csv")
    pd.DataFrame(valEpochAccuracy).to_csv(constants.saveLocation+"valEpochAccuracy.csv")


def createTrainValSplit(x_train,y_train):

    """
    TODO
    Creates the train-validation split (80-20 split for train-val). Please shuffle the data before creating the train-val split.
    """
    val_split = 0.2  # Proportion for validation
    seed = 42  # Random seed for reproducibility

    np.random.seed(seed)

    # Generate a shuffled index
    indices = np.random.permutation(len(x_train))

    # Determine the split index
    split_idx = int(len(x_train) * (1 - val_split))

    # Split data and labels based on shuffled indices
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    train_images, val_images = x_train[train_indices], x_train[val_indices]
    train_labels, val_labels = y_train[train_indices], y_train[val_indices]

    return train_images, train_labels, val_images, val_labels




def get_data():

    """
    Downloads FashionMNIST from Torch's dataset collection
    args:
    returns:
        dataset objects for both train and test data
    """

    train_data = datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor()
    )

    return train_data, test_data


def load_data(path):
    """
    Loads, splits our dataset into train, val and test sets and normalizes them

    args:
        path: Path to dataset
    returns:
        train_normalized_images, train_one_hot_labels, val_normalized_images, val_one_hot_labels,  test_normalized_images, test_one_hot_labels

    """
    if not os.path.exists(path):
        os.makedirs(path)
        print('DOWNLOADING DATA...')
        train_data, test_data = get_data() # these are Dataset objects in PyTorch


        train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

        # Store and convert data to numpy

        train_im_list = []
        train_lab_list = []
        for im, lab in train_dataloader:

            im = im.squeeze().numpy()

            train_im_list.append(im)
            train_lab_list.append(lab.item())

        train_im = np.array(train_im_list)
        train_lab = np.array(train_lab_list)

        test_im_list = []
        test_lab_list = []
        for im, lab in test_dataloader:

            im = im.squeeze().numpy()

            test_im_list.append(im)
            test_lab_list.append(lab.item())

        test_im = np.array(test_im_list)
        test_lab = np.array(test_lab_list)

        path = 'data/numpy/'
        os.makedirs(os.path.dirname(path), exist_ok=True)

        np.save(f'{path}/train_features.npy', train_im)
        np.save(f'{path}/train_labels.npy', train_lab)

        np.save(f'{path}/test_features.npy', test_im)
        np.save(f'{path}/test_labels.npy', test_lab)

        print(f'DATA DOWNLOAD COMPLETE.')

    # Load data from pickle file
    print(f'LOADING DATA ...')
    train_images = np.load('data/numpy/train_features.npy')
    train_labels = np.load('data/numpy/train_labels.npy')
    test_images = np.load('data/numpy/test_features.npy')
    test_labels = np.load('data/numpy/test_labels.npy')
    print('DATA LOADING COMPLETE.\n')

    # Reformat the images and labels
    train_images, test_images = train_images.reshape(train_images.shape[0], -1), test_images.reshape(test_images.shape[0], -1)
    train_labels, test_labels = np.expand_dims(train_labels, axis=1), np.expand_dims(test_labels, axis=1)

    # Create 80-20 train-validation split
    train_images, train_labels, val_images, val_labels = createTrainValSplit(train_images, train_labels)

    # Preprocess data
    
    train_normalized_images = normalize_data(train_images)
    train_one_hot_labels = one_hot_encoding(train_labels, num_classes=10)  # (n, 10)
    
    val_normalized_images = normalize_data(val_images)
    val_one_hot_labels = one_hot_encoding(val_labels, num_classes=10)  # (n, 10)
    
    test_normalized_images = normalize_data(test_images)
    test_one_hot_labels = one_hot_encoding(test_labels, num_classes=10)  # (n, 10)


    return train_normalized_images, train_one_hot_labels, val_normalized_images, val_one_hot_labels, test_normalized_images, test_one_hot_labels
