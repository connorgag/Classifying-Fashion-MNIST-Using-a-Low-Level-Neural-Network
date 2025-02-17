from constants import *
from train import train, modelTest
from gradient import check_grad
import argparse
import util 
from neuralnet import Activation, Neuralnetwork

# TODO
def main(args):

    # Read the required config
    # Create different config files for different experiments
    configFile = None  # Will contain the name of the config file to be loaded
    if (args.experiment == 'test_softmax'):  # Rubric #4: Softmax Regression
        configFile = "config_4.yaml"
    elif (args.experiment == 'test_gradients'):  # Rubric #5: Numerical Approximation of Gradients
        configFile = "config_5.yaml"
    elif (args.experiment == 'test_momentum'):  # Rubric #6: Momentum Experiments
        configFile = "config_6.yaml"
    elif (args.experiment == 'test_regularization'):  # Rubric #7: Regularization Experiments
        configFile = "config_7.yaml"  # Create a config file and change None to the config file name
    elif (args.experiment == 'test_activation'):  # Rubric #8: Activation Experiments
        configFile = "config_8.yaml"  # Create a config file and change None to the config file name


    # Load the data
    x_train, y_train, x_valid, y_valid, x_test, y_test = util.load_data(path=datasetDir)
    # # Load the configuration from the corresponding yaml file. Specify the file path and name
    config = util.load_config(configYamlPath + configFile)
    print(config)
    initModel = Neuralnetwork(config)


    if (args.experiment == 'test_gradients'):
        model = train(initModel, x_train, y_train, x_valid, y_valid, config, mute=True)

        # layer_idx = 0 is output layer, layer_idx = 1 is hidden layer, layer_idx = 2 is input layer (depending on layers)
        check_grad(initModel, x_train, y_train, epsilon=1e-2, layer_idx=0, bias=False)


    else:
        model = train(initModel, x_train, y_train, x_valid, y_valid, config, mute=False)

        test_acc,test_loss = modelTest(model, x_test, y_test)

        print(f"The test accuracy is {test_acc}")
        print(f"The test loss is {test_loss}")

if __name__ == "__main__":

    #Parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='test_momentum', help='Specify the experiment that you want to run')
    args = parser.parse_args()
    main(args)
