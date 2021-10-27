import json
from os import listdir

from networks.fully_connected_neural_network import NeuralNetwork

with open("config/config.json") as config_file:
    config = json.load(config_file)


def train(training_set) -> None:
    """ Functions trains networks specified in the config file

    Parameters
    ----------

    Returns
    -------

    """
    for network_config in config["models"]["fully_connected_neural_networks"]:
        model_filename = "{}.pt".format(network_config)

        network = NeuralNetwork(
            config["models"]["fully_connected_neural_networks"][network_config]["input_layer"],
            config["models"]["fully_connected_neural_networks"][network_config]["hidden_layers"],
            config["models"]["fully_connected_neural_networks"][network_config]["output_layer"],
            config["models"]["fully_connected_neural_networks"][network_config]["learning_rate"],
            config["models"]["fully_connected_neural_networks"][network_config]["momentum"],
            config["models"]["fully_connected_neural_networks"][network_config]["weight_decay"]
        )

        if model_filename in listdir("./models/fully_connected_neural_networks"):
            network.load(
                "./models/fully_connected_neural_networks/{}".format(model_filename))

        print("Training model {}".format(network_config))
        network.train_model(training_set)
        network.save(
            "./models/fully_connected_neural_networks/{}".format(model_filename))
