import os
import torch
import shutil
import numpy as np
from tqdm import tqdm
from torch import optim, nn
from torch.utils.data import DataLoader
from classes.Configuration import Configuration
from classes.GraphAttentionNetwork import GraphAttentionNetwork


def instantiate_model(config: Configuration):
    """
    config: Configuration object as loaded using Configuration()
    return: GAT model object
    """
    # Use GPU if available else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load hyper parameters
    node_features = config.get('experiment/training/node_features', 30)
    n_classes = config.get('experiment/training/n_classes', 1)
    n_hidden = config.get('experiment/training/n_hidden', 16)
    n_heads = config.get('experiment/training/n_heads', 2)
    dropout = config.get('experiment/training/dropout', 1)

    # Initialize GAT
    model = GraphAttentionNetwork(node_features=node_features, n_hidden=n_hidden, n_classes=n_classes,
                                  n_heads=n_heads,
                                  dropout=dropout).to(device)

    return model


def load_model_weights(tar_file_path: str, config: Configuration):
    """
    tar_file_path:  String path to tar file for model weights
    config:         Configuration object
    return:         GAT model object
    """

    model = instantiate_model(config)
    learning_rate = config.get('experiment/training/learning_rate', 0.001)

    # Load optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Load model parameters using fil.tar filename
    checkpoint = torch.load(tar_file_path, map_location=torch.device('cpu'))  # map_location sets mapping to cpu
    # Load the states of the model and optimizer
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optim_state'])

    return model, optimizer


def save_model_at_checkpoint(state: dict, config: Configuration):
    """
    state:  Dictionary of model state
    config: Configuration object
    return: None
    """

    # Where to save model
    save_path = config.get('experiment/save/save_path', "save_path")

    # Set filenames for saving
    # This is the configuration file you modify in the main directory where main.py exists
    base_config_filename = "configuration.yaml"
    # What we name the model weights file to
    model_filename = save_path + "/model_" + config.get('experiment/meta/name', "unlabelled") + ".pth.tar"
    # What we want to rename our copy configuration file when we save the models weights
    new_config_filename = "config_" + config.get('experiment/meta/name', "unlabelled") + ".yaml"

    # Save model
    torch.save(state, model_filename)
    # Create copy of config file with experiment name
    destination_file = os.path.join("saved_models", new_config_filename)
    shutil.copy2(base_config_filename, destination_file)

    return


def train_eval_model(train_dataloader: DataLoader, eval_dataloader: DataLoader,
                     config: Configuration, save: bool, model_weights=None):
    """
    model_weights:      Can pass model weights for a specific run as a .tar file
    save:               bool flag to save the model weights and config file
    eval_dataloader:    The dataloader to evaluate
    train_dataloader:   The dataloader to train on
    config:             Configuration object
    :return:            Tuple of average losses and average accuracies over each epoch
    """

    # Empty cuda cache memory
    torch.cuda.empty_cache()
    # Use GPU if available else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not model_weights:
        model = instantiate_model(config)
    else:
        model, _ = load_model_weights(model_weights, config)

    # Load hyper parameters
    num_epochs = config.get('experiment/training/num_epochs', 10)
    learning_rate = config.get('experiment/training/learning_rate', 0.001)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train Network
    losses = np.zeros((1, len(train_dataloader))).flatten()
    avg_epoch_losses = np.zeros((1, num_epochs)).flatten()
    avg_accuracy = np.zeros((1, num_epochs)).flatten()

    for i, epoch in enumerate(range(num_epochs)):
        for batch_index, (data, adjacency, targets) in enumerate(tqdm(train_dataloader)):
            # Get rid of the extra dimension the dataloader adds (since batch size = 1)
            data = data[0]
            adjacency = adjacency[0]
            targets = targets[0]

            # Get data to cuda if possible
            data = data.to(device=device)
            adjacency = adjacency.to(device=device)
            targets = targets.to(device=device)

            # Forward pass
            scores = model(data, adjacency)

            # Compute loss
            loss = criterion(scores, targets)
            losses[batch_index] = loss.item()

            # Backward propagation
            optimizer.zero_grad()  # Make sure to reset gradients
            loss.backward()  # Back prop

            # Gradient descent step with optimizer
            optimizer.step()

        # Save average epoch loss
        avg_epoch_losses[epoch] = np.mean(losses)
        avg_accuracy[epoch] = evaluate_model(eval_dataloader, model)
        print("\n")
        print("Epoch: " + str(epoch))
        print(avg_accuracy[epoch])

        # Upon training completion, save model as .tar file
        if epoch == num_epochs - 1 and save:
            state = {"model_state": model.state_dict(), "optim_state": optimizer.state_dict()}
            save_model_at_checkpoint(state, config)

    return avg_epoch_losses, avg_accuracy


def evaluate_model(dataloader: DataLoader, model: GraphAttentionNetwork):
    """
    dataloader:     The dataloader object to evaluate
    model:          The model object
    return:         Accuracy of the evaluation dataset
    """

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set model to evaluation mode. Prevents things like dropout from occurring
    # which are only meant to b e used during the training phase
    model.eval()

    # Instantiate a list of model predictions and truth labels to populate
    model_predictions = [0] * len(dataloader)
    actual_target = [0] * len(dataloader)

    # torch.no_grad ensures we are not computing gradients
    with torch.no_grad():
        for batch_index, (data, adjacency, target) in enumerate(tqdm(dataloader)):
            # Send data to device
            data = data[0]
            adjacency = adjacency[0]
            target = target[0]

            # Compute predictions
            score = model(data, adjacency)
            model_predictions[batch_index] = int(torch.argmax(score))
            # Grab the truth label
            actual_target[batch_index] = int(torch.argmax(target))

    model.train()
    num_correct = 0
    # Tally up how many labels the model got correct and divide by total number of labels
    for i in range(len(model_predictions)):
        if model_predictions[i] == actual_target[i]:
            num_correct += 1

    overall_accuracy = (num_correct / len(model_predictions)) * 100

    return overall_accuracy
