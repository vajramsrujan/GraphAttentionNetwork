import torch

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from functions import train_eval_model
from classes.BaceDataSet import BaceDataSet
from classes.Configuration import Configuration
from torch_geometric.datasets import MoleculeNet

# ------------ Data loading ------------ #
# Load MoleculeNet
dataset = MoleculeNet(root="src/data", name="BACE")
processed_dataset = BaceDataSet(dataset)

# Split into train and test
train_set, test_set = torch.utils.data.random_split(processed_dataset, [1134, 379])

# Create train and test loaders
# Note batch size is hard coded to 1 for Stochastic Gradient Descent. The training script currently
# does not feature a collator to pad inputs for batch processing > 1.
train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=False)
test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

# ------------ Train & Eval ------------ #
# Load the correct configuration file
config = Configuration("src/configuration.yaml")
# Train and evaluate
avg_epoch_losses, roc_auc_over_epochs = train_eval_model(train_loader, test_loader, config,
                                                         save=True,
                                                         model_weights=None)
# Plot loss curve for training
plt.figure()
plt.plot(roc_auc_over_epochs)
plt.xlabel("Epoch Number")
plt.ylabel("ROC AUC per epoch")
plt.show()
