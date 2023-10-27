import torch

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset
from functions import save_model_at_checkpoint, load_model_weights, train_eval_model, evaluate_model
from classes.CoraDataSet import CoraDataset
from classes.Configuration import Configuration

# ------------ Data loading ------------ #
# Load Cora
dataset = CoraDataset()

# Split into train and test
train_set, test_set = torch.utils.data.random_split(dataset, [2166, 542])

# Create train and test loaders
# Note batch size is hard coded to 1 for Stochastic Gradient Descent. The training script currently
# does not feature a collator to pad inputs for batch processing > 1.
train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=False)
test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

# ------------ Train & Eval ------------ #
# Load the correct configuration file
config = Configuration("configuration.yaml")
# Train and evaluate
avg_epoch_losses, overall_accuracy = train_eval_model(train_loader, test_loader, config,
                                                      save=True,
                                                      model_weights=None)
# Plot loss curve for training
plt.figure()
plt.plot(overall_accuracy)
plt.xlabel("Epoch Number")
plt.ylabel("Average class accuracies %")
plt.show()
