# Overview
A simple implementation of a static attention Graph Attention Network for node classification on the BACE classification dataset from MoleculeNet. This repository implements the original GAT paper as found here: https://arxiv.org/abs/1710.10903. Much of this code was adapted from the labmlai repositiry: https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/graphs.

My goal here was to get a simple and functional GAT architecture, something easy to tune and experiment with. While current test set results are subpar (~0.70 ROC AUC on test set) it's a good starting point to modify the architecture/data for experimentation.

## Dataset
BACE is shortform for a specific enzyme BACE-1, or Beta-Secretase 1. The BACE dataset is widely used to predict if novel molecules can inhibit BACE enzyme activity. BACE-1 is closely studied due to its role in Alzheimer's disease; it can produce the offending proteins called beta-amyloid peptides. 

## How to run this script
To train and evaluate on the BACE dataset, simply run main.py. I've included a configuration file which allows the user to change any hyperparameters quickly. After a model is done training, it's weights and associated configuration is automoatically saved in the saved_models directory. 
