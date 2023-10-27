# Overview
A simple implementation of a static attention Graph Attention Network for molecule classification on the BACE dataset from MoleculeNet. This repository implements the original GAT paper as found here: https://arxiv.org/abs/1710.10903. Much of this code was adapted from the labmlai repositiry: https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/graphs.

My goal here was to get a simple and functional GAT architecture, something easy to tune and experiment with. While current test set results are subpar (~0.68 ROC AUC on test set) it's a good starting point to modify the architecture/data for experimentation.

## Dataset
BACE is shortform for a specific enzyme BACE-1, or Beta-Secretase 1. The BACE dataset is widely used to predict if novel molecules can inhibit BACE enzyme activity. BACE-1 is closely studied due to its role in Alzheimer's disease; it can produce the offending proteins called beta-amyloid peptides. The dataset consists of a set of molecules represented as graphs, where each node is an atom with a specific number of atomic features. You can read more about it here: https://moleculenet.org/datasets-1

## Current limitations 
This implementation uses binary adjacency matrices and a static form of attention. A future version of this project aims to implement dynamic attention instead (https://arxiv.org/pdf/2105.14491.pdf) and a bond adjacency matrix which may better characterize the edge relationships between nodes.
## How to run this script
To train and evaluate on the BACE dataset, simply run main.py. I've included a configuration file which allows the user to change any hyperparameters quickly. After a model is done training, it's weights and associated configuration is automatically saved in the saved_models directory. 
