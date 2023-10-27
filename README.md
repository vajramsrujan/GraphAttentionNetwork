# Summary
A simple implementation of a static attention Graph Attention Network for node classification onthe Cora dataset. This repository implements the original GAT paper as found here: https://arxiv.org/abs/1710.10903. Much of this code was adapted from the labmlai repositiry: https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/graphs

## How to run this script
To train and evaluate on the Cora dataset, simply run main.py. I've included a configuration file which allows the user to change any hyperparameters quickly. After a model is done training, it's weights and associated configuration is automoatically saved in the saved_models directory. 
