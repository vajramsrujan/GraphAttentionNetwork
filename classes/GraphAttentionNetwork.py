from torch import nn, Tensor, mean
from .GraphAttentionLayer import GraphAttentionLayer


# ============================================================================= #
class GraphAttentionNetwork(nn.Module):

    def __init__(self, node_features: int, n_hidden: int, n_classes: int, n_heads: int, dropout: float):

        """
        :param node_features:   Number of features per node
        :param n_hidden:        Number of features in first layer
        :param n_classes:       Number of output classes
        :param n_heads:         Number of attention heads
        """

        super().__init__()

        # Define the first layer
        self.layer1 = GraphAttentionLayer(node_features, n_hidden, n_heads, dropout=dropout)

        # # Layer 2
        # self.layer2 = GraphAttentionLayer(n_hidden, n_hidden, n_heads, dropout=dropout)

        # Layer 3
        self.output_layer = GraphAttentionLayer(n_hidden, n_classes, n_heads, dropout=dropout)

        # Define an activation layer for the first layer
        self.activation_1 = nn.ELU()

        # Activation for the final layer
        self.activation_2 = nn.Softmax()

        # Define a dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: Tensor, adj_mat: Tensor):

        # Layer 1 forward pass (n X n_hidden)
        h = self.layer1(h, adj_mat)
        h = self.activation_1(h)

        # # Layer 2
        # h = self.layer2(h, adj_mat)
        # h = self.activation_1(h)

        # Output
        h = self.output_layer(h, adj_mat)

        # Average nodes from n X n_classes to  1 X n_classes
        h = mean(h, dim=0, keepdim=True)

        # softmax to scale all classes to probabilities between 0 and 1
        h = self.activation_2(h)

        return h
