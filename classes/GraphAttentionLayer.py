import torch
from torch import nn


# ============================================================================= #

# Significant credit to
# https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/graphs/gat/__init__.py
# Much of the layer was adopted and modified from the above repository
class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 dropout: float = 0.0,
                 leaky_relu_negative_slope: float = 0.2):
        """
        in_features:    Number of input features per node
        out_features:   Number of output features per node
        n_heads:        Number of attention heads
        is_concat:      Whether the multi-head results should be concatenated or averaged
        dropout:        The dropout probability
        Leaky_relu_negative_slope: Negative slope for leaky relu activation
        """

        super().__init__()

        self.n_heads = n_heads
        self.out_features = out_features

        # First linear transformation with n_head number of weight matrices side by side
        # of size (in_features, out_features)
        self.linear = nn.Linear(in_features, self.out_features * n_heads)
        nn.init.xavier_uniform_(self.linear.weight)

        # linear layer for attention
        self.attn = nn.Linear(self.out_features * 2, 1)
        nn.init.xavier_uniform_(self.attn.weight)

        # LeakyRelu activation
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)

        # Softmax to compute attention
        self.softmax = nn.Softmax(dim=1)

        # Dropout layer in attention
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        """
        h: Input collection of nodes (n x m) where n is # nodes and m is # features
        adj_mat: Matrix describing what nodes are connected to each other
        returns forward pass of nodes through linear layer & attention mechanism
        """
        n_nodes = h.shape[0]

        # First dropout to input node features
        h_dropout = self.dropout(h)

        # Multiply the weight matrices with the node matrices
        # Convert into a 3D tensor of n_head number of W (n_nodes,out_features) matrices
        h_interim = self.linear(h_dropout).view(n_nodes, self.n_heads, self.out_features)

        # Second dropout as in the paper
        h_interim = self.dropout(h_interim)

        # Calculate self-attention and neighbor attention scores
        # What we'd like is a collection of concatenated intermediate node feature vectors
        # Ex: h*1||h*1, h*1||h*2 ... h*n||h*n. Once we get this, it is easy to compute self attention
        # and the neighbor attention scores.

        # Start with repeat. This will give us: [h*1, h*1, h*1..., h*n, h*n]
        h_repeat = h_interim.repeat(n_nodes, 1, 1)

        # Next, use interleave to get [h*1, h*2, h*3...h*1, h*2, h*3...]
        h_repeat_interleave = h_interim.repeat_interleave(n_nodes, dim=0)

        # Concatenate these element wise to get all the permutations.
        h_concat = torch.cat([h_repeat_interleave, h_repeat], dim=-1)

        # Re-arrange them into a matrix such that h_concat[i][j] refers to h*i||h*j
        h_concat = h_concat.view(n_nodes, n_nodes, self.n_heads, 2 * self.out_features)

        # Compute the attention scores
        e = self.activation(self.attn(h_concat))
        # Removes last dimension
        e = e.squeeze(-1)

        # Prepare the adjacency matrix for the softmax function
        # We populate all the zeros to -inf so that our softmax does not take into account
        # non-existent node edges. If we didn't do this, any zero entry would
        # incorrectly contribute to the softmax since e^0 = 1.
        adj_mat = adj_mat.unsqueeze(2).expand(adj_mat.shape[0], adj_mat.shape[0], e.shape[2])
        e = e.masked_fill(adj_mat == 0, float('-inf'))

        # Softmax and computing final h_prime vectors
        a = self.softmax(e)
        h_primes = torch.einsum('ijh,jhf->ihf', a, h_interim)

        # Average the head values
        return h_primes.mean(dim=1)
