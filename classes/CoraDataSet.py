import numpy as np
import torch
from typing import Dict
from labml import lab, monit
from labml.utils import download
from torch.utils.data import Dataset


# As taken from:
# https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/graphs/gat/experiment.py
class CoraDataset(Dataset):
    """
    ## [Cora Dataset](https://linqs.soe.ucsc.edu/data)

    Cora dataset is a dataset of research papers.
    For each paper we are given a binary feature vector that indicates the presence of words.
    Each paper is classified into one of 7 classes.
    The dataset also has the citation network.

    The papers are the nodes of the graph and the edges are the citations.

    The task is to classify the nodes to the 7 classes with feature vectors and
    citation network as input.
    """
    # Labels for each node
    labels: torch.Tensor
    # Set of class names and an unique integer index
    classes: Dict[str, int]
    # Feature vectors for all nodes
    features: torch.Tensor
    # Adjacency matrix with the edge information.
    # `adj_mat[i][j]` is `True` if there is an edge from `i` to `j`.
    adj_mat: torch.Tensor

    @staticmethod
    def _download():
        """
        Download the dataset
        """
        if not (lab.get_data_path() / 'cora').exists():
            download.download_file('https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz',
                                   lab.get_data_path() / 'cora.tgz')
            download.extract_tar(lab.get_data_path() / 'cora.tgz', lab.get_data_path())

    def __init__(self, include_edges: bool = True):
        """
        Load the dataset
        """

        # Whether to include edges.
        # This is test how much accuracy is lost if we ignore the citation network.
        self.include_edges = include_edges

        # Download dataset
        self._download()

        # Read the paper ids, feature vectors, and labels
        with monit.section('Read content file'):
            content = np.genfromtxt(str(lab.get_data_path() / 'cora/cora.content'), dtype=np.dtype(str))
        # Load the citations, it's a list of pairs of integers.
        with monit.section('Read citations file'):
            citations = np.genfromtxt(str(lab.get_data_path() / 'cora/cora.cites'), dtype=np.int32)

        # Get the feature vectors
        features = torch.tensor(np.array(content[:, 1:-1], dtype=np.float32))
        # Normalize the feature vectors
        self.features = features / features.sum(dim=1, keepdim=True)

        # Get the class names and assign an unique integer to each of them
        self.classes = {s: i for i, s in enumerate(set(content[:, -1]))}
        # Get the labels as those integers
        self.labels = torch.tensor([self.classes[i] for i in content[:, -1]], dtype=torch.long)

        # Get the paper ids
        paper_ids = np.array(content[:, 0], dtype=np.int32)
        # Map of paper id to index
        ids_to_idx = {id_: i for i, id_ in enumerate(paper_ids)}

        # Empty adjacency matrix - an identity matrix
        self.adj_mat = torch.eye(len(self.labels), dtype=torch.bool)

        # Mark the citations in the adjacency matrix
        if self.include_edges:
            for e in citations:
                # The pair of paper indexes
                e1, e2 = ids_to_idx[e[0]], ids_to_idx[e[1]]
                # We build a symmetrical graph, where if paper $i$ referenced
                # paper $j$ we place an adge from $i$ to $j$ as well as an edge
                # from $j$ to $i$.
                self.adj_mat[e1][e2] = True
                self.adj_mat[e2][e1] = True

    def __getitem__(self, index):

        graph = []
        for j, connected in enumerate(self.adj_mat[index]):
            if connected:
                graph.append(self.features[j])

        h = torch.stack(graph)
        self_adj = torch.zeros(h.shape[0], h.shape[0])
        self_adj[0] = 1
        self_adj[:, 0] = 1
        target = torch.zeros(1, len(self.classes))
        target[0][self.labels[index] - 1] = 1

        return h, self_adj, target

    def __len__(self):
        return len(self.features)

