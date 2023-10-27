import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from torch.utils.data import Dataset


# ============================================================================= #
class SolubilityDataSet(Dataset):

    def __init__(self, csv, data_transforms=None, target_transforms=None):
        super(SolubilityDataSet, self).__init__()

        # Grab ESOL data
        self.data = pd.read_csv(csv)

        # Get targets
        self.targets = self.data['measured log solubility in mols per litre']

        # Get adjacency matrices
        self.adjacency = self.data['smiles'].apply(lambda smiles: Chem.GetAdjacencyMatrix(Chem.MolFromSmiles(smiles)))

        # Prune columns that are not features
        self.data = self.data.drop(columns=['Compound ID', 'measured log solubility in mols per litre', 'smiles'])

        # Optional transforms if provided
        self.data_transforms = data_transforms
        self.target_transforms = target_transforms

    def __getitem__(self, index):

        # Grab data, adjacency matrices and targets + convert to tensor
        data = torch.as_tensor(self.data.iloc[index], dtype=torch.float32)
        adjacency = torch.as_tensor(self.adjacency.iloc[index], dtype=torch.float32)
        target = data = torch.as_tensor(self.targets.iloc[index], dtype=torch.float32)

        # Apply optional transforms to data and target if provided
        if self.data_transforms:
            data = self.data_transforms(data)
        if self.target_transforms:
            target = self.target_transforms(target)

        return data, adjacency, target

    def __len__(self):
        return len(self.data)
