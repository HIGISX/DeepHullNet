from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import json

TOKENS = {
    '<sos>': 0,
    '<eos>': 1
}
def load_data(file_name):
    """
    Data will be loaded from json format
    Args:
        file_name: absolute path for the data file
    Return:
        cfg: configuration of the generated data
        data: data generated with ground truth
    """
    assert (os.path.isfile(file_name))
    s_data = json.load(open(file_name, 'r'))
    return  s_data['cfg'], s_data['data'][0]

class Scatter2DDataset(Dataset):
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = []


        cfg, data = load_data(file_name)
        self.cfg = cfg
        self.min_nodes = cfg['min_nodes']
        self.max_nodes = cfg['max_nodes']
        points = data['points']
        targets = data['targets']

        self.points = points
        self.targets = targets

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        points, targets, length = self.pad_item(self.points[idx], self.targets[idx])
        return points, targets, length

    def __len__(self) -> int:
        return len(self.points)

    def pad_item(
            self,
            points: list,
            targets: list
    ) -> Tuple[torch.tensor, torch.Tensor]:
        n_tokens = len(TOKENS)

        points_padded = np.zeros((self.max_nodes + n_tokens, 3 + n_tokens),
                                 dtype=np.float32)
        targets_padded = np.ones((self.max_nodes), dtype=np.int64) \
                         * TOKENS['<eos>']

        points_padded[TOKENS['<sos>'], 2] = 1.0
        points_padded[TOKENS['<eos>'], 3] = 1.0
        points_padded[n_tokens:n_tokens + len(points), :2] = points
        points_padded[n_tokens + len(points):, 4] = 1.0
        targets_padded[:len(targets)] = np.array([t + n_tokens for t in targets])

        points_padded = torch.tensor(points_padded, dtype=torch.float32)
        targets_padded = torch.tensor(targets_padded, dtype=torch.int64)
        length = torch.tensor(len(points) + 2, dtype=torch.int64)
        return points_padded, targets_padded, length