import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class SingleCell(Dataset):
    """Pytorch SingleCell
    produces an iterator that returns a
    dictionary containing a pitch data vector and description as category code

    Attributes:
        data: savant DataFrame containing relevant columns
        pitch: pitch description vector
        state: returned states as category codes
    """

    def __init__(self, data, variables):
        self.data = center_distance(
            sz_dimensions(binary_variables(run_diff(clean_up(data))))
        )
        self.data["description_codes"] = (
            self.data["description"].astype("category").cat.codes
        )
        self.pitch = self.data[variables]
        self.state = self.data["description_codes"]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data = torch.from_numpy(self.pitch.iloc[idx].to_numpy()).float()
        state = torch.tensor(self.state.iloc[idx]).long()
        output = {"pitch_vec": data, "description": state}
        return output