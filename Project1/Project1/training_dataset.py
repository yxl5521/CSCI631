import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class training_dataset(Dataset):
    def __init__(self):
        '''
        read the training data and ground truth.
        implement your code here
        '''
        self.featurestrain =
        self.groundTruthtrain =

        self.len =

    def __getitem__(self, item):
        return self.featurestrain[item], self.groundTruthtrain[item]

    def __len__(self):
        return self.len
