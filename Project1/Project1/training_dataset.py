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
        training = pd.read_csv("./Dataset/trainingData.csv", header=None)
        ground = pd.read_csv("./Dataset/newground-truth.csv", header=None)
        self.featurestrain = torch.tensor(training.to_numpy())
        self.groundTruthtrain = torch.tensor(ground.to_numpy())

        self.len = len(training)

    def __getitem__(self, item):
        return self.featurestrain[item], self.groundTruthtrain[item]

    def __len__(self):
        return self.len
