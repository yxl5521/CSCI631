import numpy as np
import pandas as pd

from functions import overlapScore

import torch

from cnn_model import cnn_model


# load test data
# implement your code here



# load model.pth and test model

model = cnn_model()
model.eval()
model.load_state_dict(torch.load('model.pth'))

# reshape your text data and feed into your model
# implement your code here


# use overlapscore function to calculate the average score
# implement your code here

# save your output in a csv file in Result directory and draw an example with bounding box
# implement your code here



