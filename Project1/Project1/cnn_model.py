import torch.nn as nn
import torch.nn.functional as f
class cnn_model(nn.Module):
    '''
    The CNN model contains three convolutional layers, one fully connected layer and one output layer with 4 nodes.
A kernel of size 5 with stride 1 is applied in each convolution layer. The first two convolutional layers are followed by a max-pooling layer with kernel size 2 and stride 2.
We need to set up a dropout rate (0.5) on the fully connected layer.
All inner layers are activated by ReLU function.
    '''
    def __init__(self):

        super(cnn_model, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=0
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=0
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=5,
            stride=1,
            padding=0
        )


        self.fc1 = nn.Linear(
            in_features=18*18*128,
            out_features=2046
        )

        self.fc2 = nn.Linear(
            in_features=2046,
            out_features=4
        )

    def forward(self, val):


        # implement your code here:


        return val