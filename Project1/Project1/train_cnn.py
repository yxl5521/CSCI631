import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from functions import overlapScore

from cnn_model import *
from training_dataset import *

'''
In this part we will train the model.
'''


def train_model(net, dataloader, batchSize, lr_rate, momentum, Epoch_num):
    '''
    training process of this model
    :param net: model
    :param dataloader: dataloader
    :param batchSize: batch size
    :param lr_rate: learning rate
    :param momentum: momentum
    :param Epoch_num: epoch number
    :return:
    '''
    '''
    setup loss function(mean squared error loss)
    optimization (stochastic gradient descent)
    scheduler(hint: optim.lr_scheduler.StepLR(), step_size=30, gamma=0.1)
    '''
    # implement your code here:
    criterion = nn.MSELoss()
    optimization = torch.optim.SGD(net.parameters(), lr=lr_rate, momentum=momentum)
    scheduler = optim.lr_scheduler.StepLR(optimization, step_size=30, gamma=0.1)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net.to(device)
    '''
    loop for training
    '''
    # implement your code here:
    for epoch in range(Epoch_num):

        scheduler.step()
        losses = []
        avgScores = 0
        for i, data in enumerate(dataloader):
            # clear the gradients for all optimized variables
            optimization.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            inputs, labels = data
            inputs, labels = (inputs.view(batchSize, 1, 100, 100)).to(device), (labels.view(batchSize, 4)).to(device)
            outputs = net(inputs)
            # calculate the loss
            loss = criterion(outputs, labels)
            losses.append(outputs.shape[0] * loss.item())
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimization.step()

            # calculate the score using overlapScore function
            pbox = outputs.cpu().detach().numpy()
            gbox = labels.cpu().detach().numpy()
            avgScore, scores = overlapScore(pbox, gbox)
            avgScores += avgScore

        '''
        print out epoch, loss and average score in following format
        epoch     1, loss: 426.835693, Average Score = 0.046756
        '''
        print('epoch     {epoch}, loss: {loss:.6f}, Average Score = {avg_score:.6f}'.format(epoch=epoch + 1,
                                                                                            loss=np.mean(losses),
                                                                                            avg_score=avgScores / len(
                                                                                                dataloader.sampler)))

    print('Finish Training')


if __name__ == '__main__':
    # hyper parameters
    # implement your code here
    learning_rate = 0.000008
    momentum = 0.9
    batch = 4
    no_of_workers = torch.get_num_threads()
    shuffle = True
    epoch = 50

    # load dataset
    # implement your code here
    data = training_dataset()

    # setup dataloader
    # implement your code here
    dataLoader = DataLoader(dataset=data, batch_size=batch, shuffle=shuffle, num_workers=no_of_workers)

    model = cnn_model()
    model.train()
    train_model(model, dataLoader, batch, learning_rate, momentum, epoch)
    # save model
    torch.save(model.state_dict(), 'model.pth')
