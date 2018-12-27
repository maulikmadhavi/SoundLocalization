#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 12:46:25 2018

@author: adminnus
"""

import matplotlib.pyplot as plt
import numpy as np
import hdf5storage
import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
import torch.nn.functional as F
import os
from torch.autograd import Variable

torch.manual_seed(7)  # For reproducibility across different computers

from sklearn.preprocessing import minmax_scale
###########################################################################################
# Data preparation
data = hdf5storage.loadmat("Training_GCCPHAT_SSLR.mat")
Xtr = data['Xtr']
Xtr = minmax_scale(Xtr, axis=1)
data = hdf5storage.loadmat("Training_lab.mat")
Ytr = data['Ytr']       # Binary target
Itr = data['Itr']       # indicator for single or 2 source
Ztr = data['Ztr']   # Gaussian-format target


data = hdf5storage.loadmat("Testing1_GCCPHAT_SSLR.mat")
Xte1 = data['Xte1']
Xte1 = minmax_scale(Xte1, axis=1)
data = hdf5storage.loadmat("Testing1_lab.mat")
Yte1 = data['Yte1']  # Binary target
Ite1 = data['Ite1']  # indicator for single or 2 source
Zte1 = data['Zte1']  # Gaussian-format target


data = hdf5storage.loadmat("Testing2_GCCPHAT_SSLR.mat")
Xte2 = data['Xte2']
Xte2 = minmax_scale(Xte2, axis=1)
data = hdf5storage.loadmat("Testing2_lab.mat")
Yte2 = data['Yte2']  # Binary target
Ite2 = data['Ite2']  # indicator for single or 2 source
Zte2 = data['Zte2']  # Gaussian-format target


##############################################################################################
# Create custom loader
class MyDataloaderClass(Dataset):

    def __init__(self, X_data, Y_data):
        self.x_data = X_data
        self.y_data = Y_data
        self.len = X_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


train_loader_obj = MyDataloaderClass(Xtr, Ztr)
train_loader = DataLoader(dataset=train_loader_obj, batch_size=256, shuffle=True, num_workers=1)


def angular_distance_compute(a1, a2):
    return 180 - abs(abs(a1 - a2) - 180)


##########################################################
# Define model and hyperparameters

class Model(torch.nn.Module):

    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(306, 1000)
        self.dense1_bn = torch.nn.BatchNorm1d(1000)
        self.linear2 = torch.nn.Linear(1000, 1000)
        self.dense2_bn = torch.nn.BatchNorm1d(1000)
        self.linear3 = torch.nn.Linear(1000, 1000)
        self.dense3_bn = torch.nn.BatchNorm1d(1000)
        self.linear4 = torch.nn.Linear(1000, 360)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        x1 = F.relu(self.dense1_bn(self.linear1(x)))
        x2 = F.relu(self.dense2_bn(self.linear2(x1)))
        x3 = F.relu(self.dense3_bn(self.linear3(x2)))
        y_pred = F.sigmoid(self.linear4(x3))
        return y_pred


def training(epoch):
    # train my model
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader, 0):
        # print('Batch#' + str(i))
        inputs, target = Variable(data), Variable(target)
        y_pred = model.forward(inputs)
        loss = criterion(y_pred, target)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.data[0]))


def testing(Xte, Yte, Ite):
    model.eval()
    MAE = []
    Y_pred_t = model.forward(torch.from_numpy(Xte))

    for i in range(Xte.shape[0]):
        hyp = Y_pred_t[i].detach().numpy()
        y_1 = Yte[i]

        if Ite[i] == 1:
            a1 = np.where(y_1 == 1)[0]
            b1 = np.argmax(hyp)
            ang = angular_distance_compute(a1, b1)[0]
            MAE.append(ang)
    return sum(MAE) / len(MAE)


# our model
model = Model()

criterion = torch.nn.MSELoss(size_average=True)
optimizer = torch.optim.Adam(model.parameters())


for epoch in range(20):
    # Train
    training(epoch)
    # save model
    modelname = '3layerMLPregre_epoch' + str(epoch) + '.model'
    torch.save(model.state_dict(), modelname)
    # Do testing
    mae1 = testing(Xte1, Yte1, Ite1)
    mae2 = testing(Xte2, Yte2, Ite2)

    print('MAE-set1 =', '{:.3f}'.format(mae1),
          'MAE-set2 =', '{:.3f}'.format(mae2))
