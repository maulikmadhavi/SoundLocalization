#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  Jan 19 12:46:25 2018

@author: adminnus
"""

import matplotlib.pyplot as plt
import numpy as np
import hdf5storage
import torch
#from torch.utils.data import Dataset, DataLoader
#import tensorflow as tf
import torch.nn.functional as F
import os
from torch.autograd import Variable

torch.manual_seed(7)  # For reproducibility across different computers
torch.cuda.manual_seed(7)
#from sklearn.preprocessing import minmax_scale
numtrain = 1567
numtest = 208
numepoch = 10
###########################################################################################
# Data preparation


def minmax_norm2d(data_in):
    dmin = data_in.min(axis=0).min(axis=0)
    dmax = data_in.max(axis=0).max(axis=0)

    data_out = ((data_in - dmin) / (dmax - dmin))
    return data_out


data = hdf5storage.loadmat("Testing1_melGCCPHAT_SSLR.mat")
Xte1 = data['Xte1']
Xte1 = minmax_norm2d(Xte1)
Xte1 = np.swapaxes(Xte1, 1, 3)
Yte1 = data['Yte1']  # Binary target
Ite1 = data['Ite1']  # indicator for single or 2 source

"""
data = hdf5storage.loadmat("Testing2_melGCCPHAT_SSLR.mat")
Xte2 = data['X']
Xte2 = minmax_norm2d(Xte2)
Xte2 = np.swapaxes(Xte2, 1, 3)
Yte2 = data['Y']  # Binary target
Ite2 = data['I']  # indicator for single or 2 source
"""


def angular_distance_compute(a1, a2):
    return 180 - abs(abs(a1 - a2) - 180)


class Flatten(torch.nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)
##########################################################
# Define model and hyperparameters


class Model(torch.nn.Module):

    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(6, 12, kernel_size=5, stride=2, padding=1)
        self.dense1_bn = torch.nn.BatchNorm2d(12)
        self.conv2 = torch.nn.Conv2d(12, 24, kernel_size=5, stride=2, padding=1)
        self.dense2_bn = torch.nn.BatchNorm2d(24)
        self.conv3 = torch.nn.Conv2d(24, 48, kernel_size=5, stride=2, padding=1)
        self.dense3_bn = torch.nn.BatchNorm2d(48)
        self.conv4 = torch.nn.Conv2d(48, 96, kernel_size=5, stride=2, padding=1)
        self.dense4_bn = torch.nn.BatchNorm2d(96)
        self.linear4 = torch.nn.Linear(192, 360)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        x1 = self.dense1_bn(F.relu(self.conv1(x)))
        x2 = self.dense2_bn(F.relu(self.conv2(x1)))
        x3 = self.dense3_bn(F.relu(self.conv3(x2)))
        x4 = self.dense4_bn(F.relu(self.conv4(x3)))
        #x4 = x4.view(-1)
        flat = Flatten()
        x4 = flat.forward(x4)
        y_pred = torch.sigmoid(self.linear4(x4))
        return y_pred


def training(epoch):
    # train my model
    model.train()

    for batch_idx in range(numtrain):
        print(batch_idx)
        data = hdf5storage.loadmat("melGCCBatch/Train_batch" + str(batch_idx + 1) + ".mat")
        Xtr = data['Xtr']
        Xtr = minmax_norm2d(Xtr)
        Xtr = np.swapaxes(Xtr, 1, 3)

        # Ytr = data['Ytr']  # Binary target
        # Itr = data['Itr']  # indicator for single or 2 source
        Ztr = data['Ztr']  # Gaussian-format target

        # print('Batch#' + str(i))
        Xtr = torch.from_numpy(Xtr)
        Ztr = torch.from_numpy(Ztr)
        inputs, target = Variable(Xtr).type(torch.FloatTensor).cuda(), Variable(Ztr).type(torch.FloatTensor).cuda()
        y_pred = model.forward(inputs)
        loss = criterion(y_pred, target)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx,
                                                                           numtrain * 256, 100. * batch_idx / numtrain, loss.data[0]))


def testing(Xte, Yte, Ite):
    model.eval()
    MAE = []
    Xte = minmax_norm2d(Xte)
    Y_pred_t = model.forward(torch.from_numpy(Xte).type(torch.FloatTensor).cuda())  # Numpy to torch data conversion

    for i in range(Xte.shape[0]):
        hyp = Y_pred_t[i].cpu().detach().numpy()  # Torch to numpy data conversion
        y_1 = Yte[i]
        if Ite[i] == 1:

            a1 = np.where(y_1 == 1)[0]
            b1 = np.argmax(hyp)
            ang = angular_distance_compute(a1, b1)[0]
            MAE.append(ang)
    return MAE


# our model
model = Model()
model = model.cuda()

criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(numepoch):
    model.train()
    for batch_idx in range(numtrain):  # 1567
        data = hdf5storage.loadmat(
            "melGCCBatch/Train_batch" + str(batch_idx + 1) + ".mat")
        Xtr = data['Xtr']
        Xtr = minmax_norm2d(Xtr)
        Xtr = np.swapaxes(Xtr, 1, 3)

        # Ytr = data['Ytr']  # Binary target
        # Itr = data['Itr']  # indicator for single or 2 source
        Ztr = data['Ztr']  # Gaussian-format target

        # print('Batch#' + str(i))
        Xtr = torch.from_numpy(Xtr)
        Ztr = torch.from_numpy(Ztr)
        inputs, target = Variable(Xtr).type(torch.FloatTensor).cuda(), Variable(Ztr).type(torch.FloatTensor).cuda()
        y_pred = model.forward(inputs)
        loss = criterion(y_pred, target)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx,
                                                                           numtrain, 100. * batch_idx / numtrain, loss.data))
    model.eval()
    MAE1 = testing(Xte1, Yte1, Ite1)
    mae1 = sum(MAE1) / len(MAE1)
    print('MAE-set1 =', '{:.3f}'.format(mae1))

    MAE2_all = []
    for batch_idx in range(numtest):  # 208
        data = hdf5storage.loadmat(
            "melGCCBatch/Test_batch" + str(batch_idx + 1) + ".mat")
        Xte = data['X']
        Xte = minmax_norm2d(Xte)
        Xte = np.swapaxes(Xte, 1, 3)
        Yte = data['Y']  # Binary target
        Ite = data['I']  # indicator for single or 2 source

        Ite = Ite.astype(np.uint8)
        MAE2 = testing(Xte, Yte, Ite)
        MAE2_all.append(MAE2)

    mae2 = np.mean(MAE2_all[0])

    try:
        print('MAE-set2 =', '{:.3f}'.format(mae2))
    except TypeError:
        print('mae2 note working')

   # mae2 = testing(Xte2, Yte2, Ite2)
   # print('MAE-set2 =', '{:.3f}'.format(mae2))
