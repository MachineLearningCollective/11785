# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

from flatten import *
from Conv1d import *
from linear import *
from activation import *
from loss import *
import numpy as np
import os
import sys

sys.path.append('mytorch')


class CNN_SimpleScanningMLP():
    def __init__(self):
        # Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # in_channels, out_channels, kernel_size, stride, padding = 0, weight_init_fn=None, bias_init_fn=None
        # <---------------------
        self.conv1 = Conv1d(in_channels=24, out_channels=8, kernel_size=8, stride=4, padding=0)
        self.conv2 = Conv1d(in_channels=8, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv3 = Conv1d(in_channels=16, out_channels=4, kernel_size=1, stride=1, padding=0)

        self.flatten = Flatten()
        self.relu1 = ReLU()
        self.relu2 = ReLU()
        self.layers = [self.conv1, self.relu1, self.conv2, self.relu2, self.conv3, self.flatten] # Add the layers in the correct order

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1, w2, w3 = weights
        # w: (kernel_size * in_channel, out_channel)
        # w1 (192, 8) W1 (8, 24, 8)
        # w2 (8, 16)  W2 (16, 8, 1)
        # w3 (16, 4)  W3 (4, 16, 1)
        # W: (out_channels, in_channels, kernel_size)
        self.conv1.conv1d_stride1.W = w1.reshape(8, 24, 8).transpose(2,1,0)
        self.conv2.conv1d_stride1.W = w2.reshape(1, 8, 16).transpose(2,1,0)
        self.conv3.conv1d_stride1.W = w3.reshape(1, 16, 4).transpose(2,1,0)

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA


class CNN_DistributedScanningMLP():
    def __init__(self):
        # Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        self.conv1 = Conv1d(in_channels=24, out_channels=2, kernel_size=2, stride=2, padding=0) # 4
        self.conv2 = Conv1d(in_channels=2, out_channels=8, kernel_size=2, stride=2, padding=0) # 2
        self.conv3 = Conv1d(in_channels=8, out_channels=4, kernel_size=2, stride=1, padding=0) # 1
        
        self.flatten = Flatten()
        self.relu1 = ReLU()
        self.relu2 = ReLU()
        self.layers = [self.conv1, self.relu1, self.conv2, self.relu2, self.conv3, self.flatten] # Add the layers in the correct order

    def __call__(self, A):
        # Do not modify this method
        return self.forward(A)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # w1 (192, 8) 
        # w2 (8, 16)  
        # w3 (16, 4)  
        # Load them appropriately into the CNN
        # w: (kernel_size * in_channel, out_channel)
        # sliced_w : (kernel_size -- * in_channel, out_channel --)
        w1, w2, w3 = weights
        sliced_w1 = w1[0:48, 0:2]  # (48, 2)
        sliced_w2 = w2[0:4, 0:8]   # (4, 8)
        sliced_w3 = w3             # (16, 4)
        print(sliced_w1.shape)
        print(sliced_w2.shape)
        # W: (out_channels, in_channels, kernel_size)
        # d w1 (2, 24, 2)
        # d w2 (8, 2, 2)
        # d w3 (4, 8, 2)
        self.conv1.conv1d_stride1.W = sliced_w1.reshape(2, 24, 2).transpose(2,1,0)
        self.conv2.conv1d_stride1.W = sliced_w2.reshape(2, 2, 8).transpose(2,1,0)
        self.conv3.conv1d_stride1.W = sliced_w3.reshape(2, 8, 4).transpose(2,1,0)

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA
