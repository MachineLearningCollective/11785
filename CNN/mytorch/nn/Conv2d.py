import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        batch_size, in_channels, input_height, input_width = A.shape
        output_height = input_height - self.kernel_size + 1
        output_width = input_width - self.kernel_size + 1
        Z = np.zeros((batch_size, self.out_channels, output_height, output_width))

        for i in range(output_height):
            for j in range(output_width): 
                A_slice = A[:, :, i: i + self.kernel_size, j: j + self.kernel_size]
                # A_slice : (batch_size, in_channels, kernel_size, kernel_size)
                # W: (out_channels, in_channels, kernel_size, kernel_size)
                Z[:,:,i,j] = np.tensordot(A_slice, self.W, axes = ([1,2,3], [1,2,3]))

        Z += self.b.reshape((1, -1, 1, 1))

        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        # get dimensions
        batch_size, _, output_height, output_width = dLdZ.shape
        _, _, input_height, input_width = self.A.shape
        padding = self.kernel_size - 1
        padded_dLdZ = np.pad(dLdZ, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant', constant_values = 0)
        # padded_dldz : (batch_size, out_channels, input_height, input_width)
        
        # Flip the weights
        flipped_W = np.flip(self.W, axis= (2,3))
        dLdA = np.zeros((batch_size, self.in_channels, input_height, input_width))
        for i in range(input_height):
            for j in range(input_width):
                dLdA_slice = padded_dLdZ[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                # dlda_slice : (batch_size, out_channels, kernel_size, kernel_size)
                # flip_w : (out_channels, in_channels, kernel_size, kernel_size)
                dLdA[:, :, i, j] = np.tensordot(dLdA_slice, flipped_W, axes=([1, 2, 3], [0, 2, 3]))
        
        # get dLdW
        self.dLdW = np.zeros(self.W.shape)
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                A_slice = self.A[:, :, i:i+output_height, j:j+output_width]
                # dldz : (batch_size, out_channels, output_height, output_width)
                # A_slice : (batch_size, in_channels, output_height, output_width)
                self.dLdW[:, :, i, j] = np.tensordot(dLdZ, A_slice, axes=([0, 2, 3], [0, 2, 3]))
        
        # To find dLdb, we simply sum dLdZ over all axes except for the channel axis
        # dldZ : (batch_size, out_channels, output_height, output_width)
        # dldb : (out_channels,)
        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))

        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """

        # Pad the input appropriately using np.pad() function
        # A : (batch_size, in_channels, input_height, input_width)
        padded_A = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), 'constant', constant_values = 0)
        # padded_A : (batch_size, in_channels, input_height + padding * 2, input_width + padding * 2)

        # Call Conv1d_stride1
        converted = self.conv2d_stride1.forward(padded_A)

        # downsample
        Z = Downsample2d.forward(self.downsample2d, converted)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        # Call downsample1d backward
        if self.stride > 1:
            upsampled_dLdZ = self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA_padded = self.conv2d_stride1.backward(upsampled_dLdZ)

        # Unpad the gradient
        if self.pad > 0:
            dLdA = dLdA_padded[:, :, self.pad:-self.pad, self.pad:-self.pad]
        else:
            dLdA = dLdA_padded

        return dLdA
