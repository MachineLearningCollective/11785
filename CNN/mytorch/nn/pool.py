import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        batch_size, in_channels, input_width, input_height = A.shape
        kernel_size = self.kernel
        self.A = A
        output_width = input_width - kernel_size + 1
        output_height = input_height - kernel_size + 1

        Z = np.zeros((batch_size, in_channels, output_width, output_height))
        for i in range(output_width):
            for j in range(output_height):
                Z[:, :, i, j] = np.max(A[:, :, i:i+kernel_size, j:j+kernel_size], axis=(2, 3))
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, out_channels, output_width, output_height = dLdZ.shape
        kernel_size = self.kernel
        dLdA = np.zeros((batch_size, out_channels, output_width + kernel_size - 1, output_height + kernel_size - 1))
        for i in range(output_width):
            for j in range(output_height):
                for b in range(batch_size):
                    for c in range(out_channels):
                        # Find the index of the maximum value in the corresponding pooling window
                        max_index = np.argmax(self.A[b, c, i:i+kernel_size, j:j+kernel_size])
                        # Convert the 1D index to 2D index
                        max_index_i, max_index_j = np.unravel_index(max_index, (kernel_size, kernel_size))
                        # Distribute the gradient to the position of the maximum value
                        dLdA[b, c, i + max_index_i, j + max_index_j] += dLdZ[b, c, i, j]
    
        return dLdA



class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        batch_size, in_channels, input_width, input_height = A.shape
        kernel_size = self.kernel
        output_width = input_width - kernel_size + 1
        output_height = input_height - kernel_size + 1
        Z = np.zeros((batch_size, in_channels, output_width, output_height))
        for i in range(output_width):
            for j in range(output_height):
                # A_slice : (batch_size, in_channels, kernel_size, kernel_size)
                Z[:, :, i, j] = np.mean(A[:, :, i:i+kernel_size, j:j+kernel_size], axis=(2, 3))
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        batch_size, out_channels, output_width, output_height = dLdZ.shape
        kernel_size = self.kernel
        dLdA = np.zeros((batch_size, out_channels, output_width + kernel_size - 1, output_height + kernel_size - 1))
        for i in range(output_width):
            for j in range(output_height):
                dLdA[:, :, i:i+kernel_size, j:j+kernel_size] += dLdZ[:, :, i:i+1, j:j+1] / (kernel_size * kernel_size)
        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        A_pooled = self.maxpool2d_stride1.forward(A)
        # Apply Downsample2d
        Z = self.downsample2d.forward(A_pooled)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ_upsampled = self.downsample2d.backward(dLdZ)
        # Backward pass through MaxPool2d_stride1
        dLdA = self.maxpool2d_stride1.backward(dLdZ_upsampled)
        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        A_pooled = self.meanpool2d_stride1.forward(A)
        # Apply Downsample2d
        Z = self.downsample2d.forward(A_pooled)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ_upsampled = self.downsample2d.backward(dLdZ)
        # Backward pass through MaxPool2d_stride1
        dLdA = self.meanpool2d_stride1.backward(dLdZ_upsampled)
        return dLdA
