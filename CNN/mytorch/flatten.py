import numpy as np

class Flatten():
    def __init__(self):
        self.original_shape = None

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """
        self.original_shape = A.shape
        batch_size, in_channels, in_width = A.shape
        Z = A.reshape(batch_size, in_channels * in_width)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """

        dLdA = dLdZ.reshape((self.original_shape))

        return dLdA
