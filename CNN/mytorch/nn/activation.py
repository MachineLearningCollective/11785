import numpy as np
import scipy
from scipy.special import erf
import math

class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self, dLdA):

        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ

        return dLdZ


class Sigmoid:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Sigmoid.
    """
    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z))
        return self.A
    
    def backward(self, dLdA):
        return dLdA * self.A * (1 - self.A)


class Tanh:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Tanh.
    """
    def forward(self, z):
        self.A = np.tanh(z)
        return self.A
    
    def backward(self, dLdA):
        return dLdA * (1 - np.square(self.A))


class ReLU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on ReLU.
    """
    def forward(self, z):
        self.A = np.maximum(0.0, z)
        return self.A
    
    def backward(self, dLdA):
        return np.where(self.A > 0, dLdA, 0)

class GELU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on GELU.
    """
    def forward(self, z):
        self.A = z
        return 0.5 * z * (1 + erf(z / math.sqrt(2)))
    
    def backward(self, dLdA):
        return dLdA * (0.5 * (1 + erf(self.A / math.sqrt(2))) + self.A / math.sqrt(2.0 * math.pi) * np.exp(- self.A * self.A * 0.5) )

class Softmax:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Softmax.
    """

    def forward(self, Z):
        """
        Remember that Softmax does not act element-wise.
        It will use an entire row of Z to compute an output element.
        """
        sumexp = np.exp(Z).sum(axis = 1, keepdims = True)
        self.A = np.exp(Z) / sumexp

        return self.A
    
    def backward(self, dLdA):

        # Calculate the batch size and number of features
        N = dLdA.shape[0]
        C = dLdA.shape[1]

        # Initialize the final output dLdZ with all zeros. Refer to the writeup and think about the shape.
        dLdZ = np.zeros((N,C))

        # Fill dLdZ one data point (row) at a time
        for i in range(N):

            # Initialize the Jacobian with all zeros.
            J = np.zeros((C,C))

            # Fill the Jacobian matrix according to the conditions described in the writeup
            for m in range(C):
                for n in range(C):
                    if m == n:
                        J[m,n] = self.A[i,m] * (1 - self.A[i,m])
                    else:
                        J[m,n] = - self.A[i,m] * self.A[i,n]

            # Calculate the derivative of the loss with respect to the i-th input
            dLdZ[i,:] = dLdA[i,:] @ J

        return dLdZ