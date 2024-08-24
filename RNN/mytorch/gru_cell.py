import numpy as np
from nn.activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, input_size, hidden_size):
        self.d = input_size
        self.h = hidden_size
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t
        
        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.

        z = self.z_act.forward(np.dot(self.Wzx, x) + np.dot(self.Wzh, h_prev_t) + self.bzx + self.bzh)

        r = self.r_act.forward(np.dot(self.Wrx, x) + np.dot(self.Wrh, h_prev_t) + self.brx + self.brh)

        self.t = np.dot(self.Wnh, h_prev_t) + self.bnh

        n = self.h_act.forward(np.dot(self.Wnx, x) + self.bnx + r * self.t)

        h_t = z * h_prev_t + (1 - z) * n
        
        self.z = z
        self.r = r
        self.n = n


        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (hidden_dim)
            derivative of the loss wrt the input hidden h.

        """

        # SOME TIPS:
        # 1) Make sure the shapes of the calculated dWs and dbs match the initalized shapes of the respective Ws and bs
        # 2) When in doubt about shapes, please refer to the table in the writeup.
        # 3) Know that the autograder grades the gradients in a certain order, and the local autograder will tell you which gradient you are currently failing.

        # delta (hidden_dim)

        # step 1 : h_t = (1-z) * n + z * h_pre
        # dldz (hidden_dim)
        dldz = delta * (self.hidden - self.n) # both in shape (hidden_dim)
        # dldn (hidden_dim)
        dldn = delta * (1 - self.z) 

        # step 2 : n = tanh (Wnh * x + bnx + rt * (Wnh * h_pre + bnh)) 
        # n (hidden_dim)
        dl_tanh = self.h_act.backward(dldn, self.n) #(h,)
        # dwnx (h, d)  x(d,)
        dwnx = np.outer(dl_tanh, self.x)  #(h, d)
        dbnx = dl_tanh   #(h,)
        dldr = dl_tanh * self.t #(h,)
        # dwnh (h, h)  h(h,)
        dwnh = np.outer(dl_tanh * self.r, self.hidden)
        dbnh = dl_tanh * self.r

        # step 3: z = sigmoid(wzh * x + bzx + wzh * hidden +bzh)
        # z (h,)
        dldz_sig = self.z_act.backward(dldz) # (h,)
        # dwzx (h, d)  x(d,)
        dwzx = np.outer(dldz_sig, self.x)
        dbzx = dldz_sig
        # dwzh (h, h)  h(h,)
        dwzh = np.outer(dldz_sig, self.hidden)
        dbzh = dldz_sig

        # step 4 : r = sigmoid(wrh * x + brx + wrh * hidden +brh)
        # r (h,)
        dldr_sig = self.r_act.backward(dldr) # (h,)
        # dwrx (h, d)  x(d,)
        dwrx = np.outer(dldr_sig, self.x)
        dbrx = dldr_sig
        # dwrh (h, h)  h(h,)
        dwrh = np.outer(dldr_sig, self.hidden)
        dbrh = dldr_sig

        # step 5 : get dldx and dldh_pre
        dx = dl_tanh @ self.Wnx + dldz_sig @ self.Wzx + dldr_sig @ self.Wrx
        dh_prev_t = delta * self.z + dl_tanh * self.r @ self.Wnh + dldz_sig @ self.Wzh + dldr_sig @ self.Wrh

        self.dWrx = dwrx
        self.dWzx = dwzx
        self.dWnx = dwnx
        
        self.dWrh = dwrh
        self.dWzh = dwzh
        self.dWnh = dwnh

        self.dbrx = dbrx
        self.dbzx = dbzx
        self.dbnx = dbnx

        self.dbrh = dbrh
        self.dbzh = dbzh
        self.dbnh = dbnh
        
        assert dx.shape == (self.d,)
        assert dh_prev_t.shape == (self.h,)

        return dx, dh_prev_t
