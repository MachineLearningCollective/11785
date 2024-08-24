import numpy as np


class CTC(object):

    def __init__(self, BLANK=0):
        """
        Initialize instance variables
        Argument(s)
        -----------
        BLANK (int, optional): blank label index. Default 0.
        """
        # No need to modify
        self.BLANK = BLANK


    def extend_target_with_blank(self, target):
        """Extend target sequence with blank.
        Input
        -----
        target: (np.array, dim = (target_len,))
                target output containing indexes of target phonemes
        ex: [1,4,4,7]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [0,1,0,4,0,4,0,7,0]

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]
        """

        extended_symbols = [self.BLANK]
        for symbol in target:
            extended_symbols.append(symbol)
            extended_symbols.append(self.BLANK)

        N = len(extended_symbols)

        # -------------------------------------------->
        skip_connect = np.zeros(N)  
        last_value = extended_symbols[1]
        
        for i, symbol in enumerate(extended_symbols):
            if symbol == 0:
                continue 
            elif symbol != last_value:
                skip_connect[i] = 1
                last_value = symbol
            else:
                continue

        # <---------------------------------------------

        extended_symbols = np.array(extended_symbols).reshape((N,))
        skip_connect = np.array(skip_connect).reshape((N,))

        return extended_symbols, skip_connect
    
    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t, qextSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """

        S, T = len(extended_symbols), len(logits)
        alpha = np.zeros(shape=(T, S))
        Sext = extended_symbols
        N = S

        # -------------------------------------------->
        # TODO: Intialize alpha[0][0]
        # The forward recursion
        # First, at t = 0
        alpha[0][0] = logits[0][Sext[0]]  # This is the blank
        alpha[0][1] = logits[0][Sext[1]]
        # Assuming alpha is a 2D list, initialize the rest of the first row to 0
        for i in range(2, N):
            alpha[0][i] = 0

        # Now for the rest of the timesteps
        for t in range(1, T):  # T is the total number of timesteps
            alpha[t][0] = alpha[t-1][0] * logits[t][Sext[0]]
            for i in range(1, N):
                alpha[t][i] = alpha[t-1][i] + alpha[t-1][i-1]
                if i > 1 and Sext[i] != Sext[i-2]:
                    alpha[t][i] += alpha[t-1][i-2]
                alpha[t][i] *= logits[t][Sext[i]]

        return alpha

    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities
    
        """
        N, T = len(extended_symbols), len(logits)
        beta = np.zeros(shape=(T, N))
        betahat = np.zeros(shape=(T, N))
        Sext = extended_symbols

        # -------------------------------------------->
        # The backward recursion
        # First, at t = T - 1
        betahat[T-1][N-1] = logits[T-1][Sext[N-1]]
        betahat[T-1][N-2] = logits[T-1][Sext[N-2]]

        for t in range(T-2, -1, -1):  # loop for time
            betahat[t][N-1] = betahat[t+1][N-1] * logits[t][Sext[N-1]]
            for i in range(N-2, -1, -1):  # loop for input 
                betahat[t][i] = betahat[t+1][i] + betahat[t+1][i+1]
                if i <= N-3 and Sext[i] != Sext[i+2]:
                    betahat[t][i] += betahat[t+1][i+2]
                betahat[t][i] *= logits[t][Sext[i]]

        # Compute beta from betahat
        for t in range(T-1, -1, -1):  # This loop goes from T-1 down to 0
            for i in range(N-1, -1, -1):  # This loop goes from N-1 down to 0
                beta[t][i] = betahat[t][i] / logits[t][Sext[i]]

        # <--------------------------------------------
        return beta


    def get_posterior_probs(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

        """

        [T, S] = alpha.shape
        gamma = np.zeros(shape=(T, S))
        sumgamma = np.zeros((T,))
        
        # -------------------------------------------->
        for t in range(T):
            sumgamma[t] = 0
            for i in range(S):
                gamma[t][i] = alpha[t][i] * beta[t][i]
                sumgamma[t] += gamma[t][i]
            for i in range(S):
                gamma[t][i] = gamma[t][i] / sumgamma[t]
        # <---------------------------------------------

        return gamma
        raise NotImplementedError


class CTCLoss(object):

    def __init__(self, BLANK=0):
        """

        Initialize instance variables

        Argument(s)
        -----------
        BLANK (int, optional): blank label index. Default 0.
        
        """
        # -------------------------------------------->
        # No need to modify
        super(CTCLoss, self).__init__()

        self.BLANK = BLANK
        self.gammas = []
        self.ctc = CTC()
        # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):

        # No need to modify
        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward

        Computes the CTC Loss by calculating forward, backward, and
        posterior proabilites, and then calculating the avg. loss between
        targets and predicted log probabilities

        Input
        -----
        logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        loss [float]:
            avg. divergence between the posterior probability and the target

        """

        # No need to modify
        self.logits = logits  # dim=(seq_length, batch_size, len(symbols)) (15, 12, 8)
        self.target = target  # dim=(batch_size, padded_target_len) (12, 4)
        self.input_lengths = input_lengths     # dim=(batch_size,) 12
        self.target_lengths = target_lengths   # dim=(batch_size,) 12

        #####  IMP:
        #####  Output losses should be the mean loss over the batch

        # No need to modify
        B, _ = target.shape  # B is batch
        total_loss = np.zeros(B)
        self.extended_symbols = []

        for batch_itr in range(B):
            # batch_itr int b = 12
            # numpy total loss [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute forward probabilities
            #     Compute backward probabilities
            #     Compute posteriors using total probability function
            #     Compute expected divergence for each batch and store it in totalLoss
            #     Take an average over all batches and return final result
            # <---------------------------------------------
            # -------------------------------------------->
            # target_batch (1, padded_target_len)  target (batch_size, padded_target_len) 2
            target_batch = target[batch_itr, :target_lengths[batch_itr]] 
            #print('target.shape')
            #print(target_batch.shape)
            # logits dim=(seq_length, batch_size, len(symbols)) (15, 12, 8)
            # logits_batch (input length, symbols) 12,8
            logits_batch = logits[:input_lengths[batch_itr], batch_itr, :]
            #print('logits.shape')
            #print(logits_batch.shape)
            
            # Extend target sequence with blank 5 target_len * 2 + 1
            extended_symbol, skip_connect = self.ctc.extend_target_with_blank(target_batch)
            #print('extend.shape')
            #print(extended_symbol.shape)

            self.extended_symbols.append(extended_symbol)
            
            # Compute forward probabilities 12*5
            alpha = self.ctc.get_forward_probs(logits_batch, extended_symbol, skip_connect)
            #print('alpha.shape')
            #print(alpha.shape)
            
            # Compute backward probabilities 12*5
            beta = self.ctc.get_backward_probs(logits_batch, extended_symbol, skip_connect)
            #print('beta.shape')
            #print(beta.shape)
            
            # Compute posteriors 
            gamma = self.ctc.get_posterior_probs(alpha, beta)
            #print('beta.shape')
            #print(gamma.shape)
            self.gammas.append(gamma)

            # --- so far so good----

            T = input_lengths[batch_itr]
            N = target_lengths[batch_itr]
            L = N*2 + 1

            # calculate divergence
            batch_loss = 0.0
            for t in range(T):
                for s in range(L): 
                    batch_loss -= gamma[t][s] * np.log(logits_batch[t][extended_symbol[s]])
                
            total_loss[batch_itr] = batch_loss

        total_loss = np.sum(total_loss) / B

        return total_loss


    def backward(self):
        """

        CTC loss backard

        Calculate the gradients w.r.t the parameters and return the derivative 
        w.r.t the inputs, xt and ht, to the cell.

        Input
        -----
        logits [np.array, dim=(seqlength, batch_size, len(Symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        dY [np.array, dim=(seq_length, batch_size, len(extended_symbols))]:
            derivative of divergence w.r.t the input symbols at each time

        """

        # No need to modify
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0.0)

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Derivative for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute derivative of divergence and store them in dY
            # <---------------------------------------------

            # get gamma from previous records
            gamma = self.gammas[batch_itr]

            T = self.input_lengths[batch_itr]
            N = self.target_lengths[batch_itr]

            logits_batch = self.logits[:self.input_lengths[batch_itr], batch_itr, :]
            extended_symbol = self.extended_symbols[batch_itr]

            # dY [np.array, dim=(seq_length t, batch_size, len(extended_symbols))]:

            for t in range(T):
                for i in range(2*N+1):
                    s_i = extended_symbol[i]
                    g = gamma[t][i]
                    y = logits_batch[t][s_i]
                    dY[t][batch_itr][s_i] -= g/y

        return dY
