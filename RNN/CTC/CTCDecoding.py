import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)
        last_symbol_index = None
        for t in range(y_probs.shape[1]):

            # get highest prb index
            symbol_index = np.argmax(y_probs[:, t, 0])
            symbol_prob = np.max(y_probs[:, t, 0])

            # Update the path probability
            path_prob *= symbol_prob

            # Append the symbol to the path if it is not a blank and not a repetition
            if symbol_index != blank and symbol_index != last_symbol_index:
                decoded_path.append(self.symbol_set[symbol_index - 1]) 
                last_symbol_index = symbol_index

        decoded_path = ''.join(decoded_path)
        return decoded_path, path_prob


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """
        T = y_probs.shape[1]  # seq_length
        beams = [('', 1.0)]

        if "-" not in self.symbol_set:
            symbol_set = ["-"] + self.symbol_set
        else:
            symbol_set = self.symbol_set
        # print(symbol_set)

        for t in range(T):
            candidates = {}
            for prefix, score in beams:
                for i, symbol in enumerate(symbol_set):
                    new_score = score * y_probs[i, t, 0]
                    if symbol == "-" and t == T-1:
                        new_seq = prefix  # Ignore "-" if it's the last symbol
                    # repeat letters?
                    elif symbol == "-" and t == T-1 and prefix[-1] == "-":
                        new_seq = prefix[:-1]
                    elif prefix.endswith(symbol):
                        new_seq = prefix  # Ignore repeating letter
                    # Check if the pattern is A-A (symbol repeats after a "-")
                    elif len(prefix) >= 2 and prefix[-1] == "-":
                        new_seq = prefix[:-1] + symbol  # Remove "-"
                    elif t == 0 and symbol =='-':
                        new_seq = prefix 
                    else:
                        new_seq = prefix + symbol
                    # update the diction
                    if new_seq in candidates:
                        candidates[new_seq] += new_score
                    else:
                        candidates[new_seq] = new_score
                    
            # Select the top `beam_width` sequences
            if t == T-1:
                beams = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
                filtered_dict = {key: value for key, value in beams}
                modified_dict = {key[:-1] if key.endswith('-') else key: value for key, value in filtered_dict.items()}
            else:
                beams = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:self.beam_width]

        # The best path is the one with the highest score
        bestPath, FinalPathScore = max(beams, key=lambda x: x[1])
        print(filtered_dict.items())
        return bestPath, modified_dict

    
