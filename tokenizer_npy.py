# Create a Tokenizer class that will be used to tokenize the input text
import struct
import sys
import numpy as np

class Tokenizer():

    def __init__(self, model_path: str, vocab_size = 32000) -> None:

        self.vocab_size = vocab_size

        # Load the tokenizer, store in dicts for fast lookup
        self.vocab2index, self.vocab_score2index = self._load_tokenizer(model_path)
        self.index2vocab = {v: k for k, v in self.vocab2index.items()}
        self.index2vocab_score = {v: k for k, v in self.vocab_score2index.items()}

    # An internal function called _load_tokenizer, takes str input, outputs a tuple (dict, dict, int)
    def _load_tokenizer(self, model_path: str) -> tuple:

        max_token_length = 0
        self.vocab2index = {}
        self.vocab_score2index = {}
        

        with open(model_path, 'rb') as file:

            max_token_length = struct.unpack('i', file.read(4))[0]

            for i in range(0, self.vocab_size):

                score = struct.unpack('f', file.read(4))[0]
                str_len = struct.unpack('i', file.read(4))[0]
                bytestr = file.read(str_len)

                if type(bytestr) is not str:
                    bytestr = bytestr.decode('utf8')

                self.vocab2index[bytestr] = i
                self.vocab_score2index[score] = i

        return self.vocab2index, self.vocab_score2index
    

    def encode(self, initial_string: str, bos: bool, eos: bool) -> list:

        # Encode the initial string character by character, assunes all characters are in vocab
        tokens = [self.vocab2index[char] for char in initial_string]
        

        # Merge consecutive pairs of tokens based on vocab_scores, stop when merging no longer increases the score
        while True:
            best_score = np.NINF
            best_id = -1
            best_idx = -1

            # Iterate over all consecutive pairs of tokens
            for i in range(len(tokens) - 1):

                # Convert the pair of tokens into a string
                string = self.index2vocab[tokens[i]] + self.index2vocab[tokens[i + 1]]
                
                # Get the ID of this merged string in vocab
                id = self.vocab2index.get(string, None)

                if id is not None:

                    if self.index2vocab_score[id] > best_score:
                        # We found a better pair to merge
                        best_score = self.index2vocab_score[id]
                        best_id = id
                        best_idx = i

            if best_idx == -1:
                break  # We couldn't find any more pairs to merge, so we're done

            # Merge the consecutive pair (best_idx, best_idx+1)

            # Replace token at position best_idx with best_id of the merged pair
            tokens[best_idx] = best_id

            # Delete token at position best_idx+1
            tokens = tokens[0:best_idx + 1] + tokens[best_idx + 2:]

        return tokens

    def decode(self, pt_tokens: list) -> str:

        # Convert the list of token IDs back into a string
        text = ''.join([self.index2vocab[token] for token in pt_tokens])

        return text
            