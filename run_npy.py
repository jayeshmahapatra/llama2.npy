import argparse
import os
import struct
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from model_npy import Transformer, ModelArgs
from tokenizer_npy import Tokenizer



if __name__ == "__main__":
    
    # Add a command line parser
    parser = argparse.ArgumentParser(description='Run the llama2 using just numpy')
    # Optional input prompt with default value
    parser.add_argument("-i", '--input', type=str, help='Input Prompt', default="Once upon a time")

    # Add optional argument num_tokens that defaults to 20
    parser.add_argument("-n", '--num_tokens', type=int, help='Number of tokens to generate', default=20)

    required_args = parser.add_argument_group('Required arguments')
    required_args.add_argument("-w", '--weight', type=str, help='Path to bin file containing the weights', required=True)
    args = parser.parse_args()

    # Get the weight filepath from parser
    weight_filepath = args.weight

    # Check if the path is valid
    
    # Check that file is a bin file
    if not weight_filepath.endswith('.bin'):
        raise ValueError('The weight file must be a bin file')

    # Check if the file exists
    if not os.path.exists(weight_filepath):
        raise ValueError('The weight file does not exist')
    
    
    # Path is valid, we read model weights from the file

    # Read header information
    header_size = struct.calcsize('iiiiiii')
    with open(weight_filepath, 'rb') as f:
        header = struct.unpack('iiiiiii', f.read(header_size))
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len = header

        # Create a model args object
        model_args = ModelArgs(dim=dim, n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv_heads, vocab_size=vocab_size, max_seq_len=max_seq_len, multiple_of=32)

    # Create a new instance of the Transformer model
    transformer = Transformer(model_args)
    
    # Function to desirialize the weights as numpy arrays
    def deserialize_np(t: np.ndarray, f):
        
        num_elements = t.size
        data = struct.unpack(f'{num_elements}f', f.read(4 * num_elements))

        return np.array(data).reshape(t.shape)
    
    with open(weight_filepath, 'rb') as f:
        # Skip header
        f.seek(header_size, 0)

        # Load embedding weights
        transformer.tok_embeddings.weight = deserialize_np(transformer.tok_embeddings.weight, f)

        # Load attention and ffn weights for each layer
        for layer in transformer.layers:
            layer.attention_norm.weight = deserialize_np(layer.attention_norm.weight, f)
        for layer in transformer.layers:
            layer.attention.wq.weight = deserialize_np(layer.attention.wq.weight, f)
        for layer in transformer.layers:
            layer.attention.wk.weight = deserialize_np(layer.attention.wk.weight, f)
        for layer in transformer.layers:
            layer.attention.wv.weight = deserialize_np(layer.attention.wv.weight, f)
        for layer in transformer.layers:
            layer.attention.wo.weight = deserialize_np(layer.attention.wo.weight, f)

        for layer in transformer.layers:
            layer.ffn_norm.weight = deserialize_np(layer.ffn_norm.weight, f)
        for layer in transformer.layers:
            layer.feed_forward.w1.weight = deserialize_np(layer.feed_forward.w1.weight, f)
        for layer in transformer.layers:
            layer.feed_forward.w2.weight = deserialize_np(layer.feed_forward.w2.weight, f)
        for layer in transformer.layers:
            layer.feed_forward.w3.weight = deserialize_np(layer.feed_forward.w3.weight, f)

        # Load final RMSNorm weight
        transformer.norm.weight = deserialize_np(transformer.norm.weight, f)

        # Load freqs_cos and freqs_sin
        head_size = dim // n_heads
        transformer.freqs_cos = deserialize_np(np.empty((max_seq_len, int(head_size/2))), f)
        transformer.freqs_sin = deserialize_np(np.empty((max_seq_len, int(head_size/2))), f)

        # Output classifier weights are shared with the embedding weights
        transformer.output.weight = transformer.tok_embeddings.weight

    # Load the tokenizer
    tokenizer_model_path = os.path.join("./", "tokenizer.bin")
    #enc = Tokenizer(tokenizer_model=tokenizer_model)
    enc = Tokenizer(tokenizer_model_path)

    # Encode the initial string
    initial_string = args.input
    x = enc.encode(initial_string, bos=True, eos=False)

    x = np.array([x])
    
    # Perform inference
    y = transformer.generate(x, max_new_tokens=args.num_tokens, temperature=0.9)

    pt_tokens = y[0].tolist()
    
    # Decode text
    text = enc.decode(pt_tokens)

    # print text
    print(text)


        



        



