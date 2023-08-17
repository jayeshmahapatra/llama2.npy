import argparse
import os
import struct
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from model import Transformer
from tokenizer import Tokenizer

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0

if __name__ == "__main__":
    
    # Add a command line parser
    parser = argparse.ArgumentParser(description='Run the llama2 using just numpy')
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
    
    # Path is valid, we read from the file

    # Read header information
    header_size = struct.calcsize('iiiiiii')
    with open(weight_filepath, 'rb') as f:
        header = struct.unpack('iiiiiii', f.read(header_size))
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len = header

        print("dim: {dim} hidden_dim: {hidden_dim} n_layers: {n_layers} n_heads: {n_heads} n_kv_heads: {n_kv_heads} vocab_size: {vocab_size} max_seq_len: {max_seq_len}".format(
            dim=dim, 
            hidden_dim=hidden_dim, 
            n_layers=n_layers, 
            n_heads=n_heads, 
            n_kv_heads=n_kv_heads, 
            vocab_size=vocab_size, 
            max_seq_len=max_seq_len))
        
        # Create a model args object
        model_args = ModelArgs(dim=dim, n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv_heads, vocab_size=vocab_size, max_seq_len=max_seq_len)

        print(model_args)

        # Create a new instance of the Transformer model
        transformer = Transformer(model_args)
        transformer.eval()

        # Read and load weights
        def deserialize(t, f):
            
            num_elements = t.numel()
            data = struct.unpack(f'{num_elements}f', f.read(4 * num_elements))

            return torch.tensor(data).view(t.shape)
        
        with open(weight_filepath, 'rb') as f:
            # Skip header
            f.seek(header_size, 0)

            # Load embedding weights
            transformer.tok_embeddings.weight = nn.Parameter(deserialize(transformer.tok_embeddings.weight, f)) 

            # Load attention and ffn weights for each layer
            for layer in transformer.layers:
                layer.attention_norm.weight = nn.Parameter(deserialize(layer.attention_norm.weight, f)) 
                layer.attention.wq.weight = nn.Parameter(deserialize(layer.attention.wq.weight, f))
                layer.attention.wk.weight = nn.Parameter(deserialize(layer.attention.wk.weight, f))
                layer.attention.wv.weight = nn.Parameter(deserialize(layer.attention.wv.weight, f))
                layer.attention.wo.weight = nn.Parameter(deserialize(layer.attention.wo.weight, f))

                layer.ffn_norm.weight = nn.Parameter(deserialize(layer.ffn_norm.weight, f))
                layer.feed_forward.w1.weight = nn.Parameter(deserialize(layer.feed_forward.w1.weight, f))
                layer.feed_forward.w2.weight = nn.Parameter(deserialize(layer.feed_forward.w2.weight, f))
                layer.feed_forward.w3.weight = nn.Parameter(deserialize(layer.feed_forward.w3.weight, f))

            # Load final RMSNorm weight
            transformer.norm.weight = nn.Parameter(deserialize(transformer.norm.weight, f))

            # Load freqs_cos and freqs_sin
            head_size = dim // n_heads
            transformer.freqs_cos = deserialize(torch.empty(max_seq_len, int(head_size/2)), f)
            transformer.freqs_sin = deserialize(torch.empty(max_seq_len, int(head_size/2)), f)
        
        print("Weights loaded successfully")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = os.path.join("./weights", "stories15M.pt")
        checkpoint_dict = torch.load(checkpoint, map_location=device)
        gptconf = ModelArgs(**checkpoint_dict['model_args'])
        model = Transformer(gptconf)
        state_dict = checkpoint_dict['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.to(device)

        

        transformer = transformer.to(device)

        # Load the tokenizer
        tokenizer_model = os.path.join("./", "tokenizer.model")
        enc = Tokenizer(tokenizer_model=tokenizer_model)

        # Encode the initial string
        initial_string = "Once upon a time "
        x = enc.encode(initial_string, bos=True, eos=False)
        x = torch.tensor([x], dtype=torch.long, device=device) # 1 is BOS
        
        with torch.inference_mode():
            y = transformer.generate(x, max_new_tokens=200, temperature=0.9)
        pt_tokens = y[0].tolist()
        
        text = enc.decode(pt_tokens)

        print(text)


        with torch.inference_mode():
            y = model.generate(x, max_new_tokens=200, temperature=0.9)
        pt_tokens = y[0].tolist()

        
        text = enc.decode(pt_tokens)

        print(text)


        



        



