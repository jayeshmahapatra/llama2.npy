import math
import struct
import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

@dataclass
class ModelArgs:
    # default hyperparameters for the Llama 7B model
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0

### Nunpy based classes for the model

# Define a numpy based RMSNorm class
class RMSNorm:
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = np.ones(dim, dtype=np.float32)

    def _norm(self, x):
        return x / np.sqrt(np.mean(np.power(x, 2), axis=-1, keepdims=True) + self.eps)


    def forward(self, x):
        output = self._norm(x.astype(float)).astype(x.dtype)
        return output * self.weight
    
    def __call__(self, x):
        return self.forward(x)
    
# Define a numpy based linear class
class NumpyLinear:
    def __init__(self, in_features: int, out_features: int, bias=True):
        self.weight = np.random.randn(out_features, in_features)
        self.bias = np.random.randn(out_features) if bias else None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.bias is not None:
            return np.dot(x, self.weight.T) + self.bias
        else:
            return np.dot(x, self.weight.T)

# Define a numpy based dropout class
class NumpyDropout:
    def __init__(self, p: float =0.5):
        self.p = p

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward_testing(x)
    
    def forward_testing(self, x: np.ndarray) -> np.ndarray:
        # scale by 1/(1-p) to match expected value
        return x * (1/(1-self.p))
    
# Numpy based softmax function
def NumpySoftmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# Numpy based sigmoid function
def NumpySigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

# Numpy based silu function
def NumpySilu(x: np.ndarray) -> np.ndarray:
    return x * NumpySigmoid(x)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=float)[: (dim // 2)].float() / dim))
    t = np.arange(end, dtype=float)
    freqs = np.outer(t, freqs).astype(float)
    freqs_cos = np.cos(freqs)
    freqs_sin = np.sin(freqs)
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: np.ndarray, x: np.ndarray):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.reshape(shape)

def apply_rotary_emb(
    xq: np.ndarray,
    xk: np.ndarray,
    freqs_cos: np.ndarray,
    freqs_sin: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    # reshape xq and xk to match the complex representation
    xq_reshaped = xq.float().reshape(xq.shape[:-1] + (-1, 2))
    xq_r = xq_reshaped[..., 0]
    xq_i = xq_reshaped[..., 1]
    
    xk_reshaped = xk.float().reshape(xk.shape[:-1] + (-1, 2))
    xk_r = xk_reshaped[..., 0]
    xk_i = xk_reshaped[..., 1]

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out_stacked = np.stack([xq_out_r, xq_out_i], axis=-1)
    xk_out_stacked = np.stack([xk_out_r, xk_out_i], axis=-1)

    xq_out = xq_out_stacked.reshape(xq.shape)
    xk_out = xk_out_stacked.reshape(xk.shape)
    
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).reshape(xq_out.shape[:3] + (-1,))
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).reshape(xk_out.shape[:3] + (-1,))

    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        np.broadcast_to(x[:, :, :, None, :], (bs, slen, n_kv_heads, n_rep, head_dim))
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention():
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = NumpyLinear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = NumpyLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = NumpyLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = NumpyLinear(args.n_heads * self.head_dim, args.dim, bias=False) 
        self.attn_dropout = NumpyDropout(args.dropout) 
        self.resid_dropout = NumpyDropout(args.dropout) 
        self.dropout = args.dropout

        
        # create causal attention mask
        mask = np.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = np.triu(mask, k=1)
        self.mask = mask

    def forward(
        self,
        x: np.ndarray,
        freqs_cos: np.ndarray,
        freqs_sin: np.ndarray,
    ):
        bsz, seqlen, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.reshape(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.reshape(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # grouped multiquery attention: expand out keys and values
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # manual implementation
        scores = np.matmul(xq, xk.transpose(2, 3)) / np.sqrt(self.head_dim)
        scores = scores + self.mask[:, :, :seqlen, :seqlen]   # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = NumpySoftmax(scores, axis=-1)
        scores = self.attn_dropout(scores)
        output = np.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).reshape(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class FeedForward():
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = NumpyLinear(dim, hidden_dim, bias=False)
        self.w2 = NumpyLinear(hidden_dim, dim, bias=False)
        self.w3 = NumpyLinear(dim, hidden_dim, bias=False)
        self.dropout = NumpyDropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(NumpySilu(self.w1(x)) * self.w3(x)))


class TransformerBlock():
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = np.random.randn(params.vocab_size, params.dim)
        self.dropout = NumpyDropout(params.dropout)
        
        self.layers = []
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = NumpyLinear(params.dim, params.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        self.tok_embeddings.weight = self.output.weight # https://paperswithcode.com/method/weight-tying

        # some useful precompute for the RoPE relative positional embeddings
        self.freqs_cos, self.freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)

        # Load all the weights
        self.load_weights()
        


    def load_weights():
        pass

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        h = self.norm(h)

        # inference-time mini-optimization: only forward the output on the very last position
        logits = self.output(h[:, [-1], :]) # note: using list [-1] to preserve the time dim
        self.last_loss = None

        return logits

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        Also note this is a super inefficient version of sampling with no key/value cache.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            logits = logits[:, -1, :] # crop to just the final time step
            if temperature == 0.0:
                # "sample" the single most likely index
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # pluck the logits at the final step and scale by desired temperature
                logits = logits / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def export(self, filepath='model.bin'):
        """export the model weights in fp32 into .bin file to be read from C"""
        f = open(filepath, 'wb')

        def serialize(t):
            d = t.detach().cpu().view(-1).numpy().astype(np.float32)
            b = struct.pack(f'{len(d)}f', *d)
            f.write(b)

        # first write out the header
        hidden_dim = self.layers[0].feed_forward.w1.weight.shape[0]
        p = self.params
        n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
        header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                                       n_kv_heads, p.vocab_size, p.max_seq_len)
        f.write(header)

        # next write out the embedding weights
        serialize(self.tok_embeddings.weight)

        # now all the layers
        # attention weights
        for layer in self.layers:
            serialize(layer.attention_norm.weight)
        for layer in self.layers:
            serialize(layer.attention.wq.weight)
        for layer in self.layers:
            serialize(layer.attention.wk.weight)
        for layer in self.layers:
            serialize(layer.attention.wv.weight)
        for layer in self.layers:
            serialize(layer.attention.wo.weight)
        # ffn weights
        for layer in self.layers:
            serialize(layer.ffn_norm.weight)
        for layer in self.layers:
            serialize(layer.feed_forward.w1.weight)
        for layer in self.layers:
            serialize(layer.feed_forward.w2.weight)
        for layer in self.layers:
            serialize(layer.feed_forward.w3.weight)
        # final rmsnorm
        serialize(self.norm.weight)
        # note: no need to write final classifier weights due to weight sharing
        # freqs_cis
        serialize(self.freqs_cos[:p.max_seq_len])
        serialize(self.freqs_sin[:p.max_seq_len])

        # write to binary file
        f.close()
        print(f"wrote {filepath}")
