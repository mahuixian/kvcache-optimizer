from torch import nn
import torch
from typing import Optional, Tuple, List
from dataclasses import dataclass
import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from kvcache.h2o import H2O


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    x: torch.Tensor,
    dim: int
):
    seq_len = x.shape[1]
    freqs_cis = precompute_freqs_cis(dim, seq_len)
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, x_) #q、k的head_dim相同 -> (1, seqlen, 1, head_dim)
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return x_out.type_as(x)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )
    

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight      


class CausalAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size #一条pipeline拥有的q的数量
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size #一条pipeline拥有的kv的数量
        self.n_rep = self.n_local_heads // self.n_local_kv_heads #kv需要复制的数量
        self.head_dim = args.dim // args.n_heads
        
        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        ) 
        
    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor],
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        
        kv_len = 0
        if past_key_value is not None:
            kv_len = past_key_value[0].shape[1]
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        
        past_key_value = (xk, xv)
    
        xq = apply_rotary_emb(xq, self.head_dim)
        xk = apply_rotary_emb(xk, self.head_dim)
        
        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)
        
        
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        scores = torch.matmul(xq, keys.transpose(2, 3)) / torch.sqrt(self.head_dim)
        if mask is not None:
            mask = torch.hstack([torch.zeros((seqlen, kv_len)), mask]).type_as(x)
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)
        return output, past_key_value 
             

class H2OAttention(CausalAttention):
    def __init__(self, args):
        super().__init__(args)
        self.h2o = H2O(
            hh_size=1,
            recent_size=128,
            k_seq_dim=1,
            v_seq_dim=1
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        
        kv_len = 0
        if past_key_value is not None:
            kv_len = past_key_value[0].shape[1]
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        
        past_key_value = (xk, xv)
        
        xq = apply_rotary_emb(xq, self.head_dim)
        xk = apply_rotary_emb(xk, self.head_dim)
        
        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)
        
        
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        scores = torch.matmul(xq, keys.transpose(2, 3)) / torch.sqrt(self.head_dim)
        if mask is not None:
            mask = torch.hstack([torch.zeros((seqlen, kv_len)), mask]).type_as(x)
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        
        past_key_value = self.h2o(past_key_value, scores.detach().clone())
        
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)
        return output, past_key_value



class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))



ATTENTIONCHOICE = {
    "h2o": H2OAttention,
    "streamingllm": CausalAttention
}

class TransformerBlock():
    def __init__(self, layer_id: int, args: ModelArgs, item: str):
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        # self.attention = AttentionFactory().create_attention(item, args)
        self.attention = ATTENTIONCHOICE[item](args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        
    
    def forward(
        self,
        x: torch.tensor,
        mask: Optional[torch.tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ):
        output, past_key_value = self.attention(self.attention_norm(x), mask, past_key_value)
        h = x + output
        return h, past_key_value


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        
        self.embeddings = VocabParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )
        
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        
        
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )
        
        
    @torch.inference_mode()
    def forward(
        self,
        tokens: torch.Tensor,
        past_key_values: Optional[List] = None
    ):
        _bsz, seqlen = tokens.shape
        tokens_embeddings = self.embeddings(tokens)
        
        if past_key_values is None:
            past_key_values = [None for _ in range(len(self.layers))]        
        
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
    
        for i, layer in enumerate(self.layers):
            output, past_key_value = layer(tokens_embeddings, mask, past_key_values[i])
            past_key_values[i] = past_key_value
        
        
        output = self.norm(output)
        output = self.output(output).float()
        return output, past_key_values
        
            
          
        
        
        
        
            
        
            
            
        
        