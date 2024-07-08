import torch


def slice1d(x, keep_indices):
    return torch.cat([x[bsz, keep_indices[bsz], ...] for bsz in range(x.shape[0])], dim=0)


def slice2d(x, keep_indices):
    return torch.cat([x[bsz, :, keep_indices[bsz], ...] for bsz in range(x.shape[0])], dim=0)


def slice3d(x, keep_indices):
    return torch.cat([x[bsz, :, :, keep_indices[bsz], ...] for bsz in range(x.shape[0])], dim=0)


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


class H2O:
    def __init__(
        self,
        hh_size,
        recent_size,
        k_seq_dim=1,
        v_seq_dim=1
    ):
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.cache_size = hh_size + recent_size
        self.k_slice = DIM_TO_SLICE(k_seq_dim)
        self.v_slice = DIM_TO_SLICE(v_seq_dim)
        
    
    def __call__(self, past_key_values, attn_scores):
        if past_key_values is None:
            return None
        
        kv_len = past_key_values[0][0].shape[self.k_seq_dim]
        if kv_len <= self.cache_size:
            return past_key_values
        
        #已测试，需要按照头将注意力合并，再去获取indices
        attn_scores = attn_scores.sum(1).sum(1) #(bsz, num_heads, seq_len, kv_len) #(bsz, seq_len, kv_len) #(bsz, kv_len)
        select_hh_scores = attn_scores[:, :-self.cache_size]
        _, topk_indices = torch.topk(select_hh_scores, self.hh_size, dim=-1) #(bsz, hh_size)
        keep_recent_indices = torch.arange(kv_len - self.cache_size, kv_len).unsqueeze(0).expand(topk_indices.shape[0], -1)
        keep_indices = torch.cat([topk_indices, keep_recent_indices], dim=-1) #(bsz, self.cache_size)
        assert keep_indices.size(-1) == self.cache_size
        return [
            (
                self.k_slice(k, keep_indices),
                self.v_slice(v, keep_indices)
            )
            for k, v in past_key_values
        ]
        
        