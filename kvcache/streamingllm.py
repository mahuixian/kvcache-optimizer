import torch

def slice1d(x, start, end):
    return x[:, start:end, ...]


def slice2d(x, start, end):
    return x[:, :, start: end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start: end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


class StreamingLLM:
    def __init__(
        self,
        initial_size=4,
        rolling_size=512,
        k_seq_dim=1,
        v_seq_dim=1,
    ):
        self.initial_size = initial_size
        self.rolling_size = rolling_size
        self.cache_size = initial_size + rolling_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]
        
    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len < self.cache_size:
            return past_key_values
        
        return [
            (
                torch.cat(
                    [
                        self.k_slice(k, 0, self.initial_size),
                        self.k_slice(k, seq_len - self.rolling_size, seq_len)
                    ], 
                    dim = self.k_seq_dim
                ), 
                torch.cat(
                    [
                        self.v_slice(v, 0, self.initial_size),
                        self.v_slice(v, seq_len - self.rolling_size, seq_len)
                    ], 
                    dim = self.v_seq_dim
                )
            )
            for k, v in past_key_values
        ]
