class H2O:
    def __init__(
        self,
        hh_size: int = 4,
        rolling_size: int = 512,
        k_seq_dim: int = 1,
        v_seq_dim: int = 1
    ):
        self.hh_size = hh_size
        self.rolling_size = rolling_size
        self.cache_size = hh_size + rolling_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_score = None
        
    def __call__(self, past_key_values, attn_score_cache):
        