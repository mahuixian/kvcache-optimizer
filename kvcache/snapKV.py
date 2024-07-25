class SnapKV:
    def __init__(self, 
                 window_size: int = 16, 
                 max_capacity_prompt: int = 256 + 16, 
                 kernel_size: int = 5, 
                 pooling: str = 'avgpool'
        ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.kernel_size = kernel_size
        self.pooling = pooling
        
    
    def reset(
            self,
            window_size: int = 16, 
            max_capacity_prompt: int = 256 + 16, 
            kernel_size: int = 5, 
            pooling: str = 'avgpool'
        ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
    
    
    def update_kv(self, q_states, k_states, v_states, attention_mask, num_key_value_groups):
        #prefill
        assert k_states.shape[-2] == q_states.shape[-2]
        bsz, num_heads, seq_len, head_dim = q_states.shape
        if seq_len <= self.max_capacity_prompt:
            return k_states, v_states
        else:
            
    