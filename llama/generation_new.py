import json
import os
import sys
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict
from llama.model import ModelArgs, Transformer
from llama.tokenizer import ChatFormat, Tokenizer
# from llama.tokenizer import ChatFormat, Dialog, Message, Tokenizer
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

class CompletionPrediction(TypedDict, total=False):
    generated: str
    tokens: List[int]
    logprobs: List[float]
    

class Llama: 
        
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        item: str,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "Llama":
        """
        Build a Llama instance by initializing and loading a model checkpoint.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.
        """
        assert 1 <= max_seq_len <= 8192, f"max_seq_len must be between 1 and 8192, got {max_seq_len}."
        assert os.path.isdir(ckpt_dir), f"Checkpoint directory '{ckpt_dir}' does not exist."
        assert os.path.isfile(tokenizer_path), f"Tokenizer file '{tokenizer_path}' does not exist."
        
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        
        tokenizer = Tokenizer(model_path=tokenizer_path)
        assert model_args.vocab_size == tokenizer.n_words
        
        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args, item)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer)
    
    
    def __init__(self,
                 model,
                 tokenizer,
                 ):
        self.model = model
        self.tokenizer = tokenizer,
        self.formatted = ChatFormat(tokenizer)
       
    
    @torch.inference_mode()
    def generate(self, 
                 input_ids,
                 max_gen_len: int,
                 temperature: float = 0.8,
                 top_p: float = 0.95,
                 past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
                 logprobs: bool = False,
                 echo: bool = False,
                 ):
        
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            past_key_values: past_key_values
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        
        bsz = len(input_ids)
        assert bsz <= self.max_batch_size, f"Batch size {bsz} exceeds max batch size {self.max_batch_size}"
        min_prompt_len = min([len(x) for x in input_ids])
        max_prompt_len = max([len(x) for x in input_ids]) 
        assert max_prompt_len <= self.max_seq_len, f"Prompt length {max_prompt_len} exceeds max sequence length {self.max_seq_len}"
        total_len = min(self.max_seq_len, max_gen_len + max_prompt_len) #min(训练长度, 生成长度+输入最大长度)
        
        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")

        for k, t in enumerate(input_ids):
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float32)
    
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens))

        #prefill
        pre_tokens = tokens[:, :min_prompt_len]
        logits, past_key_values = self.model(pre_tokens, past_key_values)
        next_token = sample(logits, temperature, top_p)
        #这边需要考虑一个问题，min_prompt_len == total_len,容易out of index
        next_token = torch.where(
            input_text_mask[:, min_prompt_len], tokens[:, min_prompt_len], next_token
        )
        next_token = next_token.to(tokens.device).to(tokens.dtype)
        tokens[:, min_prompt_len] = next_token
    
        #decode
        for step in range(min_prompt_len+1, total_len):
            logits, past_key_values = self.model(next_token, past_key_values)
            next_token = sample(logits, temperature, top_p)
            next_token = torch.where(input_text_mask[:, step], tokens[:, step], next_token)
            tokens[:, step] = next_token
            
            eos_reached |= (~input_text_mask[:, step]) & (
                torch.isin(next_token, stop_tokens)
            )
            
            if all(eos_reached):
                break
        
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(input_ids[i])
            toks = toks[start : len(input_ids[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(input_ids[i]) + max_gen_len]
            # cut to after eos tok if any
            for stop_token in self.tokenizer.stop_tokens:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                    probs = probs[:eos_idx] if logprobs else None
                except ValueError:
                    pass
            out_tokens.append(toks)
            out_logprobs.append(probs)
            
        return (past_key_values, out_tokens, out_logprobs if logprobs else None)
            


def sample(logits, temperature, top_p):
    if temperature > 0:
        probs = torch.softmax(logits[:, -1]/temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
    else:
        next_token = torch.argmax(logits[:, -1], dim=-1)  
        
    return next_token.unsqueeze(1)
        

def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
        
        
    
    def prefill(self, tokens):
        pass
        
        
    def decode(self, tokens):
        pass
        