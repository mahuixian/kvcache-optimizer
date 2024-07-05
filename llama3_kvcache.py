import json
import torch
from tqdm import tqdm
from llama.generation import Llama
from llama.tokenizer import ChatFormat, Tokenizer
from utils.utils import load_jsonl
from tools.tools import tools     


def main(model_path, tokenizer_path, testfile_path, max_gen_len):
    model = Llama(model_path, max_seq_len=8192, max_batch_size=6)
    # tokenizer = ChatFormat(Tokenizer(tokenizer_path))
    tokenizer = Tokenizer(tokenizer_path)
    formatter = ChatFormat(tokenizer)
    samples = load_jsonl(testfile_path)
    tool = tools("streamingllm")
    
    past_key_values = None
    for sample in samples:
        input_ids = [formatter.encode_dialog_prompt(sample)]
        past_key_values, generation_tokens, generation_logprobs = model.generate(
            input_ids=input_ids,
            max_gen_len=max_gen_len,
            past_key_values=past_key_values
        )
        
        answer = "".join([tokenizer.decode(token) for token in generation_tokens])
        print("assistant: ", answer)
        past_key_values = tool(past_key_values)
        

if __name__ == "__main__":
    model_path = ""
    tokenizer_path = ""
    testfile_path = ""
    max_gen_len = 1024
    main(model_path, tokenizer_path, testfile_path, max_gen_len)
    
