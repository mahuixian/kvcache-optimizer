from tqdm import tqdm
from llama.generation import Llama
from utils.utils import load_jsonl

def inference(model_path, tokenizer_path, data_path, max_batch_size, item):
    llama = Llama.build(
        ckpt_dir=model_path,
        tokenizer_path=tokenizer_path,
        max_seq_len=2048,
        max_batch_size=max_batch_size,
        item=item,
    )
    
    tokenizer = llama.tokenizer
    formatter = llama.formatter
    samples = load_jsonl(data_path)
    
    past_key_values = None
    for sample in samples:
        input_ids = [formatter.encode_dialog_prompt(sample)]
        out_tokens, past_key_values = llama.generate(
            input_ids,
            max_gen_len=1024,
            temperature=0.8,
            top_p=0.95,
            past_key_values=past_key_values,
        )
        
        answer = "".join([tokenizer.decode(token) for token in out_tokens])
        print(answer)
        

if __name__ == '__main__':
    inference(
        model_path="./models/llama-7b",
        tokenizer_path="./models/llama-7b/tokenizer.model",
        data_path="./data/test.jsonl",
        max_batch_size=1,
        item="h2o",
    )