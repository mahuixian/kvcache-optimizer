from kvcache.streamingllm import StreamingLLM

def tools(item):
    if item == 'streamingllm':
          tool = StreamingLLM(4, 512, 1, 1)
          
    return tool