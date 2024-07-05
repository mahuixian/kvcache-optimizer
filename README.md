通过优化KV Cache，提升LLM提升长文本外推性能

StreamingLLM：

https://github.com/mit-han-lab/streaming-llm/tree/main

streamingllm会单独分出一部分内存来存储和重用attention sink和rolling cache的KV状态，这样LLM的输入就可以无限长。streamingllm无法应用于使用了绝对位置编码的LLM。

![](/img/1.png)
