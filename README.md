# 通过优化KV Cache，提升LLM提升长文本外推性能

## StreamingLLM

https://github.com/mit-han-lab/streaming-llm/tree/main

streamingllm会单独分出一部分内存来存储和重用attention sink和rolling cache的KV状态，这样LLM的输入就可以无限长。streamingllm无法应用于使用了绝对位置编码的LLM。

![](/img/1.png)

在多轮对话中应用streamingllm时发现头部attention sink的kvcache貌似没有被更新，实际上是因为在拼接后的头部token也会被分配不必要的强注意力，这一部分token没有实际的语义信息，但是由于具有强注意力，所以还是进行保留。

## H2O:Heavy Hitter Oracle

https://arxiv.org/pdf/2306.14048v1

在Attention机制中，attention分数比较大的token对计算结果的影响比较大，H2O将这些tokens称为Heavy Hitters（H2），只要保证保留的KV Cache中存在H2，即可确保结果。所以H2O通过贪心算法，将当前token与其他token的注意力按列相加即得到当前token的重要性分数，然后将分数低的token丢弃即可。

![image.png](/img/1720167597632-482e02cf-a37d-4429-92af-b2f6467cbca4.png)



一般来说attention scores的size为（bsz, num_heads, seq_len, kv_len），需要先按照num_heads将attention scores相加，然后按照seq_len的维度将attention scores相加，得到最终的重要性分数。
