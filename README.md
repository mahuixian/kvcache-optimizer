# 通过优化KV Cache，提升LLM提升长文本外推性能

## StreamingLLM

https://github.com/mit-han-lab/streaming-llm/tree/main

streamingllm会单独分出一部分内存来存储和重用attention sink和rolling cache的KV状态，这样LLM的输入就可以无限长。streamingllm无法应用于使用了绝对位置编码的LLM。

![](/img/1.png)

在多轮对话中应用streamingllm时发现头部attention sink的kvcache貌似没有被更新，实际上是因为在拼接后的头部token也会被分配不必要的强注意力，这一部分token没有实际的语义信息，但是由于具有强注意力，所以还是进行保留。

## H2O:Heavy Hitter Oracle

https://arxiv.org/pdf/2306.14048v1

H2O引入了一个基于累计注意力得分的贪婪选择算法的KV缓存驱逐策略，在生成步骤中基于得分选择标记。

![image.png](/img/1720167597632-482e02cf-a37d-4429-92af-b2f6467cbca4.png)

一般来说attention scores的size为（bsz, num_heads, seq_len, kv_len），需要先按照num_heads将attention scores相加，然后按照seq_len的维度将attention scores相加，得到最终的重要性分数。

## SnapKV

SnapKV是一种高效压缩LLM中KV Cache的方法。

根据实验观察到：

- 无论生成的上下文长度如何，特定键在prompt中始终显示出较高的注意力权重，这些“活跃”的键遵循与prompt的结构和内容内在相关且稳定的模式。
- 在长摘要和问答任务中，问题在prompt中的位置（开头或结尾）不会显著改变观察到的注意力模式的稳定性。这表明，无论问题的位置如何，都可以轻松获取相关特征的注意力
- 观察到的注意力模式与用户提出的 特定指令有很强的关联，表明上下文感知的关键值（KV）压缩方法可能带来更好的性能

![image-20240710165634463](/img/image-20240710165634463.png)

关键步骤：

- 基于Observation Window计算Prompt各个位置的注意力权重之和行

- 使用1维Pooling进行权重聚类,选出Top-k的位置索引行

- 利用gather操作提取选中位置的Key和Value行

- 将压缩的KV与Observation Window对应的KV拼接,得到最终结果行



# Dual chunk Attention

块内attention，块间attention计算以及连续块attention计算。

paper：https://arxiv.org/html/2402.17463v2

code：https://github.com/HKUNLP/ChunkLlama.

![image-20240725143425244](/img/image-20240725143425244.png)
