---
layout: distill
title: "在 TPU 上训练 LLaMA 3"
# permalink: /main/
description: "让我们用上一节学到的东西，近距离看一看：在 TPU v5p 上训练 LLaMA 3 系列模型到底是什么情形。它们有多大？不同配置下训练要花多少钱？它们是怎么分片的？让我们动手做几个'纸面估算'，把前面几节的方法落到真实模型上。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 6

previous_section_url: "../training"
previous_section_name: "第 5 部分 · Transformer 训练并行"

next_section_url: ../inference
next_section_name: "第 7 部分 · 推理"

bibliography: main.bib

giscus_comments: true
giscus_discussion: 8

authors:
  - name: Jacob Austin
    url: "https://www.jacobaustin.org/"
    affiliations:
      name: Google DeepMind
  - name: Sholto Douglas
    url: "https://x.com/_sholtodouglas"
  - name: Roy Frostig
    url: "https://cs.stanford.edu/~rfrostig/"
  - name: Anselm Levskaya
    url: "https://anselmlevskaya.com/"
  - name: Charlie Chen
    url: "https://x.com/charliexychen"
  - name: Sharad Vikram
    url: "https://sharadvikram.com/"
  - name: Federico Lebron
    url: "https://fedelebron.com/"
  - name: Peter Choy
    url: "https://x.com/pchoy95"
  - name: Vinay Ramasesh
    url: "https://x.com/vinayramasesh"
  - name: Albert Webson
    url: "https://representation.ai/"
  - name: Reiner Pope<sup>*</sup>
    url: https://x.com/reinerpope

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "LLaMA 3 长什么样？"
  - name: "数参数量和 FLOPs"
  - name: "训练 LLaMA 3-70B 时该怎么分片"
  - name: "动手解题"

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

_这一节的目标，是把上一节的结果用在一个非常实际的问题上：训练 LLaMA 3 这个家族的模型。和前几节不同，我们希望你自己多动手算。所以每一部分的答案都被折叠起来了，先自己答，再对答案。强烈建议拿支笔手算一遍！_

### LLaMA 3 长什么样？

LLaMA-3 模型家族<d-cite key="llama3"></d-cite>主要有三个模型：LLaMA 3 8B、70B 和 405B。我们主要盯 70B，8B 和 405B 留给你在文末的习题里探索。这是 LLaMA 3-70B 的架构，取自 LLaMA 的 [HuggingFace 页面](https://huggingface.co/NousResearch/Meta-Llama-3-70B/blob/main/config.json)。

| **超参数**                  | **值**      |
| --------------------------- | --------- |
| $$n_\text{layers}$$ (L)     | 80        |
| $$d_\text{model}$$ (D)      | 8,192     |
| $$d_{ff}$$ (F)              | 28,672    |
| $$n_\text{heads}$$ (N)      | 64        |
| $$n_\text{kv_heads}$$ (K)   | 8         |
| $$d_\text{qkv}$$ (H)        | 128       |
| $$n_\text{embeddings}$$ (V) | 128,256   |

这些数据有多好找呢——配置文件本身就长这样，我们还标了对应关系：

{% include figure.liquid path="assets/img/llama-json.png" class="img-fluid" %}

_建议为各个开源大模型整理一张这样的大表，这样能一眼看出它们在架构上做了哪些取舍。_

### 数参数量和 FLOPs

**问题：** 看着这张表，我们能算出 LLaMA 3-70B 的参数量吗？🤫 套用[第 4 部分](../transformers)的方法，看看能不能算出 70B！

| 参数            | 公式                                                                                                                                           | 数量                                                        |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| FFW 参数       | d_model * d_ff * 3（SwiGLU 的门控、上、下三个投影）* n_layers                                                                                         | 8,192 * 8,192 * 3.5 * 3 * 80 = **56.3e9**                    |
| 词表参数     | 2（输入和输出嵌入）* n_embeddings * d_model                                                                                          | 2 * 128,256 * 8,192 = **2.1e9**                              |
| 注意力参数 | n_layers * [ 2（q 投影与拼接后的输出投影）* d_model * n_heads * d_qkv + 2（k 和 v）* d_model * n_kv_heads * d_qkv] | 80 * (2 * 8,192 * 64 * 128 + 2 * 8,192 * 8 * 128) = **12e9** |
|                  |                                                                                                                                                   | 56.3e9 + 2.1e9 + 12e9 = **70.4e9**                           |

太好了！算出来的正是我们预期的数字。如你所料，FFW 参数在总参数量中占绝对大头，不过注意力部分也不可忽略。

<p markdown=1 class="takeaway">**要点**：MLP 块里那三个大权重矩阵，比 Transformer 里其他任何数组都大得多，所以讨论模型内存或 FLOPs 时，其他参数几乎都可以忽略。对 LLaMA 3-70B 来说，70B 参数里有 56B 来自这三个矩阵。</p>

下面来看 FLOPs！*先把[第 4 部分](../transformers)里的训练通用法则想起来。*

**问题：** LLaMA-3 每个训练步、每个 token 要做多少 FLOPs？_这能帮我们算出整个训练有多贵。_

{% details 想好之后再点开答案！ %}

**答案**：如[第 4 部分](../transformers)所示，每个 token 大约做 $$6 \cdot \text{param count}$$ FLOPs，这里就是大约 `6 * 70e9 = 4.2e11` FLOPs / token——每步每 token 大约半 TFLOP。只要处于计算受限状态，单块 TPU v5p 在算力完美利用下做这一步大约需要 `4.2e11 / 4.59E+14 = 1ms`。

{% enddetails %}

**问题：** LLaMA 3 一共训练了约 15 万亿（trillion）个 token，总共是多少 FLOPs？

{% details 想好之后再点开答案！ %}

**答案**：简单，`4.2e11 * 15e12 = 6.3e24 FLOPs`。6.3 yottaFLOPs，非常庞大！在单块 TPU 上这要跑 `6.3e24 / 4.59E+14 = 435 年`。这也是个吓人的数字！

{% enddetails %}

**问题：** 假设我们要在满配 TPU v5p 整机群（16x20x28 = 8960 块芯片）上训练。bfloat16、40% MFU、计算受限，需要训练多久？

{% details 想好之后再点开答案！ %}

**答案**：每块 TPU v5p 能跑 4.59e14 FLOPs/秒。40% MFU 下需要 `T = 6.3e24 / (8960 * 4.59e14 * 0.4) = 3.8e6 秒`。**大约 44 天！** 这相当合理——前提是我们真能做到 40% MFU。

{% enddetails %}

**问题：** LLaMA 3-70B 预训练用的批大小约为 400 万 token。要用这个批大小训练，至少需要多少块 TPU？_可以假设 bfloat16 参数、float32 优化器状态，且每层做 2 次激活检查点（层开头、注意力之后各一次）。_

{% details 想好之后再点开答案！ %}

**答案**：这题主要考内存，因为内存是唯一的硬约束。训练时 HBM 的三大用户是：模型参数、优化器状态、激活检查点。假设 bfloat16 权重、float32 优化器状态、外加一个*非常*保守的检查点方案（每层 2 次），我们有：

| **参数** | 2 * 70GB | ~140GB |
| **优化器状态** | 8 * 70GB | ~560GB |
| **激活检查点** | 2 * 8192 * 4e6 * 2 * 80 | ~10.5TB |
| **总计**                |                         | ~11.2TB |

总共约 11.2TB。你会发现激活检查点才是内存大头——哪怕检查点已经放得很保守。我们其实还可以激进到每层 1 个检查点、或者做微批，但眼下这个假设是合理的。按这些假设，每块 TPU v5p 有 96GB HBM，所以需要 `11.2e12 / 96e9 = 117` 块 TPU。其实不多！

*那为什么不这么干？* 因为训练要跑 `44 天 * 8960 / 117 = 3369 天`，快十年了。**太慢了。** 不过这恰好说明：我们用这么大的集群，不是因为内存装不下，而是需要更多的 FLOPs。另外，检查点做得这么稀疏，实际计算量接近 8ND 而不是 6ND。
{% enddetails %}

**问题：** 还是上面那题的假设，如果我们用 8960 块 TPU v5p，每块芯片会用多少内存？

{% details 想好之后再点开答案！ %}

**答案**：总内存还是约 11.2TB，摊到每块芯片只要约 1.3GB，几乎可以忽略。就算检查点做得激进得多，比如每层 12 个检查点，每台也只有 8GB。在这个规模下训练，离内存受限还远得很。

{% enddetails %}

<p markdown=1 class="takeaway">**要点**：哪怕非常小的拓扑，理论上也能训练非常大的模型——只是大概率要跑到天荒地老。会算训练总 FLOPs 之后，给定一个适中的 MFU 和已知的拓扑，我们就能对训练时长来个 ballpark 估算。</p>

### 训练 LLaMA 3-70B 时该怎么分片

沿用上面的设定：在 8960 块芯片的 TPU v5p 整机群上，以 400 万 token 的批大小（每批 1024 条长度为 4096 的序列）训练 LLaMA 3-70B。我们来讨论这个模型最好的分片策略。

**问题：** 在上面的假设下，只用 FSDP 能训练吗？先假设不能做任何序列/上下文并行。_这应该是你冒出的第一个念头——它最简单，而且如果行得通，不引入任何额外通信。_

{% details 想好之后再点开答案！ %}

**答案**：这个答案会有点较真。上面说过，LLaMA 3-70B 一开始是用长度 4K 的序列训练的，所以 400 万 token 的批对应*序列批大小* 1024。这意味着纯数据并行/FSDP 最多只能做到 1024 块芯片——因为我们只有这么多条序列可以拿来切。所以在“完全不额外通信的完全数据并行”这个严格意义上，答案是：不行。下一题会回答一个不那么较真的版本。

{% enddetails %}

**问题：** 放宽“不做序列切分”的要求。如果允许沿批维*和*序列维一起做 FSDP，只用 FSDP 能在 8960 块芯片上训练 LLaMA 3-70B 吗？

{% details 想好之后再点开答案！ %}

**答案**：允许序列/上下文并行后，能扩的规模大多了。先算每设备批大小：做 8960 路 FSDP 的话，每台 TPU 的批是 `4 * 1024 * 1024 / 8960 = 468 tokens`。上一节知道，FSDP 在 $$\text{per device batch size} < 2550 / M_X$$ 时被 ICI 卡住。满配 3D 整机群允许我们用 3 条轴，下限是 850，而我们远低于它。**所以答案是不行——哪怕用满 3 条轴，也会被通信结结实实地卡住。**

{% enddetails %}

**问题：** 现在来看张量并行 + FSDP 的混合方案。存在某种组合让我们保持计算受限吗？如果存在，FSDP 和张量并行各做多少？

{% details 想好之后再点开答案！ %}

**答案**：先确认这能不能行。已知每芯片批大小低于 $2550^2 / 2F = 113$ 时会被通信卡住；上面算过我们略高于这个线，太好了！接着用这个公式选 FSDP 的最优规模：

$$X_{opt} = \sqrt{\frac{2BN}{F}} = \sqrt{\frac{2 \cdot 4.19e6 \cdot 8960}{28672}} = 1618$$

取个合适的 2 的倍数，大约就是 2048 路 FSDP + 4 路张量并行。这样应该跑得不错！

{% enddetails %}

<p markdown=1 class="takeaway">**要点**：用“数据并行（1024 路）+ 序列并行（2 路）+ 张量并行（4 路）”的组合，我们就能在满配 TPU v5p 整机群上以 400 万 token 的批训练 LLaMA-3，且不被通信卡住。若只用纯 FSDP、或 FSDP+序列并行，就会被通信卡住。上一节推的那些公式真的很有用。</p>

## 动手解题

**问题 1 [把 LLaMA 70B 扩到更多芯片]：** 假设要在 4 个整机群上以同样的批大小训练 LLaMA 3-70B。该用什么并行方案？会是计算受限还是通信受限？训练大概要多久？*注意用对 roofline 界限。*

**问题 2 [LLaMA 405B]：**

(a) 用 LLaMA 3-405B 的[配置文件](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-405B/blob/main/config.json)（它是带门控的模型，可能需要登录申请权限才能看到），照上文做一张关键超参数的大表。这个模型一共有多少参数？每个训练步多少 FLOPs？训练 15T token 一共多少 FLOPs？

(b) 假设要在 8 个 TPU v5p 整机群上训练。该用什么并行方案？训练要多久？会是计算受限还是通信受限？

<h3 markdown=1 class="next-section">第 6 部分到此结束！第 7 部分讲 Transformer 推理，[点击这里](../inference)。</h3>
