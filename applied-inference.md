---
layout: distill
title: "在 TPU 上把 LLaMA 3-70B 服务起来"
# permalink: /main/
description: "让我们近距离看一看：怎么在 TPU v5e 上把 LLaMA 3-70B 服务起来。按屋顶线算，服务一个模型的各种配置分别要花多少钱？KV 缓存有多大？该用多大的批大小？推理时参数和激活怎么分片？让我们来几发“生产级”的纸面估算，把延迟和吞吐都算出来。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 8

previous_section_url: "../inference"
previous_section_name: "第 7 部分 · 推理"

next_section_url: ../profiling
next_section_name: "第 9 部分 · 性能剖析"

giscus_comments: true
giscus_discussion: 10

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
  - name: "LLaMA 的服务化是怎么一回事？"
  - subsections:
    - name: "想想吞吐量"
    - name: "预填充呢？"
  - name: "把“延迟-吞吐”的取舍画出来看"
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

*这一节我们来看看：要把 LLaMA-3 服务起来需要付出什么、能做得多高效。和上一个"实战"章节一样，请先用纸笔自己算一遍，再点开答案！*

## LLaMA 的服务化是怎么一回事？

先温习一下 LLaMA 3-70B 长什么样（参见[第 6 部分](../applied-training)）：

| **超参数**                  |  **值**   |
| --------------------------- | :-------: |
| $$n_\text{layers}$$ (L)     |    80     |
| $$d_\text{model}$$ (D)      |   8,192   |
| $$d_{ff}$$ (F)              |  28,672   |
| $$n_\text{heads}$$ (N)      |    64     |
| $$n_\text{kv heads}$$ (K)   |     8     |
| $$d_\text{qkv}$$ (H)        |    128    |
| $$n_\text{embeddings}$$ (V) |  128,256  |

先从最简单的问题开始：**该用什么硬件来服务？** 答案基本是：每美元能买到最多 FLOPs 的那个<d-footnote>并不总是这样——有时 HBM 容量或 ICI 带宽才是命门，FLOPs/美元反倒次要。但作为启发式，这条很好用。</d-footnote>所以我们通常会选 TPU v5e——我们当前的专职推理芯片（价格取自 2025 年 2 月的 [Google Cloud 价目表](https://cloud.google.com/tpu/pricing)）：

| **TPU 型号** | **bfloat16 FLOPs/s** | **Google Cloud 美元 / 小时** | **FLOPs / 美元** |
| ------------ | :------------------: | :-------------------------: | :-----------: |
| H100         |        9.9e14        |            $10.8            |    3.3e17     |
| v5p          |       4.59e14        |            $4.2             |    3.9e17    |
| v5e          |       1.97e14        |            $1.2             |  **5.8e17**  |

每块 TPU v5e 只有 16GB HBM，这会逼着我们把模型切得狠一点。先算几个绕不开的基本量：

**问题：** LLaMA 3-70B 每 token 的 KV 缓存有多大？*可以假设 KV 用 int8 存。这个数决定了给定拓扑下我们的批能做到多大。*

{% details 想过一遍之后再点开！ %}

LLaMA 3-70B 有 8 个 KV 头，所以每 token 的大小是 `2 * K * H * L = 2 * 8 * 128 * 80 = 160kB`。

**感受一下这是多大！** 序列长度取常见的 32k 时，一条序列就要 `160e3 * 32,768 = 5.3GB`。批 240 的话，一共 1.3TB！TPU v5e 每块才 16GB，光是装下这些内存就要 `(70e9 + 1.3e12) / 16e9 = 86` 块。再对比一下：模型参数才 70GB——KV 缓存比参数还大得多。

{% enddetails %}

**问题：** 假设要以批 32、序列长度 8192 服务 L3 70B，参数和 KV 全用 int8。总共要占多少内存？最小的服务切片是多大？

{% details 答案 %}

int8 下 KV 是每 token `160e3` 字节，KV 总内存 `160e3 * 8192 * 32 = 41.9e9` 字节。每个参数 1 字节，参数共 `70e9` 字节。合计 `41.9e9 + 70e9 = 112GB`。

最小的切片是 `112e9 / 16e9 = 7` 块 TPU，向上取整成规整尺寸就是 TPU v5e `4x2`。这会挤得很紧，算上别的开销可能真装不下，所以八成至少要 `4x4`（或者把批调小）。

{% enddetails %}

**问题：** 在这个批大小和量化方式下，在 TPU v5e `4x2` 上每步解码大约什么延迟？吞吐（tokens/秒/芯片）呢？`4x4` 呢？*假设 FLOPs 用 bfloat16 计算，一切都完全分片。*

{% details 答案 %}

直接套上一节的公式：

$$\begin{align*}
\tiny \text{Theoretical Step Time (General)} = \underbrace{\frac{\text{Batch Size} \times \text{KV Cache Size}}{\tiny \text{Total Memory Bandwidth}}}_{\text{Attention (always bandwidth-bound)}} + \underbrace{\max\left(\frac{2 \times \text{Batch Size} \times \text{Parameter Count}}{\text{Total FLOPs/s}}, \frac{\text{Parameter Size}}{\text{Total Memory Bandwidth}}\right)}_{\tiny \text{MLP (can be compute-bound)}}
\end{align*}$$

参数是 int8、FLOPs 是 bfloat16，临界批大小约 120。右边那个 max 也可以手动比一遍，但那笔账我们已经算过好几遍了。**总之，无论矩阵乘还是 FLOPs，我们都深陷内存受限区。**

只看内存带宽，每步时间就是 `(KV size + param size) / (8 * HBM bandwidth) = 112e9 / (8 * 8.2e11) = 17ms`。**理论上每步约 17ms**，吞吐 `32 / .017 = 1882 tokens / sec`，即每芯片 `1882 / 8 = 235 tokens / sec`。

这里有个环节要检查：矩阵乘会不会被 ICI 卡住。这里最多能占用 2 条轴，理论上 ICI 受限的界限是 $Y > 2 * F / 2200 = 2 * 28672 / 2200 = 26$，我们远没到，稳！

换到 `4x4` 上，ICI 依然没问题，延迟会降到 `17 / 2 = 8.5ms`，但每芯片吞吐不变。

{% enddetails %}

### 想想吞吐量

专门花点时间只想吞吐这一件事。追求吞吐时，我们想进入计算受限状态，把 TPU 的 MXU 榨干。通常这意味着批越大越好——每块芯片干的活越多。

**问题：** 在 TPU v5e 上用 bfloat16 权重和激活，批要多大矩阵乘才计算受限？int8 权重、bfloat16 计算呢？int8 权重、int8 计算呢？

{% details 答案 %}

第 7 部分讲过，对任何 $B \ll D, F$ 的 bfloat16 矩阵乘，有

$$\begin{equation*}
T_\text{math} > T_\text{comms} \leftrightarrow \frac{2BDF}{2DF} \geq \frac{\text{TPU bfloat16 FLOPs/s}}{\text{HBM bandwidth}} = 240
\end{equation*}$$

权重换成 int8 后，分母少了一个因子 2，于是 $2BDF / DF = 2B > 240$，即 $B > 120$——临界批直接减半，太有帮助了！而 int8 权重配 int8 FLOPs 时，要用 int8 的算力值：TPU 从 bfloat16 的 1.97e14 涨到 3.94e14，几乎翻倍。于是又绕回 $B > 240$。

int8 权重 + bfloat16 计算是常见组合：无损量化参数，往往比直接做低精度算术容易得多。

{% enddetails %}

**问题：** 用 bfloat16、int8、int4（KV 和参数同精度）在 8k 上下文下服务 LLaMA 3-70B，最小的 TPU v5e 拓扑分别要多大？*这道题里可以把 KV 缓存忽略成零。*

{% details 答案 %}

简单！批小到无所谓的话，唯一的限制是参数装得进 HBM，即 `ceil(num_params * sizeof(dtype) / HBM per TPU)`，再向上取个规整拓扑（2 的倍数）：`ceil(70e9 * sizeof(dtype) / 16e9)`：

| 精度  | 参数大小 | 每 token KV 大小（字节） | 最少 v5e 数 | 实际最小切片 | 剩余 KV 用 HBM | 8k 下可存 KV 份数 |
| :---: | :--------: | :---------------------: | :----------: | :--------------: | :-------------------------: | :----------------: |
| bf16  |   140GB    |          324kB          |     8.75     |  4x4 = 16 块  |             116             |         43         |
| int8  |    70GB    |          162kB          |     4.38     |  4x2 = 8 块   |             58              |         43         |
| int4  |    35GB    |          81kB           |     2.81     |  2x2 = 4 块   |             29              |         43         |

挺有意思！也就是说真想塞的话，LLaMA 70B 能挤进一个 TPU v5e 2x2。只是你会注意到"可存 KV 份数"小得可怜——那就是我们的批大小！这意味着 FLOPs 利用率会非常难看。为了把批推到 240，我们完全乐意用更大的拓扑。

{% enddetails %}

**问题：** 假设就用这些拓扑装得下的最大批，每步生成的延迟会是多少？

{% details 答案 %}

也简单——我们把批选成刚好塞满 HBM 了！问题等价于"把一块 TPU v5e 的全部字节装进 MXU 要多久"：`v5e HBM / v5e HBM memory bandwidth = 16GB / 8.2e11 = 19ms`，即**每步 19ms**。若生成长度中位数 512 个 token，一次完整解码约 9 秒。注意小一点的批其实能换来更快的延迟：比如只装 int8 模型参数，每步的最小延迟约 10ms——因为 HBM 没被塞满。

{% enddetails %}

<p markdown=1 class="takeaway">**要点**：解码延迟永远可以用"把所有参数从 HBM 装进 MXU 要多久"来设下界。KV 缓存小的时候，可以把每一层理解成"按块搬进权重、用完就扔"。只要批不大、设备间通信不多，这个界通常靠谱（误差 1.5 倍以内）。批更大时，就得把 KV 缓存的读取也建模进来——那时它比参数更占头。</p>

同样，在计算受限区（比如训练或大批推理），可以用 $$\text{Total FLOPs} / (N \cdot C) = 2 \cdot \text{param count} \cdot B / (N \cdot C)$$ 设下界——它假设零通信。

**问题：** 上述每种配置，每芯片吞吐是多少（按"每芯片每秒几个请求"算）？*可以假设解码长度中位数 512 个 token。*

{% details 答案 %}

这个问题很要紧，因为它和每 token 成本严格挂钩。

按上面的解码长度假设，吞吐就是 $$B / (\text{per-step latency} \cdot \text{median steps} \cdot N) \approx 43 / (0.019 * 512 * N)$$，约为 $$(4.42 / N)$$ QPS。代入 $$N$$：

|   精度   | QPS / 芯片 |
| :------: | :--------: |
| bfloat16 |    0.27    |
|   int8   |    0.55    |
|   int4   |    1.11    |

注意这相当乐观——它完全没算前向的工作内存（激活和注意力要占的内存）。有 Flash Attention 在，这笔账不算离谱，但也绝不便宜。真实数字大约只有这里的一半。想要极限吞吐，我们多半得把芯片数翻上一倍还多、再把批显著拉大。

{% enddetails %}

**问题：** 如果上面每个例子的拓扑都翻倍，峰值吞吐会怎么变？

{% details 答案 %}

若用 4x8 切片跑 bfloat16，就有 372GB 余量给 KV 缓存，批可以拉到 140。每步时间不变，吞吐就是 `14.39 / num_chips`：

|      精度       | QPS / 芯片 |
| :---------------: | :--------: |
| bfloat16（4x8 上） |    0.44    |
|   int8（4x4 上）   |    0.90    |
|   int4（2x4 上）   |    1.80    |

再翻倍还有更大的甜头！最重要的结论是：**受 KV 缓存大小限制时，最小拓扑不等于最省钱的拓扑。**

{% enddetails %}

**问题：** 现在来聊分片。假设要在 TPU v5e 4x8 上用 bfloat16 服务。生成阶段该用什么模型分片？能不被通信卡住吗？

{% details 答案 %}

上一节说过，生成阶段的分片选择其实只有一个：模型并行。能做到多狠才被通信卡住？也讲过，被通信卡住大约在

$$Y > \frac{F \cdot M_Y}{2200}$$

LLaMA 3-70B 的 `F = 28,672`。若模型分片占 2 条轴，得 $$Y = 28672 \cdot 2 / 2200 = 26$$。也就是说一般能扩到 16 块芯片左右不被通信卡住——`4x4` 可以，`4x8` 不行。而且通常计算做不到完美重叠，连这个估计都偏乐观。

**要点：纯模型并行实际上服务不了 4x8。**最多 4x2，`4x4` 得碰运气。

不过前面也讲过，批小的时候可以多做模型并行而不太伤吞吐——模型卡在内存带宽而不是 FLOPs 上。那个界大约是 $Y=F / (8\cdot B)$：批取 64 时，理论上 `Y = 28,672 / (8 * 64) = 56` 路模型并行才被 ICI 卡住。验证一下：对单个矩阵乘比较 $T_\text{ici comms}$、$T_\text{hbm comms}$ 和 $T_\text{math}$，显然有

$$\begin{align*}T_\text{ici comms} = \frac{2BD}{W_\text{ici}} && T_\text{hbm comms} = \frac{2DF}{Y \cdot W_\text{hbm}} && T_\text{math} = \frac{2BDF}{Y \cdot C}\end{align*}$$

`4x8` 下：$T_\text{ici comms}$ = `(2 * 64 * 8192) / 9e10 = 11us`；$T_\text{hbm comms}$ = `(2 * 8192 * 28,672) / (32 * 8.2e11) = 18us`；$T_\text{math}$ = `(2 * 64 * 8192 * 28,672) / (32 * 1.97e14) = 4us`。理论上仍由 HBM 带宽定乾坤，完美！*注意从 `4x4` 扩到 `4x8`，吞吐上多半没好处，但延迟会降！*

int8 和 int4 那两套配置则 _可以_ 纯模型并行。于是我们撞上了一个有意思的节点：量化带来的好处不止"算得更快"——它让我们在被通信卡住之前能用上更大的批。**所以这一段的结论：4x8 上达不到峰值吞吐；但 int8 和 int4 配置可以纯模型并行。**

{% enddetails %}

<p markdown=1 class="takeaway">**提示**：有用的模型并行的最大路数取决于 $$d_{ff}$$ 和你用了多少条切分轴，通常在 8 到 32 之间，随模型大小而异。越过这条线继续切也能切——那就是拿吞吐换延迟了。</p>

### 预填充呢？

前面基本没提预填充，因为它简单得多。把几个概念拼起来，看看端到端的图景。

**问题：** 假设预填充能做到 40% 算力利用率。在 16 块 TPU v5e 上做一次长度 8192 的预填充要多久？

{% details 答案 %}

8k 个 token，稳处计算受限区，只算 FLOPs 就行。模型 `70e9` 参数，一次前向 `2 * 70e9 * B` FLOPs。按 40% MFU 算，耗时约 `2 * 70e9 * 8192 / (16 * 1.97e14 * 0.4) = 0.91s`。和前面的数字比，这可真不少！

{% enddetails %}

**问题：** 设预填充长度中位数 8192、解码长度中位数 4096，生成批大小为 32。平均每步有多少条序列解码完成？平均每步要从 KV 缓存里清掉多少 token？

{% details 答案 %}

这题挺直白。解码长度中位数 4096，意味着平均每 4096 个 token 完成一条序列。批 32，则每步淘汰 `32 / 4096` 条。KV 缓存长度约 `8192 + 4096`，所以每步清掉 `32 * (8192 + 4096) / 4096 = 96` 个 token。通式是 $B * (P + G) / G$，$P$、$G$ 分别是预填充和生成长度。

{% enddetails %}

**问题：** 假设做分离式服务，预填充长度中位数 8192、解码长度中位数 512，bfloat16，延迟就用上面算出来的那些。要让预填充和生成两组服务器都不闲着，预填充:生成的服务器数量比例该是多少？

{% details 答案 %}

这题有点意思。设预填充服务器 $P$ 台、生成服务器 $G$ 台。本质是个流水线问题：注入速率是 `P / prefill_latency`，消耗速率是 `B * G / (generate_latency * median_decode_length)`。前面算过一次预填充 `910ms`、批 43（就当 32）时每步解码 `19ms`。于是要 `P / 0.91 = 32 * G / (0.019 * 512)`，即 `P = 3G`——**预填充服务器要配成生成服务器的约 3 倍！**

{% enddetails %}

## 把“延迟-吞吐”的取舍画出来看

还是拿 LLaMA 70B 开刀，看看生成阶段不同批大小下的延迟和吞吐。上一节对 PaLM 已经展示过，这会画出一条吞吐/延迟的帕累托前沿。设 16 路张量并行——想保住 MLP 块的计算受限，这是个合理的上限——拓扑取 TPU v5e 4x4。**滑块控制序列长度，可以直观看到大 KV 缓存的效果。**

<figure class="plotly-embed">
  <iframe src="{{ 'assets/plotly/pareto.html' | relative_url }}" title="Latency/throughput Pareto frontier for LLaMA 3-70B" loading="lazy" scrolling="no"></iframe>
  <figcaption><b>图：</b>LLaMA 3-70B（int8 权重、16 路张量并行）在 TPU v5e 4x4 上，每芯片吞吐对每 token 延迟，沿每条曲线扫描批大小。拖动滑块改变上下文长度；每条曲线在"KV 缓存还装得进 HBM 的最大批"处截止。</figcaption>
</figure>

* **感受一下成本与延迟的交换有多剧烈。**每 token 延迟翻一倍，每 token 成本就能降大约 100 倍。另外，延迟从小批时的 5.5ms 一路到大批的 20ms，跨度巨大。
* 注意 2k 上下文时，吞吐在撞到 BS 120 的屋顶线后基本停在 1 token / ms / 芯片（120 是因为 int8 权重配 bf16 计算）。而上下文一长，这个批就装不进内存了，我们再也到不了完全饱和的那一点。
* 注意同样的吞吐下，大批的延迟高得多——因为这时是 KV 加载当家，不再是参数加载。

把成本与延迟拆成"参数加载、KV 加载、FLOPs"三块，会看得更清楚。阴影区是我们预期 MLP 块计算受限的区域。

<figure class="plotly-embed">
  <iframe src="{{ 'assets/plotly/latency_breakdown_log.html' | relative_url }}" title="Decode step time breakdown for LLaMA 3-70B" loading="lazy" scrolling="no"></iframe>
  <figcaption><b>图：</b>LLaMA 3-70B 在 TPU v5e 4x4 上每步解码耗时随批大小的变化，拆分为参数加载、KV 缓存加载与 FLOPs 三部分。总时间 = KV 加载时间 + （参数加载与 FLOPs 中较大的那个）。拖动滑块改变上下文长度。</figcaption>
</figure>

这一图讲了一个很清楚的故事：起初延迟几乎全是参数加载；批足够大后，FLOPs 和 KV 加载才变得要紧。值得注意的是，只要序列长度超过 2048，KV 缓存加载的时间就全程超过 FLOPs 时间！**加大批确实能提高硬件利用率——但在长上下文下，KV 加载永远统治着每步总时间。**

<p markdown=1 class="takeaway">**要点**：LLaMA 3-70B 在以上几乎所有配置里都是被 KV 缓存的内存带宽（以及 HBM 容量）死死卡住的。这正说明：想提生成吞吐，压缩 KV 缓存最关键。另外也感受一下这里的延迟/吞吐取舍有多陡峭。</p>

{% details 这部分的代码其实很简单。 %}

计算这些屋顶线的代码：

```py
import numpy as np

num_chips = 16  # we fix 16 as the amount of total model parallelism we do
bytes_per_param = 1  # int8 means 1 byte per param
param_count = 70e9
param_size = bytes_per_param * param_count
sequence_length = 8192  # can vary this

hbm_bandwidth = 8.20E+11  # v5e
flops = 1.97E+14  # v5e

def kv_cache_size(bs):
    return 2 * bs * 128 * 8 * 80

def min_topology(bytes):
    return 2 ** np.ceil(np.log2(bytes / 16e9))

def get_max_batch_size(
    num_chips: int,
    sequence_length: int,
    param_size: float,
) -> int:
  batch_sizes = np.arange(1, 1024, 4)
  kv_sizes = kv_cache_size(sequence_length * batch_sizes)
  required_chips = min_topology(kv_sizes + param_size)
  max_idx = np.where(required_chips <= num_chips)[0][-1]
  return max_idx

max_idx = get_max_batch_size(
    num_chips=num_chips,
    sequence_length=sequence_length,
    param_size=param_size,
)  # get the largest batch size that can fit
batch_sizes = np.arange(1, 512, 1)[:max_idx]
kv_sizes = kv_cache_size(sequence_length * batch_sizes)

kv_comms_time = kv_sizes / (num_chips * hbm_bandwidth)

param_comms_time = param_size / (num_chips * hbm_bandwidth)
param_comms_time = np.asarray([param_comms_time] * batch_sizes.shape[0])

flops_time = 2 * param_size * batch_sizes / (num_chips * flops)  # roughly true in a 2ND sense

mlp_time = np.maximum(flops_time, param_comms_time)
attn_time = kv_comms_time  # always bandwidth-bound for generate

latency = 1000 * (mlp_time + attn_time)
throughput = batch_sizes / (latency * num_chips)
```

注意我们把延迟明确拆成了两个来源：KV 加载和参数加载；而延迟取的是 FLOPs 和通信中较大的那个。

{% enddetails %}

## 动手解题

下面几道题。有的和上面做过的重复，但当练习也许更有教学意义。

**问题 1：** LLaMA 3-405B 每次前向每 token 用多少 FLOPs？若计算受限，在 N 块 TPU v5e 上一次前向的下界是多少？若通信受限呢？*不用管模型装不进单块芯片这件事。*

**问题 2：** 假设要用 int8 权重、int8 KV、批 240 服务 LLaMA 3-8B。(a) 模型参数、(b) KV 缓存、(c) 峰值工作激活各占多少字节（数量级即可）？能跑起来的最小拓扑是多大？

**问题 3：** 你会怎么在 TPU v5e 上服务 LLaMA 3-405B？假设 int8 权重、bfloat16 计算。设每 token 15ms 是硬上限，能达到的最高吞吐配置是什么？理论最小步长是多少？

**问题 4：** 了解 LLM 最好的办法是亲手实现一个。搭完整的训练管线又麻烦又贵，但把从 [HuggingFace 下载](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/)的训练好的权重变成一个能跑的推理实现，收获会大得惊人。大致来说，你应该试着做完下面这些事：

1. 下载 LLaMA 3 8B 权重，在 Colab 里加载。把权重画出来看看（我好用 [TreeScope](https://github.com/google-deepmind/treescope)），看看能不能认出每个张量。数一数参数量，和你预期的一致吗？MLP 和注意力各占多少？

2. 实现模型的完整前向。先不用管预填充/解码之类的花样，只要能喂进一段序列、得到说得过去的下一 token 概率就行：给个 prompt，出概率，挑最大的，重复。这很慢，但已经是能用的采样了。尽量只拿 LLaMA-3 论文当参考，但细节上可以对照 [`jax-ml/jax-llm-examples/llama3`](https://github.com/jax-ml/jax-llm-examples/tree/main/llama3) 仓库（位置编码和因果掩码要小心）。目标是让模型输出连贯的 token。*只要能拿到 16GiB 以上 HBM，这一切在一块 TPU 上就能跑。*

3. 现在来提速。实现 KV 缓存：把一次前向的键/值激活存下来，后续步骤直接去关注它们。采样循环会快一大截。

4. 把预填充和解码拆成两个服务，跑在不同芯片上。预填充一次只处理一条 prompt，然后交给一个按批处理 token 的解码服务器。

5. 用 Pallas 实现 Flash Attention，注意力会高效得多。*我觉得值得一试的是：全程不抄参考实现。*

<h3 markdown=1 class="next-section">第 8 部分到此结束！第 9 部分深入 XLA 与 TPU 性能剖析，[点击这里](../profiling)。</h3>
