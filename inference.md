---
layout: distill
title: "关于 Transformer 推理的一切"
# permalink: /main/
description: "做推理和做训练可以非常不一样。部分原因在于推理多了一个新因素要考虑：延迟。这一节我们从'采样一个新 token'讲起，一路讲到如何把一个大型 Transformer 高效地铺在很多加速器切片上、组成一台推理引擎。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 7

previous_section_url: "../applied-training"
previous_section_name: "第 6 部分 · 训练 LLaMA 3"

next_section_url: ../applied-inference
next_section_name: "第 8 部分 · 服务化 LLaMA"

bibliography: main.bib

giscus_comments: true
giscus_discussion: 9

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
  - name: "Transformer 推理基础"
  - subsections:
    - name: "我们到底想优化什么？"
    - name: "线性运算：是什么卡住了我们？"
    - name: "注意力呢？"
    - name: "LLM 延迟与吞吐的理论估算"
    - name: "内存呢？"
    - name: "给 LLaMA 2-13B 算一笔吞吐和延迟的账"
  - name: "提升生成吞吐与延迟的招数"
  - name: "把推理铺到多块加速器上"
  - subsections:
    - name: "预填充"
    - name: "生成"
    - name: "KV 缓存的分片"
  - name: "设计一台高效的推理引擎"
  - subsections:
    - name: "连续批处理"
    - name: "前缀缓存"
    - name: "看一个真实实现：JetStream"
  - name: "动手解题"
  - name: "附录"

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

## Transformer 推理基础

Transformer 训练好了，你想拿它生成一些新序列。_说到底，跑分变高、损失下降都只是代理指标——真正要看的是到了真刀真枪的那一刻，它能不能说出点东西！_<d-footnote>历史上，做 Transformer 研究完全可以不碰推理——选择题的打分基准不用生成，效率高得多。但预训练规模化到一定程度之后，推理的重要性只会一路上升，所以我们现在就把它讲清楚。</d-footnote>

采样在概念上很简单。我们喂进一段序列，心爱的 Transformer 会吐出 $$\log p(\text{next token}_i \vert \text{previous tokens})$$，也就是所有可能的下一个 token 的对数概率。从这个分布里抽样，就得到一个 token。把它接上、重复这个过程，就得到一段接着提示写下去的序列。

{% include figure.liquid path="assets/img/naive-inference.png" class="img-fluid" caption="<b>图：</b>朴素的 Transformer 采样。蓝色的 logits 给出下一个 token 的分布，从中抽样即可。注意每一步都重算了整个前缀，所以这个算法的运行时间是 $\Theta(n^2)$。" %}

上面就是朴素的 Transformer 采样实现。它能用，但**实践中从不这么干**——因为每生成一个 token 都要把整段序列重新算一遍。生成 $$n$$ 个 token，光 FFW 部分就要 $$O(n^2)$$，注意力部分更是 $$O(n^3)$$！

**怎么避免？** 不必每次都做完整前向。我们可以把每次前向中的一些中间激活存下来，从而跳过对旧 token 的重新计算。具体来说，既然每个 token 在做点积注意力时只关注它前面的 token，那我们只要把每个 token 的键、值投影写进一个叫 **KV 缓存**的新数据结构里就行。存好旧 token 的键/值投影之后，新 token 只需计算自己的 $$q_i \cdot k_j$$ 点积，早先的 token 一个 FLOPs 都不用再算。漂亮！

这样一想，推理就分成两个关键部分：

* <b style="color: red;">预填充（prefill）</b>：拿到一段很长的提示，把它的所有 token 一次性算完，并把结果激活（具体就是键、值投影）存进 **KV 缓存**，同时保存最后一个 token 的 logits。
* <b style="color: blue;">生成（generation）</b>：手里有 KV 缓存和上一步的 logits，就从 logits 里采样一个 token，把它喂回 Transformer，得到下一步的 logits，同时把这个新 token 的 KV 激活追加进 KV 缓存。如此重复，直到遇到特殊的 `<EOS>` token 或达到长度上限。

带 KV 缓存的采样长这样：

{% include figure.liquid path="assets/img/cached-inference.png" class="img-fluid" caption="<b>图：</b>带 KV 缓存的高效 Transformer 采样。<b style=\"color: red;\">预填充</b>处理整段提示，把每个 token 的键值激活存进缓存。<b style=\"color: blue;\">生成</b>拿着缓存（和最后一个 token 的 logits）采样新 token，把新 token 送过模型、去关注 KV 缓存，再把它自己的键值投影存回缓存。MLP 块部分的复杂度因此是 $O(n)$ 的。" %}

有了 KV 缓存，生成 $n$ 个 token 的复杂度降到了：FFW 部分 $$O(n)$$、注意力部分 $$O(n^2)$$——旧 token 再不用重算。但生成一段序列仍需要很多次前向——你在 Gemini 或 ChatGPT 上看到回答逐字“流”出来，就是这个在发生：每个 token（通常）都是一次独立的（只是部分复用了缓存）巨型模型调用。

我们马上会看到，<b style="color: red;">预填充</b>和<b style="color: blue;">生成</b>是两种完全不同的动物——Transformer 推理其实是伪装成一件事的两件事！和训练相比，KV 缓存还是一个全新的、重量级的复杂度来源。

### 我们到底想优化什么？

往下走之前，先点出推理里一个全新的东西：延迟。训练时我们只关心吞吐量（每秒每芯片处理多少 token）；推理时还必须关心 token 出来有多快——既包括**首 token 时间（TTFT，Time To First Token）**，也包括**每 token 延迟**。比如：

* **离线批量推理**（跑评测、造数据）只在意推理的总成本，完全不在乎单条样本的快慢。
* **聊天界面/流式任务**要在低成本、大规模的同时做到低 TTFT，并且出字速度要快过人类的阅读速度。
* **端侧推理**（比如笔记本上跑 `llama.cpp`）同一时刻只服务一个用户，只求延迟尽可能低，硬件条件还可能非常苛刻。

提高硬件利用率依然关键，对成本和 TTFT 都有帮助；但与训练不同，它*未必*在所有场景下都转化为更好的用户体验。加速器层、系统层、模型架构层的许多优化，都在延迟、吞吐、上下文长度甚至模型质量之间做取舍。

### 更细粒度地看 Transformer

到目前为止，我们基本把 Transformer 当成一摞前馈块。从 FLOPs 和内存角度看这常常没问题，但要认真给推理建模就不够了。<d-footnote>你会在整个这一节里注意到：推理比训练“好欺负”得多——FLOPs 少得多、批量化的机会少得多、对延迟敏感得多，KV 缓存更是让事情复杂了一大截。</d-footnote>如[第 4 部分](../transformers)所见，前向的主要组成部分是：

1. **注意力投影**。也就是 $W_Q$、$W_K$、$W_V$、$W_O$ 这四个。它们都是同一类活：从 HBM 读参数和一批激活，做点 FLOPs，把结果写回 HBM。
2. **点积注意力**。从 HBM 读一批键值投影和一批查询激活，做几个内积、一些 softmax，再把注意力结果写回 HBM。
3. **其余部分**：层归一化、激活函数、token 采样、更新 KV 缓存、位置嵌入等等。它们也耗一点 FLOPs，但要么被上面两项完全盖过，要么被融合进去了。

接下来几小节，我们在预填充和生成两种场景下分别审视这些操作，问同一个问题：性能会被什么卡住？在单块加速器内部，我们是计算受限还是内存受限？要强调的是：预填充的答案和生成的答案会非常不一样。

### 线性运算：是什么卡住了我们？

不管住在 MLP 块里还是注意力里，所有线性运算在概念上都是一样的：算术强度取决于批大小。这笔账在[第 1 部分](../roofline)算过，但值得再来一遍。看一次矩阵乘：$\text{bf16[B, D]}$ 的一批激活乘以 $\text{bf16[D, F]}$ 的矩阵。它可以是 MLP 里的大矩阵（$W_\text{in}$ 或 $W_\text{out}$），也可以是注意力里的小投影（$W_Q$、$W_K$、$W_V$、$W_O$）。做这次矩阵乘，要把两个数组都从 HBM 读进 MXU，乘完再把结果写回 HBM。和以前一样：

$$T_\text{math} = \frac{\text{Computation FLOPs}}{\text{Accelerator FLOPs/s}} = \frac{2BDF}{\text{Accelerator FLOPs/s}}$$

$$T_\text{comms} = \frac{\text{Communication Bytes}}{\text{Bandwidth Bytes/s}} = \frac{2BD + 2FD + 2BF}{\text{Bandwidth Bytes/s}}$$

TPU 或 GPU 可以边读边算，把两者重叠起来。要计算受限，需要 $$T_\text{math} \geq T_\text{comms}$$，即：

$$\frac{2BDF}{2BD + 2DF + 2BF} \geq \frac{\text{Accelerator FLOPs/s}}{\text{Bandwidth Bytes/s}} \underset{\text{TPU v5e}}{=} \frac{1.97E+14}{8.20E+11} = 240$$

右边是硬件的算术强度。现在假设 $D$、$F$ 都远大于 $B$（我们的批一般最多 500，而 $D$、$F$ 大于 10k），分母就可以近似：$\small{2BD + 2DF + 2BF \approx 2DF}$，于是

$$\begin{align*}
\frac{2BDF}{2BD + 2DF + 2BF} \approx \frac{2BDF}{2DF} \geq \frac{\text{Accelerator FLOPs/s}}{\text{Bandwidth Bytes/s}} \\
\underset{\text{TPU v5e}}{=} \frac{1.97E+14}{8.20E+11} \implies B \geq 240 = B_{\text{crit}}
\end{align*}$$

如果量化权重、或者用更低精度做矩阵乘，这个临界批大小会变。比如把权重量化成 int8 或 fp8，$B_\text{crit}$ 缩小 2 倍；如果用 int8 或 fp8 做 FLOPs，$B_\text{crit}$ 增大 2 倍。所以若记 $\beta = \text{bits per param} / \text{bits per activation}$、$\alpha_\text{hbm} = C / W_\text{hbm}$，真正的临界批大小是 $B_\text{crit} = \beta \alpha_\text{hbm}$。

<p markdown=1 class="takeaway">**要点**：Transformer 的矩阵乘是计算受限，*当且仅当*每副本的**token**批大小大于 $B_\text{crit} = C / W_\text{hbm} \cdot (\text{bits per param} / \text{bits per activation}) = \beta \cdot \alpha_\text{hbm}$。bf16 激活在 TPU v5e 上是 240 个 token，在 H100 上约 280 个。</p>

训练时，所有矩阵乘的算术强度都很高，因为同一份权重会被巨大的批反复复用。**这个高强度会延续到预填充**，因为用户的提示通常有成百上千个 token。前面算过 TPUv5e 的硬件算术强度是 240，所以只要把一条超过 240 token 的序列喂进跑在这块硬件上、用 bf16 的稠密模型，我们就是计算受限的，一切安好。比这更短的提示当然也可以凑批来提高利用率，但通常没有必要。

<p markdown=1 class="takeaway">**要点**：预填充时，矩阵乘基本总是计算受限的。所以只要把硬件利用率/MFU（模型算力利用率）拉满，就等于最大化了每芯片吞吐（成本）和延迟（体现为 TTFT）。除非提示极短，逐条请求级别的凑批只会增加延迟，换来一点点预填充吞吐的提升。</p>

可是生成阶段，每个请求每次只能前向一个 token——步骤之间有先后依赖！所以我们唯一能做的，就是把许多请求的批叠起来。因此对稠密模型，生成阶段在矩阵乘上*基本总是内存带宽受限的*，除非每芯片的总 token 批大小能超过 $B_\text{crit}$。

<p markdown=1 class="takeaway">**要点**：生成阶段，总 token 批大小必须大于 $B_{\text{crit}}$，线性/前馈部分才可能计算受限（bf16 参数在 TPU v5e 上是 240）。而生成只能逐 token 串行进行，这就逼着我们必须把许多请求凑批——而这很难！</p>

*值得感受一下 240 有多大！* 生成批大小 240，意味着 240 个请求同时在生成，稠密模型还得同时摆着 240 份 KV 缓存。除了少数批量推理场景，这在实践中几乎做不到。相比之下，预填充一次推过 240 个 token 稀松平常——当然稀疏度高的时候要小心些。

**注意，这个数确实随量化方式和硬件而变。** 低精度下加速器往往能提供更强的算术吞吐。比如 int8 参数、bf16 计算时，临界批大小降到 120；而 int8 参数、int8 激活时又回到 240——因为 TPUv5e 能提供 400 TOPs/s 的 int8×int8。

### 注意力呢？

看真正的点积注意力运算就复杂多了，何况还得算上 KV 缓存。只看一个注意力头、标准多头注意力。在一次 Flash Attention 融合里，我们会<d-footnote>这里简化了不少：忽略了 softmax、掩码等非矩阵乘 FLOPs。它们本应与计算或 HBM 读取重叠，但在某些 TPU 代次上并不好做。这些细节不改变主要结论——KV 缓存通常是内存受限的——但值得留意。</d-footnote>：

1. 从 HBM 读形状为 $\text{bf16[B, T, D]}$ 的 $Q$ 激活。
2. 从 HBM 读 $KV$ 缓存——一对 $\text{bf16[B, S, D]}$ 张量。
3. 在 $$QK$$ 矩阵乘中做 $2BSTD$ 个 FLOPs。有 Flash Attention，$\text{bf16[B, S, T]}$ 的注意力矩阵不用写回 HBM。
4. 在注意力 $$AV$$ 矩阵乘中做 $2BSTD$ 个 FLOPs。
5. 把结果 $\text{bf16[B, T, D]}$ 张量写回 HBM。

合起来得到：

$$\text{Multiheaded Attention Arithmetic Intensity} = \frac{4BSTD}{4BSD + 4BTD} = \frac{ST}{S+T}$$

预填充时 $S=T$（自注意力），式子化简为 $T^2 / 2T = T / 2$。很好，这意味着**预填充时注意力的算术强度是 $\Theta(T)$ 的**——想计算受限相当容易，序列长度够大就行！

但生成时序列维恒为 1，$B$ 和 $D$ 又消掉了，于是可以近似：

$$S \gg T = 1 \implies \frac{ST}{S+T} \approx 1$$

这就糟了：我们没有任何办法提高生成时注意力的算术强度。搬着巨大的 KV 缓存，却只做了极少的 FLOPs。**所以注意力在生成时几乎永远是内存带宽受限的！**

<p markdown=1 class="takeaway">**要点**：预填充时，只要序列长度说得过去（大约大于 480 个 token），注意力通常是计算受限的；而生成时算术强度低且恒定，永远是内存带宽受限。</p>

*从概念上讲，这是为什么？* 线性部分计算受限，是因为参数（内存带宽的大头）被许多批样本共用了。可每个批样本都有自己的 KV 缓存——批越大，缓存越多，读取量随批一起涨。除非对架构动大手术，这里几乎*永远*是内存受限的。

这也意味着靠加大批大小换吞吐，收益递减得很快：批每大一点，就要多搬对应的 KV 缓存。这正是各种 KV 缓存优化（见下一章）能带来数量级吞吐提升的原因。

### LLM 延迟与吞吐的理论估算

从这些公式，我们可以为“每步时间优化该奔着什么目标”给出很好的界。**（如果整章只让读者记住一样东西，就是下面这两条。）** 生成阶段的小批（最常见的情形）下，假设注意力和 MLP 块都是内存带宽受限，每步延迟有个下界：

$$\begin{equation*}
\text{Theoretical Min Step Time} = \frac{\text{Batch Size} \times \text{KV Cache Size} + \text{Parameter Size}}{\text{Total Memory Bandwidth}}
\end{equation*}$$

吞吐同理：

$$\begin{equation*}
\text{Theoretical Max Tokens/s} = \frac{\text{Batch Size} \times \text{Total Memory Bandwidth}}{\text{Batch Size} \times \text{KV Cache Size} + \text{Parameter Size}}
\end{equation*}$$

批再大下去，FLOPs 开始压过参数搬运，实际更通用的公式是：

$$\begin{align}
\tiny \text{Theoretical Step Time (General)} = \underbrace{\frac{\text{Batch Size} \times \text{KV Cache Size}}{\tiny \text{Total Memory Bandwidth}}}_{\text{Attention (always bandwidth-bound)}} + \underbrace{\max\left(\frac{2 \times \text{Batch Size} \times \text{Parameter Count}}{\text{Total FLOPs/s}}, \frac{\text{Parameter Size}}{\text{Total Memory Bandwidth}}\right)}_{\tiny \text{MLP (can be compute-bound)}}
\end{align}$$

其中注意力部分（左边）永远不会计算受限，所以不需要 FLOPs 屋顶线。这些式子做“拍脑袋估算”非常好用，例如：

<b markdown=1 style="color: #57cf57;">小测验：</b>假设要在 TPU v5e 4x4 切片上，对一个 30B 参数的稠密模型做批大小为 4 token 的生成步：int8 参数、bf16 计算、上下文 8192、每 token 的 KV 缓存 100 kB。这个操作的延迟合理下界是多少？如果想一次采样 256 个 token 呢？

{% details 点此查看答案。 %}

**答案**：int8 下参数占 30e9 字节；按题设，每份 KV 缓存占 `100e3 * 8192 = 819MB`。我们有 16 块芯片，每块带宽 `8.2e11` 字节/秒、bf16 算力 `1.97e14` FLOPs/秒。代入上面的公式：批很小，每步至少要 `(4 * 819e6 + 30e9) / (16 * 8.2e11) = 2.5 ms`。到 256 个 token，MLP 块已深入计算受限区，每步大约 `(256 * 819e6) / (16 * 8.2e11) + (2 * 256 * 30e9) / (16 * 1.97e14) = 21ms`。

{% enddetails %}

可以看到，吞吐和延迟的取舍一目了然：批小了快但浪费硬件，批大了高效但变慢。下面是用老一代 PaLM 模型算出的延迟-吞吐帕累托前沿（出自 [ESTI 论文](https://arxiv.org/pdf/2211.05102)<d-cite key="esti"></d-cite>）：

{% include figure.liquid path="assets/img/latency-cost.png" class="img-fluid" caption="<b>图：</b>若干 PaLM 模型的成本（读作吞吐）对延迟的帕累托前沿。注意芯片数（C）和批大小（B）如何带着你沿前沿移动。唯一的例外是绿点（PaLM 540B，C:32 B:16）：内存不够，撑不起像样的批大小，吞吐受了拖累。注意吞吐一般在批大小 240 之后趋于平缓。int8 权重能带来更好的延迟-吞吐帕累托点，但换不来更高的吞吐上限。" %}

我们不仅拿批大小当“延迟-吞吐”的旋钮；当被 HBM 容量限制时，还可能宁可换更大的拓扑，好装下更大的批。[下一节](../applied-inference)会展开这一点。

<p markdown=1 class="takeaway">**要点**：想要生成吞吐，就把每芯片批大小拉到尽可能大。只要每芯片批超过 TPU 的算术强度（$B_\text{crit}$，通常是 120 或 240），吞吐就到顶了。为此你可能需要扩大拓扑。较小的批大小则以牺牲吞吐为代价换来更好的延迟。</p>

{% details 硬件上还有一些注意事项，点此看几条“挑刺”。 %}

以上都很理论。实践中往往看不到一条锋利的屋顶线，原因有几条：

* “HBM 读取能和 FLOPs 完美重叠”的假设不现实，因为编译器（XLA）是会犯错的。
* 对分片模型，XLA 也常常没能把模型分片矩阵乘的 ICI 通信和计算高效重叠，所以线性部分往往从 $$\text{BS}=32$$ 开始就开始吃延迟的亏。
* 超过理论屋顶线的批大小仍能靠“重叠得不完美”捡回一点吞吐——但上面的界仍是好启发式。

{% enddetails %}

### 内存呢？

带宽和 FLOPs 聊了不少，内存还没聊。推理时的内存图景大不相同，因为我们多了一个新数据结构：KV 缓存。这一节挑个真实模型（LLaMA 2-13B）来演示差别有多大：

| 超参数             | 值     |
| ------------------ | ------ |
| L（层数）          | 40     |
| D（d_model）       | 5,120  |
| F（ffw_dimension） | 13,824 |
| N（注意力头数）    | 40     |
| K（KV 头数）       | 40     |
| H（qkv_dim）       | 128    |
| V（词表大小）      | 32,000 |

推理时什么东西在吃内存？首先当然是参数。数一数：

| 参数           | 公式                                                                                                          | 大小（字节）                                                   |
| -------------- | ------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| FFW 参数       | d_model x d_ff x 2.7（SwiGLU 的门控、上、下三个投影的放大倍数）x n_layers                                  | 5,120 x 5,120 x 2.7 x 3 x 40 = **8.5e9**                       |
| 词表参数       | 2（输入输出嵌入）x n_embeddings x d_model                                                                     | 2 x 32,000 x 5,120 = **0.3e9**                                 |
| 注意力参数     | [2（*q 和输出投影*）x d_model x n_heads x d_qkv + 2（*k 和 v*）x d_model x n\_kv\_heads x d_qkv] x n_layers | (2 x 5,120 x 40 x 128 + 2 x 5,120 x 40 x 128) x 40 = **4.2e9** |

加起来是 8.5e9 + 4.2e9 + 0.3e9 = **共 13e9 个参数**，正与名字相符。前几节讲过，训练时参数可能用 bfloat16 存、优化器状态用 float32，大概要占 100GB 内存；这和动辄几 TB 的梯度检查点比根本不算什么。

推理呢？假设参数不加任何量化、就用 bfloat16，那就是 26GB——实践中靠量化常常能做得比这好得多。没有优化器状态，也没有梯度要伺候。因为不做检查点（不需要为反向保留激活），激活的占用在预填充和生成阶段都可以忽略<d-footnote>尤其要感谢 Flash Attention，它避免把注意力矩阵整个写出来。</d-footnote>。就算一次预填充 8k 个 token，单条激活也不过 `8,192 x 5,120 x 2 bytes = 80MB`。更长的预填充可以拆成多次较小的前向，长上下文也不成问题。生成用的 token 更少，激活更可以忽略。

**真正的区别在 KV 缓存**。它是历史所有 token 的键、值投影，大小只受最大序列长度约束。$$T$$ 个 token 的总大小是

$$\text{KV cache size} = 2 \cdot \text{bytes per float} \cdot H \cdot K \cdot L \cdot T$$

其中 $$H$$ 是每个头的维度，$$K$$ 是 KV 头数，$$L$$ 是层数，2 来自键和值都要存。

**这个东西涨得非常快**，哪怕批和上下文都不夸张。LLaMA-13B 在 bf16 下，单条 8192 序列的 KV 缓存是

$$8192\ (T) \times 40\ (K) \times 128\ (H) \times 40\ (L) \times 2\ (\text{bytes}) \times 2 = 6.7 \text{GB}$$

**只要 4 份，就超过了参数本身的占用！**说清楚：LLaMA 2 并没有针对长上下文的 KV 缓存大小做优化（局面通常没这么糟，一般 $K$ 会小得多，比如 LLaMA-3），但这个例子足够说明问题。无论估内存还是估延迟，都不能无视它。

### 给 LLaMA 2-13B 算一笔吞吐和延迟的账

来看：在 8 块 TPU v5e 上，以不同批大小做“完全高效”的生成，一直加到我们之前推出的临界批大小 240（理论最大吞吐的位置）。

| 批大小                            |      1 |      8 |     16 |     32 |     64 |    240 |
| :-------------------------------- | -----: | -----: | -----: | -----: | -----: | -----: |
| KV 缓存内存（GiB）                |    6.7 |   53.6 |  107.2 |  214.4 |  428.8 |   1608 |
| 总内存（GiB）                     |   32.7 |   79.6 |  133.2 |  240.4 |  454.8 |   1634 |
| 理论每步时间（ms）                |   4.98 |  12.13 |  20.30 |  36.65 |  69.33 | 249.09 |
| 理论吞吐（tokens/s）              | 200.61 | 659.30 | 787.99 | 873.21 | 923.13 | 963.53 |

8 块 TPU v5e 共有 128GiB HBM、6.5TiB/s 的 HBM 带宽（每块 0.82TiB/s）、1600TF/s 算力。

对这个模型，加大批确实能提高吞吐，但回报衰减得很快。批到 16 就内存溢出（OOM）了；想到 240 附近，还得再加一个数量级的内存。更大的拓扑能改善延迟，但每芯片吞吐已经撞到墙上。

现在假设参数总量不变，但神奇地把 KV 缓存缩小 5 倍（比如用 1:5 的 [GMQA](#tricks-for-improving-generation-throughput-and-latency)，即 40 个 Q 头共享 8 个 KV 头——细节见下一节）。

| 批大小                            |      1 |        8 |       16 |       32 |       64 |      240 |
| :-------------------------------- | -----: | -------: | -------: | -------: | -------: | -------: |
| KV 缓存内存（GiB）                |   1.34 |    10.72 |    21.44 |    42.88 |    85.76 |    321.6 |
| 总内存（GiB）                     |  27.34 |    36.72 |    47.44 |    68.88 |   111.76 |    347.6 |
| 理论每步时间（ms）                |   4.17 |     5.60 |     7.23 |    10.50 |    17.04 |    52.99 |
| 理论吞吐（tokens/s）              | 239.94 | 1,429.19 | 2,212.48 | 3,047.62 | 3,756.62 | 4,529.34 |

KV 缓存变小后，回报依然递减，但每芯片理论吞吐能一路涨到批 240。我们装得下 64 的大批，而且所有批大小下的延迟都更好。延迟、最大吞吐、最大批大小全都大幅改善！事实上，后来的 LLaMA 正是这么改的——LLaMA-3 8B 就是 32 个查询头配 8 个 KV 头（[来源](https://huggingface.co/MaziyarPanahi/Llama-3-13B-Instruct-v0.1/blob/dfdeb40bdb2c149dfa399ea2be0d56eb120f0831/config.json)）。

<p markdown=1 class="takeaway">**要点**：除了参数量，KV 缓存的大小对模型最终的推理性能影响极大。要靠架构设计和运行时优化双管齐下，把它压住。</p>

## 提升生成吞吐与延迟的招数 {#tricks-for-improving-generation-throughput-and-latency}

自最初的 [Attention is All You Need 论文](https://arxiv.org/abs/1706.03762)以来，人们发明了许多让模型更高效的技术，很多就是冲着 KV 缓存去的。总的原则是：KV 缓存越小，越容易在不掉延迟的前提下加大生成的批和上下文，围绕 Transformer 的那些系统设施（比如请求缓存）也更好过。抛开对质量的影响不谈，我们可能看到：

**分组多查询注意力（也叫 GMQA、GQA）：** 减少 KV 头的数量，让多个 Q 头共享同一组 KV。极端情况下所有 Q 头共用一个 KV 头。相比标准 MHA，KV 缓存按 Q:KV 的比例缩小，而实践表明模型效果对这类改动相当不敏感。

{% include figure.liquid path="assets/img/gmqa.png" class="img-fluid" %}

这还顺带提高了注意力计算的算术强度（见[第 4 部分](../transformers)的问题 4）。

**掺入一些局部注意力层：** 局部注意力把上下文截断到一个较小到适中的最大长度。训练和预填充时，注意力矩阵被掩码成一条对角带而不是一个三角形，等于给这些层的 KV 缓存最大长度设了上限。在全局层之间掺一些局部层，超过局部窗口之后 KV 缓存会大幅缩小。

**跨层共享 KV：** 模型可以学会按某种模式跨层复用同一份 KV 缓存。这确实能缩小 KV 缓存、方便加大批大小、做缓存、做离线存储等等；但共享的 KV 可能要多次从 HBM 读取，*所以每步时间未必更快。*

{% include figure.liquid path="assets/img/kv-sharing.png" class="img-fluid" caption="
 <b>左：</b>多层纯全局注意力。<b>右：</b>一种全局/局部交替并与相邻层共享的模式。来源：<a href=\"https://research.character.ai/optimizing-inference/?ref=blog.character.ai\">Character.ai 博客</a>。"%}

**量化：** 推理对参数和 KV 的精度通常没那么敏感。把参数和 KV 缓存量化（比如 int8、int4、`fp8` 等），两边都省内存带宽，达到计算屋顶线所需的批大小更小，省下的内存又能支撑更大的批。量化还有个额外好处：即使模型训练时没用量化，训练后也常常能直接套上。

**用不规则（ragged）HBM 读取和 Paged Attention：** 上面的计算里每个 KV 缓存都按 8k 上下文预留，但实际常常不需要把整段 KV 缓存从内存读出来——请求的长度分布很宽，很少用满模型的最大上下文，所以可以实现只读非 padding 部分的内核（比如各种 Flash Attention 变体）。

Paged Attention<d-cite key="paged"></d-cite>是它的升级：把 KV 缓存按操作系统页表那样分页存放，基本完全免去 padding。这增加了不少复杂度，但每个批只占它真正需要的内存。这是运行时优化，对模型架构是“无感”的。

{% include figure.liquid path="assets/img/paged-attention.png" class="img-fluid img-small" caption="<b>图：</b>生成时，一个 token（\"forth\"）要关注多个 KV 缓存块/页。把 KV 缓存分页后，我们不再多读、多存任何不需要的内存。取自 <a href=\"https://arxiv.org/pdf/2309.06180\">PagedAttention 论文</a>。" %}

<p markdown=1 class="takeaway">**大局观**：这些 KV 缓存优化加起来，能把 KV 缓存压到标准 MHA Transformer 的十分之一以下，整个 Transformer 的服务成本随之可能降低一个数量级。</p>

## 把推理铺到多块加速器上

前面我们把“如何扩展到多块芯片”糊弄过去了。按照[第 5 部分](../training)的路子，我们来梳理可用的各种策略和取舍。照例，预填充和生成分开看。

### 预填充

从屋顶线的角度看，**预填充和训练几乎一模一样**，训练的全部技巧和取舍几乎都适用——模型（Megatron）并行、序列切分（上下文足够长时）、流水线，甚至 FSDP 都可行！只要把 KV 留着，后面还要用它做生成。和训练一样：加芯片能换来更多 FLOPs/s（TTFT 可能更低），但通信开销也更大（每芯片吞吐可能下降）。

**预填充的分片总原则：** 下面这套规则假设只预填充单条序列（没有批维）：

1. *模型分片：* 通常先做一定程度的模型并行，做到逼近 ICI 界限为止。[第 5 部分](../training)讲过，单轴大约是 $F / 2200$（一般是 4-8 路）。
2. *序列并行：* 再往上就做序列并行（类似数据并行，但切的是序列维）。序列并行虽会在注意力里引入额外通信，长上下文下通常很小。和训练一样，通信和计算可以重叠（Megatron 用集合式矩阵乘，注意力用 ring attention）。

<p markdown=1 class="takeaway">**要点**：预填充时，训练里能用的分片几乎都能用。模型并行做到 ICI 界限，然后上序列并行。</p>

### 生成

生成比预填充难伺候得多。一来，想要大批就得凑许多请求，不容易；二来延迟指标更苛刻。这两点合起来，意味着我们更常处于内存受限状态、对通信开销也更敏感，可选的分片策略因此很受限：

1. **FSDP 不可能：** 我们本来就卡在“把参数和 KV 从 HBM 搬进 MXU”的内存带宽上，绝不想再通过比 HBM 慢好几个数量级的 ICI 搬它们。*要搬的是激活，不是权重。*所以 FSDP 那一路在生成阶段通常完全不可行<d-footnote>训练完忘了把 FSDP 关掉，是性能莫名掉一个数量级最常见的原因之一。</d-footnote>

2. **没有任何理由做数据并行：** 纯数据并行复制参数，也不会让参数读得更快，没有用处。与其这样，不如多开几份模型副本。<d-footnote>这里说的是多开几台各自持有完整模型副本、批更小的服务器。在模型层面做数据并行严格更差。</d-footnote>

3. **没有序列，就没有序列切分。**祝你好运，没得可切。

_稠密模型的生成，剩下的选项基本就是模型分片的各种变体_。和预填充一样，最简单的是朴素的模型并行（激活完全复制，MLP 权重沿隐层维完全分片），做到 4-8 路就会撞上 ICI 界限。不过，由于我们经常是内存带宽受限的，其实可以越过这条界限继续加模型分片来改善延迟！

**关于生成阶段 ICI 界限的说明：** 训练要计算受限，所以屋顶线看的是“ICI 通信何时超过 FLOPs”。可生成阶段若卡在参数搬运的内存带宽上，我们可以把模型分片做得更多——用极小的吞吐代价（按每芯片 tokens/s 算）换更好的延迟。模型多分几片，就多几块 HBM 可以并行搬权重，反正 FLOPs 根本不是瓶颈<d-footnote>意思是：既然 FLOPs 时间卡不住我们，要担心的就只剩“ICI 时间超过参数搬运时间”这一件事。</d-footnote>来看看在 ICI 成为瓶颈之前，模型并行最多能做到多少。

$$\begin{align*}T_\text{HBM comms} = \frac{2DF}{Y \cdot W_\text{hbm}} && T_\text{ICI comms} = \frac{2BD}{W_\text{ici}}\end{align*}$$

$$T_\text{ICI comms} > T_\text{HBM comms} \rightarrow \frac{W_\text{hbm}}{W_\text{ici}} > \frac{F}{Y \cdot B} \rightarrow Y > F / (B \cdot \beta)$$

其中 $\beta = W_\text{hbm} / W_\text{ici}$。这个数在 TPU v5e 和 TPU v6e 上通常约为 8。也就是说，若 $F$ 是 16,384、$B$ 是 32，理论上模型并行可以做到 `16384 / (32 * 8) = 64` 路而几乎不损失吞吐。前提是 KV 缓存也能完全切成 64 份——这很难，下面就来讨论。

注意力层同理，我们把 $$W_Q$$ 和 $$W_O$$ 按 Megatron 风格沿注意力头做模型分片。KV 的权重相当小，超过 $K$ 路之后再继续分，往往不如直接复制便宜。

<p markdown=1 class="takeaway">**要点**：生成阶段仅有的选项是模型并行的各种变体。目标是搬激活，不搬更大的 KV 缓存和参数。批大时，模型并行做到 FLOPs-ICI 界限（$F / \alpha$）为止；批小时，可以更激进地切模型来换延迟（付出少量吞吐代价）；想切的份数超过 KV 头数时，把 KV 沿批维一起切。</p>

### KV 缓存的分片

**我们还多了一个需要分片的数据结构——KV 缓存。**同样，尽量别复制缓存，因为它是注意力延迟的主要来源。做法是先沿头维对 KV 做 Megatron 式分片。但这最多只能切 $K$ 路；对 KV 头不多的模型，先把头维能切尽切，再沿批维切，即 $\text{KV}[2, B_Z, S, K_Y, H]$。这样 KV 缓存就彻底分布式化了。

{% include figure.liquid path="assets/img/esta-figure.png" class="img-fluid" caption="<b>图：</b>两种注意力实现的对比：(a) 多头注意力 + 纯模型分片；(b) 多查询注意力 + KV 缓存按批分片。注意 (b) 需要两次额外的 AllToAll，把激活从'模型分片'格式转成'按批分片'格式，才能去和 KV 缓存做运算。" %}

代价是每个注意力层两次 AllToAll——一次把 Q 激活转到批分片格式以便按批算注意力，一次把按批分片的注意力输出转回纯模型分片格式。

{% details 这是完整算法！ %}

这里写出对 $Y$、$Z$ 两轴同时做模型并行的完整注意力算法。抱歉用 $K$ 同时表示键张量和 KV 头维。记 $M=N/K$。

<div markdown=1 class="algorithm">

1. X[B, D] = ...（上一层传来的未分片激活）
2. K[B<sub>Z</sub>, S, K<sub>Y</sub>, H], V[B<sub>Z</sub>, S, K<sub>Y</sub>, H] = ...（已有的 KV 缓存，按批分片）
3. Q[B, N<sub>YZ</sub>, H] = X[B, D] \* W<sub>Q</sub>[D, N<sub>YZ</sub>, H]
4. Q[B<sub>Z</sub>, N<sub>Y</sub>, H] = **AllToAll**<sub>Z->B</sub>(Q[B, N<sub>YZ</sub>, H])
5. Q[B<sub>Z</sub>, K<sub>Y</sub>, M, H] = **Reshape**(Q[B<sub>Z</sub>, N<sub>Y</sub>, H])
6. O[B<sub>Z</sub>, S, K<sub>Y</sub>, M] = Q[B<sub>Z</sub>, K<sub>Y</sub>, M, H] \*<sub>H</sub> K[B<sub>Z</sub>, S, K<sub>Y</sub>, H]
7. O[B<sub>Z</sub>, S, K<sub>Y</sub>, M] = **Softmax**<sub>S</sub>(O[B<sub>Z</sub>, S, K<sub>Y</sub>, M])
8. O[B<sub>Z</sub>, K<sub>Y</sub>, M, H] = O[B<sub>Z</sub>, S, K<sub>Y</sub>, M] \*<sub>S</sub> V[B<sub>Z</sub>, S, K<sub>Y</sub>, H]
9. O[B, K<sub>Y</sub>, M<sub>Z</sub>, H] = **AllToAll**<sub>B->Z</sub>(O[B<sub>Z</sub>, K<sub>Y</sub>, M, H])
10. O[B, N<sub>YZ</sub>, H] = **Reshape**(O[B, K<sub>Y</sub>, M<sub>Z</sub>, H])
11. X[B, D] {U<sub>YZ</sub>} = W<sub>O</sub>[N<sub>YZ</sub>, H, D] \*<sub>N,H</sub> O[B, N<sub>YZ</sub>, H]
12. X[B, D] = **AllReduce**(X[B, D] { U<sub>YZ</sub>})

这挺绕的，但大体思路你应该能看出来。新增的通信不贵，因为动的是我们的小激活；换来的却是搬 K/V（它们是驻留不动的）省下的巨大内存带宽。

</div>

{% enddetails %}

* **序列切分：** 批太小、上下文太长时，可以对 KV 缓存做序列切分。同样，跨片累加注意力要付一次集合通信：先把 Q 激活 AllGather 过来，再仿照 Flash Attention 的方式跨片累加 KV。

## 设计一台高效的推理引擎

到目前为止，我们讨论的都是如何单独地把一次预填充/生成做好。要真正用好它们，需要设计一台推理引擎，把这两路操作喂到我们选定的“延迟/吞吐帕累托前沿”上的那一点。

最省事的办法是先跑一批预填充、再跑一批生成：

{% include figure.liquid path="assets/img/batched-prefill.png" class="img-fluid" caption="<b>图：</b>最简单的做法：把请求攒起来，服务器交替执行'一批预填充'和'对所有序列反复调用生成直到完成'。" %}

这很好实现，多数代码库的第一版推理就是这么写的。但它有几个毛病：

1. **延迟很差。**预填充批和生成批被绑在一起：预填充批一大，TTFT 就难看——所有请求都得排完预填充的队，谁也别想见到第一个 token；生成批一小，吞吐就难看。
2. **短生成被长生成拖住。**很多序列会比别的先结束，生成时批里留下空位，进一步拖累生成吞吐。批越大、生成长度越长，问题越严重。
3. **预填充要 padding。**一批预填充要 pad 到最长的那条，浪费大量计算。虽有对策，但历史上 XLA 很难跳过这些白算的 FLOPs。同样，批越大、预填充越长，浪费越大。
4. **预填充和生成被迫共用一套分片。**两者活在同一切片上，只能共用同一拓扑和同一分片（除非你存两份权重），这对性能基本没好处——比如生成想要的是更重的模型切分。

所以这种方式只推荐给两类场景：端侧应用（通常只服务一个用户、硬件 FLOPs/字节比更低），以及 Transformer 代码库早期的快速迭代（因为它简单）。

稍好一点的做法：预填充按批 1 执行（此时它计算受限，延迟也可接受），生成阶段再把多个请求凑批：

{% include figure.liquid path="assets/img/interleaving.png" class="img-fluid" %}

这既躲开了批量预填充拖长的 TTFT，又保住了生成吞吐。我们叫它**交替式（interleaved）**配置，因为预填充步和生成步是“交替”排布的。对以吞吐为王的批量生成任务（比如跑评测），这非常强：调度器一发现生成位有空缺就立刻插队做预填充，哪怕生成批很大也能保持高利用率。而且预填充不再需要 pad 到最大长度——反正不和别的需求凑一批。

主要缺点是：服务器一旦开始做预填充，其他所有请求的生成就得停一停——所有计算资源都被那次预填充占走了。正在解码的用户 A 会被正在预填充的用户 B 堵住。TTFT 是好看了，但大家的出字会一顿一顿、平均更慢，对很多应用来说用户体验不好——别人的预填充跑在了我这次请求的延迟关键路径上。

要绕开这一点，就把解码和预填充分开。虽然 Transformer 推理完全可以在一台服务器上跑完，但从延迟角度看，把两种任务放到两组 TPU/GPU 上往往更好：预填充服务器产出 KV 缓存，经网络发给生成服务器；生成服务器把多份缓存凑批，逐份生成 token。这叫**分离式（disaggregated）**服务。

{% include figure.liquid path="assets/img/disaggregation.png" class="img-fluid" %}

好处有几条：

1. **大规模下的低延迟**：用户不会排到别的用户后面——除非预填充容量不够。请求应当马上被预填充、马上送到生成服务器、马上插进生成队列。预期并发高时，可以单独扩预填充服务器的数量，不让用户长时间排在预填充队里。

2. **各司其职：**预填充和生成在延迟上的最优参数切分/硬件拓扑常常差别很大（比如模型并行对生成有用、对预填充没用）。强迫两者共用一套切分，两边性能都受损；而维护两份权重又费内存。另外，预填充独占服务器后，除了手头正在处理的那条，它不需要保存任何 KV 缓存——省出大量内存用于历史缓存（见下一节）或继续优化预填充延迟。

一个坏处是 KV 缓存现在要过一遍网络。这通常可以接受，但也再次构成了“把 KV 缓存做小”的动力。

<p markdown=1 class="takeaway">**要点**：对延迟敏感、又要高吞吐的服务，我们通常必须把预填充和生成拆到两组服务器：预填充按批 1 跑，生成把许多并发请求凑在一起跑。</p>

### 连续批处理

上面的毛病 (2) 引出了**连续批处理**的概念。我们优化并编译两个函数：

* 一个预填充函数：适配长短不一的上下文，把结果写进一个 KV 缓冲区（缓冲区有最大批大小、最大上下文长度/页数）。
* 一个生成函数：接收 KV 缓存，对所有当前活跃的请求各执行一步生成。

再把这两个函数和一个编排器（orchestrator）配合起来：编排器给到来的请求排队，按可用的生成位安排调用预填充和生成，处理历史缓存（见下一节），并把 token 流式发出。

{% include figure.liquid path="assets/img/continuous-batching.gif" class="img-fluid" %}

### 前缀缓存

预填充昂贵且计算受限（余量小），所以省钱的最佳途径之一是：少做一点。LLM 是自回归的，提问 ["I", "like", "dogs"] 和 ["I", "like", "cats"] 的 KV 缓存前两个 token 完全相同。也就是说原则上，先算 "I like dogs" 的缓存、再算 "I like cats" 时，后者只需要做 1/3 的计算——大部分活靠复用缓存省下了。在几个特定场景里这特别有效：

1. **聊天机器人**：对话就是一条不断往后追加的序列。只要存下每轮对话的 KV 缓存，除了最新几个 token，其他全都不用重算。
2. **Few-shot 提示**：任何少样本示例都可以存下来免费复用。系统提示词（system prompt）常常也是这个形态。

唯一的拦路虎是内存。前面见过，KV 缓存很大（动辄好几 GB），而且要有后续请求命中才有价值，得一直留着。实践中，预填充服务器上闲置的 HBM 可以拿来做本地缓存；而且加速器的 CPU 主机上通常还有很多内存（比如一台 8xTPUv5e 服务器有 128GiB HBM，但主机 DRAM 约 450GiB）。这些内存比 HBM 慢得多——做生成步是来不及的——但读一次缓存绰绰有余。落地时：

* KV 缓存生在最初处理该请求的那组 TPU 上，所以需要某种亲和性路由，保证后续查询落回同一个副本。这会给负载均衡添麻烦。
* KV 缓存做小又有好处（又来了）——同样的空间能存更多份，读得也更快。
* KV 缓存及其查找天然适合存在树或字典树（trie）里，按 LRU 淘汰。

{% include figure.liquid path="assets/img/prefix-caching-trie.png" class="img-fluid" caption="<b>图：</b>用 LRU 字典树实现的 KV 前缀缓存。共享前缀，KV 内存就不必重复存。来源：<a href=\"https://research.character.ai/optimizing-inference/?ref=blog.character.ai\">Character.ai 博客</a>。" %}

### 看一个真实实现：JetStream

Google 开源了一个实现了这套逻辑的库，叫 [JetStream](https://github.com/google/JetStream)。服务器有一组“预填充引擎”和“生成引擎”，通常在不同的 TPU 切片上，由一个总控（controller）统一编排。预填充在“[prefill thread](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L499)”里跑，生成在“[generate thread](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L629)”里跑，还有一条“[transfer thread](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L592)”负责把 KV 缓存从预填充切片搬到生成切片。

Engine 接口（[实现在这里](https://github.com/google/JetStream/blob/445f1aa8e857d0a09d72618e365daf80723bdf4c/jetstream/engine/engine_api.py#L138)）是任何 LLM 都要提供的一套通用接口，关键方法有：

* **prefill：**接收一批输入 token，产出 KV 缓存。
* **insert：**接收一份 KV 缓存，插进 generate 正在生成的那批 KV 缓存里。
* **generate：**接收一批 KV 缓存，每个批元素各生成一个 token，并把这一个 token 的 KV 追加进各自的解码状态。

JetStream 还有 PyTorch 版本，在[这里](https://github.com/google/jetstream-pytorch)。

## 动手解题

这一节我基于 LLaMA-2 13B 编一个新模型。参数如下：

| 超参数           | 值     |
| :--------------- | :----- |
| L（层数）        | 64     |
| D（d_model）     | 4,096  |
| F（ffw_dimension）| 16,384 |
| N（注意力头数）  | 32     |
| K（KV 头数）     | 8      |
| H（qkv_dim）     | 256    |
| V（词表大小）    | 32,128 |

**问题 1：** 上面这个模型有多少参数？int8 下每 token 的 KV 缓存多大？*可以假设输入输出投影矩阵共享。*

{% details 点此查看答案。 %}

**参数量：**

* MLP 参数：$L * D * F * 3$
* 注意力参数：$L * 2 * D * H * (N + K)$
* 词表参数：$D * V$（矩阵共享）

总参数量 $L * D * (3F + 2H * (N + K)) + D * V$。代入数字：`64 * 4096 * (3*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 18.4e9`。这个模型大约有 184 亿参数。

int8 下 KV 缓存每 token 为 $2 * L * K * H$，即 `2 * 64 * 8 * 256 = 262kB`。

{% enddetails %}

**问题 2：** 假设要在 TPUv5e 4x4 切片上服务这个模型，且 KV 缓存可以在整个拓扑上完全分片。若全部用 int8、要支持 128k 的序列，最大的批能开到多大？如果把 KV 头数降到 1 呢？

{% details 点此查看答案。 %}

int8 下每 token 的 KV 缓存是 $2 \cdot L \cdot K \cdot H$，即 `2 * 64 * 8 * 256 = 262kB`。128k 的序列意味着每个批元素 `262e3 * 128e3 = 33.5GB`。每块 TPU 只有 16GB HBM，还要放下参数，所以最大批是 `(16 * 16e9 - 18.4e9) / 33.5e9 = 7`。若 $K=1$，则是它的 8 倍，约 56。

{% enddetails %}

**问题 3：** 参数完全分片在 TPU v5e 4x4 切片上，全部从 HBM 装进 MXU 要多久？假设 int8 参数。*这是每步延迟的一个很好的下界。*

{% details 点此查看答案。 %}

共 18.4B 参数，int8 下即 18.4e9 字节。每块芯片 HBM 带宽 8.2e11，若带宽全部用满，大约 `18e9 / (8.2e11 * 16) = 1.4ms`。

{% enddetails %}

**问题 4：** 假设要在 TPUv5e 4x4 切片上用 int8 的 FLOPs、参数和激活来服务这个模型。预填充和解码分别该怎么分片？*提示：可以先回答这几个小问题：*

1. 4x4 切片上的 ICI 长什么样？
2. 张量并行的屋顶线界限是多少？
3. KV 缓存可以怎么切？

按你选的分法，生成阶段的每步大致延迟是多少？

**问题 5：** 假设上面的模型其实是 MoE。MoE 模型相当于一个把 FFW 块复制 $E$ 份的稠密模型：每个 token 只走 $k$ 个 FFW 块，输出取这 `k` 个的平均。仍用上面的设置，取 `E=16`、`k=2`。

1. 总参数量和激活参数量各是多少？*激活指任意给定 token 会真正用到的部分。*
2. 在 TPU v5e 上要多大的批才能进入计算受限区？
3. 每 token 的 KV 缓存多大？
4. T 个 token 的一次前向要多少 FLOPs？

{% details 点此查看答案。 %}

(1) 作为 MoE，每个 MLP 块现在有 $3 * E * D * F$ 个参数，是稠密版的 $E$ 倍。总参数为 $L * D * (3EF + 2H * (N + K)) + D * V$，即 `64 * 4096 * (3*16*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 212e9`，涨了约 12 倍。激活参数用的是 $k$ 而不是 $E$，共 `64 * 4096 * (3*2*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 31.2e9`，比稠密版涨了不到 2 倍。

(2) 参数多了 $E$ 倍、FLOPs 只多 $k$ 倍，所以 HBM 屋顶线抬高 $E/k$ 倍。TPU v5e 上大约需要 `240 * (16 / 2) = 1920` 个 token。

(3) KV 缓存大小不变——MoE 不改变注意力机制的任何东西。

(4) 仍然是 $2 \cdot \text{activated params} \cdot T$，即 $2 * \text{31.2e9} * T$。

{% enddetails %}

**问题 6：** 对 MoE 可以做“专家分片”：把专家切到网格的一条轴上。按我们的标准记号，第一个 FFW 权重形状是 `[E, D, F]`，切法是 [E<sub>Z</sub>, D<sub>X</sub>, F<sub>Y</sub>]，其中 `X` 只在训练里用作 FSDP 维。假设在 TPU v5e 上推理：

1. 上面这个模型在 TPU v5e 8x16 切片、Y=8、Z=16 时的 HBM 权重装载时间是多久？每块 TPU 还剩多少空闲 HBM？
2. 我们的模型最小能塞进多大的切片？

**问题 7 [二维模型分片]：** 我们来算一算 [ESTI 论文](https://arxiv.org/pdf/2211.05102)所称“二维权重驻留分片（2D weight-stationary sharding）”背后的数学。附录 B 会简略介绍它，但建议先自己做一遍这道题，试试能不能自己推出来。基本思路：把权重同时沿 $D$ 和 $F$ 两轴切开，让每块权重近似正方形。这能减轻通信负担，让我们能多撑一会儿。

二维权重驻留的算法：

<div markdown=1 class="algorithm">

1.  In[B, D<sub>X</sub>] = **AllGather**<sub>YZ</sub>(In[B, D<sub>XYZ</sub>])
2.  Tmp[B, F<sub>YZ</sub>] {U<sub>X</sub>} = In[B, D<sub>X</sub>] \*<sub>D</sub> W<sub>in</sub>[D<sub>X</sub>, F<sub>YZ</sub>]
3.  Tmp[B, F<sub>YZ</sub>] = **AllReduce**<sub>X</sub>(Tmp[B, F<sub>YZ</sub>] {U<sub>X</sub>})
4.  Out[B, D<sub>X</sub>] {U<sub>YZ</sub>} = Tmp[B, F<sub>YZ</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>YZ</sub>, D<sub>X</sub>]
5.  Out[B, D<sub>XYZ</sub>] = **ReduceScatter**<sub>YZ</sub>(Out[B, D<sub>X</sub>] {U<sub>YZ</sub>})
</div>

你的目标是算出这个算法的 $T_\text{math}$ 和 $T_\text{comms}$，找出它什么时候会胜过传统的一维模型分片。

{% details 点此查看答案！ %}

来算 $T_\text{math}$ 和 $T_\text{comms}$。所有 FLOPs 都是完全分片的，所以照旧 $T_\text{math} = 4BDF / (N \cdot C)$；而现在是

$$\begin{align*}
T_\text{2D comms} = \frac{2BD}{2X \cdot W_\text{ici}} + \frac{4BF}{YZ \cdot W_\text{ici}} + \frac{2BD}{2X \cdot W_\text{ici}} = \frac{2BD}{X \cdot W_\text{ici}} + \frac{4BF}{YZ \cdot W_\text{ici}}
\end{align*}$$

注意 AllReduce 贵一倍，而每个操作要按它横跨的轴数来摊薄通信。假设拓扑可以自由选择，并设 $F=4D$（如 LLaMA-2），我们断言（用一点初等微积分）最优取值为 $X = \sqrt{N / 8}$、$YZ = \sqrt{8N}$，总通信为

$$T_\text{2D comms} = \frac{2B}{W_\text{ici}} \left(\frac{D}{X} + \frac{8D}{YZ}\right) = \frac{\sqrt{128} BD}{\sqrt{N} \cdot W_\text{ici}} \approx \frac{11.3 BD}{\sqrt{N} \cdot W_\text{ici}}$$

首先，照搬上文：普通一维模型并行的 $T_\text{model parallel comms} = 4BD / (3 \cdot W_\text{ici})$。新方案的通信什么时候更小？有

$$\begin{align*}
T_\text{model parallel comms} > T_\text{2D comms} \iff \frac{4BD}{3 \cdot W_\text{ici}} > \frac{\sqrt{128} BD}{\sqrt{N} \cdot W_\text{ici}} \\
\iff N > 128 \cdot \left(\frac{3}{4}\right)^2 = 72
\end{align*}$$

一般的 $F$ 下，我们断言条件是

$$N > 32 \cdot \left(\frac{F}{D}\right) \cdot \left(\frac{3}{4}\right)^2$$

也就是说，芯片超过 72 块就该改用这个新方案。这个结果有点怪——因为我们历史上大约 20 路张量并行就被 ICI 卡住了。可这里即使处于通信受限区，总通信量仍随芯片总数持续下降！它告诉我们：可以继续加芯片、加批、把参数做得更大，而延迟还能降。

{% enddetails %}

<h3 markdown=1 class="next-section">第 7 部分到此结束！第 8 部分看看怎么在 TPU 上把 LLaMA 3 服务起来，[点击这里](../applied-inference)。</h3>

## 附录

### 附录 A：批大于 240 这条规则靠谱吗？

上面给的简单规则——批要大于 240 个 token 才能计算受限——大体成立，但低估了 TPU 的一种能力：当其他操作（比如设备间通信）没有用满 HBM 带宽时，它会偷偷预取权重。

这里有一张实测图：一个小 Transformer 的逐层耗时（微秒），d<sub>model</sub> 8192、d<sub>ff</sub> 32768、每层只有 2 个矩阵乘。数据出自[这个 Colab notebook](https://colab.sandbox.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing)。可以看到：步长在批到 240 之前涨得极慢，之后就线性上涨。

{% include figure.liquid path="assets/img/batch-scaling-latency.png" class="img-fluid img-small" %}

这是实测吞吐（token/微秒）。道理讲得很清楚了：这层约 600M 参数、切了 4 片，最小延迟理应约 365us。

{% include figure.liquid path="assets/img/batch-scaling-throughput.png" class="img-fluid img-small" %}

所以至少在这个模型上，吞吐确实在涨到“每个数据并行分片批 240”之前一直涨。

### 附录 B：二维权重驻留分片

拓扑继续变大时，如果能用到高维网格（比如 TPU），还可以用“**二维权重切分**”再改良一次——引入第二条切分轴。我们叫它“**二维权重驻留**”，在 [Efficiently Scaling Transformer Inference 论文](https://arxiv.org/abs/2211.05102)里有更详细的描述。

Megatron 只切隐层 $$F$$ 维：芯片数一大、又是单维切分时，切到每片剩下的部分会远小于 $$E$$（即 $$d_\text{model}$$ 维）。批较大时，在 MLP 第一层算完之后把一部分集合通信挪到隐层维去做，会更划算。

{% include figure.liquid path="assets/img/2d-weight-stationary.png" class="img-fluid img-small" %}

这张图展示了：

1. 一维权重驻留分片，即纯 Megatron 分片：激活经 AllGather 后完全复制，权重只沿隐层 F 维完全分片。
2. 二维权重驻留分片：权重同时沿隐层 F 维和收缩 E 维切分，激活沿 E 维切分。第一层矩阵乘前在 (yz) 轴做 AllGather，之后在 (x) 轴做 ReduceScatter。

注意力层同理，芯片数不多时，Megatron 式分片也相对简单。只是 Megatron 切的是 $$n_\text{heads}$$ 维，能切的份数有限。把二维方案套到注意力上（不切隐层，改切 $$n_\text{heads}$$ 维），就能继续往上扩。

### 附录 C：延迟受限的通信

复习一下[第 3 部分](../sharding)的推导：在 X 块芯片、1D 环网（链路全双工带宽 WICI、单次延迟 Tmin）上，向每块 TPU 做一次规模 B 的 AllGather 需要的时间是

$$T_{total} = \max\left(\frac{T_{min} \cdot |X|}{2}, \frac{B}{W_{ICI}}\right)$$

B 很大时，墙钟时间基本不变：加芯片的同时，要搬的数据量和总带宽是同步增长的。

{% include figure.liquid path="assets/img/all-gather.gif" class="img-fluid" %}

延迟优先的推理搬的数据很少，所以激活上的集合通信常常落在延迟项里（小批尤其如此）。很好直观感受延迟：数一数完成一次通信要“跳”过多少跳。

在 TPU 上，如果通信里“与张量大小有关”的部分每跳不足 1 微秒（一跳 = 一对相邻设备间的通信），真正卡我们的就是分发这次集合通信本身的固定开销。ICI 单向带宽 `4.5e10`，于是当 $$(\text{bytes} / n_\text{shards}) / 4.5e10 < 1e-6$$ 时通信进入延迟受限。8 路 Megatron 分片下，这就是 `buffer_size < 360kB`。**这个数在推理里真不算小：**`BS=16`、`D=8192` 的 int8 激活才 `16*8192=131kB`——我们已经延迟受限制了。

<p markdown=1 class="takeaway">**要点**：当 $$\text{total bytes} < W_{ICI} \times 1e-6$$ 时，通信进入延迟受限。比如沿 $$Y$$ 做模型并行时，int8 下 $$Y > BD / 45,000$$ 就会被卡住。</p>

这和计算屋顶线如出一辙——都是固定成本在坑我们：通信的固定成本是延迟，矩阵乘的固定成本是内存带宽。

### 附录 D：投机采样

当我们*真的*非常在意端到端延迟时，还有一招可用：投机采样<d-cite key="spec1"></d-cite><d-cite key="spec2"></d-cite>。回顾一下，平时我们是一个 token 一个 token 地从大 Transformer 里往外生成：

{% include figure.liquid path="assets/img/spec-sampling1.png" class="img-fluid" %}

投机采样用一个更小的便宜模型先写 token，再拿大模型来批改作业。用*贪心解码*最容易讲清楚：

{% include figure.liquid path="assets/img/spec-sampling2.png" class="img-fluid" %}

1. 用某个更小更便宜的模型贪心生成。最好用大模型蒸馏出来的模型，不过哪怕只是 n-gram、或在小语料上做 token 匹配也行。
2. 攒够 K 个 token 后，用大模型对已生成的整段算一遍 next-token logits。
3. 既然是贪心解码，只要检查小模型出的每个 token 是不是该位置上概率最大的那个。若某个错了，就取最长正确前缀、把第一个错的 token 换成正确的，回到 (1)；若全对，就顺手用最后那个 logits 多采样一个 token，再回到 (1)。

**为什么这是延迟上的胜利？**这套做法算下来，每个 token 仍要付出相当于大模型一次前向的 FLOPs——但许多 token 可以凑在一起，一次前向全算完。我们利用了“本来就没到计算受限”这个事实，多给几个 token 打分几乎不花额外时间。

平均到每个被接受的 token，FLOPs 其实更贵了（有的会被拒绝，还得跑草稿模型），但硬件里榨出的 FLOPs 更多，而小模型又便宜，总体是赚的。另外多个步共享 KV 缓存的读取，**长上下文下投机解码还能提升吞吐。**由于所有 token 都经过大模型校验，采样分布完全没有改变（非贪心时具体轨迹会不同，但分布不变）。

传统上，投机解码需要一个与目标模型分布接近的小模型，比如 LLaMA-2 70B 配 LLaMA-2 2B——这种模型常常不存在。就算有，接受率低时草稿模型也还是太贵。于是另一种思路是把草稿头嵌进主模型里：比如给靠后的某一层加一个专门的草稿头<d-cite key="eagle"></d-cite><d-cite key="medusa"></d-cite><d-cite key="DeepSeek3"></d-cite>。这种草稿头和主模型共享大部分参数，跑起来更快，分布也更贴合。

普通自回归采样下，每秒 token 数就等于步长的倒数。我们依然逃不出上文“算术强度”一节给出的最小步长。（实际上投机采样的单步往往比常规自回归慢不少——但平均每步能产出多个 token，每秒 token 数反而好得多。）

{% include figure.liquid path="assets/img/spec-sampling3.png" class="img-fluid" caption="<b>图：</b>Chinchilla（DeepMind 的 70B 模型）配 4B 参数草稿模型的每步延迟和草稿接受率。对 XSum（自然语言数据集），最理想的提前量约 3-4 个 token；HumanEval（代码数据集）更可预测，冒进一些的投机还能一直赚。" %}

**非贪心解码怎么办？**复杂一些，但本质上是一个源自 Metropolis-Hastings 的算法：从 logits 得出 $$P_{\text{draft model}}(\text{chosen token})$$ 和 $$P_{\text{target model}}(\text{chosen token})$$，若两者比值小于某个阈值，就按概率拒绝这个 token。

这两篇[论文](https://arxiv.org/abs/2211.17192) [1](https://arxiv.org/abs/2302.01318)各自独立给出了该算法，里面有很好看的实际例子。

<p markdown=1 class="takeaway">**要点**：投机采样又是一根“用吞吐换单 token 延迟”的杠杆。而在批大小受限的场景（硬件规模小、或 KV 缓存很大），它是双赢。</p>
