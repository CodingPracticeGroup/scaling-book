---
layout: distill
title: "漫谈 Roofline"
# permalink: /main/
description: "当我们在硬件上运行算法时，会受到三样东西的限制：计算机做数学运算的速度（OPs/秒）、搬运数据可用的带宽（字节/秒），以及可用于存数据的总内存（字节）。这些“roofline”式约束让我们能够给出一次给定计算耗时的上下界。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 1

previous_section_url: ".."
previous_section_name: "第 0 部分 · 引言"

next_section_url: ../tpus
next_section_name: "第 2 部分 · 漫谈 TPU"

bibliography: main.bib

giscus_comments: true
giscus_discussion: 3

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

  - name: 时间都花到哪里去了？
  - subsections:
    - name: "Roofline 的可视化"
    - name: "矩阵乘法"
    - name: "网络通信 roofline"
  - name: 一些练习题

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

## 时间都花到哪里去了？

从一个极简单的问题开始：*为什么某个算法要花 50 毫秒，而不是 50 秒或者 5 毫秒*？模型里到底是哪部分在吃时间、它本该吃多久？

**计算：** 深度学习模型拆开看，就是一大堆矩阵乘法；每个矩阵乘法都由浮点的乘法和加法“操作”（FLOPs）组成。计算器一秒能算多少次，就决定了这一步要跑多久：

$$\begin{equation}
T_\text{math} = \frac{\text{Computation FLOPs}}{\text{Accelerator FLOPs/s}}
\end{equation}$$

举个例子：一块 NVIDIA H100 一秒能做约 9.89e14 个 bfloat16 FLOPs<d-footnote>bf16 是 <a href="https://en.wikipedia.org/wiki/Bfloat16_floating-point_format">bfloat16</a> 的缩写，一种常用于机器学习的 16 位浮点格式。</d-footnote>，TPU v6e 一秒能做约 9.1e14 个。<d-footnote>H100 和 B200 通常只能达到标称峰值 FLOPs 的约 80-85%，而 TPU 在正常使用下能接近 95%。</d-footnote> 也就是说，做 `1e12` 个 FLOPs，在 H100 上大约要 `1e12 / 9.89e14 = 1.01ms`，在 TPU v6e 上要 `1e12 / 9.1e14 = 1.1ms`。<d-footnote>注意这些芯片的定价不同，这个比较没有按成本归一化。</d-footnote>

**片内通信：** *在一块加速器内部*，数据要在加速器的内存（HBM）和计算核心之间来回搬。这条“搬运通道”的带宽，就叫“HBM 带宽”。<d-footnote>NVIDIA 也称其为“memory bandwidth”（内存带宽）。</d-footnote> 在 H100 上，[这大约是 3.35TB/s](https://www.nvidia.com/en-us/data-center/h100/)，在 TPU v6e 上[这大约是 1.6TB/s](https://cloud.google.com/tpu/docs/v6e)。

**芯片间通信：** 当我们把一个模型分布到*多块*加速器上时，张量经常需要在它们之间搬运。硬件上这通常有几种选择（ICI、DCN 和 PCIe），各自带宽不同。

无论是片内还是片间通信，我们都以 字节/秒 来衡量它，并用下式估算总的通信时间：

$$\begin{equation}
T_\text{comms} = \frac{\text{Communication Bytes}}{\text{Network/Memory Bandwidth Bytes/s}}
\end{equation}$$

通常（虽有例外），计算可以和片内、片间的通信同时进行、互相打掩护。这意味着**取计算时间和通信时间里较大的那个，就能给训练和推理时间定一个下界**；**把两者相加，则得到一个上界**。实践中我们按下界来优化：一是这样算起来简单，二是只要让通信躲在计算背后进行，通常就真能跑到那么快。按最大值来算，上下界至多差 2 倍，因为 $T_\text{math} + T_\text{comms} \leq 2 * \max(T_\text{math}, T_\text{comms})$。之后我们再靠建模“重叠区间”和各类开销来算得更准——这些可以对你的模型和目标系统做性能剖析（profiling）拿到。

$$\begin{equation}
T_\text{lower}=\max(T_\text{math}, T_\text{comms})
\end{equation}$$

$$\begin{equation}
T_\text{upper} = T_\text{math} + T_\text{comms}
\end{equation}$$

假设通信和计算真能完美配合，那么当 $T_\text{math} > T_\text{comms}$ 时，硬件就是满载的，我们叫它“计算受限（compute-bound）”。反过来 $T_\text{comms} > T_\text{math}$ 时，就是“通信受限（communication-bound）”<d-footnote>在本书中，“communication-bound”“comms-bound”“memory-bound” 和 “bandwidth-bound” 会交替使用。</d-footnote>——算力中至少有一部分，白白耗在了等数据上。要判断一个操作属于哪种，可以看它的“**算术强度（arithmetic intensity）**”，也叫“**操作强度（operational intensity）**”。

**定义：** 一个算法的算术强度等于它所执行的总 FLOPs 与它需要通信的字节数（无论是片内还是片间）之比。

$$\begin{equation}
\text{Arithmetic Intensity} = \frac{\text{Computation FLOPs}}{\text{Communication Bytes}}
\end{equation}$$

算术强度说的就是“每搬一个字节，能换来几次浮点运算”。简单讲：强度越高，$T_\text{math}$ 相对 $T_\text{comms}$ 就越大，算力通常能被充分用起来；强度越低，时间就越多地耗在通信上，算力跟着闲置。从“够用”翻到“不够用”的那个临界点，叫硬件的“峰值算术强度”——拿峰值算力除以内存带宽就得到它。

$$\begin{align*}
T_\text{math} > T_\text{comms} \Leftrightarrow \frac{\text{Computation FLOPs}} {\text{Accelerator FLOPs/s}} > \frac{\text{Communication Bytes}}{\text{Bandwidth Bytes/s}} & \\[0.5em]
\Leftrightarrow \frac{\text{Computation FLOPs}}{\text{Communication Bytes}} > \frac{\text{Accelerator FLOPs/s}}{\text{Bandwidth Bytes/s}} & \\[0.5em]
\Leftrightarrow \text{Intensity}(\text{Computation}) > \text{Intensity}(\text{Accelerator}) & \\
\end{align*}$$

量 $\text{Intensity}(\text{Accelerator})$ 是使加速器达到峰值 FLOPs/s 的算术强度。**对于 TPU v5e 的 MXU，这约为 240 FLOPs/字节**，因为该 TPU 能以 `1.97e14` FLOPs/s 运行，并能以 `8.2e11` 字节/s 从 HBM 载入数据。<d-footnote>MXU 是 TPU 上的矩阵乘法单元。这里特别指明，是因为 TPU 还有别的加速器，比如负责逐元素操作、峰值 FLOPs/s 不同的 VPU。</d-footnote> 也就是说，如果一个算法的算术强度低于 240 FLOPs/字节，它就会受字节加载的限制，因此我们无法好好利用硬件。<d-footnote>这只在算法从 HBM 读取权重并在 MXU 上运行时成立。正如我们下一节会讨论的，有时我们能把参数存放在带宽高得多的 VMEM 里。许多算法还会运行在 VPU 上，其性能特性各不相同。</d-footnote> 我们来看一个这样的例子：

**<span style="color:#7ab5ff">示例（点积）：</span>** 要以 bfloat16 精度计算两个向量的点积 `x • y: bf16[N], bf16[N] → bf16[1]`，我们需要从内存载入 $x$ 和 $y$，每个都有 $2 * N = 2N$ 字节，执行 $N$ 次乘法和 $N-1$ 次加法，并写回 $2$ 字节到 HBM
$$\begin{equation}
\text{Intensity}(\text{dot product}) = \frac{\text{Total FLOPs}}{\text{Total Bytes}} = \frac{N + N - 1}{2N + 2N + 2} = \frac{2N - 1}{4N + 2} \rightarrow \frac{1}{2}
\end{equation}$$

令 $N\rightarrow\infty$ 取极限。也就是说，点积的算术强度只有 $\frac{1}{2}$：每从内存搬一个字节，只换来 0.5 次浮点运算。这比硬件的算术强度低得多，所以点积注定是通信受限的。<d-footnote>上面那个 240 在这里并不是正确的比较对象，因为正如你将在下一节看到的，点积是在 VPU 而非 MXU 上执行的。TPU v5p 的 VPU 每核大约能做 7e12 FLOPs / 秒，所以它的临界强度约为 3，这意味着我们在这里仍然有些通信受限。无论哪种情况，我们的强度都既低又是常数，这意味着在大多数硬件上都很难做到计算受限。</d-footnote>

### Roofline 的可视化

我们可以用 **roofline 图**把内存和计算之间的这层关系画出来：纵轴是某个算法在我们硬件上能跑出的峰值 FLOPs/s（也就是吞吐量），横轴是这个算法的算术强度。每个算法都能在这张图上占一个点。下面是一张对数-对数（log-log）坐标的示例图：

{% include figure.liquid path="assets/img/roofline-improved.png" class="img-fluid" caption="<b>图：</b>一个 roofline 图示例，展示了两个具有不同算术强度的算法（Algo 1 与 Algo 2），以及它们在不同带宽（BW1 与 BW2）下对应的理论峰值吞吐量。红色区域中，算法在两种带宽下都是带宽受限的，浪费了硬件峰值 FLOPs/s 的一部分。黄色区域只在较低带宽（BW1）下是带宽受限的。绿色区域在所有带宽下都是计算受限的。这里我们使用的是加速器的峰值 FLOPs/s，增加带宽或提高强度都不会带来收益。" %}

如上图，随着强度增大（从左往右），我们起初会看到算法性能（以 FLOPs/s 计）线性上升，直到撞上硬件的临界算术强度——对 TPU v5e 而言是 240。任何强度更低的算法都会受带宽（BW）限制，被峰值内存带宽卡住（图中红色）。任何更靠右的算法都会充分利用我们的 FLOPs（图中绿色）。这里，Algo 1 是通信受限的，只用到了硬件总 FLOPs/s 的一小部分；Algo 2 是计算受限的。我们通常可以通过提高算法的算术强度、或提高可用的内存带宽（从 BW1 变到 BW2）来改善一个算法的性能。

### 矩阵乘法

我们来看接下来将成为我们最爱的算法：矩阵乘法（简称 matmul）。给定两个矩阵 $X$ 和 $Y$，我们可以写作 $X * Y \rightarrow Z$，其中 $X$ 的形状是 $\text{bf16}[B, D]$，$Y$ 的形状是 $\text{bf16}[D, F]$，$Z$ 的形状是 $\text{bf16}[B, F]$。要做这个 matmul，我们需要载入 $2DF + 2BD$ 字节，执行 $2BDF$ FLOPs，并写回 $2BF$ 字节。<d-footnote>严格地说，我们执行的是 $BF \times (2D - 1)$ 次 FLOPs，但这已经足够接近。它来自 $BDF$ 次乘法和 $BF * (D-1)$ 次加法。第 4 节有更多细节。</d-footnote> <d-footnote>尽管 matmul 的输出严格来说是 float32，我们通常会在拷回 HBM 前把它下转型（downcast）成 bfloat16。</d-footnote> 于是：

$$\begin{equation}
\text{Intensity}(\text{matmul}) = \frac{2BDF}{2BD + 2DF + 2BF} = \frac{BDF}{BD + DF + BF}
\end{equation}$$

如果我们假设“批大小（batch size）” $B$ 相对于 $D$ 和 $F$ 很小，就能得到一个很简洁的简化。于是我们得到

$$\begin{equation}
\frac{BDF}{BD + DF + BF} \approx \frac{BDF}{DF} = B
\end{equation}$$

$$\begin{equation}
\text{Intensity}(\text{matmul}) > \text{Intensity}(\text{TPU}) \implies B > \frac{1.97e14}{8.20e11} = 240
\end{equation}$$

对 Transformer 的 matmul 来说，这是一个合理的假设，因为我们通常的本地（每个副本 per-replica）token 批大小 $B < 1024$（*注意，不是序列数 sequences*），而 $D$ 和 $F > 8000$。因此，当我们每个副本的<d-footnote>我们说每个副本，是因为如果我们通过某种模型分片来增加参与 matmul 的芯片数，我们的可用计算和内存带宽会按同一比例一起放大。所以这个临界批大小是针对每一份独立的模型权重副本而言的。</d-footnote> 批大小大于 240 个 token 时，我们通常就会变成计算受限——一条非常简单的规则！

<p markdown=1 class="takeaway">**要点：** 要让一个 bfloat16 的 matmul 在大多数 TPU 上变成计算受限，我们的批大小需要大于 240（个 token）。<d-footnote>注意这**不是**通常意义上的批大小（即按序列数计的批大小）。事实证明，大多数 roofline 只取决于 token 的数量，而不管它们属于同一序列还是不同序列。例如，如果你有 512 条序列、每条 4096 个 token，分布在 2048 块 GPU 上，那么你的总批大小是 `512 * 4096 = 2M` 个 token，本地批大小则是 1k 个 token。</d-footnote></p>

这条规则有几个值得注意的例外，我们会在下面的练习题里探讨，尤其是量化方面（例如，如果我们量化了激活值但仍做全精度的 FLOPs），但它是一条值得记住的好规则。对 GPU 而言，这个数字会稍高一些（接近 300），但同样的结论总体成立。当我们[把一个大 matmul 分解成多个小 matmul](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html#your-first-matrix-multiplication-kernel)时，tile 大小也很重要。<d-footnote>当我们做一次大矩阵乘法时，需要把它拆成能放进 VMEM/SMEM/TMEM（带宽更高的片上内存）的小 tile。这会导致我们多次载入数据块，因此“我们只载入 $O(N^2)$ 字节”就不再严格成立。考虑一个 $(m, k) \cdot (k, n)$ 的 matmul，tile 大小为 $bm$、$bk$、$bn$。令 $tm = m / bm$，以此类推。那么总 FLOPs 是 $2 \cdot tm \cdot tn \cdot tk \cdot bm \cdot bn \cdot bk$，总字节数是 $2 \cdot tm \cdot tn \cdot (tk \cdot (bm \cdot bk + bk \cdot bn) + bm \cdot bn)$。忽略最后一项，我们得到的强度为 $bm \cdot bn / (bm + bn)$，与上面的形式类似。</d-footnote> 我们会在[下一节](../tpus)讨论更底层的 GPU 和 TPU 细节。

### 网络通信 roofline

到目前为止，我们算的都是内存带宽的 roofline，*全都发生在单块芯片之内*。但别把这当成天经地义。实际上，本书真正关心的大多数 roofline 都涉及芯片之间的通信——多半是那种矩阵被拆到多块 TPU 上做的 matmul。

这里做个略人造的小例子：想相乘两个大矩阵 $X\sim \text{bf16}[B, D]$ 和 $Y \sim \text{bf16}[D, F]$，它们沿 $D$ 维被均分给了 2 块 TPU/GPU。要做这个乘法（做法我们在[第 3 节](../sharding)会细讲），可以每块 TPU 各算一半：TPU 0 算 `Z0 = X[:, :D // 2] @ Y[:D // 2, :]`，TPU 1 算 `Z1 = X[:, D // 2:] @ Y[D // 2:, :]`，再把各自的“部分和（partial sums）”发给对方、加在一起。假设拷贝带宽每个方向 `4.5e10` 字节/s，每块芯片算力 `1.97e14` FLOPs/s。那么 $T_\text{math}$ 和 $T_\text{comms}$ 各是多少？

$T_\text{math}$ 显然只有之前的一半，因为每块 TPU 只做一半的工作，即<d-footnote>我们忽略了把两个部分和相加所需的 FLOPs（又是 BF 次加法），但这基本可以忽略。</d-footnote>

$$T_\text{math} = \frac{2BDF}{2 \cdot \text{Accelerator FLOPs/s}} = \frac{BDF}{1.97e14}$$

那 $T_\text{comms}$ 呢？现在它指的是芯片之间的通信时间！这就是发送的总字节数除以网络带宽，即

$$T_\text{comms} = \frac{2BF}{\text{Network Bandwidth}} = \frac{2BF}{4.5e10}$$

因此，当我们有 $$\text{Intensity}(\text{matmul (2-chips)}) > \text{Intensity}(\text{TPU w.r.t. inter-chip network})$$ 时，或者说等价地当 $\frac{BDF}{2BF} = \frac{D}{2} > \frac{1.97e14}{4.5e10} = 4377$（即 $D > 8755$）时，我们就变成计算受限的（此时是相对于芯片间网络而言）。注意，与之前不同，临界阈值现在取决于 $D$ 而不是 $B$！试着想想这是为什么。这只是一个这样的例子，但我们想强调的是，这类 roofline 对于判断我们何时能把一个操作并行到多块 TPU 上至关重要。

## 一些练习题

**问题 1 [int8 matmul]：** 假设我们想以 int8 精度（每个参数 1 字节）而非 bfloat16（每个参数 2 字节）来做 matmul $X[B, D] \cdot_D Y[D, F] \rightarrow Z[B, F]$<d-footnote>在此及全书中，我们用记号 $A \cdot_D B$ 来表示这次乘法是沿 D 维做收缩（contraction）。这是对 einsum 记号的一种借用（不严谨用法）。</d-footnote>，因为 TPU/GPU 在更低精度下做 matmul 更快。

1. 需要从内存载入多少字节？需要写回多少字节？
2. 总共执行了多少次 OP？
3. 算术强度是多少？
4. 对 $T_\text{math}$ 和 $T_\text{comms}$ 做一个 roofline 估计？整个运行时的合理上下界是什么？

假设我们的 HBM 带宽是 `8.2e11` 字节/s，int8 峰值 OPs/s 是 `3.94e14`（约为 bfloat16 的 2 倍）。

{% details 点此查看答案。 %}

1. 因为我们的参数以 int8 存放，每个参数 1 字节，所以我们从 HBM 载入 $$BD + DF$$ 字节，写回 $$BF$$ 字节。
2. 这与 bfloat16 相同，但理论上 int8 OPs/s 应该更快。所以这仍然是 $2BDF$ 次 OP。
3. 算术强度是 $$2BDF / (BD + DF + BF)$$。如果我们沿用上面关于 $$B \ll D$$ 和 $$B \ll F$$ 的假设，得到的算术强度就是 $$2B$$，于是我们的规则变成 $B > \text{HBM int8 算术强度} / 2$。用给出的数字，这个 int8 强度是 `3.94e14 / 8.2e11 = 480`，所以规则是 $B > 480 / 2 = 240$。注意这基本上没什么变化！
4. $$T_\text{math} = 2BDF / 3.94e14$$ 且 $$T_\text{comms} = (BD + DF + BF) / 8.2e11$$，因此一个合理的下界是 $$\max(T_\text{math}, T_\text{comms})$$，上界是 $$T_\text{math} + T_\text{comms}$$。

{% enddetails %}

**问题 2 [int8 + bf16 matmul]：** 在实践中，我们常常对权重和激活值做不同的量化，所以可能把权重以极低精度存放，而把激活值（和计算）保持较高的精度。假设我们想把权重量化成 int8，但把激活值（和计算）保持为 bfloat16。我们在多大的批大小下会变成计算受限？假设 bfloat16 为 `1.97e14` FLOPs/s。

*提示：这里具体指 `bf16[B, D] * int8[D, F] -> bf16[B, F]`，其中 $B$ 是“批大小”。*

{% details 点此查看答案。 %}

同样假设 B 很小，我们有 2BDF 次 bfloat16 FLOPs，但只有 DF 个权重（而不是 bfloat16 下的 2DF）。这意味着当 $$2B > 240$$，即 $$B > 120$$ 时我们就变成计算受限。这个值要低得多，意味着如果我们能做 int8 权重量化（这相当容易做到）但仍做 bfloat16 FLOPs，就能在效率上获得可观的收益（尽管 int8 OP 会更好）。

{% enddetails %}

**问题 3：** 沿用问题 2 的设置，针对 $F = D = 4096$ 和 $F = D = 1024$，画一张峰值 FLOPs/s 关于 $B$ 的 roofline 图。*请使用载入字节的确切数量，不要用近似。*

{% details 点此查看答案。 %}

下面是那张图：

{% include figure.liquid path="assets/img/roofline-plot-q3.png" class="img-fluid img-small" %}

注意两个模型最终都能达到硬件的峰值 FLOPs/s，但更大的 D/F 会更早达到。D=F=1024 几乎让临界批大小翻了一倍。生成这张图的代码在这里：

```py
import matplotlib.pyplot as plt
import numpy as np

bs = np.arange(1, 512)

def roofline(B, D, F):
  total_flops = 2*B*D*F
  flops_time = total_flops / 1.97e14
  comms_time = (2*B*D + D*F + 2*B*F) / 8.2e11
  total_time = np.maximum(flops_time, comms_time)
  return total_flops / total_time

roofline_big = roofline(bs, 4096, 4096)
roofline_small = roofline(bs, 1024, 1024)

plt.figure(figsize=(8, 4))
plt.plot(bs, roofline_big, label='F=D=4096')
plt.plot(bs, roofline_small, label='F=D=1024')
plt.legend()
plt.xlabel('batch size')
plt.ylabel('peak bfloat16 FLOPs/s on TPU v5e')
plt.grid()
```

{% enddetails %}

**问题 4：** 如果我们想做 $\text{int8}[B, D] \cdot_D \text{int8}[B, D, F] \rightarrow \text{int8}[B, F]$，即设想每个批元素都有一个不同的矩阵，会怎样？这个操作的算术强度是多少？

{% details 点此查看答案。 %}

我们先来看总的 FLOPs 和通信量。

1. 总 FLOPs：FLOPs 基本相同，因为我们在做 $$B$$ 个独立的 $$[D] \times [D, F]$$ 乘积，其总工作量与单个 $$[B, D] \times [D, F]$$ 的 matmul 相同（这会在第 4 节进一步讨论）。所以这就是 $$2BDF$$。
2. 总通信量：这里的通信量大得多：$$BD + BDF + BF$$。
3. 因此，我们的算术强度现在实际上是 $$2BDF / (BD + BDF + BF)$$。由于 $$BDF$$ 在分母中占主导，它大致等于 $$2$$。于是它不再取决于批大小，而是本质上恒定的。这很糟糕，因为这意味着无论怎样我们基本上总会是通信受限的。

{% enddetails %}

**问题 5 [GPU 的内存 roofline]：** 使用 [NVIDIA 提供的 H100 SXM 规格表](https://www.nvidia.com/en-us/data-center/h100/)，计算 bfloat16 矩阵乘法在多大的批大小下会变成计算受限。*注意，由于 Tensor Core FLOPs 的数值只有在结构化稀疏下才能达到，所以它是真实值的两倍。*

{% details 点此查看答案。 %}

从规格表中我们看到，标称的 bfloat16 FLOPs 值是 `1.979e15` FLOPs/s，并带有一个注明“with sparsity”（启用稀疏）的星号。在没有稀疏的真实情况下，这个值是其一半，即 `9.89e14` FLOPs/s。内存带宽是 3.35TB/s，即 `3.35e12` 字节 / 秒。于是 $B_\text{crit}$ 是 `9.89e14 / 3.35e12 = 295`，与 TPU 相当接近。

{% enddetails %}

<h3 markdown=1 class="next-section">第 1 部分到此结束！第 2 部分将看看真实的 TPU 如何处理 FLOPs 与通信，[点击这里](../tpus)。</h3>
