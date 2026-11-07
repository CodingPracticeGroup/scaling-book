---
layout: distill
title: "如何理解 TPU"
# permalink: /main/
description: "这一节专门讲 TPU 是怎么工作的、它们是怎么联网起来支撑多芯片的训练和推理的，以及这一切会怎样影响我们那些常用算法的性能。对 GPU 用户来说，这里面也有不少有用的内容！"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting

section_number: 2

previous_section_url: "../roofline"
previous_section_name: "第 1 部分 · Roofline 入门"

next_section_url: ../sharding
next_section_name: "第 3 部分 · 分片矩阵乘法"

bibliography: main.bib

giscus_comments: true
giscus_discussion: 4

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
  - name: TPU 是什么？
  - name: TPU 的网络互联
  - name: 重点小结
  - subsections:
    - name: TPU 规格参数
  - name: 动手解题
  - name: 附录
  - subsections:
    - name: "附录 A：更多 TPU 内部细节"
    - name: "附录 B：脉动阵列是怎么工作的？"

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

<p markdown=1 class="announce">你可能也会喜欢我们新写的[第 12 节](../gpus)——它讲的是 NVIDIA GPU！</p>

## TPU 是什么？

**简单说，TPU 就是一个专门做矩阵乘法的计算核心（叫 TensorCore），配上一大摞高速内存（叫高带宽内存，HBM）<d-cite key="tpu_paper"></d-cite>。** 看这张图：

{% include figure.liquid path="assets/img/tpu-chip.png" class="img-fluid" caption="<b>图：</b>一块 TPU 芯片的基本组成。左边灰色的那个框就是 TensorCore，里面包含矩阵乘法单元（MXU）、向量单元（VPU）和向量内存（VMEM）。" %}

你可以把 TensorCore 看成一台"矩阵乘法做得特别好的机器"，但它还有几个值得一提的功能。TensorCore 里有三个关键部件：

* **MXU**（矩阵乘法单元）是 TensorCore 的核心。在 TPU7x 上，它靠一种叫"脉动阵列"的结构，每 `4` 个时钟周期做一次 `bf16[8,256] @ bf16[256,256] -> f32[8,256]` 的矩阵乘法（细节见<a href="#appendix-b-how-does-a-systolic-array-work">附录 B</a>）。
  * 在 TPU7x 上，2.2GHz 主频下，每个 MXU 大约能跑出 `5.8e14` bf16 FLOPs/秒。每块 TPU7x 芯片有 2 个 TensorCore，每个 TensorCore 有 2 个 MXU，所以整块 TPU7x 的 bf16 算力约为 `2.3e15` FLOPs/秒。
  * TPU 还支持用更低的精度做矩阵乘法，从而更快（比如每块 TPU7x 能做 `4.6e15` fp8 OP/秒）。

* **VPU**（向量处理单元）负责做那些"通用"的数学运算，比如 ReLU 激活、向量之间的逐元素加减乘除，求和这类"归约"操作也在这里做。<a href="#appendix-a-more-on-tpu-internals">附录 A</a> 有更多细节。
* **VMEM**（向量内存）是 TensorCore 里的一块"临时草稿纸"内存，紧挨着计算单元。它比 HBM 小得多（比如 TPU7x 上每芯片只有 128 MiB，而 HBM 有 192 GiB），但它到 MXU 的带宽要高得多。VMEM 有点像 CPU 上的 L1/L2 缓存，但更大、而且由程序员自己控制。数据想被 TensorCore 拿来计算，必须先复制进 VMEM。

**TPU 做矩阵乘法非常快、非常快**。这基本就是它擅长的事，而且做得很好。[TPU7x](https://cloud.google.com/tpu/docs/tpu7x#system_architecture) 是有史以来最强的 TPU 之一，每个核心能做到 `1.15e15` bf16 FLOPs/秒，整块芯片 `2.3e15`。一个由 9216 块芯片组成的整机群（pod）能做到 21 bf16 exaFLOPs/秒——那可是*天文数字*，是世界上最强的超算之一。而 Google 有一大堆这样的机器。<d-footnote>TPU——尤其是它的脉动阵列——之所以是这么强的硬件加速器，是因为矩阵乘法是少数几种"计算量按 $O(n^3)$ 增长、而内存只按 $O(n^2)$ 增长"的算法之一。正因如此，普通计算单元很容易被"算力"卡住，而不会被"内存带宽"卡住。</d-footnote>

上面那张图里还画了几个别的部件，比如 SMEM 和标量单元，它们主要用来处理控制流，我们在<a href="#appendix-a-more-on-tpu-internals">附录 A</a>里简单提一下，不必深究。但 HBM 既重要又比较简单：

* **HBM**（高带宽内存）是一大块高速内存，用来存放 TensorCore 要用的数据（张量）。HBM 容量通常是几十到几百 GB（比如 TPU7x 的 HBM 是 [192GiB](https://cloud.google.com/tpu/docs/tpu7x#system_architecture)）。

  * 当某个计算需要这些数据时，张量会从 HBM 流出，经过 VMEM（见下）进入 MXU 计算，算完的结果再从 VMEM 写回 HBM。

  * HBM 和 TensorCore 之间（经 VMEM）的带宽，就是常说的"HBM 带宽"（TPU7x 约 7TB/秒）。在那些被内存卡脖子的任务里，它决定了计算最快能有多快。

**通常，TPU 上的所有操作都是"流水线化、互相重叠"的。** 要做一次矩阵乘法 $X \cdot A \to Y$，TPU 会先把矩阵 $A$ 和 $X$ 的一块块数据从 HBM 复制进 VMEM，再送进 MXU，让它按每次 8x256（对 $X$）和 256x256（对 $A$）的小块去算，然后一块块把结果写回 HBM。为了做得高效，整个流程会"流水线化"：让进出 VMEM 的搬运和 MXU 的计算同时进行、互相重叠。这样 MXU 就一直有活干、不用干等内存搬运，矩阵乘法就能是"计算受限"而不是"内存受限"。

下面是一个例子，演示从 HBM 做一次逐元素相乘：

{% include figure.liquid path="assets/img/pointwise-product.gif" caption="<b>图：</b>一段动画，展示在 TPU 上做一次逐元素相乘（数据从 HBM 载入）。注意数据是分块从内存流出来的，结果也是一边算一边流水线式写回，而不用等整个数组都准备好。" %}

矩阵乘法看起来几乎一模一样，只是数据是送进 MXU 而不是 VPU/向量单元，而且读写的顺序会不同——因为同一块权重会被多块激活反复用到。你能看到数据一块块流进 VMEM，再进 VREG（向量寄存器），再进向量单元，然后又回到 VMEM 和 HBM。正如我们马上要看到的：如果"从 HBM 搬到 VMEM"比"向量单元（或 MXU）里的计算"还慢，我们就会变成"带宽受限"，因为 VPU 或 MXU 在饿着肚子等数据。

<p markdown=1 class="takeaway">**要点：** TPU 很简单。它把权重从 HBM 搬进 VMEM，再从 VMEM 送进一个脉动阵列——那个阵列每秒能做大约 290 万亿次乘加。HBM $\leftrightarrow$ VMEM、以及 VMEM $\leftrightarrow$ 脉动阵列 这两段带宽，从根本上决定了 TPU 能高效地做哪些计算。</p>

**VMEM 与算术强度：** VMEM 比 HBM 小得多，但它到 MXU 的带宽高得多。正如我们在[第 1 节](../roofline)看到的，这意味着：如果一个算法能把所有输入/输出都塞进 VMEM，它就很难撞上通信瓶颈。对那种算术强度不高的计算，这一点尤其有用：TPU7x 上，VMEM 带宽比 HBM 带宽高约 7 倍，这意味着一个读写 VMEM 的 MXU 运算，要达到峰值算力，所需的算术强度只要用 HBM 时的约 1/7。换句话说，如果能把权重放进 VMEM 而不是 HBM，我们的矩阵乘法就能在更小的批大小下做到"算力受限"。那些天生算术强度就低的算法，也照样能跑得高效。只是 VMEM 实在太小了，要做到这一点往往是个难题。<d-footnote>我们有时会提到"VMEM 预取"，意思是提前把权重搬进 VMEM，从而把矩阵乘法的搬运开销藏起来。比如在一个普通 Transformer 里，我们有时可以在做注意力的时候，顺手把那些很大的前馈权重搬进 VMEM；如果我们正被内存带宽卡着，这样就能把搬权重的耗时掩盖掉。前提是我们的权重足够小、或切分得足够细，能让一层的权重连同余量一起放进 VMEM。</d-footnote>

{% include figure.liquid path="assets/img/tpu-bandwidth.png" class="img-fluid" %}

**一块 TPU 芯片通常（但不总是）包含两个 TPU 核心，它们共享内存，可以当作一台更大的加速器来用**——算力翻倍（这种配置叫"megacore"）。v4、v5、v6 这几代都是这样（TPU v7 取消了 megacore，改为在两个核心之间加一条高带宽链路）。更早的 TPU 芯片各有独立内存，被当作两台独立的加速器（TPU v3 及更早）。面向推理优化的芯片，比如 TPU v5e，每块芯片只有一个 TPU 核心。

{% include figure.liquid path="assets/img/cores.png" class="img-fluid img-small" %}

**芯片**是**每 4 块装在一个"托盘（tray）"上**，通过 **PCIe 网络连到一台 CPU 主机**。这是大多数人熟悉的形式：4 块芯片（在 TPU7x 上其实是 8 个核心，因为它每块芯片有 2 个 TensorCore；不过有些老型号会把这些合并成 4 个逻辑 megacore），通过 Colab 或单台 TPU-VM 暴露给你。对于像 TPU v6e 这样的推理芯片，每台主机配 2 个托盘而不是 1 个，但每块芯片只有 1 个核心，所以是 8 块芯片 = 8 个核心。<d-footnote>在 Cloud TPU VM 上，每个托盘被当作一台独立 VM 的一部分暴露出来，所以又是能看到 4 个核心。</d-footnote>

{% include figure.liquid path="assets/img/pcie.png" class="img-fluid" %}

**PCIe 带宽是有限的：** 就像 HBM $\leftrightarrow$ VMEM 那条链路一样，CPU $\leftrightarrow$ HBM 的 PCIe 连接也有一个固定的带宽上限，限制了"从主机内存往 HBM 搬"或反过来的速度。比如 TPU v4 的 PCIe 带宽是单向 16GB/秒，比 HBM 慢近 100 倍。我们*可以*把数据搬到主机（CPU）内存里，但快不了。

## TPU 的网络互联

**芯片之间通过 ICI 网络连接，组成一个"整机群（Pod）"。** 在 Trillium（TPU v6e）上，ICI（"芯片间互连"）把每块芯片连到它最近的 4 个邻居（并在边缘加上绕回的链路，形成一个 2D 环面网络，也就是"甜甜圈"拓扑），这是面向推理优化芯片的典型做法。而在 TPU7x 上，每块芯片连到最近的 6 个邻居（形成一个 3D 环面网络），为训练任务提供更多 ICI 带宽。注意这些连接**不经过**主机，是芯片之间直接相连。

{% include figure.liquid path="assets/img/ici-wraparound.png" class="img-fluid img-small" %}

这种环面（torus）结构能把任意两个节点之间的最大距离从 $N$ 缩小到 $N / 2$，让通信快得多。TPU 还有一种"扭曲环面（twisted torus）"配置，把环面按莫比乌斯环那样扭一下再连起来，进一步缩短节点之间的平均距离。

**TPU 整机群（用 ICI 连起来）可以做得非常大：** TPU7x 的最大整机群（叫 **superpod**）由 9216 块芯片组成，它是若干个 `4x4x4` 的"立方块"拼起来的<d-footnote>所谓"可重构"，是指可以用一台共享的光路交换机（OCS），把这些立方块动态地拼成任意 3D 拓扑，如 4nx4mx4k。OCS 本质上就是一条可重新配置的连接，ICI 带宽不变。它只是让我们能把立方块连起来，同时保留那条绕回的链路。</d-footnote>，再通过[光纤绕回链路](https://arxiv.org/pdf/2208.10041)相连，我们可以重新配置这些链路，去连成非常巨大的拓扑。

{% include figure.liquid path="assets/img/tpu-rack.png" class="img-fluid" %}

也可以申请更小的拓扑（比如 `2x2x1`、`2x2x2`），但那样就没有绕回链路了。这一点很关键，因为少了绕回链路，多数通信的耗时通常会翻倍。任何"整立方块的倍数"（比如 `4x4x4` 或 `4x4x8`）都会有光交换机提供的绕回链路。<d-footnote>注意，`2x2x4` 不会有绕回链路，因为绕回链路是光交换机提供的，而光交换机只在整立方块上才有。但一台 TPU v6e 的 8x16 *会*在较长的那条轴上有绕回链路，因为它用的是不可重构的光网络。</d-footnote>

{% include figure.liquid path="assets/img/subslices.png" class="img-fluid" %}

TPU v5e 和 Trillium（TPU v6e）的整机群是一个单一的 `16x16` 二维环面，任何长度为 16 的轴都带绕回链路（也就是说 `8x16` 会在较长那条轴上有绕回链路）。这几代的拓扑不能再扩过 16x16 环面，但不同整机群之间还能通过标准的数据中心网络（DCN）通信，DCN 把各台 TPU 主机连起来。同样地，可以申请比 16 更小的拓扑，那些长度 $<16$ 的轴就没有绕回链路。

{% include figure.liquid path="assets/img/more-subslices.png" class="img-fluid" %}

**这种"只连最近邻居"的连法，是 TPU 和 GPU 的一个关键区别。** GPU 用的是层层交换机，近似于让每块 GPU 之间都能点对点直连，而不是像 TPU 那样只连本地邻居。通常，同一个节点内的 GPU（H100 是 8 块，B200 NVL72 多达 72 块）是直接相连的，而更大的拓扑里，任意两块 GPU 之间要走 O(log(N)) 跳。一方面，这意味着 GPU 能在很少几跳内把任意数据送到任意地方。另一方面，TPU 要便宜得多（因为 NVLink 交换机很贵）、连线更简单，而且能扩到长得多的拓扑——因为每台设备的链路数和每台设备的带宽是恒定的。更多见[这里](../gpus#networking)。

**相比 DCN，ICI 很快，但仍比 HBM 带宽慢。** 比如一块 [TPU7x](https://cloud.google.com/tpu/docs/tpu7x#system_architecture) 有：

* 每芯片 `7.4e12` 字节/秒（7380 GB/s）的 HBM 带宽。
* 每芯片 `1.2e12` 字节/秒（1200 GB/s）的双向 ICI 带宽，也就是每个方向轴 `4e11` 字节/秒（400 GB/s）。<d-footnote>TPU 的 ICI 链路带宽会随具体操作略有不同。一般直接用本文的这些数字就够了，不用纠结。</d-footnote>
* 每芯片 `1.25e10` 字节/秒（12.5 GB/s）的 DCN（出网）带宽。

这意味着，当我们把模型切分到多块芯片上时，要小心别让较慢的跨设备通信把 MXU 拖成瓶颈。

**多切片训练：** 一组用 ICI 连起来的 TPU 叫一个 **切片（slice）**。不同的切片之间可以用 DCN 连起来，比如把不同整机群上的切片打通。因为 DCN 比 ICI 慢得多，我们要尽量少让计算去等来自 DCN 的数据。DCN 是主机到主机的，所以要把数据从一块 TPU 经 DCN 传到另一块 TPU，得先经 PCIe 上到主机，再从主机出网，然后从目标主机的网络入网，最后经 PCIe 进入它的 HBM。

## 重点小结

* TPU 很简单，多数情况下你可以把它理解成："一个矩阵乘法单元 + 几种速度不同的连接"：连自己的内存（超快）、连最近的邻居芯片走 ICI（挺快）、连整个数据中心的其余部分走 DCN（有点慢）。

* 通信受各种网络的带宽限制，按速度从快到慢排：
  * HBM 带宽：一个 TensorCore 和它自己的 HBM 之间。
  * ICI 带宽：一块 TPU 芯片和它最近的 4 个或 6 个邻居之间。
  * PCIe 带宽：一台 CPU 主机和它挂的那些托盘之间。
  * DCN 带宽：多台 CPU 主机之间（通常是没有用 ICI 连起来的主机）。

* **在一个切片内部，TPU 只通过 ICI 连着最近的邻居。** 这意味着切片里相距很远的两块芯片之间通信，要先把数据一跳跳接力传过去。

* **权重矩阵的每条边都要补齐到能被 256 整除**（TPU v6e 之前的老代是补齐到 128），这样才能填满 MXU（实际上，比较短的那条边会被补齐到 256）。

* **更低精度的矩阵乘法往往更快。** TPU v6e 做 int8/int4 矩阵乘法比 bfloat16 快 2 倍/4 倍；TPU7x 做 fp8 矩阵乘法的算力是 bfloat16 的 2 倍。VPU 运算仍然是用 fp32 或 bf16 做的（不过 TPU v6e 之前的老代，VPU 只支持 fp32）。

* 要避免把 TPU 的计算单元拖成瓶颈，我们必须**保证走每条通道的通信量，和那条通道的速度成比例**。

### TPU 规格参数

下面是一些具体数字：

| 型号                                      | 整机群(Pod)规模 | 单机规模 | 每芯片 HBM 容量 | 每芯片 HBM 带宽（字节/秒） | 每芯片 bf16 FLOPs/秒 | 每芯片 int8 FLOPs/秒 |
| :----------------------------------------- | :------: | :-------: | :---------------: | :-------------------: | :-----------------: | :-----------------: |
| <span class="nowrap-header">TPU v3</span>  |  32x32   |    4x2    |       32GiB       |        9.0e11         |       1.4e14        |       1.4e14        |
| <span class="nowrap-header">TPU v4p</span> | 16x16x16 |   2x2x1   |       32GiB       |        1.2e12         |       2.75e14       |       2.75e14       |
| <span class="nowrap-header">TPU v5p</span> | 16x20x28 |   2x2x1   |       96GiB       |        2.8e12         |       4.59e14       |       9.18e14       |
| <span class="nowrap-header">TPU v5e</span> |  16x16   |    4x2    |       16GiB       |        8.2e11         |       1.97e14       |       3.94e14       |
| <span class="nowrap-header">TPU v6e</span> |  16x16   |    4x2    |       32GiB       |        1.6e12         |       9.20e14       |       1.84e15       |
| <span class="nowrap-header">TPU7x</span>   | 4x4x576  |   2x2x1   |      192GiB       |        7.4e12         |       2.31e15       |       4.61e15 (fp8) |

"单机规模"指的是一台主机上连着的 TPU 拓扑（比如 TPU v5e 是一台 CPU 主机连 8 块 TPU，排成 4x2）。关于最新一代的更多细节，见 [TPU7x 文档](https://docs.cloud.google.com/tpu/docs/tpu7x)。下面是各种互连带宽的数字：

| 型号       | 每条 ICI 链路单向带宽（字节/秒） | 每条 ICI 链路双向带宽（字节/秒） |
| :---------- | :----------------------------: | :-------------------------: |
| **TPU v3**  |             1.0e11             |           2.0e11            |
| **TPU v4p** |             4.5e10             |           9.0e10            |
| **TPU v5p** |             9.0e10             |           1.8e11            |
| **TPU v5e** |             4.5e10             |           9.0e10            |
| **TPU v6e** |             9.0e10             |           1.8e11            |
| **TPU7x**   |             9.0e10             |           1.8e11            |

我们同时列出单向带宽和双向带宽，因为单向带宽更贴近硬件的真实值，而双向带宽在涉及"整条环路"的公式里更常出现。<d-footnote>这里说的双向（bidi, bidirectional）带宽，指的是一条链路两个方向能一共送出多少字节；等价地，在一台 TPU 上、沿某条轴，假设两条链路都用满，它一共能往外发多少字节。当这条轴形成了一个真正工作的环（也就是这条轴上有绕回连接）时，就可以这么算。在推理芯片上，这要求那条轴是一条完整的 16；在训练芯片（v*p）上，这要求那条轴是 4 的倍数。我们更爱用双向带宽，因为它在涉及双向通信的计算里经常出现。</d-footnote>

PCIe 带宽通常约为每块 TPU `1.6e10` 字节/秒（TPU v6e 是 `3.2e10`），而 DCN 带宽通常约为每块 TPU `6.25e9` 字节/秒（TPU v6e 和 TPU7x 是 `12.5e9`，TPU v5e 是 `3.125e9`）。

## 动手解题

这些数字有点枯燥，但它们能让你对模型性能做出基本的 roofline 估算。我们来算几道题，你就明白这有多有用了。第 3 部分里会有更多例子。

**问题 1 [估算大模型推理的延迟下限]：** 假设你想从一个用 bf16 表示、有 200B（2000 亿）参数、切分到 32 块 TPU v6e 上的模型里采样（生成）。把所有参数从 HBM 读进脉动阵列，要多久？*提示：用上面的数字。*

{% details 点此查看答案。 %}

**答案：** 我们要在 32 块芯片上读入 `sizeof(bf16) * 200e9 = 400e9` 字节，也就是每块 12.5e9 字节，每块 HBM 带宽是 1.6e12。所以读取大约要 7.8ms。

这挺有意思，因为*这就是模型一次采样延迟的一个合理下限*。每个采样步骤都要把所有参数从 HBM 读一遍，所以不可能快过 7.8ms。实际上，在小批大小下，这个下限是很接近能达到的。

{% enddetails %}

**问题 2 [TPU 细节]：** 考虑一整个 TPU v5e 整机群。一共有多少台 CPU 主机？多少个 TPU TensorCore？整个整机群的总 FLOPs/秒是多少？总 HBM 是多少？再对一整个 TPU v5p 整机群做同样的计算。

{% details 点此查看答案。 %}

**答案：** 对 TPU v5e，每个整机群是 `16x16`，每台主机是一个 4x2 的切片，所以主机数是 `16*16 / 8 = 32`。TPU v5e 每块 TPU 只有一个核心，所以有 256 个 TensorCore。总 FLOPs/秒是 `16*16*2e14 = 5.1e16`（bfloat16）。每块芯片有 16GiB HBM，所以是 `256 * 16 = 4TiB` 内存。

对一整个 TPU v5p 整机群，我们有 `16x20x28` 块芯片，每台主机是 2x2x1，所以主机数是 `(16*20*28) / (2*2) = 2,240`。TPU v5p 每块 TPU 有两个 TensorCore，所以有 `8960 * 2 = 17,920` 个核心。总 FLOPs/秒是 `8960 * 4.59e14 = 4.1e18`（bfloat16）。每块芯片有 96GiB HBM，所以是 `8960 * 96 = 840TiB` 内存。

{% enddetails %}

**问题 3 [PCIe 的操作强度]：** 假设我们被迫把一个很大的权重矩阵 $A$（类型 $\text{bf16}[D, F]$）和一批激活 $x$（类型 $\text{bf16}[B, D]$）存在主机内存（DRAM）里，然后想对它们做一次矩阵乘法。这一切跑在一台主机上，只用其中一块 TPU v6e。你可以假设 $B \ll D$、$F = 4D$（后面几章会说明这些假设为什么合理）。要想在 PCIe 上保持"算力受限"，我们需要的最小批大小 $B$ 是多少？假设 PCIe 带宽为 1.6e10 字节/秒。

{% details 点此查看答案。 %}

**答案：** 我们要做 $2BDF$ 次浮点运算，每块芯片每秒能做 `9.2e14` 次，所以这需要 $2BDF / 9.2e14$ 秒。我们要从 DRAM 读入 $2DF + 2BD$ 字节，并把 $2BF$ 字节写回去。瓶颈在 PCIe 传输速度，所以数据进出这块 TPU 一共需要 $2 \cdot (BD + DF + BF) / 1.6e10$ 秒。我们希望"计算"比"搬权重"花的时间更长（假设我们能把所有搬权重都藏在计算背后），也就是要 $2BDF / 9.2e14 > 2 \cdot (BD + DF + BF) / 1.6e10$。用我们的假设 $B \ll D$、$F = 4D$ 化简，得到

$$\frac{8BD^2}{9.2 \times 10^{14}} > \frac{8D^2}{1.6 \times 10^{10}}$$

也就是

$$B > \frac{9.2 \times 10^{14}}{1.6 \times 10^{10}} \simeq 57{,}500$$

{% enddetails %}

**问题 4 [一般矩阵乘法的耗时]：** 假设我们想拿一个权重矩阵 int8[16384, 4096] 去乘一个大小未知（批大小为 B）的激活矩阵 int8[B, 4096]。一开始我们只用 1 块 TPU v5e。

1. 这次乘法要花多久（用 B 表示）？*提示：分别算一下"从 HBM 读数组要多久"和"真正做乘法要多久"，看谁在卡你。*
2. 如果我们改成从 VMEM 来做这个运算呢？要花多久（用 B 表示）？

{% details 点此查看答案。 %}

**答案：**（1）要做的运算量是 $2 \cdot 4096 \cdot 16384 \cdot B = 1.3 \times 10^{8} \cdot B$。所以 $T_{\text{math}} = (1.3 \times 10^{8} \cdot B) / 3.94 \times 10^{14}$ 秒。我们要从 HBM 读进 VMEM $16384 \cdot 4096 + 4096 \cdot B$ 字节，并把 $16384 \cdot B$ 字节从 VMEM 写回 HBM。这意味着 $T_{\text{comms}} = (6.7 \times 10^{7} + 2 \times 10^{4} \cdot B) / 8.2 \times 10^{11}$ 秒。假设能最大程度地让通信和计算重叠，那么整个乘法大约要花

$$\max\{T_{\text{math}}, T_{\text{comms}}\} = \max\left\{\frac{1.3 \times 10^{8} \cdot B}{3.94 \times 10^{14}}, \frac{6.7 \times 10^{7} + 2 \times 10^{4} \cdot B}{8.2 \times 10^{11}}\right\}$$

当 $\frac{6.7 \times 10^{7} + 2 \times 10^{4} \cdot B}{8.2 \times 10^{11}} < \frac{1.3 \times 10^{8} \cdot B}{3.94 \times 10^{14}}$ 时，我们是"算力受限"的，等价于 $B > 267$。这比我们在[第 1 节](../roofline)算出的 240 略大，因为我们这次把 $D$ 和 $F$ 的完整影响都算进去了。

（2）如果改成从 VMEM 读，我们把 VMEM 到 MXU 的带宽看成 HBM $\leftrightarrow$ VMEM 带宽的 22 倍。这会把读数据的分母从 8.2e11 变成 1.80e13，于是得到 $B > 11$。注意实际上我们不可能把全部 VMEM 带宽都用来到权重，所以实际会更接近 20。

{% enddetails %}

**问题 5 [ICI 带宽]：** 假设我们有一个 TPU v5e 的 `4x4` 切片。想把一个类型 `bf16[8, 128, 8192]` 的数组从 `TPU{0,0}` 发到 `TPU{3, 3}`。假设 TPU v5e 每跳延迟是 $1\mu s$。

1. 第一个字节最快多久能到目的地？
2. 整个传输总共要花多久？

{% details 点此查看答案。 %}

**答案：** TPU v5e 是二维连通。因为我们只有 `4x4` 切片（没有任何一条轴长度达到 16），所以没有绕回链路。因此目标芯片有两个端口能收数据，源芯片也有两个端口能发数据。要传的数据量是 `2 * 8 * 128 * 8192 = 1.7e7` 字节。我们可以两个端口同时发（也就是一半往右、一半往下），于是每秒能传 `2 * 4.5e10 = 9e10` 字节，也就是传完整个数组大约要 `1.7e7 / 9e10 = 188us`（假设是带宽受限）。在 `4x4` 切片里，芯片 $(0, 0)$ 和 $(3, 3)$ 之间要走六跳，因为长度不足 16 的轴没有绕回链路。每跳延迟约 $1\mu s$，所以第一个字节大约 `6us` 到，整个传输大约 `188 + 6 = 194us`——因为最后一个字节发出后，同样要再走完那六跳（一般延迟和带宽这两项是相加的，只是这里延迟只是个很小的修正）。

{% enddetails %}

**问题 6 [综合题，较难]：** 假设你有一个大矩阵 **A**：`int8[128 * 1024, 128 * 1024]`，均匀切分在一个 TPU v5e 4x4 切片上，但被卸载到了每块芯片的主机内存（DRAM）里。假设你想把整个数组拷贝到 TPU{0, 0}，然后拿它去乘一个向量 `bf16[8, 128 * 1024]`。这要花多久？*提示：用上面的数字。*

{% details 点此查看答案。 %}

**答案：** 我们先梳理一下要做哪些操作。我们的数组大约 16GB。从上面的表看，TPU v5e 一台主机是 4x2 拓扑，所以一个 4x4 有 2 台主机。既然数组是均匀切分的，每台主机上实际存着整个数组 1/2 的一块，也就是 8GB。我们要把这些块全都拷到 TPU{0,0}，这有两个办法：

1. 先经 DCN 传过去，再把整个不切分的数组经 PCIe 读进 HBM。
2. 先把各自切分的数组读到对应的 TPU 上，再经 ICI 做一次收集（gather），然后在 TPU{0,0} 上做矩阵乘法。

显然方案 (2) 更好。相比 ICI，DCN 太慢；而且我们更愿意把一大堆数据分摊到许多条 PCIe 链接上读，而不是只走少数几条（主机 0 上的那 8 条）。下面是这套系统的一部分示意图。如前所述，TPU 之间靠 ICI 连邻居（跨主机也是），所有 TPU 都经 PCIe 连到自己的主机 CPU，而主机之间靠 DCN 相连。

{% include figure.liquid path="assets/img/challenge-problem.png" class="img-fluid img-small" caption="每块芯片其实都有各自一条到主机的 PCIe 链路，只是这里为了清晰只画了一条。" %}

现在我们逐块算每段要花多久：

1. **PCIe 读取**：我们要经 16 条 PCIe 链路读入各 16GB 的块，每条带宽 `1.6e10` 字节/秒。所以大约要 63ms。

2. **ICI 拷贝**：现在每块 TPU 有 16GB / 16 = 1GB 的我们的数组。ICI 带宽是每条链路*双向* 9e10 字节/秒，而且从上面的图能看出，在这个拓扑里对 TPU{0,0} 来说，TPU v5e 的 4 条 ICI 链路中只有 2 条真正在用。由于 TPU{0,0} 要从两条轴上共收 15GB，每条轴 `4.5e10` 字节/秒，所以耗时下限是 `15e9 / (4.5e10 * 2) = 167ms`。实际上这个下限大概率达不到，因为负载很不均，但真实值大概在这个的 2 倍以内。你在第 3 节会看到，做一次完整的 AllGather 也大约要 `16e9 / (4.5e10 * 2)`，所以这已经接近最优了。

3. **HBM $\rightarrow$ MXU 读取**：要做最后那次矩阵乘法，我们得把这 16e9 字节加上那个 bf16[8, 128 \* 1024] 数组（还有 2MB，可忽略）经 HBM 带宽读进 MXU，这将花 `16e9 / 8.2e11 = 20ms`。

4. **FLOPs**：我们总共要做 $$2 \cdot 8 \cdot 128 \cdot 1024 \cdot 128 \cdot 1024 = 2.7 \times 10^{11}$$ 次 FLOPs，而我们能做 `1.97e14` bf16 FLOPs/秒，于是得到 1.4ms。

总时间的上限是这些时间之和；但因为 TPU 通常能让这些操作互相重叠，我们可以把它当成一个"被最慢环节卡住"的流水线问题。假设如此，那么答案至少是 167ms，考虑到重叠不完美，很可能接近 200ms。

{% enddetails %}

<h3 markdown=1 class="next-section">第 2 部分到此结束！第 3 部分会讲切分（partitioning）和跨 TPU 通信，[点击这里](../sharding)。</h3>

## 附录

### 附录 A：更多 TPU 内部细节 {#appendix-a-more-on-tpu-internals}

这里我们更深入地看看 TPU 的内部运作。除非特别说明，给出的规格都是 TPU7x 的。

### VPU（向量处理单元）

VPU 是 TPU 做向量运算的核心。它包含一台二维的 SIMD 向量机（也就是 **VPU**），负责做逐元素的算术运算，比如 vadd（向量加法）、vmax（逐元素取最大）；还有一组叫 **VREG**（向量寄存器）的寄存器，给 VPU 和 MXU 暂存数据。

**VREG：** TPU7x 每个核心有 64 个 32 位的 VREG（TPU v4 是 32 个），所以每核心大约 `64 * 8 * 128 * 4 = 256kB` 的 VREG 内存（整块芯片是它的 2 倍，因为有两个核心）。TPU7x 每个周期可以从 VMEM 读 3 个寄存器、往 VMEM 写 2 个寄存器，但总共读写不超过 4 个，也就是要么"3 读 1 写"，要么"2 读 2 写"。

对以 float32 输入为准的程序，一个 VREG 里的每个 32 位字正好放一个 float32 元素。用更低精度的数据类型时，可以把多个元素"打包"塞进同一个 32 位字里，更省 VMEM 和 VREG；有些运算本来就要求操作数这样打包好。

**VPU：** VPU 是一个形状为 `(8, 128)` 的二维向量运算单元，其中长度为 128 的那一维叫"通道（lane）轴"，长度为 8 的那一维叫"子通道（sublane）轴"。在 TPU7x 上，每个（lane, sublane）位置包含 4 个彼此独立的标准浮点 ALU。VPU 的绝大多数算术指令（比如 vadd 向量加）在每个 ALU 里一个周期就能执行、延迟为 2 个周期，所以在 TPU7x 上，每个周期你可以从 VREG 里把 4 对 f32 值相加。一条典型的 VPU 指令长这样：`{v2 = vadd.8x128.f32 v0, v1}`，其中 v0、v1 是输入的 VREG，v2 是输出的 VREG。所有代 TPU 的 ALU 也都支持一组逐元素的整数、按位、逻辑运算；TPU v6e 和 TPU7x 还能对打包好的 bfloat16 VREG 做逐元素算术运算。

所有 lane 和 sublane 每个周期都执行同一个程序（纯 SIMD），但每个 ALU 可以做不同的运算。所以比如我们可以一个周期里同时做 1 次 vadd 和 1 次 vsub，每个都读满两个 VREG、把结果写到第三个。

**小测验 [计算 VPU 吞吐]：** 用上文信息，算算一块 TPU7x 能做多少向量 FLOPs/秒。TPU7x 主频约 2.2GHz。

{% details 点此查看答案。 %}

*答案*：每个周期，每个核心能在 `8 * 128` 个 ALU 上执行 4 条向量指令。也就是每核心 `8 * 128 * 4` FLOPs/周期，即 `8 * 128 * 4 * 2.2e9 = 9e12 FLOPs/s`（如果是 bfloat16，则是 1.8e13 bfloat16 FLOPs/s，因为打包好的 bfloat16 VREG 装的下元素是 float32 的两倍）。注意这比 MXU 的算力（每核心约 `1.2e15`）小得多（bfloat16 下大约差 64 倍）。

{% enddetails %}

**归约（Reductions）：** 一般来说，在子通道（sublane）维度上通信或归约，比在通道（lane）维度上容易。比如，VPU 支持一种"lane 内洗牌（shuffle）"操作，能在大约一个周期里沿着长度为 8 的那条轴滚动。可以用来高效地沿子通道维度做归约（只要依次滚 4、2、1，做 3 轮逐元素相加即可）。

跨通道的归约难得多，要动用一块叫 XLU（"跨通道单元"）的独立硬件，它又慢又贵。

**和 GPU 的对比：** 对熟悉 NVIDIA GPU 的读者：VPU 里的每个 ALU 相当于一个 CUDA core，一个 VPU lane 相当于一个"Warp 调度器"，也就是通常 32 个 CUDA core 一起做 SIMD 算术的那一组。在 lane 内部做归约很容易，可一旦要跨 lane，就得走 VMEM/XLU/SMEM，慢得多。更多见 [GPU 一节](../gpus)。

### 标量核心（Scalar Core）

标量核心是 TPU 的"指挥"。它负责取指、派发所有指令、执行"把数据从 HBM 搬进 VMEM"这类搬运，还可以被用来做标量类的辅助工作。因为标量核心是单线程的，一个副作用是：TPU 的每个核心每周期只能发起一次 DMA（直接内存搬运）请求。

给你一个体感：一个标量核心要指挥一台 VPU（包含 4096 个 ALU）、2 个 MXU、2 个 XLU 和多个 DMA 引擎。这种"一份控制对应海量计算"的悬殊比例，是硬件高效率的来源，但也限制了"以数据为条件、做有意思的向量化"的能力。

### 附录 B：脉动阵列是怎么工作的？ {#appendix-b-how-does-a-systolic-array-work}

TPU 的 MXU 核心是一个 `256x256` 的脉动阵列（TPU v6e 之前是 `128x128`）。当它被喂满时，每 4 个时钟周期可以做一次 `bf16[8,256] @ bf16[256,256] -> f32[8,256]`<d-footnote>如果你不熟悉这种记号，它的意思是：把一个 `8x256`、元素为 bfloat16 的矩阵，乘以一个 `256x256`、元素为 bfloat16 的矩阵，并把结果存进一个 `8x256`、元素为 float32 的矩阵。</d-footnote>的乘法（TPUv6e 之前是 8 个周期做一次）<d-footnote>MXU 每 4 个周期可以载入 8 行 `f32` 输入并把它们转成 `bf16`；或者每 8 个周期载入 16 行 `bf16` 输入。无论哪种，都相当于每 4 个周期载入 8 行 `bf16` 输入。</d-footnote>。

* 说到底，脉动阵列就是一个 `256x256`（`=65,536`）的二维 ALU 网格，每个 ALU 都能做一次"乘加"运算。
* 权重（**W**，那个 `256x256` 输入）从上方往下传（称为 RHS，右边操作数），输入（**X**，那个 `8x256` 输入）从左方传入（称为 LHS，左边操作数）。

下面是一段简化的动画，演示把一组权重（蓝色）和一组激活（绿色）相乘。你会注意到，权重（RHS）先斜着被部分装入，然后激活也斜着喂进来。在下面每一帧里，我们把所有重叠的绿色单元和蓝色单元相乘，把结果和从上方传下来的"累加残值"相加，然后再把结果往下传一格。

{% include figure.liquid path="assets/img/systolic-array.gif" %}

下面是这段动画更通用的一版，展示结果是如何从计算中被"流水线式"送出来的：

{% include figure.liquid path="assets/img/systolic-array2.gif" class="img-small" %}

下面这张图展示了怎样把多个 RHS 和 LHS 数组流水线地喂进来：

{% include figure.liquid path="assets/img/systolic-array-pipelining.png" class="img-fluid" %}

一开始会有一个"流水线气泡"（空档），那是权重（RHS）和激活（LHS）在往里装载造成的。挺过这个初始空档后，新的输入和权重就能不断喂进来，不再产生额外空档。

下面这段是一个 bf16[2, 3] x bf16[3, 3] 矩阵乘法的、做得不太好的动画，你可以把它想象成：拿一个 2x3 的权重矩阵，去乘一个批大小为 1、长度为 3 的激活。这张图相对前面的幻灯片旋转了方向，输入是往右边流出去而不是往下，但你大致能看出那个结构。

{% include figure.liquid path="assets/img/systolic-array-bad.gif" class="img-small" %}

我们可以高效地把它流水线化，去乘大矩阵，而不会产生太大的流水线空档。话虽如此，重要的是我们的矩阵形状要大于 MXU 的边长（即 128x128 或 256x256）。有些 TPU（从 TPU v3 起）有多个 MXU：TPU v3、TPU v6e、TPU7x 各有 2 个，TPU v4/5 有 4 个。所以我们要保证分块（tile）的尺寸大于 256（老代是 128）再乘以 MXU 个数。[这里](https://www.youtube.com/watch?v=sJltBQ4MOHA)有一段不错的动画。

[这篇博客](https://fleetwood.dev/posts/domain-specific-architectures#google-tpu)还有一段很棒的动画，演示固定权重矩阵下的脉动阵列乘法。
