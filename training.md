---
layout: distill
title: "怎么把 Transformer 并行开来做训练"
# permalink: /main/
description: "这一节讲 LLM 训练中最常用的四种并行方案：数据并行、全分片数据并行（FSDP）、张量并行和流水线并行。对每一种，我们都会算出它在什么情况下开始被通信卡住脖子。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 5

previous_section_url: "../transformers"
previous_section_name: "第 4 部分 · Transformer 的数学"

next_section_url: ../applied-training
next_section_name: "第 6 部分 · 训练 LLaMA"

bibliography: main.bib

giscus_comments: true
giscus_discussion: 7

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
  - name: "“扩大规模”到底指什么？"
  - subsections:
    - name: "数据并行"
    - name: "全分片数据并行（FSDP）"
    - name: "张量并行"
    - name: "把 FSDP 和张量并行组合起来"
    - name: "流水线并行"
    - name: "跨整机群（Pod）扩大规模"
  - name: "TPU 上训练 LLM 的重点小结"
  - name: "动手解题"
  - name: "附录"
  - subsections:
    - name: "附录 A：反向传播的通信是怎么推出来的"

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

## “扩大规模”到底指什么？

“模型扩大规模”的目标，是在训练或推理中越用越多的芯片时，吞吐量也成比例地线性上涨（这叫*强扩展*）。单块芯片上的性能，取决于内存带宽和 FLOPs 的权衡；而整个集群的性能，取决于能否把芯片之间的通信“藏”进有用的计算里。这件事并不容易：芯片越多，通信负担越大，而每台设备上能拿来“遮掩”通信的计算反而越少。正如[第 3 部分](../sharding)看到的，分片的矩阵乘法常常需要昂贵的 AllGather 或 ReduceScatter，会把 TPU 挡在“没活可干”的状态。这一节的目标，就是搞清楚这些通信什么时候会*贵得离谱*。

本节会讨论四种常见的并行方案：（纯）**数据并行、全分片数据并行**（FSDP / ZeRO 分片）、**张量并行**（也叫模型并行），以及（简略介绍的）**流水线并行**。每一种我们都会算清楚：要付多少通信代价，这个代价什么时候开始卡住计算。<d-footnote>我们只关注通信上限——内存容量约束当然重要，但在预训练里配上重算（激活检查点）和海量芯片后，容量通常不会卡住我们。这里也不讨论 MoE 的专家并行——那会大大扩展设计空间——只讨论稠密 Transformer 的基本情形。</d-footnote>本节可以只看芯片之间的通信开销：只要单芯片的批足够大，“HBM 到 MXU”的数据搬运早已被计算盖住了。

为简化计算，全节使用下面的记号。

| 记号 | 含义（模型参数）                                             |
| :------- | :--------------------------------------------------------------------- |
| D        | **d**<sub>model</sub>（隐层维度/残差流的宽度）      |
| F        | **d**<sub>ff</sub>（前馈网络的维度）                        |
| B        | 批维度（一个批次里的 token 总数；是全局的，不是单设备的） |
| T        | 序列长度                                                        |
| L        | 模型的层数                                          |

| 记号 | 含义（硬件特性）                                                                 |
| :------- | :------------------------------------------------------------------------------------------------ |
| C        | 每芯片的 FLOPS/s                                                                                  |
| W        | 网络带宽（双向，常加下标区分，如 $W_{\text{ici}}$ 或 $W_{\text{dcn}}$） |
| X        | 沿网格轴 X 的芯片数                                                                 |
| Y        | 沿另一条网格轴 Y 的芯片数                                           |
| Z        | 沿第三条网格轴 Z 的芯片数                                                |

为了简单起见，**我们把 Transformer 近似成一摞 MLP 块**——如[第 4 部分](../transformers)所见，对稍大一点的模型，注意力只占 FLOPs 里很小的一块。我们还忽略门控矩阵乘。于是每一层就是下面这个简单结构：

{% include figure.liquid path="assets/img/transformer-layer.png" class="img-fluid" caption="<b>图：</b>一个简化的 Transformer 层。我们把每个 FFW 块看成两个矩阵叠在一起：<b>W<sub>in</sub></b>: <code>bf16[D, F]</code>（上投影）和 <b>W<sub>out</sub></b>: <code>bf16[F, D]</code>（下投影），输入是 <b>In</b>: <code>bf16[B, D]</code>。" %}

{% details 这里先给出完全不并行的"小 Transformer"的完整算法。 %}

<div markdown=1 class="algorithm">

**前向传播：** 需要计算 Loss[B]

1.  Tmp[B, F] = In[B, D] *<sub>D</sub> W<sub>in</sub>[D, F]
2.  Out[B, D] = Tmp[B, F] *<sub>F</sub> W<sub>out</sub>[F, D]
3.  Loss[B] = ...

**反向传播：** 需要计算 dW<sub>out</sub>[F, D], dW<sub>in</sub>[D, F]

1.  dOut[B, D] = ...
2.  dW<sub>out</sub>[F, D] = Tmp[B, F] *<sub>B</sub> dOut[B, D]
3.  dTmp[B, F] = dOut[B, D] *<sub>D</sub> W<sub>out</sub>[F, D]
4.  dW<sub>in</sub>[D, F] = In[B, D] *<sub>B</sub> dTmp[B, F]
5.  dIn[B, D] = dTmp[B, F] \*<sub>F</sub> W<sub>in</sub>[D, F]（*给更前面的层用*）

</div>

放在一起，方便和后面“加了通信”的算法做对比。

{% enddetails %}

下面是我们要讲的 4 种并行方案。每种方案都可以由“**In**、**W<sub>in</sub>、W<sub>out</sub> 和 Out** 在上图中的分片方式”唯一确定。

**1. 数据并行：** *激活沿批维分片，参数和优化器状态在每台设备上完整复制。只有反向传播才有通信。*

$$\text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D]$$

**2. 全分片数据并行（FSDP，即 ZeRO-3）：** *激活沿批维分片（同纯数据并行），参数沿同一条网格轴分片，并在前向传播用到之前“现用现取”地 AllGather 起来。优化器状态也沿批维分片。大幅减少重复占用的内存。*

$$\text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D]$$

**3. 张量并行（也叫 Megatron 分片或模型并行）：** *激活沿 D（$d_\text{model}$）分片，参数沿 F（$d_{ff}$）分片。每个块前后分别对激活做 AllGather 和 ReduceScatter。可与 FSDP 组合。*

$$\text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y]$$

**4. 流水线并行：** *权重沿层维分片，激活按微批沿层维往前“滚动”。流水线各阶段之间通信极少（只是把激活传一跳）。借用一下记号：*

$$\text{In}[L_Z, B, D][i] \cdot_D W_\text{in}[L_Z, D, F][i] \cdot_F W_\text{out}[L_Z, F, D][i] \rightarrow \text{Out}[L_Z, B, D][i]$$

### 数据并行

**记号：** $$\text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D]$$

只要模型哪怕用很小的批（大于 240 个 token 即可计算受限）就能塞进单块芯片，**你就应该用简单数据并行**。纯数据并行可以把激活分到任意多的 TPU 上——只要芯片数不超过批大小。前向传播完全零通信，但每一步结束时，**每台 TPU 对自己的本地梯度做一次 AllReduce，把梯度对齐后再更新参数。**

{% include figure.liquid path="assets/img/data-parallelism.png" class="img-fluid" caption="<b>图：</b>纯数据并行（前向传播）的示意图。激活（左）沿批维彻底分片，权重完全复制，每块 TPU 拿到一模一样的权重。这样做的代价是权重总内存被放大 N 倍，但前向传播不需要任何通信。" %}

{% details 这是前向和反向的完整算法。为了简短，我们把 dL/dOut 简写成 dOut。 %}

<div markdown=1 class="algorithm">

**纯数据并行算法：**

**前向传播：** 需要计算 Loss[B<sub>X</sub>]

1.  Tmp[B<sub>X</sub>, F] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F]
2.  Out[B<sub>X</sub>, D] = Tmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>out</sub>[F, D]
3.  Loss[B<sub>X</sub>] = ...

**反向传播：** 需要计算 dW<sub>out</sub>[F, D], dW<sub>in</sub>[D, F]

1.  dOut[B<sub>X</sub>, D] = ...
2.  dW<sub>out</sub>[F, D] {U<sub>X</sub>} = Tmp[B<sub>X</sub>, F] \*<sub>B</sub> dOut[B<sub>X</sub>, D]
3.  dW<sub>out</sub>[F, D] = **AllReduce**(dW<sub>out</sub>[F, D] {U<sub>X</sub>})（*不在关键路径上，可以异步做*）
4.  dTmp[B<sub>X</sub>, F] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F, D]
5.  dW<sub>in</sub>[D, F] {U<sub>X</sub>} = In[B<sub>X</sub>, D] \*<sub>B</sub> dTmp[B<sub>X</sub>, F]
6.  dW<sub>in</sub>[D, F] = **AllReduce**(dW<sub>in</sub>[D, F] {U<sub>X</sub>})（*不在关键路径上，可以异步做*）
7.  dIn[B<sub>X</sub>, D] = dTmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>in</sub>[D, F]（*给更前面的层用*）

</div>

我们略去损失函数的细节，并把 $\text{Tmp} = W_\text{in} \cdot \text{In}$ 当作简写。注意：虽然最终的损失应该是 **AllReduce**(Loss[B<sub>X</sub>]) 的平均，但我们只需在反向传播平均权重梯度时做这次 AllReduce。

{% enddetails %}

注意前向传播没有任何通信——**通信全在反向**！而且反向还有个极好的性质：AllReduce 不在“关键路径”上。意思是每次 AllReduce 可以找空闲的时候做，不会挡住后面的计算。通信总量*仍可能*超过计算时间而卡住我们，但从实现角度看要宽松得多。后面会看到，模型/张量并行就没有这个好脾气。

**为什么用它？** 纯数据并行把激活沿批维切开，大幅降低激活的内存压力；只要芯片够多，批大小几乎可以无限往上加。训练时激活往往是大头，这一点特别有用。

**为什么不够用？** 纯数据并行对参数和优化器状态的内存压力毫无办法。这意味着对真正有规模、参数+优化器状态装不进一块 TPU 的模型，纯数据并行几乎没用。给个体感：用 Adam 训练、参数 bf16、优化器状态 fp32<d-footnote>Adam 要存参数、一阶矩和二阶矩累加器。参数是 bfloat16、优化器状态是 float32，所以每个参数要 `2 + 8 = 10` 字节。</d-footnote>，能装下的最大模型是 $$\text{TPU memory} / 10$$ 个参数（TPU 内存除以十）。比如在一块 96GB HBM 的 TPUv5p 上，用纯数据并行大约只能装 9B 参数。

<p markdown=1 class="takeaway">**要点**：用 Adam + 纯数据并行能训练的最大模型是 $$\text{num_params} = \text{HBM per device} / 10$$（每台设备的 HBM 除以 10）。对 TPU v5p 大约是 9B 参数。<d-footnote>这还没算梯度检查点的内存，所以其实根本不够用。这是"批只有 1 个 token"时的绝对下限。</d-footnote></p>

*要让它对真实模型有用，至少得把参数或优化器状态分片掉一部分。*

**什么时候会被通信卡住？** 如上所述，每层有两次 AllReduce，各涉及 $$2DF$$ 字节（bf16 权重）。数据并行什么时候变成通信受限？

沿用上面的记号：$C$ = 每芯片 FLOPs，$W_{\text{ici}}$ = **双向**网络带宽，$X$ = 批被切成的份数<d-footnote>我们假设这个切分在 ICI 网格内完成，所以相关带宽是 $W_\text{ici}$。</d-footnote>。分别算出矩阵乘法需要的时间 $$T_\text{math}$$ 和通信需要的时间 $$T_\text{comms}$$。由于这个方案前向无通信，只需要算反向的这两个量。

*通信时间：* 前面讲过，一维网格上做一次 AllReduce 只取决于数组总字节数和 ICI 带宽 $W_\text{ici}$，时间是 $2 \cdot \text{总字节} / W_\text{ici}$。$W_\text{in}$ 和 $W_\text{out}$ 各要一次，所以每层两次 AllReduce。每次的对象是一个权重矩阵，即 $DF$ 个参数、$2DF$ 字节。合起来，单层一次 AllReduce 的总时间是

$$\begin{align}
T_\text{comms} &= \frac{2 \cdot 2 \cdot 2 \cdot D \cdot F}{W_\text{ici}}. \\
\end{align}$$

*矩阵乘时间：* 每层前向有两次矩阵乘、反向有四次，每次需要 $2(B/X)DF$ FLOPs。于是反向的单层计算时间是

$$\begin{align}
T_\text{math} &= \frac{2 \cdot 2 \cdot 2 \cdot B \cdot D \cdot F}{X \cdot C} \\
\end{align}$$

两者会重叠，所以单层总时间取二者的较大值：

$$\begin{aligned}
T &\approx \max(\frac{8 \cdot B \cdot D \cdot F}{X \cdot C}, \frac{8 \cdot D \cdot F}{W_\text{ici}}) \\
T &\approx 8 \cdot D \cdot F \cdot \max(\frac{B}{X \cdot C}, \frac{1}{W_\text{ici}})
\end{aligned}$$

当 $$T_\text{math}/T_\text{comms} > 1$$ 时是计算受限，即

$$\begin{align}
\frac{B}{X} > \frac{C}{W_\text{ici}}.
\end{align}$$

结论：数据并行想保持计算受限，每台设备的批大小 $$B / X$$ 必须超过 ICI 的算术强度 $C / W_\text{ici}$。道理在于：计算时间随每设备批大小增长，通信时间却与它无关（搬的是模型权重）。注意 $B/X > C/W_\text{ici}$ 和单芯片计算受限法则 $B > 240$ 长得很像——那条法则同样来自“计算随批大小增长、而（$B \ll F, D$ 时）搬运量与批大小无关”。

代入真实数字感受下量级。TPUv5p 上 `C=4.6e14`，一维 ICI 数据并行的 `W=2 * 9e10`，所以**每台芯片的批至少要 2,550 个 token，否则就被通信卡住**。数据并行可以跨多条轴做：如果把 TPUv5p 的三条轴都拿来做纯数据并行，带宽 $W_\text{ici}$ 翻 3 倍，每台只要 850 个 token，一个整机群（8960 块芯片）每批 760 万个 token！**也就是说：想被纯数据并行卡住脖子，还真不容易！**

<p markdown=1 class="takeaway">**注 [上下文并行]：** 本节里 $B$ 始终指**按 token 计**的总批大小。可我们的批由许多条不同序列组成，这怎么算？对 MLP 来说，**token 就是 token**！管它们属不属于同一条序列。所以我们可以对批维和序列维一起做数据并行：这叫上下文并行或序列并行，你可以把它当成另一种数据并行。注意力比 MLP 麻烦些，因为存在跨序列的计算，但可以在注意力内部收集 KV 或 Q，并仔细重叠计算与通信（典型做法叫"ring attention"）。本节后面会完全忽略序列维，只假设存在某种程度的批或序列并行。</p>

**关于多条网格轴的说明：** 顺便说清多轴如何影响可用带宽。用多条网格轴做同一种并行策略，就能拿到更多带宽。

* **定义：** $M_X$（$M_Y$、$M_Z$ 等）= 某种并行策略横跨的硬件网格轴条数。
* **效果（带宽受限侧）：** 用 $M$ 条轴，聚合链路带宽约变成 $M$ 倍，集合通信耗时按 $\propto 1/M_X$ 缩小。

### 全分片数据并行（FSDP）

**记号：** $$\text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D]$$

全分片数据并行（常叫 FSDP 或 ZeRO 分片<d-cite key="zero"></d-cite>）把优化器状态和权重切到各数据并行分片上，需要时高效地收集（gather）和散开（scatter）。**和纯数据并行相比，FSDP 大幅降低每台设备的内存，还省掉一部分反向 FLOPs，开销却几乎没有。**

{% include figure.liquid path="assets/img/fsdp.png" class="img-fluid" caption="<b>图：</b>FSDP 沿数据维对 Win 的收缩维和 Wout 的输出维做分片。这能省内存，但（由第 3 部分）做矩阵乘前必须先把权重收集起来。注意激活（左）*没有沿收缩维分片*，正是这一点逼着我们去做收集。**另外，权重的优化器状态同样沿收缩维分片。**" %}

你还记得（[第 3 部分](../sharding)）AllReduce 可以拆成 AllGather + ReduceScatter。这意味着：不必像标准数据并行那样做完整的梯度 AllReduce，而可以把权重和优化器状态分片到各芯片上，前向每层用到时再 AllGather、反向再对梯度做 ReduceScatter——总量一分不多花。

{% details 这是 FSDP 的完整算法。 %}

<div markdown=1 class="algorithm">

**全分片数据并行（FSDP）：**

**前向传播：** 需要计算 Loss[B<sub>X</sub>]

1.  W<sub>in</sub>[D, F] = **AllGather**(W<sub>in</sub>[D<sub>X</sub>, F])（*不在关键路径上，可以在上一层时提前做*）
2.  Tmp[B<sub>X</sub>, F] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F]（*现在可以把 W<sub>in</sub>[D, F] 丢掉了*）
3.  W<sub>out</sub>[F, D] = **AllGather**(W<sub>out</sub>[F, D<sub>X</sub>])（*不在关键路径上，可以在上一层时提前做*）
4.  Out[B<sub>X</sub>, D] = Tmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>out</sub>[F, D]
5.  Loss[B<sub>X</sub>] = ...

**反向传播：** 需要计算 dW<sub>out</sub>[F, D<sub>X</sub>], dW<sub>in</sub>[D<sub>X</sub>, F]

1.  dOut[B<sub>X</sub>, D] = ...
2.  dW<sub>out</sub>[F, D] {U<sub>X</sub>} = Tmp[B<sub>X</sub>, F] \*<sub>B</sub> dOut[B<sub>X</sub>, D]
3.  dW<sub>out</sub>[F, D<sub>X</sub>] = **ReduceScatter**(dW<sub>out</sub>[F, D] {U<sub>X</sub>})（*不在关键路径上，可以异步做*）
4.  W<sub>out</sub>[F, D] = **AllGather**(W<sub>out</sub>[F, D<sub>X</sub>])（*可以提前做*）
5.  dTmp[B<sub>X</sub>, F] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F, D]（*这里可以把 W<sub>out</sub>[F, D] 丢掉*）
6.  dW<sub>in</sub>[D,F] {U<sub>X</sub>} = In[B<sub>X</sub>, D] \*<sub>B</sub> dTmp[B<sub>X</sub>, F]
7.  dW<sub>in</sub>[D<sub>X</sub>, F] = **ReduceScatter**(dW<sub>in</sub>[D, F] {U<sub>X</sub>})（*不在关键路径上，可以异步做*）
8.  W<sub>in</sub>[D, F] = **AllGather**(W<sub>in</sub>[D<sub>X</sub>, F])（*可以提前做*）
9.  dIn[B<sub>X</sub>, D] = dTmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>in</sub>[D, F]（*给更前面的层用*）（这里可以把 W<sub>in</sub>[D, F] 丢掉）

</div>

{% enddetails %}

这也叫 “ZeRO Sharding”，来自 “Zero Redundancy Optimizer”（零冗余优化器）——不做任何多余的计算，也不存任何多余的状态。ZeRO-{1,2,3} 分别指按这种方式分片优化器状态、梯度、以及权重。三者的通信代价相同<d-footnote>严格说 FSDP 在前向加了纯 DP 没有的通信，但它和反向成同样的比例，不影响通信 roofline。关键是 ZeRO-3 把反向的一次 AllReduce 换成了 AllGather + ReduceScatter，总通信量完全相同。</d-footnote>，所以我们基本可以放心用 ZeRO-3：把参数、梯度、优化器状态都分片到一组设备上。

**为什么要这么做？** 标准数据并行重复做了大量工作：每台 TPU AllReduce 完整的梯度，然后更新完整的优化器状态（各台做的是一模一样的活），再更新参数（又是完全重复）。而用 ZeRO 分片（把梯度/优化器状态切开）后，不必做 AllReduce，只需对梯度做 ReduceScatter、只更新自己那一份优化器状态、更新那一份参数，需要时再把参数 AllGather 回来供前向使用。

**什么时候会被通信卡住？** 计算和通信的相对代价与纯数据并行完全一样——反向的每次 AllReduce 只是变成了 AllGather + ReduceScatter。回想一下：AllReduce 本来就是由 AllGather 和 ReduceScatter 实现的，各占一半代价。这里我们改用前向来建模（它和反向的“计算/通信比”相同）：

$$\begin{aligned}
T_\text{math} &= \frac{2 \cdot 2 \cdot B \cdot D \cdot F}{X \cdot C} \\
T_\text{comms} &= \frac{2 \cdot 2 \cdot D \cdot F}{W_\text{ici}} \\
T &\approx \max\left(\frac{4 \cdot B \cdot D \cdot F}{X \cdot C}, \frac{4 \cdot D \cdot F}{W_\text{ici}}\right) \\
T &\approx 4 \cdot D \cdot F \cdot \max\left(\frac{B}{X \cdot C}, \frac{1}{W_\text{ici}}\right)
\end{aligned}$$

所以，与纯数据并行一样，当 $$B / X > C / W_\text{ici}$$ 时我们是计算受限的——即每台设备的批大小 $B/X$ 超过“ICI 算术强度” $C/W_\text{ici}$（v5p 上是 `4.59e14 / 1.8e11 = 2550`）。这对我们是大好事：只要每台设备的批大小足以让纯数据并行计算受限，我们就可以*放心地*升级成 FSDP——不用担心掉出计算受限区间——顺手省下巨量的参数和优化器状态内存！虽然前向确实多了通信，但没关系，它会被前向的计算盖住。

<p markdown=1 class="takeaway">**要点：** 在 TPUv5 上，FSDP 和纯数据并行都在"每台设备批大小小于 $2550 / M_X$"时变成带宽受限，$M_X$ 是所用的网格轴条数。</p>

举个例子：DeepSeek-V2（为数不多公开了训练批大小的强模型之一）用的批大小约 4000 万个 token。**这允许我们扩展到约 47,000 块芯片——大约 5 个 TPUv5 整机群——才会撞上带宽上限。**

LLaMA-3 70B 的训练总量约为 `6.3e24（15e12 * 70e9 * 6）` FLOPs。把 1600 万 token 的批分到约 `16e6 / (2550 / 3) = 18,823` 块芯片（大约 2 个 8960 芯片的整机群）上，每块跑 `4.59e14` FLOPs、达到峰值算力的 50%（常称 MFU），**大约 17 天就能训完**。不错！但我们来看看怎么能更好。

<p markdown=1 class="takeaway">**关于临界批大小的说明**：有点反直觉——总批越小（芯片数固定），我们反而越容易被通信卡住。只要批能一直变大，数据并行和 FSDP 就能让我们扩到任意多的芯片！但实际上，批变大后训练收益会递减（梯度几乎不带噪声了），有时还会训练不稳。所以在"算力无限"的世界里找最优分片方案，通常是这样开始的：批大小由缩放定律定死，芯片数量已知（很大），然后去找一种分法，让这份"不大的批"能铺在这么多芯片上。</p>

### 张量并行

**记号：** $$\text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y]$$（用 $$Y$$ 是为了之后和 FSDP 组合）

全分片数据并行的 AllReduce 搬的是*权重*。我们还可以把模型的前馈维切开、让*激活*在层内流动——这叫“一维模型并行”或 Megatron 分片<d-cite key="megatron"></d-cite>。它能把单个整机群的有效批大小再往下压。下图是一个矩阵这样分片的例子：

{% include figure.liquid path="assets/img/model-parallelism.png" class="img-fluid" caption="<b>图：</b>基础张量并行的例子。我们只沿 Y 切激活（不同于 FSDP 的沿 X 切），所以激活沿 X 是复制的。用标准记号写就是 <b>A</b>[B, D<sub>Y</sub>] * <b>B</b>[D, F<sub>Y</sub>] -> <b>C</b>[B, F<sub>Y</sub>]。因为只切了收缩维之一，矩阵乘法前通常要对激活 <b>A</b> 做 AllGather。" %}

如前所述，**In\[B, D<sub>Y</sub>\] \*<sub>D</sub> W<sub>in</sub>\[D, F<sub>Y</sub>\] \*<sub>F</sub> W<sub>out</sub>\[F<sub>Y</sub>, D\] \-\> Out\[B, D<sub>Y</sub>\] 意味着第一次矩阵乘前要把激活收集起来。当激活比权重小的时候，这比 ZeRO 分片更便宜。**而这通常只在搭配了一些 ZeRO 分片（把收集量变小）时才成立。这也是我们倾向于把 ZeRO 分片和张量并行混着用的原因之一。

{% details 这是张量并行的算法！ %}

<div markdown=1 class="algorithm">

**张量并行：**

**前向传播：** 需要计算 Loss[B]

1.  In[B, D] = **AllGather**(In[B, D<sub>Y</sub>])（*在关键路径上*）
2.  Tmp[B, F<sub>Y</sub>] = In[B, D] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>]（*收缩维没切，无需通信*）
3.  Out[B, D] {U<sub>Y</sub>} = Tmp[B, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D]
4.  Out[B, D<sub>Y</sub>] = **ReduceScatter**(Out[B, D] {U<sub>Y</sub>})（*在关键路径上*）
5.  Loss[B] = ...

**反向传播：** 需要计算 dW<sub>out</sub>[F<sub>Y</sub>, D], dW<sub>in</sub>[D, F<sub>Y</sub>]

1.  dOut[B, D<sub>Y</sub>] = ...
2.  dOut[B, D] = **AllGather**(dOut[B, D<sub>Y</sub>])（*在关键路径上*）
3.  dW<sub>out</sub>[F<sub>Y</sub>, D] = Tmp[B, F<sub>Y</sub>] \*<sub>B</sub> dOut[B, D]
4.  dTmp[B, F<sub>Y</sub>] = dOut[B, D] \*<sub>D</sub> W<sub>out</sub>[F<sub>Y</sub>, D]（*这里可以把 dOut[B, D] 丢掉*）
5.  In[B, D] = **AllGather**(In[B, D<sub>Y</sub>])（*这一步可以和前向的 (1) 共享，省掉*）
6.  dW<sub>in</sub>[D, F<sub>Y</sub>] = In[B, D] \*<sub>B</sub> dTmp[B, F<sub>Y</sub>]
7.  dIn[B, D] {U<sub>Y</sub>} = dTmp[B, F<sub>Y</sub>] \*<sub>F</sub> W<sub>in</sub>[D, F<sub>Y</sub>]（*给更前面的层用*）
8.  dIn[B, D<sub>Y</sub>] = **ReduceScatter**(dIn[B, D] {U<sub>Y</sub>})（*在关键路径上*）

</div>

{% enddetails %}

张量并行有个妙处：它和 Transformer 前向里的两个矩阵配合得刚好。天真做法是两个矩阵后各来一次 AllReduce。但按我们的排法：先 **In[B, D<sub>Y</sub>] \* W<sub>in</sub>[D, F<sub>Y</sub>] -> Tmp[B, F<sub>Y</sub>]**，再 **Tmp[B, F<sub>Y</sub>] \* W<sub>out</sub>[F<sub>Y</sub>, D] -> Out[B, D<sub>Y</sub>]**——只需开头 AllGather 一次 **In**、结尾 ReduceScatter 一次 **Out**，不用 AllReduce。

**这有多贵？** 只建模前向即可——反向就是每个运算转置一遍。一维张量并行在第一次矩阵乘前 AllGather 激活、第二次后 ReduceScatter，每次一个字节对（bf16）。来算算什么时候被通信卡住。

$$\begin{align}
T_\text{math} & = \frac{4 \cdot B \cdot D \cdot F}{Y \cdot C} \\
T_\text{comms} & =
\frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}\\
\textnormal{T} & \approx \max \left(\frac{4 \cdot B \cdot D \cdot F}{Y \cdot C}, \frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}\right)
\end{align}$$

要求计算时间大于通信时间：

$$\begin{align}
\frac{4 \cdot B \cdot D \cdot F}{Y \cdot C} > \frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}
\end{align}$$

$$\begin{align}
\frac{F}{Y \cdot C} > \frac{1}{W_\text{ici}}
\end{align}$$

$$\begin{align}
F > Y \cdot \frac{C}{W_\text{ici}}
\end{align}$$

例如 TPUv5p 上 $C / W_{ici}$ 在 bf16 下是 2550，所以张量并行最多做到 $Y < F / 2550$。若有多条 ICI 轴，$T_\text{comms}$ 会缩小 $M_Y$ 倍，于是 $Y < M_Y \cdot F / 2550$。

<p markdown=1 class="takeaway">**要点**：当 $Y > M_Y \cdot F / 2550$ 时，张量并行变成通信受限。对大多数模型，这意味着张量并行做到 8 到 16 路就到头了。</p>

**注意这与计算的数值精度无关**：比如 int8 下，TPUv5p 的 $$C_\text{int8} / W_{ici}$$ 是 $$5100$$ 而不是 $$2550$$，但通信量也减半了，两个 2 正好抵消。

**来看几个例子：**

* TPUv5p 上跑 LLaMA 3-70B，$$D = 8192,$$ $$F \approx 30,000$$：8 路张量并行轻轻松松，16 路就会被通信卡住。支撑 8 路模型分片所需的 F 是 20k。

* Gemma 7B 的 $$F \approx 50k$$，到 19 路张量并行才被卡住。也就是说做到 16 路应该仍有好性能。

### 把 FSDP 和张量并行组合起来

**记号：** $$\text{In}[B_X, D_Y] \cdot_D W_\text{in}[D_X, F_Y] \cdot_F W_\text{out}[F_Y, D_X] \rightarrow \text{Out}[B_X, D_Y]$$

FSDP 和张量并行的好处在于可以叠加。沿两条轴分别切 **W<sub>in</sub>** 和 **W<sub>out</sub>**，省内存又省计算。沿 X 切 B，让模型并行的 AllGather 变小；沿 Y 切 F，让 FSDP 的通信开销变小。两者结合，能把有效批大小压到比前面单独的任何一种都低。

{% include figure.liquid path="assets/img/mixed-fsdp-model-parallelism.png" class="img-fluid" caption="<b>图：</b>FSDP 与张量并行组合的示意图。与前面几种不同，模型参数没有任何重复。" %}

{% details 这是 FSDP + 张量并行混合的完整算法。通信看着很多，但因为我们把激活按批切得更碎、权重按张量切得更碎，每次 AllGather 和 ReduceScatter 的量都小得多！ %}

<div markdown=1 class="algorithm">

**前向传播：** 需要计算 Loss[B]

1.  In[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(In[B<sub>X</sub>, D<sub>Y</sub>])（*在关键路径上*）
2.  W<sub>in</sub>[D, F<sub>Y</sub>] = **AllGather**<sub>X</sub>(W<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>])（*可以提前做*）
3.  Tmp[B<sub>X</sub>, F<sub>Y</sub>] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>]
4.  W<sub>out</sub>[F<sub>Y</sub>, D] = **AllGather**<sub>X</sub>(W<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>])（*可以提前做*）
5.  Out[B<sub>X</sub>, D] {U<sub>Y</sub>} = Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D]
6.  Out[B<sub>X</sub>, D<sub>Y</sub>] = **ReduceScatter**<sub>Y</sub>(Out[B<sub>X</sub>, D] {U<sub>Y</sub>})（*在关键路径上*）
7.  Loss[B<sub>X</sub>] = ...

**反向传播：** 需要计算 dW<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>], dW<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>]

1.  dOut[B<sub>X</sub>, D<sub>Y</sub>] = ...
2.  dOut[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(dOut[B<sub>X</sub>, D<sub>Y</sub>])（*在关键路径上*）
3.  dW<sub>out</sub>[F<sub>Y</sub>, D] {U<sub>X</sub>} = Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>B</sub> dOut[B<sub>X</sub>, D]
4.  dW<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>] = **ReduceScatter**<sub>X</sub>(dW<sub>out</sub>[F<sub>Y</sub>, D] {U<sub>X</sub>})
5.  W<sub>out</sub>[F<sub>Y</sub>, D] = **AllGather**<sub>X</sub>(W<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>])（*可以提前做*）
6.  dTmp[B<sub>X</sub>, F<sub>Y</sub>] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F<sub>Y</sub>, D]（*这里可以把 dOut[B, D] 丢掉*）
7. In[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(In[B<sub>X</sub>, D<sub>Y</sub>])（*不在关键路径上，且可与上一层的 (2) 共享*）
8.  dW<sub>in</sub>[D, F<sub>Y</sub>] {U<sub>X</sub>} = In[B<sub>X</sub>, D] \*<sub>B</sub> dTmp[B<sub>X</sub>, F<sub>Y</sub>]
9.  dW<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>] = **ReduceScatter**<sub>X</sub>(dW<sub>in</sub>[D, F<sub>Y</sub>] {U<sub>X</sub>})
10. W<sub>in</sub>[D, F<sub>Y</sub>] = **AllGather**<sub>X</sub>(W<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>])（*可以提前做*）
11. dIn[B<sub>X</sub>, D] {U<sub>Y</sub>} = dTmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W<sub>in</sub>[D, F<sub>Y</sub>]（*给更前面的层用*）
12. dIn[B<sub>X</sub>, D<sub>Y</sub>] = **ReduceScatter**<sub>Y</sub>(dIn[B<sub>X</sub>, D] {U<sub>Y</sub>})（*在关键路径上*）

</div>

{% enddetails %}

**FSDP 和张量并行的正确配比是什么？** 有一条朴素但关键的准则：FSDP 搬的是权重，张量并行搬的是激活。因此批越小（尤其数据并行做得越多），张量并行反而越便宜——因为每片的激活变小了。

* 张量并行做的 $$\mathbf{AllGather}_Y([B_X, D_Y])$$ 随 $$X$$ 变大而变小。
* FSDP 做的 $$\mathbf{AllGather}_X([D_X, F_Y])$$ 随 $$Y$$ 变大而变小。

所以把两者组合，能把每副本的最小批大小进一步压低。最优配比可以照上面的方法算：

设 $$X$$ 为用于 FSDP 的芯片数、$$Y$$ 为用于张量并行的芯片数，$$N$$ 为切片的总芯片数、$$N=XY$$。设 $$M_X$$、$$M_Y$$ 为 FSDP 和张量并行各自横跨的网格轴条数（两者之和大致等于 3）。只建模前向（它每个 FLOPs 对应的通信最多）。把上面算法的通信加起来：

$$T_\text{FSDP comms}(B, X, Y) = \frac{2\cdot 2\cdot D \cdot F}{Y \cdot W_\text{ici} \cdot M_X}$$

$$T_\text{TP comms}(B, X, Y) = \frac{2 \cdot 2 \cdot B \cdot D}{X \cdot W_\text{ici} \cdot M_Y}$$

计算时间则

$$T_\text{math} = \frac{2\cdot 2 \cdot B \cdot D \cdot F}{N \cdot C}.$$

为简化分析，做两个假设：其一，允许 $X$、$Y$ 取非整数（只要为正且满足 $XY=N$）；其二，X 轴和 Y 轴上的通信可以彼此完全重叠。第二个假设下，总通信时间为

$$T_\text{comms} = \max\left(T_\text{FSDP comms}, T_\text{TP comms}\right)$$

在问“什么时候计算受限”之前，先找让通信最小的 $X$、$Y$。既然 FLOPs 与 $X$、$Y$ 无关，最优解就是单纯最小化通信。把 $T_\text{comms}$ 写成 $X$ 和 $N$（固定，是系统的芯片总数）的函数：

$$T_\text{comms} (X) = \frac{4D}{W_\text{ici}} \max\left(\frac{F \cdot X}{N \cdot M_X}, \frac{B}{X \cdot M_Y}\right)$$

$T_\text{FSDP comms}$ 随 $X$ 单调上升，$T_\text{TP comms}$ 随 $X$ 单调下降，所以两者的较大值必在 $T_\text{FSDP comms} = T_\text{TP comms}$ 处最小，此时

$$\begin{align*}
\frac{FX_{opt}}{M_X} = \frac{BN}{X_{opt} M_Y} \rightarrow \\
X_{opt} = \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N}
\end{align*}$$

这非常有用！给定 $B$、$F$、$N$，它直接告诉我们 FSDP 的最优规模。感受一下量级：代入比较真实的数值——$N = 64$（4x4x4 的芯片方阵）、$B=48,000$、$F=32768$——得到 $X\approx 13.9$。于是选 $X = 16$、$Y = 4$，就很接近最优了。

<p markdown=1 class="takeaway">**要点**：训练时，FSDP 的最优规模一般是 $$X_{opt} = \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N}$$。 </p>

现在回到我们对每种并行方案都要问的那个问题：**什么条件下才能计算受限？** 由于计算和通信可以重叠，计算受限的条件是

$$\max\left(T_\text{FSDP comms}, T_\text{TP comms}\right) < T_\text{math}$$

记 $\alpha \equiv C / W_\text{ici}$（ICI 算术强度），可化简为：

$$\max\left(\frac{F}{Y \cdot M_X}, \frac{B}{X \cdot M_Y}\right) < \frac{B \cdot F}{N \cdot \alpha}$$

既然 $X_{opt}$ 正是让左边两项相等的取值，直接代进去任一项（注意 $Y_{opt} = N/X_{opt}$）：

$$\frac{F}{N \cdot W_\text{ici} \cdot M_X} \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N} < \frac{B \cdot F}{N \cdot C}$$

继续化简得

$$ \sqrt{\frac{B\cdot F}{M_X \cdot M_Y \cdot N}} < \frac{B \cdot F}{N \cdot \alpha},$$

左边正比于通信时间，右边正比于计算时间。注意：计算时间随批大小线性增长（不管怎么并行都如此），通信时间却只随批大小的*平方根*增长。两者之比因此也按批大小的平方根增长：

$$ \frac{T_\text{math}}{T_\text{comms}} = \frac{\sqrt{BF}\sqrt{M_X M_Y}}{\alpha \sqrt{N}}. $$

要让这个比值大于 1（计算受限），需要

$$ \frac{B}{N} > \frac{\alpha^2}{M_X M_Y F}$$

代入近似数字：$F=32,768$、$\alpha=2550$、$M_X M_Y=2$（三维网格必然如此），得到 $B/N > 99$ 左右。相比纯数据并行（或 FSDP）——三维网格下 $B/N$ 要超过约 $850$ 才计算受限——我们大约赢了 8 倍。

<p markdown=1 class="takeaway">**要点**：把张量并行和 FSDP 组合，能把 $B/N$ 压到 $$2550^2 / 2F$$——每台芯片只需约 100 个 token，比只用 FSDP 能做到的约小 8 倍。</p>

下图画出“混合 FSDP + 张量并行”在代表性的 4x4x4 芯片方阵上，FLOPs 时间与通信时间之比，并与“只用张量并行（TP）”“只用数据并行（FSDP）”对比。批极大时纯 FSDP 占优；但当“每芯片批大小”落在大约 100 到 850 之间时，必须混合 FSDP + TP 才能计算受限。

{% include figure.liquid path="assets/img/mixed-fsdp-comms-2.png" class="img-fluid" caption="<b>图：</b>TPUv5p 4x4x4 切片、F=30k 时，最优 FSDP/TP 混合的 FLOPs 与通信时间之比。如预期：张量并行的比值不随批大小变化；最优 FSDP + TP 按 $\sqrt{B}$ 增长；FSDP 按 $B$ 增长。但在中间批大小区间，只有 FSDP + TP 能让比值超过 1。"%}

再来一个 TPU v5p 16x16x16 的例子，展示不同分片方案的 FLOPs 与通信时间随批大小的变化。

{% include figure.liquid path="assets/img/math-comms-time.png" class="img-fluid" caption="<b>图：</b>不同并行方案下的通信耗时。黑色虚线是矩阵乘法计算所需时间，任何在这条线上方的曲线都是通信受限。可以看到，所有方案在批小于 6e5 时都会通信受限——与我们预计的 4096 * 2550^2 / (2 * 8192 * 4) = 4e5 相符。" %}

黑色曲线是模型计算所需时间：只要某批大小下它比所有通信时间都低，就必然通信受限。可以看到黑线在约 `4e5` 处与绿色曲线相交，和预测一致。

这里还有一个可交互的动画，可以拖动不同批大小，看总计算时间和通信时间：

<figure class="plotly-embed">
  <iframe src="{{ 'assets/plotly/training-roofline.html' | relative_url }}" title="MLP roofline for mixed FSDP/TP" loading="lazy" scrolling="no"></iframe>
  <figcaption><b>图：</b>TPU v5p 16x16x16（$D=8192$、$F=32768$）上，前向 MLP 的计算与通信耗时随批大小的变化，用的是上面的 $T_\text{FSDP comms}$、$T_\text{TP comms}$ 和 $T_\text{math}$。拖动滑块改变 FSDP/TP 的配比。阴影区内的任何批大小都是通信受限的。</figcaption>
</figure>

你会看到它与前面的结论基本一致：最佳配比 FSDP=512、TP=8，在批大小约 4e5 以上就计算受限；任何加大张量并行的配比，在任何批大小下都被通信卡着；减少张量并行的配比则需要更大的批。

### 流水线并行

你可能已经注意到，前面完全没提流水线并行。流水线（pipelining）是 GPU 并行中的主流策略，但在 TPU 上没那么不可或缺。简单说：流水线训练把模型的层拆开放到多台设备上，前向和反向时把激活在各流水线阶段之间传来传去。算法大意为：

1. 数据从 TPU 0 开始，权重沿层维分片（若与 FSDP、张量并行组合，就是 $W_\text{in}[L_Z, D_X, F_Y]$）。
2. 在 TPU 0 上算第一层，把得到的激活复制给 TPU 1，如此重复直到最后一台 TPU。
3. 计算损失函数及其导数 $\partial L / \partial x_L$。
4. 对最后一个流水线阶段，算出导数 $\partial L / \partial W_L$ 和 $\partial L / \partial x_{L-1}$，再把 $\partial L / \partial x_{L-1}$ 复制给上一个阶段，如此重复直到 TPU 0。

{% details 这里有一段可以运行的 Python 伪代码 %}

这段伪代码可以在 Cloud TPU VM 上跑。它既不高效也不真实，但能让你体会数据是怎么在各设备间传递的。

```python
batch_size = 32
d_model = 128
d_ff = 4 * d_model

num_layers = len(jax.devices())

key = jax.random.PRNGKey(0)

# Pretend each layer is just a single matmul.
x = jax.random.normal(key, (batch_size, d_model))
weights = jax.random.normal(key, (num_layers, d_model, d_model))

def layer_fn(x, weight):
  return x @ weight

# Assume we have num_layers == num_pipeline_stages
intermediates = [x]
for i in range(num_layers):
  x = layer_fn(x, weights[i])
  intermediates.append(x)

  if i != num_layers - 1:
    x = jax.device_put(x, jax.devices()[i+1])

def loss_fn(batch):
  return jnp.mean(batch ** 2)  # make up some fake loss function

loss, dx = jax.value_and_grad(loss_fn)(x)

for i in range(num_layers - 1, -1, -1):
  _, f_vjp = jax.vjp(layer_fn, intermediates[i], weights[i])
  dx, dw = f_vjp(dx)  # compute the jvp dx @ J(L)(x[i], W[i])
  weights[i] = weights[i] - 0.01 * dw  # update our weights

  if i != 0:
    dx = jax.device_put(dx, jax.devices()[i-1])
```

{% enddetails %}

**为什么这是个好主意？** 流水线并行有很多好处：阶段之间通信开销极低，哪怕互连带很差也能训练超大模型。这在 GPU 上特别有用，因为 GPU 不像 TPU 有大范围的高速互联。

**难处和麻烦在哪？** 你可能发现了：上面伪代码里的 TPU 0 几乎一直在闲着！它只在流水线的头和尾干了点活。这段空转叫“流水线气泡”（pipeline bubble），非常棘手。通常先靠微批次（microbatching）缓解：把多个小批一起灌进流水线，让 TPU 0 至少在更大比例的时间里保持忙碌。

第二种办法是仔细重叠三种计算：前向矩阵乘 $W_i @ x_i$、反向的 $dx$ 矩阵乘 $W_i @ \partial L / \partial x_{i+1}$、以及 $dW$ 矩阵乘 $\partial L / \partial x_{i+1} @ x_i$。它们各自都有计算量，互相错开填满空隙，就能把气泡完全藏起来。下面是 DeepSeek v3 论文<d-cite key="DeepSeek3"></d-cite>里的“无气泡”流水线调度图：

{% include figure.liquid path="assets/img/deepseek-pipeline.png" class="img-fluid" caption="<b>图：</b>DeepSeek v3 的流水线调度（出自其<a href=\"https://arxiv.org/pdf/2412.19437\">论文</a>）。橙色是前向矩阵乘，绿色是 dL/dx 矩阵乘，蓝色是 dL/dW 矩阵乘。优先安排反向的 dL/dx 乘法，就能避免让 FLOPs 被'晾'在那里。" %}

由于 TPU（整机群互联范围大）对流水线的依赖较低，我们不再深挖，但弄懂流水线的关键瓶颈仍是个很好的练习。

### 跨整机群（Pod）扩大规模

单片 TPU 能组到的最大规模是 8960 块芯片的 TPU v5p SuperPod（2240 台主机）。要再大，就得越过数据中心网络（DCN）的边界。每台 TPU 主机都配有一块或几块网卡（NIC），通过以太网连向其他 TPU v5p 整机群。如 [TPU 一节](../tpus)所说，每台主机约有 200Gbps（25GB/s）的全双工 DCN 带宽，摊到每块 TPU 约是 6.25GB/s 的全双工（出站）带宽。

通常，超出单个整机群时，我们会在 ICI 域内做某种模型并行或 FSDP，再在多个整机群之间做纯数据并行。设要扩到 $N$ 块 TPU，每个 ICI 互联切片含 $M$ 块 TPU。要跨 DCN 做 AllReduce，可以在整机群之间做环形归约，反向传播于是有：

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{N \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF}{M \cdot W_\text{dcn}}$$

通信带宽随 $M$ 增长——与 ICI 不同，扩大 ICI 域的同时也拿到了更多网卡。化简可得 $T_\text{math} > T_\text{comms}$ 当

$$\frac{B}{\text{slice}} > \frac{C}{W_\text{dcn}}$$

TPU v5p 上 $\frac{C}{W_\text{dcn}}$ 约为 `4.59e14 / 6.25e9 = 73,440`。也就是说：想高效跨 DCN 扩展，每个 ICI 域（切片）必须承担至少这么大的批。

**这问题有多严重？** 举个例子：用 TPU v5p 训练 LLaMA-3 70B，批 2M token，$F\approx 30,000$。由前面的结论：

* 张量并行最多做到 $Y = M_Y \cdot F / 2550 \approx 11 \cdot M_Y$。
* FSDP 只要 $B / N > 2550 / M_X$ 就能做。若要以批 2M、三条数据并行轴来训练，最多只能用到 $\approx 2400$ 块芯片——大约四分之一个 TPU v5p 整机群。
* 组合 FSDP + 张量并行后，$B / N < 2550^2 / (2 \cdot 30000) = 108$ 才通信受限，这能扩到约 18k 块芯片！但 TPU v5p 单整机群上限是 8k 芯片，再往上就得走 DCN。

总之我们有一个训练批 1M 的漂亮配方：X（FSDP）≈ 1024、Y（TP）≈ 8；而批 2M 就必须用上 DCN。前面算过 DCN 算术强度是 $\text{73,440}$，只要保证每个 ICI 域的批大于它就行。这很容易：两个整机群各担 1M 批，每块 TPU 摊到 111 个 token，完全够用（稍微贴边，但理论上没问题）。

<p markdown=1 class="takeaway">**要点**：跨多个 TPU 整机群扩展其实相当直白——用纯数据并行，只要每个整机群的批不小于 73k token。</p>

## TPU 上训练 LLM 的重点小结

* 加大并行度或减小批大小，都会让我们更容易被通信卡住——因为它们都减少了每块芯片上的计算量。

* 在合理的上下文长度（约 32k）内，我们可以把 Transformer 建模成一摞 MLP 块，并用“每层那两三处主要矩阵乘怎么分片”来定义各种并行方案。

* 训练中有 4 种主要并行方案，各有各的带宽与计算要求（数据并行、FSDP、张量并行、FSDP + 张量并行混合）。

| **策略**                                 | **描述**                                                                                                                                                                            |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **数据并行**                         | 激活沿批分片，其余全部完整复制；反向时对梯度做 all-reduce。                                                                      |
| **FSDP**                                     | 激活、权重、优化器状态都沿批分片；权重用到之前才收集（gather），梯度做 reduce-scatter。                                                               |
| **张量并行（又叫 Megatron、模型并行）** | 激活沿 $$d_\text{model}$$ 分片，权重沿 $$d_{ff}$$ 分片；W<sub>in</sub> 之前收集激活，W<sub>out</sub> 之后对结果做 reduce-scatter。 |
| **FSDP + 张量并行混合**          | 以上两者叠加，由 FSDP 负责收集模型并行分过片的权重。                                                                                                                           |

每种方法的“公式”如下：

$$\small
\begin{array}{cc}
\text{Strategy} & \text{Formula}\\
\hline
\text{DP} & \text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D] \\
\text{FSDP} & \text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D] \\
\text{TP} & \text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y] \\
\text{TP + FSDP}  & \text{In}[B_X, D_Y] \cdot_D W_\text{in}[D_X, F_Y] \cdot_F W_\text{out}[F_Y, D_X] \rightarrow \text{Out}[B_X, D_Y] \\
\hline
\end{array}$$

* 每种策略都有一个“开始被网络/通信卡住”的界限，由每设备的计算与通信决定。下面是逐层的计算量与通信量，设 $$X$$ 为 FSDP、$$Y$$ 为张量并行：

$$
\small
\begin{array}{ccc}
\text{Strategy} & \text{Compute per layer} & \text{Comms per layer} \\
& \text{(ignoring gating einsum)} & \text{(bytes, forward + backward pass)}\\
\hline
\text{DP} & 4BDF/X + 8BDF/X & 0 + 8DF \\
\text{FSDP} & 4BDF/X + 8BDF/X & 4DF + 8DF \\
\text{TP} & 4BDF/Y + 8BDF/Y & 4BD + 4BD \\
\text{FSDP + TP} & 4BDF/(XY) + 8BDF/(XY) & (4BD/X + 4DF/Y) + (8BD/X + 8DF/Y) \\
\hline
\end{array}$$

* 纯数据并行很少够用：模型加上优化器状态就要占“10 倍参数量”的字节数，内存里往往装不下超过几十亿参数。

* 当 $$\text{batch size per shard} < C / W$$（每分片批大小小于网络的算术强度）时，数据并行和 FSDP 会通信受限。ICI 上是 2,550，DCN 上约 71,000。多几条并行轴可以改善。

* 当 $$\lvert Y\rvert > F / 2550$$ 时，张量并行通信受限。**对多数模型这大约是 8-16 路。** 它与批大小无关。

* FSDP + 张量并行混合，能把每芯片批大小压到 $$2550^2 / 2F \approx 100$$ 这么低，低得惊人。

* 跨整机群的数据并行，要求每个整机群的批不小于约 71,000，否则会被 DCN 卡住。

* 总之：批够大或模型够小，事情就简单——做数据并行，或 FSDP 再跨 DCN 做数据并行。难的是中间地带。

## 动手解题

本节以 LLaMA-2 13B 为基本模型。它的配置如下：

| 超参数 | 值  |
| ---------- | ------ |
| L          | 40     |
| D          | 5,120  |
| F          | 13824  |
| N          | 40     |
| K          | 40     |
| H          | 128    |
| V          | 32,000 |

LLaMA-2 的嵌入矩阵和输出矩阵是分开的，MLP 块带门控。

**问题 1：** LLaMA-2 13B 有多少参数（虽然明知它就叫 13B，还是请动手算一遍）？*注意：如[Transformer 的数学](../transformers)所述，LLaMA-3 有三个大 FFW 矩阵——两个上投影、一个下投影。本节忽略了两个“门控”einsum 矩阵，它们的行为与 W<sub>in</sub> 相同。*

{% details 点此查看答案。 %}

* FFW 参数：$$3LDF$$ = `8.5e9`
* 注意力参数：$$4DNHL$$ = `4.2e9`
* 词表参数：$$2VD$$ = `0.33e9`
* 合计：`8.5e9 + 4.2e9 + 0.33e9 = 13.0e9`，和名字对上了！

{% enddetails %}

**问题 2：** 假设以批 16M token、Adam 优化器训练。先不管并行，模型的参数、优化器状态和激活一共要用多少内存？*假设参数用 bf16、优化器状态用 fp32，且每层只在三个大 FFW 矩阵乘之后做激活检查点（各存 3 份）。*

{% details 点此查看答案。 %}

参数（bf16）加两个优化器状态（fp32 的一阶、二阶矩累加器）共 `(2 + 4 + 4) * 13e9 ~ 130GB`。前两个矩阵乘之后的激活形状是 $BF$、最后一个之后是 $BD$（按前面的 Transformer 图），所以 bf16 激活总量是 $2 \cdot L \cdot (BD + 2 * BF) = 2LB \cdot (D + 2F)$，即 `2 * 40 * 16e6 * 5,120 * (1 + 2 * 2.7) ~ 4.2e13 = 42TB`（`B=16e6`）。其余激活基本可以忽略。

{% enddetails %}

**问题 3：** 假设要在 TPUv5p 16x16x16 切片上，以 32k 序列长度、总批 3M token 训练，权重 bfloat16、优化器 float32，同上。

1. 能用纯数据并行吗？为什么？
2. 只用 FSDP 会不会被通信卡住？为什么？纯 FSDP 下每台设备用多少内存（假设只在 3 个大 FFW 矩阵乘之后做梯度检查点）。
3. 能用 FSDP + 张量并行混合吗？为什么？如果可以，$X$ 和 $Y$ 应该取多少？每台设备存多少数据？只按 roofline 的 FLOPs 估算、忽略注意力的话，40% MFU 下每一步训练要花多久？

{% details 点此查看答案。 %}

先把数字摆出来。32k 序列长度、批 3M，序列批大小为 96。TPU v5p 16x16x16 切片共有 `393TB` HBM。

1. 纯数据并行不行：它把参数和优化器状态在每块芯片上完整复制，约 130GB（见问题 2），超过单芯片的 96GB HBM。

2. 先看内存。把问题 2 的批从 16M 换成 3M，检查点激活共 `~7.86e12`，加上 1.3e11 的优化器状态，几乎正好 8e12 = 8TB。切片共有 `393TB` HBM，内存妥妥够用。再看会不会被通信卡住：4096 块芯片、三条并行轴，最小批是 `850 * 4096 = 3.48M` token，略高于我们的 3M。也就是说会被通信卡住，有点可惜。结论：**只用 FSDP 做不到不被通信卡住。**

3. 现在主要矛盾就是通信。前面算过，混合 FSDP + 张量并行的每芯片批要高于 $2550^2 / 2F = 235$。也就是说理论上可行！来看各取多少。

规则是 $X_{opt} = \sqrt{(B / F) \cdot (M_X / M_Y) \cdot N}$，代入得 `sqrt(3e6 * 2 * 4096 / 13824) = 1333`，于是做大约 1024 路数据并行 + 4 路张量并行。每台 TPU 的内存同 (2)，每步时间就是 `6 * 3e6 * 13e9 / (4096 * 4.6e14 * 0.4) = 300ms`。

{% enddetails %}

<h3 markdown=1 class="next-section">第 5 部分到此结束！第 6 部分会把这些方法用到真实的 LLaMA 模型上，[点击这里](../applied-training)！</h3>

## 附录

### 附录 A：反向传播的通信是怎么推出来的

前面把 Transformer 层的前向简化成了 Out[B, D] = In[B, D] *<sub>D</sub> W<sub>in</sub>[D, F] *<sub>F</sub> W<sub>out</sub>[F, D]。那反向传播需要哪些通信是怎么来的？

这直接来自上一节单个矩阵乘 **Y = X * A** 的规则：

$$\frac{dL}{dA} = \frac{dL}{dY}\frac{dY}{dA} = X^T \left(\frac{dL}{dY}\right)$$

$$\frac{dL}{dX} = \frac{dL}{dY}\frac{dY}{dX} = \left(\frac{dL}{dY}\right) A^T$$

由此得到下面几条公式（Tmp[B, F] 代表 In[B, D] * W<sub>in</sub>[D, F]）：

<div markdown=1 class="algorithm">

1. dW<sub>out</sub>[F, D] = Tmp[B, F] *<sub>B</sub> dOut[B, D]
2. dTmp[B, F] = dOut[B, D] *<sub>D</sub> W<sub>out</sub>[F, D]
3. dW<sub>in</sub>[D, F] = In[B, D] *<sub>B</sub> dTmp[B, F]
4. dIn[B, D] = dTmp[B, F] *<sub>F</sub> W<sub>in</sub>[D, F]

</div>

注意：这些只是数学等式，完全没提分片。反向传播的任务就是把这四个量算出来。所以，要弄清需要哪些通信，只需取上面四个等式里所有要做矩阵乘的量（Tmp、dOut、W<sub>out</sub>、W<sub>in</sub>）的分片状态——它们由你选的并行方案决定——再按分片矩阵乘的规则推出必须做的通信即可。注意 dOut 的分片方式与 Out 相同。
