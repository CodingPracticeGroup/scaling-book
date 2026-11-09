---
layout: distill
title: "分片的矩阵，以及怎么把它们相乘"
# permalink: /main/
description: "训练大型机器学习模型时，我们必须把模型的参数或输入分片（shard，也叫切分、划分）到许多加速器上。由于大模型几乎全由矩阵乘法组成，这件事最终就归结为一个问题：矩阵被拆散放在多台设备上时，还怎么把它们相乘？我们从 TPU 通信原语的开销出发，建立一套关于分片矩阵乘法的简单理论。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 3

previous_section_url: "../tpus"
previous_section_name: "第 2 部分 · 如何理解 TPU"

next_section_url: ../transformers
next_section_name: "第 4 部分 · Transformer 的数学"

giscus_comments: true
giscus_discussion: 5

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
  - name: "分片的记号与集合通信操作"
  - subsections:
    - name: "一套统一的分片记号"
    - name: "怎么用代码来描述这件事？"
  - name: "用分片后的数组做计算"
  - subsections:
    - name: "情形 1：两个乘数的收缩维都没有被分片"
    - name: "情形 2：有一个乘数的收缩维被分片了"
    - name: "情形 3：两个乘数的收缩维都被分片了"
    - name: "情形 4：两个乘数都有一个非收缩维沿同一条网格轴被分片"
  - name: "更深入地看 TPU 的通信原语"
  - subsections:
    - name: "最后一个通信原语：AllToAll"
    - name: "再深入谈谈 ReduceScatter"
    - name: "怎样让矩阵乘法的通信和计算重叠"
  - name: "我们学到了什么？"
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

## 分片的记号与集合通信操作

用一万块 TPU 或 GPU 训练大模型时，我们做的计算和只在一块芯片上训练时其实是同一回事。区别在于：**我们的数组塞不进任何一块 TPU/GPU 的 HBM**，所以只能把它们拆开。<d-footnote>值得说明的是，有时我们为了更快也会主动选择并行。哪怕模型在一小批芯片上就放得下，多拿一些芯片也能简单地换来更多 FLOPs/秒。比如推理时，我们有时明明能塞进更小的拓扑，却故意分到更大的拓扑上，好让延迟更低。同样，训练时也常常是多拿一些芯片，好把每一步的时间压短。</d-footnote>我们管这叫给数组做“*分片（sharding）*”或“*划分（partitioning）*”。扩大规模的“艺术”，就在于想清楚怎么分片，才能让计算一直保持高效。

下面这个例子，是一个二维数组 **A** 被分片到了 4 块 TPU 上：

{% include figure.liquid path="assets/img/sharding-example.png" class="img-fluid" caption="<b>图：</b>一个形状为 <b>A</b>[I, J] 的数组被分片到 4 台设备上。两个维度各切成两半，分别放在 2 台设备上，分片方式记作 <b>A</b>[I<sub>X</sub>, J<sub>Y</sub>]。每块 TPU 只持有整个数组的 1/4。" %}

注意：分片后的数组仍然有一个*全局*的、也叫*逻辑*的形状，比如 `(4, 128)`，跟没分片时一模一样；但它同时还有一个*设备本地形状*，比如 `(2, 64)`，它告诉你每块 TPU 实际拿着多大的数据（上面这张图里，每块 TPU 持有整个数组的 ¼）。下面把它推广到任意数组。

### 一套统一的分片记号

我们用一种“*命名轴记号*”的变体来描述张量是怎么按块分到各设备上的：我们假设存在一个二维或三维的设备格子，叫**设备网格（device mesh）**，它的每条轴都有自己的**网格轴名**，比如 **X**、**Y** 和 **Z**。只要说明数组的每个被命名的维度沿着哪条物理网格轴划分，就能描述矩阵数据在整个网格上的摆法。这个“哪个维度沿哪条网格轴分”的分配方案，就叫一个**分片（sharding）**。

**例子（上面那张图）**：对上图来说，我们有：
* **网格：** 设备网格为 `Mesh(devices=((0, 1), (2, 3)), axis_names=('X', 'Y'))`，意思是：有一台 2x2 格子排布的 4 块 TPU，两条轴分别叫 $X$ 和 $Y$。
* **分片：** $A[I_X, J_Y]$，意思是：把第一个轴 $I$ 沿网格轴 $X$ 分片，把第二个轴 $J$ 沿网格轴 $Y$ 分片。这个分片告诉我们，每片只持有数组的 $1 / (\lvert X\rvert \cdot \lvert Y\rvert)$。

把两者合起来，我们就知道数组的本地形状（单台设备持有的那片有多大）是 $(\lvert I\rvert / 2, \lvert J\rvert / 2)$，其中 $$\lvert I\rvert$$ 是 A 第一个维度的大小，$$\lvert J\rvert$$ 是 A 第二个维度的大小。

<b markdown=1 style="color: #048affff;">小测验 [二维数据只沿一条轴分片]：</b> 有一个数组 `fp32[1024, 4096]`，分片方式 $A[I_{XY}, J]$，网格 `{'X': 8, 'Y': 2}`。每台设备持有多大的数据？在 H100 上把这整个数组从 HBM 读出来要多久（设每芯片内存带宽 `3.4e12`）？

{% details 点此查看答案。 %}

$A[I_{XY}, J]$ 表示把第一个维度（I）沿 X 和 Y 两条硬件轴一起分片。这个例子里，本地形状就是 $(\lvert I\rvert /(\lvert X\rvert \cdot \lvert Y\rvert), \lvert J\rvert)$。代入题目数据，全局形状是 `fp32[1024, 4096]`，所以本地形状是 `fp32[64, 4096]`。

每台 GPU 上有 `4 * 64 * 4096 = 1MiB` 字节，读取大约需要 `1e6 / 3.4e12 = 294ns`——不过这么小的数据，各种固定开销会占大头，实际时间大概率会明显更长。

{% enddetails %}

**把这些分片画出来：** 我们试着用一张“2 行 4 列”、分给 4 台设备的数据图来想象这些分片：

{% include figure.liquid path="assets/img/sharding-colored1.png" class="img-fluid img-small" %}

**完全复制**（完全不分片）的形式就简单写成 $A[I, J]$，不带任何分片标记。意思是*每台*设备都存着整个矩阵的一份完整副本。

{% include figure.liquid path="assets/img/sharding-colored2.png" class="img-fluid img-small" %}

要说某条维度沿某条网格轴做了划分，就在下标上标出网格轴。比如 $A[I_X, J]$ 表示 **I** 这条逻辑轴沿 **X** 网格维做了划分，而 **J** 维*没有*被划分，数据块沿 **Y** 网格轴*部分复制*。

{% include figure.liquid path="assets/img/sharding-colored3.png" class="img-fluid img-small" %}

$A[I_X, J_Y]$ 表示 **I** 逻辑轴沿 **X** 网格轴划分、**J** 维沿 **Y** 网格轴划分。

{% include figure.liquid path="assets/img/sharding-colored4.png" class="img-fluid img-small" %}

其余的可能性画在下面的图里：

{% include figure.liquid path="assets/img/sharding-colored5.png" class="img-fluid" %}

这里 $A[I_{XY}, J]$ 表示把 **X** 和 **Y** 两条网格轴看成一条更大的、拉平的维度，把 **I** 命名轴沿所有设备切开。多个网格轴下标的*顺序是有讲究的*：它规定了划分沿网格扫描的先后次序。

{% include figure.liquid path="assets/img/sharding-colored6.png" class="img-fluid img-small" %}

最后请注意：*不能*把多条命名轴往*同一条*网格维上分片。比如 $A[I_X, J_X]$ 是不合法、被禁止的分片。一条网格维一旦用来给某个数组的某条维做分片，它就在某种意义上“用完”了。

<b markdown=1 style="color: #57cf57;">小测验：</b> 设 **A** 是一个形状为 `int8[128, 2048]` 的数组，分片方式 $A[I_{XY}, J]$，网格 `Mesh({'X': 2, 'Y': 8, 'Z': 2})`（共 32 台设备）。**A** 在每台设备上占多少内存？在所有设备上一共占多少内存？

{% details 点此查看答案。 %}

**答案：** 数组 **A** 沿 X 和 Y 分片、沿 Z 复制，所以每台设备上的形状是 `int8[128 / (2 * 8), 2048] = int8[8, 2048]`，大小 `8 * 2048 = 16,384` 字节。由于它沿 Z 复制，而在每个 Z 平面内部又沿 X、Y 彻底分片，所以整个数组一共有 2 份完整副本（每个 Z 平面一份）。因此所有设备加起来的大小是：原数组大小 × Z 向副本数 = 128 * 2048 * 2 = 共 512 KiB。另一种算法：32 台设备 × 每台 16,384 字节 = 共 512 KiB，能对上。

{% enddetails %}

### 怎么用代码来描述这件事？

前面我们一直没碰代码，现在正好抢先看一眼。JAX 有一套命名式分片语法，和我们上面讲的抽象记号几乎一模一样。[第 10 部分](../jax-stuff)会细讲，这里先给个预告。你可以在[这个 Google Colab](https://colab.research.google.com/drive/15cxw66eABwZPG-V4QFmbLfiykPFf_gaP?usp=sharing)里亲手玩一玩，并用性能分析工具看看 JAX 如何处理各种分片。这段代码做了 3 件事：

1. 建一个 **jax.Mesh**，把 8 块 TPU 组成一个 4x2 的网格，并给两条轴起名叫 'X' 和 'Y'。
2. 建矩阵 A 和 B：A 沿两个维度都分片，B 沿输出维分片。
3. 编译并执行一次简单的矩阵乘法，返回一个分片数组。

```py
import jax
import jax.numpy as jnp

# Create our mesh! We're running on a TPU v2-8 4x2 slice with names 'X' and 'Y'.
# The Auto axis type tells JAX to let the XLA compiler infer intermediate shardings.
assert len(jax.devices()) == 8
Auto = jax.sharding.AxisType.Auto
mesh = jax.make_mesh(axis_sizes=(4, 2), axis_names=('X', 'Y'), axis_types=(Auto, Auto))

# A little utility function to help define our sharding. A PartitionSpec is our
# sharding (a mapping from axes to names).
def P(*args):
  return jax.NamedSharding(mesh, jax.sharding.PartitionSpec(*args))

# We shard both A and B over the non-contracting dimension and A over the contracting dim.
A = jnp.zeros((8, 2048), dtype=jnp.bfloat16, device=P('X', 'Y'))
B = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, device=P(None, 'Y'))

# We can perform a matmul on these sharded arrays! out_shardings tells us how we want
# the output to be sharded. JAX/XLA handles the rest of the sharding for us.
y = jax.jit(lambda A, B: jnp.einsum('BD,DF->BF', A, B), out_shardings=P('X', 'Y'))(A, B)
```

JAX 妙就妙在：这些数组用起来就像没被分片过一样！`B.shape` 会告诉你全局的、逻辑上的形状 (2048, 8192)。你得专门去看 `B.addressable_shards`，才知道它本地是怎么切的。在这些数组上做运算，JAX 会自己想办法把它们广播或重排成能算的样子。比如上面这段例子里，**A** 的本地形状是 `[2, 1024]`，**B** 是 `[2048, 4096]`。为了完成最后的乘法，JAX/XLA 会自动在这些数组之间加上需要的通信。

## 用分片后的数组做计算

如果一份数据分布在许多台设备上，你想对它做数学运算，那么“数据和计算都被分片”会带来哪些额外开销？

这当然取决于要做什么计算。

* 对*逐元素*运算，在分布式数组上计算**没有任何额外开销**。
* 当我们想做那种要跨越多台设备上元素的运算时，事情就复杂了。好在机器学习中几乎所有计算都是矩阵乘法，而它相对好分析。

本节余下部分讲的就是怎么把分片后的矩阵相乘。粗略地说，就是把矩阵的一块块数据搬来搬去，好让每一块都能被完整地乘完或加完。**不同的分片方式对应不同的通信。** 比如 $A[I_X, J] \cdot B[J, K_Y] \to C[I_X, K_Y]$ 可以完全不通信就乘出来，因为*收缩维*（就是 J——矩阵乘法里真正要做求和、被“缩掉”的那一维）没有被分片。但如果我们想要输出不分片（即 $A[I_X, J] \cdot B[J, K_Y] \to C[I, K]$），就得把 $A$、$B$ 或 $C$ 中的某一方复制到每台设备上（用一次 *AllGather*）。这两种选择的通信代价不同，所以我们要把它们都算出来，挑最便宜的那个。

{% details 你可以用“分块矩阵乘法”来理解这件事。 %}

要理解这一点，可以先回忆“*分块矩阵*”的概念——也就是“矩阵套矩阵”的嵌套矩阵：

$$\begin{equation}
\begin{pmatrix}
a_{00} & a_{01} & a_{02} & a_{03} \\
a_{10} & a_{11} & a_{12} & a_{13} \\
a_{20} & a_{21} & a_{22} & a_{23} \\
a_{30} & a_{31} & a_{32} & a_{33}
\end{pmatrix}
=
\left(
\begin{matrix}
\begin{bmatrix}
a_{00} & a_{01} \\
a_{10} & a_{11}
\end{bmatrix} \\
\begin{bmatrix}
a_{20} & a_{21} \\
a_{30} & a_{31}
\end{bmatrix}
\end{matrix}
\begin{matrix}
\begin{bmatrix}
a_{02} & a_{03} \\
a_{12} & a_{13}
\end{bmatrix} \\
\begin{bmatrix}
a_{22} & a_{23} \\
a_{32} & a_{33}
\end{bmatrix}
\end{matrix}
\right)
=
\begin{pmatrix}
\mathbf{A_{00}} & \mathbf{A_{01}} \\
\mathbf{A_{10}} & \mathbf{A_{11}}
\end{pmatrix}
\end{equation}$$

矩阵乘法有一个很好的性质：当两个乘数都按分块写出来时，乘积可以按同样规则写成“分块的矩阵乘法”：

$$\begin{equation}
\begin{pmatrix}
A_{00} & A_{01} \\
A_{10} & A_{11}
\end{pmatrix}
\cdot
\begin{pmatrix}
B_{00} & B_{01} \\
B_{10} & B_{11}
\end{pmatrix}
=
\begin{pmatrix}
A_{00}B_{00} + A_{01}B_{10} & A_{00}B_{01} + A_{01}B_{11} \\
A_{10}B_{00} + A_{11}B_{10} & A_{10}B_{01} + A_{11}B_{11}
\end{pmatrix}
\end{equation}$$

这意味着：实现分布式矩阵乘法，就归结为把这些分片块通过网络搬来搬去、在块上做*本地*的矩阵乘法、再把结果加起来。**剩下的问题就是：需要加什么通信，它有多贵。**

{% enddetails %}

好在，所有可能的分片都能归成大约 4 种情形，每种情形都有一条对应“要加什么通信”的规则：
1. **[情形 1](#情形-1-两个乘数的收缩维都没有被分片)：** 两个输入沿收缩维都没有分片。_完全不用通信，本地直接乘。_
2. **[情形 2](#情形-2-有一个乘数的收缩维被分片了)：** 有一个输入沿收缩维分片了。_通常先沿收缩维对这个输入做一次 “AllGather”。_
3. **[情形 3](#情形-3-两个乘数的收缩维都被分片了)：** 两个输入沿收缩维都被分片了。_本地先把分片乘起来，再对结果做一次 “AllReduce”。_
4. **[情形 4](#情形-4-两个乘数都有一个非收缩维沿同一条网格轴被分片)：** 两个输入都有一个非收缩维沿同一条网格轴被分片。不先把其中一个输入 AllGather 一下，就没法往下做。

你可以只把这几条当成“照做就行”的规则，但理解它们为什么成立、各自多贵，也很值得。下面逐个细讲。

### 情形 1：两个乘数的收缩维都没有被分片 {#情形-1-两个乘数的收缩维都没有被分片}

**引理：** 给分片矩阵做乘法时，计算是合法的，输出会自动继承输入的分片——*除非*收缩维被分片了，或者两个矩阵沿同一条轴都分了片。比如下面这样完全没问题：

$$\begin{equation*}
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K_Y] \rightarrow \mathbf{C}[I_X, K_Y]
\end{equation*}$$

完全不需要任何通信，结果就是一个沿 X、Y 两条硬件维都分好片的张量。不妨自己想想为什么。说白了，计算和分片方式是*互不相干*的：每个批次样本手里都拿着“要被求和那一维”的一段本地数据，自己乘完自己加就行。下面这些组合全都成立、全都符合这条规则：

$$\begin{align*}
\mathbf{A}[I, J] \cdot \mathbf{B}[J, K] \rightarrow &\ \mathbf{C}[I, K] \\
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K] \rightarrow &\ \mathbf{C}[I_X, K]\\
\mathbf{A}[I, J] \cdot \mathbf{B}[J, K_Y] \rightarrow &\ \mathbf{C}[I, K_Y]\\
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K_Y] \rightarrow &\ \mathbf{C}[I_X, K_Y]
\end{align*}$$

由于 **A** 和 **B** 的收缩维 **J** 都没被分片，我们只要把各个输入的分块在本地乘起来，结果*自然就已经*按照想要的输出分片摆好了。当两个乘数都有非收缩维沿同一条轴被分片时，这话就不再成立（详见[非法分片](#情形-4-两个乘数都有一个非收缩维沿同一条网格轴被分片)一节）。

### 情形 2：有一个乘数的收缩维被分片了 {#情形-2-有一个乘数的收缩维被分片了}

看看输入 **A** 沿收缩维 **J** 被分片、而 **B** 完全复制时该怎么办：

$$\mathbf{A}[I, J_X] \cdot \mathbf{B}[J, K] \rightarrow \mathbf{C}[I, K]$$

我们不能拿 **A** 和 **B** 的本地块直接乘，因为必须对 **A** *完整的*收缩维求和，而它被拆散在 X 轴各台设备上。通常的做法是先对 **A** 的分片做一次 “**AllGather**”，让每台设备都拿到完整的一份，然后再和 **B** 相乘：

$$\textbf{AllGather}_X[I, J_X] \rightarrow \mathbf{A}[I, J]$$

$$\mathbf{A}[I, J] \cdot \mathbf{B}[J, K] \rightarrow \mathbf{C}[I, K]$$

这样真正的乘法就能完全在每台设备本地完成了。

<p markdown=1 class="takeaway">**要点：** 乘矩阵时，若其中一方的收缩维被分片了，一般先把它 AllGather 起来，让收缩维不再分片，然后在本地乘。</p>

注意，若 **B** 没有沿 X 分片，我们也可以先在本地乘出“部分和”，再把这些部分和加起来（即 *AllReduce*）。这样计算也能分到各设备上，只是通信通常更贵。某些情况下这样反而更快，不过现实中 **B** 往往也是分过片的，所以一般不这么干。下面[第 4 题](#动手解题)会算清楚什么时候这样更划算。

**什么是 AllGather？** AllGather 是我们要讲的第一个核心 [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) 通信原语。AllGather 会*消除*某条轴上的分片：沿那条轴散落在各设备上的分片，会被拼回完整的一份，放到这条轴上的*每台*设备上。用我们的记号，AllGather 就是“把下标擦掉”，例如：

$$\textbf{AllGather}_{XY}(A[I_{XY}, J]) \rightarrow A[I, J]$$

不必把一条维的所有下标一次全擦掉，例如 $$A[I_{XY}, J] \rightarrow A[I_Y, J]$$ 也是 AllGather，只是只沿一条轴做。另外还要注意：AllGather 也可以用来消除*非收缩维*的分片。比如矩阵乘法：

$$A[I_X, J] \cdot B[J, K] \rightarrow C[I, K]$$

我们可以一开始就把 **A** AllGather 起来、消掉输入的分片；也可以先按分片乘完，再对结果 **C** 做 AllGather。

**AllGather 具体怎么做？** 在一条 TPU 轴（一个环）上做一维 AllGather，基本上就是让每台 TPU 把自己那片沿环接力传递，直到每台设备都拿到完整一份。<d-footnote>GPU 上的 AllGather 也能这样工作：把一台机器里的 GPU 连成一个环，按（任意的）顺序沿环传递数据块。</d-footnote> 看这段动画：

{% include figure.liquid path="assets/img/all-gather.gif" caption="<b>图：</b>一段动画，展示如何在 8 台 TPU 或 GPU 设备上做一次 AllGather。每台设备起初只有数组的 1/8，结束时每台都拿到完整一份。" %}

AllGather 可以沿单向做，也可以双向同时做（上面演示的是双向）。单向做时，每台 TPU 沿环传 $N-1$ 跳，每跳传 $\text{bytes} / N$ 大小的块。双向做时，只需 $\lfloor \frac{N}{2} \rfloor$ 跳，每跳每个方向传 $\text{bytes} / N$。

**这要花多久？** 我们来算双向 AllGather 的耗时。设 $$V$$ 是数组的字节数，$X$ 是收缩维上的分片数。按上面的图，每一跳在两个方向各传 $V / \lvert X\rvert$ 字节，所以每跳耗时

$$T_{hop} = \frac{2 \cdot V}{\lvert X \rvert \cdot W_\text{ici}}$$

其中 $W_\text{ici}$ 是**双向** ICI 带宽。<d-footnote>分子上那个 2，是因为我们用的是双向带宽：两个方向各传 $V / X$，合计 $2V / X$。</d-footnote> 要覆盖所有 TPU 总共需要传 $\lvert X\rvert / 2$ 跳<d-footnote>严格说是 $\lfloor X / 2 \rfloor$</d-footnote>，所以整个过程总耗时

$$T_{total} = \frac{2 \cdot V \cdot X}{2 \cdot X \cdot W_\text{ici}}$$

$$T_{total} = \frac{V}{W_\text{ici}}$$

注意这**和 $X$ 无关！** 这挺惊人——它意味着虽然我们的 TPU 只是“就近相连”的，但这种就近连接并不影响速度，瓶颈只在每条链路本身的快慢。

<p markdown=1 class="takeaway">**要点：** 在“带宽受限”的区间里做 AllGather（或 ReduceScatter、AllReduce），实际通信时间只取决于数组大小和可用带宽，跟数组被分到了多少台设备上无关！</p>

**关于 ICI 延迟的说明：** 每跳 ICI 链路都有一点“起步开销”，跟传多少数据无关，通常约 1us。所以当数组 $$A$$ 非常小、每跳数据本来用不了 1us 时，就会进入“延迟受限”区间，这时计算结果*会*跟 $X$ 有关。

{% details 完整推导点此展开。 %}

设 $$T_\text{min}$$ 是单跳的最小耗时。那么

$$T_{hop} = \max \left[ T_{min}, \frac{2 \cdot V}{X \cdot W_\text{ici}} \right]$$

$$T_{total} = \max \left[ \frac{T_{min} \cdot X}{2}, \frac{V}{W_\text{ici}} \right]$$

因为我们要走 $X / 2$ 跳。对大的归约或收集操作，我们稳稳处于带宽受限区间：数据多到每跳的固定开销几乎可以忽略。但对很小的数组（比如模型采样时），这笔开销不能忽略，ICI 带宽帮不上忙——我们就纯被延迟卡着。换个说法：拿一块具体的 TPU，比如 TPU v5e，单向 ICI 带宽 `4.5e10`，那么任何小于 `4.5e10 * 1e-6 = 45kB` 的缓冲区都会是延迟受限的。

{% enddetails %}

下面是在一块 TPU v5e 8x16 切片上实测的 AllGather 带宽。数组沿长度为 16 的那条轴分片，所以有一个完整的双向环。

{% include figure.liquid path="assets/img/all-gather-bandwidth.png" class="img-small" caption="<b>图：</b>TPU v5e 做 AllGather 时的实测带宽与推算链路带宽。橙色是按秒实际收集到的字节数；蓝色是按该集合通信的已知开销算出的实测单向链路带宽。" %}

注意我们不但达到了标称峰值带宽（`4.5e10`）的约 95%，而且在约 10MB 处就到顶了——按 16 路分片摊，每设备才 625kB（*旁注*：这比 GPU 好得多）。

**沿多条轴同时 AllGather 会怎样？** 沿多条轴收集时，我们就多出了几个可以做收集的 ICI 维度。比如 AllGather<sub>XY</sub>([B, D<sub>XY</sub>]) 同时利用两条硬件网格轴。可用带宽因此变成原来的 $N_\text{axes}$ 倍。

算上延迟因素，通用规则是：

$$T_{total} = \max \left[ \frac{T_{min} \cdot \sum_{i} |X_i|}{2}, \frac{V}{W_\text{ici} \cdot N_\text{axes}} \right]$$

其中 $$\sum_i \lvert X_i \rvert / 2$$ 是 TPU 网格中最长路径的长度。

<b markdown=1 style="color:rgb(144, 92, 255);">小测验 2 [AllGather 的时间]：</b> 用[第 2 部分](../tpus)里的数字，在 TPU v5e、二维网格 `{'X': 8, 'Y': 4}` 上做 AllGather<sub>Y</sub>([E<sub>Y</sub>, F]) → [E, F] 要多久？设 $$E = 2048$$、$$F = 8192$$，bfloat16。换成 $$E=256, F=256$$ 呢？

{% details 点此查看答案。 %}

**答案：** 先算几个基本量：

1) TPU v5e 每条轴有 4.5e10 字节/秒的单向 ICI 带宽，共两条轴。
2) bfloat16 下情形 (a)：$A[E_Y, F]$，每台设备持有 bf16[512, 8192]，即 512 * 8192 * 2 = 8.4MB。整个数组的大小是 2048 * 8192 * 2 = 34MB。

*第 (1) 问*，直接套上面的公式。因为只沿一条轴收集，$T_{\text{comms}} = \text{34e6} / \text{9e10} = \text{377us}$。检查是否延迟受限：沿一条大小为 4 的轴最多 3 跳，延迟下限约 3us，差得远，不是延迟受限。不过 TPU v5e 只有当某条轴长为 16 时才有绕回链路，所以*这里其实做不了真正的双向 AllGather*。数据从边缘要传 3 跳才能到达另一端，所以理论值更接近 $T_{\text{comms}} = 3 * \text{8.4e6} / \text{4.5e10} = 560\mu s$。[**这里**](https://imgur.com/a/RkvpRGQ)**有一份真实的性能分析**（来自 [这个 Colab](https://colab.research.google.com/drive/15tDZMfNqm2vJjvSzw5VC9qtSwc5td-oV?usp=sharing)），显示 $680 \mu s$——很合理，因为我们拿不到 100% 的理论带宽！*第 (2) 问*：每个分片 `64 * 256 * 2 = 32kB`。`32e3 / 4.5e10 = 0.7us`，所以是延迟受限。共 3 跳，大约 3 * 1us = 3us。[实测接近 8us。](https://imgur.com/a/HZLQmYs)

{% enddetails %}

<p markdown=1 class="takeaway">**注意：** 对于 `{'X': 16, 'Y': 4}` 这样的二维网格，网格轴并不要求一一对应某条具体的*硬件*轴。比如它也可以描述一个 4x4x4 的 TPU v5p 立方体、其中两条轴都挂在 $X$ 名下。后文讲“跨多轴的数据并行”时会用到这一点。</p>

### 情形 3：两个乘数的收缩维都被分片了 {#情形-3-两个乘数的收缩维都被分片了}

第三种基本情形：两个乘数沿收缩维、且沿同一条网格轴都被分片了：

$$\textbf{A}[I, J_X] \cdot \textbf{B}[J_X, K] \rightarrow C[I, K]$$

这时*本地*的分块矩阵乘法至少在“能做”上是没问题的——两边用的收缩下标是同一段。但每台设备乘出来的只是完整结果的一个*部分和*，沿 **X** 维的每台设备拿到的部分还各不相同。这种情形太常见了，我们专门扩展记号来标记它：

$$\textbf{A}[I, J_X] \cdot_\text{LOCAL} \textbf{B}[J_X, K] \rightarrow C[I, K] \{\ U_X \}$$

记号 **{ U<sub>X</sub> }** 读作“沿 X 网格轴*未归约*”，意思是这个运算在某种意义上“没做完”：就差最后加总一下。$\cdot_\text{LOCAL}$ 表示“做本地那份乘加，但结果先不归约”。

这背后的事实，其实是矩阵乘法与外积的关系：

$$A \cdot B = \sum_{i=1}^{P} \underbrace{A_{:,i} \otimes B_{i,:}}_{\in \mathbb{R}^{n \times m}}$$

其中 ⊗ 是外积。这样，如果 **X** 轴上的第 **i** 块 TPU 存着 **A** 的第 **i** 列和 **B** 的第 **i** 行，它就能本地算出 $$A_{:,i} \otimes B_{i,:} \in \mathbb{R}_{n\times m}$$。这个矩阵的每个元素，恰好是 **A • B** 对应元素求和式里的第 **i** 项。还缺的是对 **P** 的那一步求和——它被拆到了网格轴 **X** 的各台设备上——补上它，才能得到完整的 **A • B**。把 **A**、**B** 按块（也就是按分片）写、再对结果的每一块求和，道理完全一样。

用一次沿 **X** 轴的完整 **AllReduce** 就能完成这个求和：

$$\begin{align*}
A[I, J_X] \cdot_\text{LOCAL} B[J_X, K] \rightarrow &\ C[I, K] \{ U_X \} \\
\textbf{AllReduce}_X C[I, K] \{ U_X \} \rightarrow &\ C[I, K]
\end{align*}$$

AllReduce 消掉“未归约”状态，让沿那条轴的*每台*设备都拿到完全加好的同一个值。AllReduce 是本节要讲的第二个关键通信原语（第一个是 AllGather，后面还有 ReduceScatter 和 AllToAll）。AllReduce 接收一个带有未归约（部分和）轴的数组，把各分片沿那条轴传着累加，完成求和。它的签名是

$$\textbf{AllReduce}_Y A[I_X, J] \{U_Y\} \rightarrow A[I_X, J]$$

也就是它只是去掉 $\{U_Y\}$ 这个后缀，别的一概不变。

**AllReduce 有多贵？** 可以这样想象它的工作方式：每台设备把自己这份发给邻居，把收到的所有份加在一起。显然它比 AllGather 更贵，因为每个“分片”都有整个数组那么大。一般说，**AllReduce 的代价是 AllGather 的两倍。** 有一种看法很直观：**AllReduce** 可以由两个别的原语拼成：一次 **ReduceScatter** 加一次 **AllGather**。和 AllReduce 一样，ReduceScatter 也能消掉部分和，但输出的结果是沿某条维“散开”（分片）的状态。接着 AllGather 再把这些碎片收集起来，沿那条逻辑轴取消分片、复制成完整。

$$\begin{align*}
\textbf{ReduceScatter}_{Y,J} : A[I_X,J] \{U_Y\} \rightarrow &\ A[I_X, J_Y] \\
\textbf{AllGather}_Y : A[I_X, J_Y] \rightarrow &\ A[I_X, J]
\end{align*}$$

**ReduceScatter 又是什么？** 就像 AllGather 是把分好片的数组拼回来（擦掉下标），ReduceScatter 是把“未归约/部分和”数组加起来，然后沿*同一条网格轴*把另一条逻辑轴散开（分片）：$X[F]\\{U_Y\\} \to X[F_Y]$。动画演示了过程：注意它和 AllGather 非常像，区别是每台设备不保留收到的每份数据，而是把它们加在一起。所以它的延迟和 AllGather 差不多（再加一点点做加法的时间）。

{% include figure.liquid path="assets/img/reduce-scatter.gif" class="img-fluid" %}

每跳的通信时间就是每片的字节数 $V / Y$ 除以带宽 $W_\text{ici}$，和 AllGather 一样，所以有

$$T_{\text{comms per AllGather or ReduceScatter}} = \frac{V}{W_\text{ici}}$$

$$T_{\text{comms per AllReduce}} = 2 \cdot \frac{V}{W_\text{ici}}$$

其中 $$W_\text{ici}$$ 是双向带宽，前提是有一条完整的归约环。

### 情形 4：两个乘数都有一个非收缩维沿同一条网格轴被分片 {#情形-4-两个乘数都有一个非收缩维沿同一条网格轴被分片}

给一个张量分片时，每条网格维最多只能出现一次。按上面的规则做，有时会撞上违反这条规则的情形，比如：

$$A[I_X, J] \cdot B[J, K_X] \rightarrow C[I_X, K_X]$$

这不合规则因为：沿 **X** 的第 **i** 片如果直接乘，只能算出 **C** 的第 **(i, i)** 块——一个对角块。光靠这些分片，除了对角线上的块，其他什么都拼不出来，所以这样的分片不被允许。

解决的办法是把其中一些维 AllGather 掉。这里有两个选择：

$$\begin{align*}
\textbf{AllGather}_X A[I_X, J] \rightarrow &\ A[I, J] \\
A[I, J] \cdot B[J, K_X] \rightarrow &\ C[I, K_X]
\end{align*}$$

或者

$$\begin{align*}
\textbf{AllGather}_X B[J, K_X] \rightarrow &\ B[J, K] \\
A[I_X, J] \cdot B[J, K] \rightarrow &\ C[I_X, K]
\end{align*}$$

无论哪种，结果里 **X** 都只会出现一次。选哪种，取决于后面的运算想要什么样的分片。

## 更深入地看 TPU 的通信原语

前面 4 种情形已经引出了几个做分片矩阵乘法要用到的“核心通信原语”：

1. **AllGather：** 擦掉分片记号里的一个下标，把分片收集成完整一份。
2. **ReduceScatter：** 消掉“未归约”后缀——沿那条轴把各份加起来，让数组转而沿另一条维分片。
3. **AllReduce：** 消掉“未归约”后缀，让数组在那条轴上恢复完整。

还有一个核心通信原语要讲，它出现在混合专家（MoE）等模型的计算里：**AllToAll**。

### 最后一个通信原语：AllToAll {#最后一个通信原语-alltoall}

最后这个基本集合通信，在分片矩阵乘法里不会自然出现，但在实践中频繁遇到：**AllToAll** 集合通信，更准确说是它的特例——*分片转置*（resharding，重新分片）操作。例如：

$$\textbf{AllToAll}_{X, J} A[I_X, J] \rightarrow A[I, J_X]$$

当一段分片计算的相邻区域用了互不兼容的排布方案时，就需要 AllToAll 来重排分片布局。它在分片的混合专家模型里很自然地出现。*你可以把 AllToAll 理解为“把一个下标从一条轴挪到另一条轴”*。由于 AllToAll 不需要把每个分片的完整数据沿环复制，它实际上比 AllGather *更便宜*（只有 ¼ 的代价）<d-footnote>对偶数大小的双向环，每台设备向右发 $(N/2 + (N/2-1) + … + 1)$ 块、向左发 $((N/2-1) + … + 1)$ 块 $= 0.5 \cdot (N / 2) \cdot (N/2 + 1) + 0.5 \cdot (N / 2) \cdot (N/2 - 1) = N^2/4$。每块（即“分片的分片”）的大小是 $\text{bytes} / N^2$，所以每台设备的代价是 $(\text{bytes} / N^2) \cdot N^2 / 4 = \text{bytes} / 4$。所有设备合起来的总量随设备数的增加方式，与总带宽随设备数的增加方式一致。</d-footnote>。

{% include figure.liquid path="assets/img/all-to-all.gif" class="img-fluid" %}

推广到 N 维，在 AxBxC 网格上，对一个总共 $V$ 字节（所有设备加在一起）的数组做一次 AllToAll，总代价是

$$T_\text{comms per AllToAll} = \frac{V \cdot \max(A, B, C, ...)}{4 \cdot N \cdot W_\text{ici}}$$

照例，$W_\text{ici}$ 是双向 ICI 带宽，$N = A \cdot B \cdot C \cdot \ldots$ 是设备总数。等价地，用每设备字节数 $V / N$ 表示，代价是 $(V / N) \cdot \max(A, B, C, ...) / (4 \cdot W_\text{ici})$。对一维网格，它化简为 $V / (4 \cdot W_\text{ici})$，正好是 AllGather 的 1/4。在二维里，代价反而随*最小*那条轴的大小下降。

*旁注：想要一个不太严格的推导，可以从一维环面 $\mathbb{Z} / N\mathbb{Z}$ 出发。随机挑一个源节点和目标节点，它们平均相距 N / 4 跳，于是代价是 $(V \cdot N) / (4 * N)$。再看 N 维环面：每条轴基本互不相干。每台设备持有 $1 / N$ 的字节，平均要把数据传 $\max(A, B, C, …) / 4$ 跳。也可以从对分带宽推：AllToAll 中，网格的一半要把自己一半的数据（$V / 4$ 字节）发给另一半。最窄的对分切面垂直于最长轴，切过 $2 \cdot N / \max(A, B, …)$ 条链路（两个切面，算上绕回），单向带宽即 $N \cdot W_\text{ici} / \max(A, B, …)$。两者相除就是上面的公式。*

### 再深入谈谈 ReduceScatter

ReduceScatter 比乍看起来更根本：它其实是 AllGather 的“导数”，反过来也是。也就是说，若前向传播里有：

$$\textbf{AllGather}_X A[I_X] \rightarrow A[I]$$

那么反向传播时，把反向模式导数 **A'**（各分片一般各不相同）做一次 ReduceScatter，就得到分片状的 **A'**：

$$\textbf{ReduceScatter}_X A'[I] \{ U_X \} \rightarrow A'[I_X]$$

同理，前向的 $$\text{ReduceScatter}_X(A[I] \{U_X\}) \to A[I_X]$$ 意味着反向的 $$\text{AllGather}_{X}(A'[I_X]) \to A'[I]$$。

{% details AllGather 和 ReduceScatter 为何互为导数，点此展开。 %}

这源于一个事实：广播和归约作为线性算子互为转置，而 AllGather 和 ReduceScatter 分别是广播和归约的外积（也叫 [Kronecker 积](https://en.wikipedia.org/wiki/Kronecker_product)）。具体地说，给一个向量 $x \in \mathbb{R}^n$、任意设备台数 $p \in \mathbb{N}$，并记 $u = (1, \ldots, 1) \in \mathbb{R}^p$，我们可以如下定义广播和归约——它应该和你的直觉一致：

$$
\begin{align*}
\text{broadcast} &: \mathbb{R}^n \rightarrow \mathbb{R}^{p n} \\
\text{broadcast} &= u \otimes \mathbf{I}_n \\
\text{reduce} &: \mathbb{R}^{p n} \rightarrow \mathbb{R}^n \\
\text{reduce} &= u^T \otimes \mathbf{I}_n
\end{align*}
$$

看一个例子，$n = 1$，$p = 2$。若 $x = (7)$，则 $$\text{broadcast}(x) = \left(\begin{pmatrix} 1 \\ 1 \end{pmatrix} \otimes \begin{pmatrix} 1 \end{pmatrix}\right) x = \begin{pmatrix} 1 \\ 1 \end{pmatrix} x = \begin{pmatrix}  7\\  7  \end{pmatrix} \in \mathbb{R}^{p n}$$。这正符合预期：把 $\mathbb{R}^n$ 里的向量广播成 $\mathbb{R}^{pn}$ 里的向量。再取 $y = (8, 9)$，则 $$\text{reduce}(y) = \left(\begin{pmatrix} 1 & 1 \end{pmatrix} \otimes \begin{pmatrix} 1\end{pmatrix}\right) y = \begin{pmatrix} 1 & 1  \end{pmatrix} \begin{pmatrix}  8 \\ 9  \end{pmatrix} = \begin{pmatrix}   17    \end{pmatrix}$$。这也符合预期：把 $\mathbb{R}^{p n}$ 里的向量归约成 $\mathbb{R}^{n}$ 里的向量。由于对任意矩阵 $A$、$B$ 都有 $(A \otimes B)^T = A^T \otimes B^T$，我们看到 $\text{reduce} = \text{broadcast}^T$。而 AllGather 和 ReduceScatter 就是下面这些外积：

$$
\begin{align*}
\text{AllGather} &: \mathbb{R}^{p n} \rightarrow \mathbb{R}^{p^2 n} \\
\text{AllGather} &= \text{broadcast} \otimes \mathbf{I}_p \\
\text{ReduceScatter} &= \mathbb{R}^{p^2 n} \rightarrow \mathbb{R}^{p n} \\
\text{ReduceScatter} &= \text{reduce} \otimes \mathbf{I}_p
\end{align*}
$$

这里把 $\mathbb{R}^{p^2 n}$ 看作 $\mathbb{R}^{p \times p n}$，即 $p$ 台设备各持有一个 $\mathbb{R}^{p n}$ 向量。建议拿 $n = 2$、$p = 3$ 这样的小例子亲手算一算，看看这些算子长什么样。再用一次转置性质，就得到 $\text{AllGather}^T = \text{ReduceScatter}$，当然也有 $\text{ReduceScatter}^T = \text{AllGather}$。反向传播时这个转置就会出现：若 $y = Ax$，其中 $A$ 是某个线性算子（比如 AllGather 或 ReduceScatter），反向传播时我们手上有损失对 $y$ 的导数 $\frac{\partial L}{\partial y}$，而 $\frac{\partial L}{\partial x} = A^T \frac{\partial L}{\partial y}$。这就说明 AllGather 的导数是 ReduceScatter，反之亦然。

{% enddetails %}

把 AllReduce 拆成 ReduceScatter + AllGather 还有个方便之处：最后那次 AllGather 可以先不做，留到以后。我们经常并不愿意付“把整个矩阵乘积重新拼成完整、每台一份”的代价，而宁愿在“两个乘数的收缩维都被分片”的情形下保持分片状态：

$$A[I, J_X] \cdot B[J_X, K] \rightarrow C[I, K_X]$$

这时我们可以只做一次 ReduceScatter 而不做 AllReduce，最后需要时再做 AllGather，即：

$$\begin{align*}
A[I, J_X] \cdot_{LOCAL} B[J_X, K] \rightarrow &\ C[I, K] \{ U_X \} \\
\textbf{ReduceScatter}_{X,K} C[I, K] \{ U_X \} \rightarrow &\ C[I, K_X]
\end{align*}$$

注意 ReduceScatter 会*引入*一个新的分片维，所以这里有“沿 **I** 还是沿 **K** 命名维分片”的选择自由。用 ReduceScatter 时我们一般得*指定*往哪条命名维引入新分片（当然，选择通常由整个模型的上下文决定）。这就是我们写 **ReduceScatter<sub>X,K</sub>** 来指定分片轴的原因。

### 怎样让矩阵乘法的通信和计算重叠

正如[第 1 部分](../roofline)讲的，只要通信足够快，我们通常可以假设通信总能和一些有用的计算重叠。本节的集合通信原则上也能和矩阵乘法的计算本身重叠，但做起来不容易。我们用的算法叫**集合通信矩阵乘法（collective matmul）**，最早出自 [Wang et al.](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959)。下面是这种重叠的简化动画：

{% include figure.liquid path="assets/img/ag_matmul.gif" caption="<b>图：</b>一段动画，展示一次分片矩阵-向量乘法如何与随后的 AllReduce（上面的情形 3）重叠。完整的矩阵乘法由多次矩阵-向量乘法组成。" %}

说白了，就是在做某一块矩阵乘法的*同时*，启动前面那些块的环状归约。某些情况下，我们还可以沿批或矩阵输入维做分块流水。[第 10 部分](../jax-stuff)给出了一个简单的 JAX 实现，[Mosaic 文档](https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html)也有一个不错的 GPU 版例子。鼓励你亲手实现一次。

## 我们学到了什么？

* 一个数组的分片，由两部分描述：**Mesh** 给 TPU 网格的物理硬件轴起名字；**Sharding** 把网格轴名指派给数组的逻辑轴。
  * 例如 **A**[I<sub>XY</sub>, J] 描述一个抽象数组 **A**，第一维沿 X、Y 两条网格轴分片。再配上 Mesh(mesh_shape=(4, 8), axis_names=('X', 'Y'))（简写 Mesh({'X': 4, 'Y': 8})），就说明数组的第一维被 32 路分片了。

* **对分片数组做算术，结果和不分片时完全一样——除非你沿被分片的轴做了收缩（求和）**。那种时候就得引入一些通信。我们归纳了四种情形：

  1. *两个数组沿收缩维都没分片*：不需要通信。
  2. *有一个数组沿收缩维分片*（或两个收缩维沿不同轴分片）：先把其中一个输入 AllGather，再做运算。
  3. *两个数组沿收缩维、沿同一条轴以同样方式分片*：本地先乘，再做一次 AllReduce 或 ReduceScatter。
  4. *两个数组都有非收缩维沿同一条网格轴分片*：先把其中一个输入 AllGather。

* TPU 大致用到 **4 种核心通信原语**：
  1. AllGather：$[A_X, B] \to [A, B]$
  2. ReduceScatter：$[A, B] \\{U_X\\} \to [A_X, B]$
  3. AllToAll：$[A, B_X] \to [A_X, B]$
  4. AllReduce：$[A_X, B]\\{U_Y\\} \to [A_X, B]$（严格说不算原语，它是 ReduceScatter + AllGather 的组合）

{% include figure.liquid path="assets/img/all-collectives.png" class="img-fluid" %}

* 这些操作的时间与延迟，**在带宽受限区间里不依赖轴的大小**，只取决于输入数组的大小和链路带宽。对一次单向 AllGather/ReduceScatter：

$$T_{\text{comm per AllGather or ReduceScatter}} = \frac{\text{Data volume}}{\text{bandwidth}} \cdot \frac{\text{Axis} - 1}{\text{Axis}}
\longrightarrow \frac{\text{Data volume}}{\text{bandwidth (bidirectional)}}$$

* AllReduce 由一次 ReduceScatter 加一次 AllGather 组成，因此是上面代价的 2 倍。AllToAll 只需把分片沿环传半程，因此只有 AllGather 的 ¼。总结如下：

| 操作              | 描述                                                                                                        | 语法                           | 耗时                                          |
| :---------------- | :----------------------------------------------------------------------------------------------------------------- | :------------------------------- | :----------------------------------------------- |
| **AllGather**     | 沿某轴收集分片、去掉一个下标。                                     | $[A_X, B] \to [A, B]$            | 字节数 /（双向 ICI 带宽 × 轴数） |
| **ReduceScatter** | 沿某轴把部分和加起来，转而沿另一条维分片（加一个下标）。                 | $[A, B] \\{U_X\\} \to [A_X, B]$  | 与 AllGather 相同                                |
| **AllReduce**     | 沿某轴把部分和加起来，去掉一个 { U<sub>x</sub> }。是 AllGather 和 ReduceScatter 的组合。 | $[A_X, B]\\{U_Y\\} \to [A_X, B]$ | AllGather 的 2 倍                                    |
| **AllToAll**      | 收集（复制）一条轴，同时沿同一条轴分片另一条维。                                 | $[A, B_X] \to [A_X, B]$          | 双向环上为 AllGather 的 1/4           |

## 动手解题 {#动手解题}

*这里有一些与本节内容配套的练习题。目前不会给出全部答案，但我们会陆续补上。*

**问题 1 [复制式分片]**：数组按 $A[I_X, J, K, \ldots]$ 分片（只沿 $X$ 分片），网格 `Mesh({'X': 4, 'Y': 8, 'Z': 2})`。$A$ 在所有芯片上占的总字节数，与单份数组大小之比是多少？

{% details 点此查看答案。 %}

我们的数组只沿 X（大小 4）分片，所以每片实际大小是 $[I / 4, J, K, \ldots] = \text{sizeof}(A) / 4$。又因为沿 Y 和 Z 复制，总大小是 $Y \cdot Z \cdot \text{sizeof}(A)$。于是总大小与单份大小之比为 $Y \cdot Z \cdot \text{sizeof}(A) / \text{sizeof}(A) = 16$。

{% enddetails %}

**问题 2 [AllGather 延迟]**：在 TPU v4p 4x4x4 切片、网格 `Mesh({'X': 4, 'Y': 4, 'Z': 4})` 上，$\text{AllGather}_X([B_X, D_Y])$ 应该要多久？设 $B=1024$、$D=4096$，bfloat16。那 $$\text{AllGather}_{XY}([B_X, D_Y])$$ 呢？$$\text{AllReduce}_Z([B_X, D_Y] \{U_Z \})$$ 呢？

{% details 点此查看答案。 %}

完整的 `4x4x4` 立方体在每条轴上都有绕回链路，所以可用的双向带宽是 9e10。

1. 我们只沿一条轴收集，另一条维是分片状态，所以实际上是对 $2BD / Y$ 字节做单轴收集。*只盯着沿 Y 轴的单个分片看，这次 AllGather 相当于一次字节数只有 1 / Y 的、不分片的 AllGather。* TPU v4p 的双向 ICI 带宽是 9e10 字节/秒，因此耗时 $2BD / (\text{9e10} \cdot Y) = 2 \cdot 1024 \cdot 4096 / (\text{9e10} \cdot 4) = 23 \mu s$。

2. 带宽翻倍，但要对整个数组做 AllGather，所以 `T = 2BD / (2 * W) = 2*1024*4096 / (2 * 9e10) = 46us`。这离延迟下限 4us（每跳 1us）很远，没问题。

3. AllReduce 的代价是 AllGather 的两倍。每片大小 $2BD / (X * Y)$，所以代价约 $4BD / (X * Y * W)$，即 `4 * 1024 * 4096 / (16 * 9e10) = 11.6us`。

*有趣的事实：* 第 (1)、(2) 问的算法其实不是最优的——数组沿没用到的 Z 轴也是复制的，这些闲置链路可以利用：先免费把 $[B_X, D_Y] \to [B_{XZ}, D_Y]$ 重新分片（每台设备直接丢掉自己一部分数据就行），再做 $$\text{AllGather}_{XZ}$$（或 $$\text{AllGather}_{XYZ}$$），到达同样的最终状态，但沿更多轴收集。这能把第 (1) 问降到 11.5us、第 (2) 问降到 31us——实践中你只要一开始就往更多轴上分片就能得到这个效果。这也是“数组要尽可能切细”的一个理由。

{% enddetails %}

**问题 3 [延迟受限的 AllGather]**：做一次 $\text{AllGather}_X([B_X])$，但 $B$ 非常小（比如 128）。在 TPU v4p 4x4x4 切片、网格 `Mesh({'X': 4, 'Y': 4, 'Z': 4})` 上，bfloat16，应该要多久？*提示：多半是延迟受限的。*

{% details 点此查看答案。 %}

这个数组 bfloat16 下总共才 256 字节，每台设备只有 64 字节。轴长为 4 的 TPU v4p 有绕回链路，可以双向发送。单向带宽 `4.5e10`，每跳传输耗时 `64 / 4.5e10 ~ 0`，所以妥妥的延迟受限。数一下跳数：整个收集只需 2 跳，所以大约 2us 是个不错的估计。

{% enddetails %}

**问题 4 [矩阵乘法的策略]**：要计算 $X[B, D] \cdot_D Y[D_X, F] \to Z[B, F]$，本节教你的做法是先 $\text{AllGather}_X(Y[D_X, F])$、再乘完整矩阵（情形 2，*策略 1*）。你也可以先本地乘：$X[B, D_X] \cdot_D Y[D_X, F] \to Z[B, F] \\{U_X\\}$（情形 3，*策略 2*），再 $\text{AllReduce}_X(Z[B, F] \\{ U_X\\})$。两种策略各做多少 FLOPs、多少通信？哪个更好，为什么？

{% details 点此查看答案。 %}

先看基准方案（*策略 1*）。如前所述，AllGather 的代价是 $2DF / W_\text{ici}$。拿到完整数组后，计算时间是 $2BDF / C$（$C$ 是加速器的 FLOPs/s，每台 TPU 做同样的 FLOPs）。于是

$$T_\text{total (Strategy 1)} = \max\left(\frac{2BDF}{C}, \frac{2DF}{W_\text{ici}}\right)$$

相比之下，新策略（策略 2）要对 $2BF$ 字节做 AllReduce，代价 $4BF / W_\text{ici}$，但 FLOPs 少做 $1 / X$（计算是分片的）。也就是说做 $2\cdot B\cdot D\cdot F / X$ 次 FLOPs，随后的 AllReduce 在 bfloat16 下通信 $$2 \cdot 2 \cdot B \cdot F$$ 字节。于是*策略 2*（没有 AllGather，最后只做 AllReduce）的总时间大致是

$$T_\text{total} = \max\left(\frac{2BDF}{X \cdot C}, \frac{4BF}{W_\text{ici}}\right)$$

问题是：*这两个哪个更大？* 当 $D / (X \cdot C) > 2 / W_\text{ici}$，即 $D / 2X > C / W_\text{ici} \approx 2550 \rightarrow X < D / (2 * 2550)$ 时，策略 (2) 是计算受限的。合理的预期是 $D \approx 8k$，那么 $X < 2$——不太可能。所以策略 2 基本上总是被通信卡住。而基准方案（策略 1）在 $$B < C / W_\text{ici} = 2550$$ 时被通信卡住，这经常但并非总是成立。

于是当 $B < 2550$ 时，两个方案都是通信受限，此时

$$T_\text{comms for Strategy 2} < T_\text{comms for Strategy 1} \Leftrightarrow \frac{4BF}{W_\text{ici}} < \frac{2DF}{W_\text{ici}}$$

在 $D > 2B$（而 $2B < 5100$）时成立。这经常成立，所以批大小不大时策略 2 有时更好。批很大（$B > 2550$）时：

$$T_\text{comms for Strategy 2} < T_\text{math for Strategy 1} \Leftrightarrow \frac{4BF}{W_\text{ici}} < \frac{2BDF}{C}$$

这在 $2 / W_\text{ici} < D / C$、即 $D > 2 * 2550 = 5100$ 时成立——对大模型通常成立。所以对大模型，这个替代策略通常是更好的，除非 $D$ 很小。

*为什么不总这么干？* 实践中有时确实会这么做，但“矩阵乘法某个输入的收缩维沿一条轴分片、另一个输入却不沿这条轴分片”这种布局本来就少见。比如做 FSDP（[第 5 部分](../training)会讲）时，参数沿数据维分片，激活*同样也沿数据维分片*。所以从这个意义上说，这个情形不常出现。

{% enddetails %}

**问题 5 [最低延迟]**：假设我要在 TPU v4p 4x4x4 上做矩阵乘法 $A[I, J] \cdot_J B[J, K] \to C[I, K]$，并希望延迟尽可能低。输入可以随意分片，但结果必须是完整复制的。输入应该怎么分片？总的 FLOPs 和通信时间是多少？

{% details 点此查看（部分）答案。 %}

这里不给完整答案，只列出 4 个最可能的方案：

1. $A[I_{XYZ}, J] \cdot B[J, K]$ + 最后做一次 AG
2. $A[I, J] \cdot B[J, K_{XYZ}]$ + 最后做一次 AG
3. $A[I, J_{XYZ}] \cdot B[J_{XYZ}, K]$ + 最后做一次 AR
4. $A[I, J] \cdot B[J, K]$（完全复制）

也可以考虑把不同轴分片到不同网格轴上，但那不太可能改变最终成本。除 (4) 外，每块 TPU 的总 FLOPs 都一样，只是通信各不相同。逐一算出通信代价、挑最低的即可。结论是 (1) 和 (2) 一样好。

{% enddetails %}

**问题 6：** 假设要在 TPU v5e 4x4 上做 $A[I_X, J_Y] \cdot_J B[J_Y, K] \to C[I_X, K]$。要做哪些通信？通信和计算各花多少时间？

* 换成 $A[I_X, J] \cdot_J B[J_X, K_Y] \to C[I_X, K_Y]$ 呢？这是训练中最标准的设置：数据并行、张量并行、ZeRO 分片三者结合。
* 换成 $A[I_X, J] \cdot_J B[J, K_Y] \to C[I_X, K_Y]$ 呢？这是推理的标准形态：纯张量并行（+数据并行）。

**问题 7：** 典型 Transformer 层有两个矩阵 $W_\text{in}[D, F]$ 和 $W_\text{out}[F, D]$，$F \gg D$。设批大小为 B，整个层就是 $In[B, D] \cdot W_\text{in}[D, F] \cdot W_\text{out}[F, D]$。取 $D=8192$、$F=32768$、$B=128$，全部 bfloat16。假设运行在 TPU v5e 2x2 切片上，并且*假设*每块 TPU 只有 300MB 空闲内存。In、$W_\text{in}$、$W_\text{out}$、Out 应该怎么分片，才能既不超过内存上限又让总时间最小？通信和 FLOPs 各花多少时间？*提示：最终输出不必完整复制，但应与输入的分片方式相同，这样这一“层”才能重复堆叠。*

{% details 点此查看（部分）答案。 %}

先看内存。两个大矩阵各占 `2 * 8192 * 32768 = 536MB`。激活 `In` 的大小是 `2 * 128 * 8192 = 2MB`（小到不用担心）。每台设备只有 300MB 空闲内存，显然必须对矩阵乘法做分片。

1. $In[B_X, D] * W_\text{in}[D_{XY}, F] * W_\text{out}[F, D_{XY}] \rightarrow Out[B_X, D]$（这就是常说的 FSDP）
2. $In[B, D_{XY}] * W_\text{in}[D, F_{XY}] * W_\text{out}[F_{XY}, D] \rightarrow Out[B, D_{XY}]$（这叫张量并行）

方案 1 很不划算：得先 AllGather 大权重或激活。方案 2 需要开头一次 AllGather、结尾一次 ReduceScatter（这比 AllReduce 便宜）。剩下的数学留作练习。

{% enddetails %}

**问题 8 [挑战题]**：用上面的代码片段做模板，分配一个分片数组，用 pmap 或 shard_map 分别给 4 种主要通信原语（AllGather、AllReduce、ReduceScatter、AllToAll）做基准测试。要用到 `jax.lax.all_gather`、`jax.lax.psum`、`jax.lax.psum_scatter` 和 `jax.lax.all_to_all`。你理解这些函数的语义吗？它们各要花多长时间？

**问题 9 [分片矩阵乘法的另一种策略？]**：我们在上面[情形 2](#情形-2-有一个乘数的收缩维被分片了)说过：矩阵乘法只有一个输入沿收缩维分片时，应当把分片的矩阵 AllGather 起来，再在本地做收缩。你可能想到的另一种做法是：先分片乘、再把结果 AllReduce（把两边都当成沿收缩维分片了），即 $A[I, J_X] *_J B[J, K] \to C[I, K]$ 通过：

1. $C[I, K] \\{ U_X \\} = A[I, J_X] \cdot B[J_X, K]$
2. $C[I, K] = \text{AllReduce}(C[I, K] \\{ U_X\\})$

回答以下问题：

1. 对矩阵 $A[N, M]$ 和 $B[M, K]$，把这个算法用下标明确写出来，写清哪台设备做了什么计算。设 $A$ 按 $A[I, J_X]$ 分片在 ND 设备上，输出要在所有设备上完整复制。
2. 如果允许最终结果不逐台复制、而是分片（沿 N 或 K 维分片），算法要怎么改？
3. 只看上面这个策略（第 2 问而非第 1 问）的通信成本：和“先 AllGather A 再做矩阵乘法”的算法相比，通信成本相差多少？

{% details 点此查看答案。 %}


1. 先算外积，结果存入 $$O[N, K]: o_{kj} = \sum_i a_{ki} b_{ij}$$。注意重复的下标并不是要被收缩的那个——我们做的是外积。这里的求和只遍历当前设备上存着的那些 i。比如收缩轴长为 16、共 4 台设备，那么设备 0 上 i 取 {0, 1, 2, 3}；设备 1 上取 {4, 5, 6, 7}；设备 2 上取 {8, 9, 10, 11}；设备 3 上取 {12, 13, 14, 15}。然后对各设备上 $O[N, K]$ 的部分和做 AllReduce，得到完整的 $O[N, K]$。
2. 第 2 步不做 AllReduce，可以用更便宜的 ReduceScatter，沿哪条轴都行：$[N, K] \\{ U_X \\} \to [N_X, K]$ 或 $[N, K] \\{ U_X \\} \to [N, K_X]$。
3. 如正文所说，带宽受限时 AllGather 的成本与 ReduceScatter 相同，都由所处理的整个矩阵的大小决定。于是“先收集再乘”的方案按 $NM$ 伸缩（因为我们 AllGather 的是 $A$）；“先乘再 ReduceScatter”的方案按 NK 伸缩（因为 ReduceScatter 的是 $O$）。两个算法的通信成本之比是 `M/K`。

{% enddetails %}

**问题 10：AllToAll 里的乐趣：** 上面的表里提到，带宽受限时 AllToAll 的时间比 AllGather 或 ReduceScatter 低 4 倍。这道题带你找出这个 4 倍从哪来，并看看如果 ICI 链路只有单向（而非双向），这个倍数会怎么变。

1. 先看单向的情形。设想 *D* 台设备连成一个环，要对 $N \times N$ 矩阵 $A[I_X, J]$ 做 AllGather 或 ReduceScatter（设 $D$ 整除 $N$）。描述这两个集合通信涉及的通信过程，算出整个算法中流经**单条** ICI 链路的标量（浮点数或整数）总数。
2. 再看 AllToAll，仍然假设 ICI 单向。这个算法和 AllGather 有什么不同？算出流经单条 ICI 链路的标量数。
3. 你会发现第 (1) 问和第 (2) 问的答案之比是个很整齐的数。用简单的话解释这个倍数从哪来。
4. 现在加入双向通信。这对 AllGather 的总时间有什么影响？
5. 双向通信对 AllToAll 的总时间又有什么影响？
6. 最后，用一句话解释双向环上 AllGather 与 AllToAll 的时间之比。

{% details 点此查看答案。 %}

(1) **解答：** 过程很简单：算法的每一步，每台设备把矩阵的一个分片“条带”（大小 $$\frac{N}{D} \times N$$ 个元素）发给自己的邻居。这样的步骤要进行 $$D-1$$ 次，因为每个分片都要传给除原产地以外的所有设备。总计每台设备转发 $$\frac{N^2(D-1)}{D}$$ 个标量，也就是单条 ICI 链路上流过这么多。

**答案：** $$N^2 (1-\frac{1}{D})$$；当 $$D >> 1$$ 时约等于 $$N^2$$。

(2) **解答：** 从通信的角度看，AllToAll 与 AllGather 的关键区别是：AllToAll 中，某台设备上存的那个分片，*不需要*把它的全部内容都发给其他每台设备。设想某台设备（叫它设备 0）存的分片是 $$[A, B, C, D]$$（这里 A,B,C,D 是矩阵，我们拿 4 台设备的环来打比方）。矩阵 $$A$$ 哪儿都不用去；$$B$$ 要去设备 1；$$C$$ 要去设备 2；$$D$$ 要去设备 3。于是第一步把 $$B$$、$$C$$、$$D$$ 发给设备 1；下一步设备 1 把 $$C$$、$$D$$ 继续转给设备 2；最后设备 2 把 $$D$$ 转给设备 3。总共传的参数个数是 $$(\text{size of A/B/C/D}) * (3 + 2 + 1)$$。A/B/C/D 的大小（这次回到一般情形）是 $$\frac{N^2}{D^2}$$，而 $$(3 + 2 + 1)$$ 一般化为 $$((D-1) + (D-2) + … + 1)$$，即 $$\frac{(D)(D-1)}{2}$$。于是单条 ICI 链路上流过的字节总数是 $$\frac{N^2(D-1)}{D \times 2}$$。

**答案：** $$\frac{N^2}{2}(1-\frac{1}{D})$$；当 $$D >> 1$$ 时约等于 $$\frac{N^2}{2}$$。

(3) **解答：** 这个倍数就是 $$\frac{1}{2}$$：单向环上，AllToAll 的代价是 AllGather/ReduceScatter 的一半。回看上面的推导，根源在于：AllGather 是“同样大小的一块乘 $$(D-1)$$ 次”，即 $$ \text{tiny block size} * (D + D + D + … + D)$$；而 AllToAll 是 $$\text{tiny block size} * (D + D-1 + D-2 + … + 1)$$。这个 2 倍差别说到底来自 $$1 + 2 + \ldots + n = n(n+1)/2$$ 这个求和公式。

(4) **解答**：双向环里，每条链路要承载的标量总数少了一半——每个“分片条带”现在可以同时往两个方向送。

(5) **解答**：这种情况下，相比单向我们一共赚了 4 倍。最容易的理解方式：看单个分片条带里那些大小 (N²/D²) 的块最终的去处，比如从设备 0 出发的那一条。单向时，一个块走 D-1 的距离、另一个走 D-2，一路排到 1；双向时，把条带拆成向左走和向右走的块，最远距离只有 floor(D/2)。对应的求和变成 $$D/2 + D/2 - 1 + D/2 - 2 + … = D/2 \cdot (D/2+1)/2$$，大 $$D$$ 极限下是 $$D^2/8$$。对比单向情形的 $$D^2/2$$，正好是 4 倍。

(6) **解答：** 单向环上我们已经看到 AllToAll 本来就比 AllGather 快 2 倍——因为我们不需要把整个条带发给每台设备。加上双向之后，我们看到 AllToAll 赚 4 倍、AllGather 只赚 2 倍。把这两个比值合起来，就得到了我们要的 4 倍。

{% enddetails %}

<h3 markdown=1 class="next-section">第 3 部分到此结束！第 4 部分讲 Transformer 的数学，[点击这里](../transformers)。</h3>
