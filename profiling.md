---
layout: distill
title: "怎么给 TPU 程序做性能剖析"
# permalink: /main/
description: "到目前为止，这个系列讲的全是理论：基于硬件屋顶线的纸面估算。这套理解很有用，但大量优化拼的是实践细节：XLA 编译器怎么工作，以及当事情不对劲时，怎么用 JAX/TensorBoard Profiler 这类剖析工具找出问题在哪。这一节就来谈这个。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 9

previous_section_url: "../applied-inference"
previous_section_name: "第 8 部分 · 服务化 LLaMA"

next_section_url: ../jax-stuff
next_section_name: "第 10 部分 · JAX 那些事"

giscus_comments: true
giscus_discussion: 11

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
  - name: "站在千尺高空看 TPU 软件栈"
  - name: "JAX Profiler：TPU 的多用途剖析器"
  - subsections:
    - name: "Trace Viewer"
    - name: "一条 XLA 算子怎么读"
    - name: "Graph Viewer"
    - name: "看一个（稍微）真实的 profile"
    - name: "Memory Profile"
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

## 站在千尺高空看 TPU 软件栈

Google 给 TPU 编程开放了一整排 API，从高层的 JAX 代码提到底层的 Pallas 或 HLO。绝大多数程序员只写 JAX 代码：它让你用类 NumPy 的写法描述抽象的线性代数程序，再由编译器自动把它们编译成在 TPU 上高效运行的代码。

看个简单例子——一个把两个矩阵乘起来的 JAX 程序：

```py
import jax
import jax.numpy as jnp

def multiply(x, y):
  return jnp.einsum('bf,fd->db', x, y)

y = jax.jit(multiply)(jnp.ones((128, 256)), jnp.ones((256, 16), dtype=jnp.bfloat16))
```

调用 `jax.jit`，就是告诉 JAX：跟踪这个函数，吐出一份叫 [StableHLO](https://openxla.org/stablehlo) 的更低层 IR（一种平台无关的机器学习中间表示），再由 XLA 编译器进一步下降（lower）成 HLO。编译器会跑很多遍（pass）来决定融合、布局等等，最后得到 JAX profile 里能看到的那份 HLO。这份 HLO 用一种 LLVM 风格的图视图，表示 JAX 代码里所有核心线性代数操作（矩阵乘、逐元素运算、卷积等等）。比如，上面程序去掉无关部分后的 HLO 长这样<d-footnote>想拿到这份 HLO，可以运行 `jax.jit(f).lower(*args, **kwargs).compile().as_text()`。</d-footnote>：

```c
ENTRY %main.5 (Arg_0.1: f32[128,256], Arg_1.2: bf16[256,16]) -> f32[16,128] {
  %Arg_1.2 = bf16[256,16]{1,0} parameter(1), metadata={op_name="y"}
  %convert.3 = f32[256,16]{1,0} convert(bf16[256,16]{1,0} %Arg_1.2),
  %Arg_0.1 = f32[128,256]{1,0} parameter(0), metadata={op_name="x"}
  ROOT %dot.4 = f32[16,128]{1,0} dot(f32[256,16]{1,0} %convert.3, f32[128,256]{1,0} %Arg_0.1), lhs_contracting_dims={0}, rhs_contracting_dims={1},
}
```

HLO 的语法马上讲。先注意到一件事：它和上面的 JAX 代码其实相当对得上。比如这一行

```c
ROOT %dot.4 = f32[16,128]{1,0} dot(f32[256,16]{1,0} %convert.3, f32[128,256]{1,0} %Arg_0.1), lhs_contracting_dims={0}, rhs_contracting_dims={1}
```

就是上面那个真正的矩阵乘：两个 f32 矩阵，各沿第一个矩阵的第 0 维、第二个矩阵的第 1 维做收缩，再相乘。

**要把这份 HLO 变成 TPU 上能执行的代码，XLA 编译器先把它下降为 LLO**（低层优化器）IR。LLO 直接给 TPU 编程：安排存储之间的搬运、把数组送进脉动阵列，等等。LLO 里有各种原语：把缓冲区送进脉动阵列、把结果取出来、调度 TPU 各块存储之间的 DMA 传输。下降成 LLO 之后，再编译成机器码，装进 TPU 的 IMEM 执行。

程序跑得比预期慢时，我们主要在 JAX 这一层动手改进性能。但这通常要求我们懂一些 HLO 的语义、了解代码在 TPU 上到底是怎么跑的。如果问题出在更底层，我们就掏出最后一个逃生舱：用 [Pallas](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html) 写自定义内核。想看一个程序的 HLO 和运行时统计，就用 JAX profiler。

## JAX Profiler：TPU 的多用途剖析器 {#jax-profiler-tpu-的多用途剖析器}

JAX 自带一个多用途 TPU 剖析器，里面一堆好用的工具，帮你搞清楚程序运行时 TPU 上发生了什么。用 `jax.profiler` 模块可以在程序运行时打点，记录从每个子过程的耗时、每个程序的 HLO、到内存占用的一切。比如这段代码会把一份 trace 写到 `/tmp/tensorboard`，可以在 TensorBoard 里查看（[这里](https://docs.jax.dev/en/latest/profiling.html#tensorboard-profiling)有一份手把手指南）：

```py
import jax
with jax.profiler.trace("/tmp/tensorboard"):
  key = jax.random.key(0)
  x = jax.random.normal(key, (1024, 1024))
  y = x @ x
  y.block_until_ready()

# Now you can load TensorBoard in a Google Colab with
#
# !pip install -U xprof
# !pip install -U protobuf
# %load_ext tensorboard
# %tensorboard --logdir=/tmp/tensorboard
#
# or externally with
#
# > tensorboard --logdir=/tmp/tensorboard
#
```

剖析器能做的事，先给个全景：

{% include figure.liquid path="assets/img/xprof-overview.png" class="img-fluid" %}

在 TensorBoard 里，有几个关键标签页帮你读懂程序：

1. **Trace Viewer**：按时间线展示 TPU 上真实发生的一切，非常细。
2. **Graph Viewer**：展示 HLO 图，能看清程序各部分谁喂给谁、东西是怎么分片的。
3. **Memory Profile 和 Memory Viewer**：展示程序用了多少内存。

profile 文件不太方便分享，不过[这里](https://ui.perfetto.dev/#!/?s=fa9f13b487bde622707c1a503f9227c34594760a)有一个 Perfetto 链接，至少包含一个简单 Transformer 的 Trace Viewer 部分。[这个 Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing)能生成完整的 JAX/TensorBoard trace，随便你玩。

### Trace Viewer

**Trace Viewer 大概是剖析器里最有用的部分。**下面这张图是一个加了标注的简单 Transformer。标注名字来自代码里写的标签。

{% include figure.liquid path="assets/img/trace-viewer.png" class="img-fluid" %}

Trace Viewer 按时间顺序列出每块 TPU 核上的所有动作。这里只看了 TPU:0——因为通常所有 TPU 执行的指令是一样的。几个要点：

1. 最上面一行（XLA Ops）是真正的 TPU 操作（名字就是 HLO 名字）。其余各行是基于 `jax.named_scope`、`jax.named_call` 和 Python 调用栈还原出来的近似轨迹。
2. 认准重复的方块，就能把单层单独拎出来。结合代码（以及对 Transformer 的了解），也能认出哪段是注意力、哪段是 MLP。
3. 点一条 XLA 算子，可以看到它来自代码的哪一行（对理解 trace 很有用），还有通往 Graph Viewer 的链接。

<p markdown=1 class="takeaway">**提示**：Trace Viewer 支持“体感游戏”式操作：A/D 左右平移，W/S 缩放。熟练之后导航快得多。</p>

### 一条 XLA 算子怎么读

HLO 其实不难读，而且它对弄清 trace 里某一段到底对应什么都很有帮助。比如有个叫 fusion.3 的算子：

```c
%fusion.3 = bf16[32,32,4096]{2,1,0:T(8,128)(2,1)S(1)} fusion(bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)} %fusion.32), kind=kCustom, calls=%all-reduce-scatter.3
```

拆开来看：

* **算子名**：fusion.3
  * 一个 dot 或 fusion 算子是一组运算，其中至多含一个矩阵乘，外加若干相关的、在 VPU 上做的逐元素运算。
* **形状（Shape）**：`bf16[32,32,4096]`
  * 这是该算子的输出形状：数据类型 bf16（每元素 2 字节），形状 `[32,32,4096]`。
* **布局（Layout）**：`{2,1,0:T(8,128)(2,1)}`
  * `{2,1,0:T(8,128)(2,1)}` 说明各轴在内存中的排列次序（列主序、行主序等）以及数组的补齐方式。下面细说。
* **内存位置：** S(1)
  * S(1) 表示这个数组住在 VMEM。S(0)（有时省略不写）是 HBM，S(2)、S(3) 是其他存储空间。
* **参数（Arguments）**：`bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)} %fusion.32`
  * 该算子只有一个输入：一个叫 fusion.32、特定形状的 bf16 数组。这告诉我们是哪个函数在给它喂料。

再把这个记号讲透一点。看个小例子：

`f32[3,5]{1,0:T(2,2)}`

它说的是：该算子返回一个 float32 数组，形状 `[3, 5]`，分块（tiling）方式是 `{1,0:T(2,2)}`。虽然分块不是最要紧的事——简单说，它描述一个 N 维数组如何顺序铺在内存里。这个数组的铺法如下图：

{% include figure.liquid path="assets/img/tiling.png" class="img-fluid" %}

`{1,0:T(2,2)}` 里，`1,0` 表示各维在物理内存中的排列顺序，从最“小”轴到最“大”轴。从右往左读这部分、对应到 `f32[3,5]` 的各维，就能推出数组的物理排布。这个例子里，物理排布就是 `[3,5]`，和逻辑形状一致。接下来，`T(2,2)` 表示数组按 `(2, 2)` 的小块铺开，每个小块内部先铺行（**行主序**）再铺列，即 `(0, 0)` 后面是 `(0, 1)`，然后 `(1, 0)`、`(1, 1)`。由于 `T(2, 2)` 分块，数组被补齐成 `[4, 6]`，内存占用约膨胀 1.6 倍。再看前面那个大 bf16 数组 `bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)}`：它的 `T(8,128)(2,1)` 表示两层分块——外层 `(8, 128)`、内层 `(2, 1)`（bf16 专用，保证每次读取都是 4 字节的整数倍）。比如 `bf16[4,8]{1,0:T(2,4)(2,1)}` 长这样（颜色是 (2,4) 块，红框是 (2,1) 块）：

{% include figure.liquid path="assets/img/tiling2.png" class="img-fluid img-small" %}

分块会影响张量各块装进 VMEM 的效率。XLA 有时会在程序里插入“改块”/“改布局”的拷贝，开销有时不小。<d-footnote>JAX 提供了一个<a href="https://docs.jax.dev/en/latest/notebooks/layout.html">实验特性</a>来绕开这个问题：让 XLA 自己算出程序输入“最喜欢”的布局。用 `jax.jit` 即时编译时，你通常要传“假”输入来告诉 JAX 期望的形状和数据类型——它们往往也带着未必最优的分块信息。你可以改为把输入布局标成 AUTO，`jax.jit` 会返回被编译程序偏好的布局，然后你按那个布局显式加载张量，程序内部就不会再被塞进额外拷贝。</d-footnote>

### Graph Viewer

上面的融合算子看着也许复杂，XLA Graph Viewer 能让它们好懂很多。比如下面是一个相当复杂的融合：

{% include figure.liquid path="assets/img/graph-viewer.png" class="img-fluid" %}

非常建议多盯着几张 HLO 图看，练习把 HLO 算子对回你的代码。把鼠标悬在方框上，常常能直接看到定义这个函数的代码行。

### 看一个（稍微）真实的 profile {#看一个-稍微-真实的-profile}

[这个 Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing)里有一个假 Transformer 的 profile。赶时间的话，[这个](https://ui.perfetto.dev/#!/?s=fa9f13b487bde622707c1a503f9227c34594760a) Perfetto 链接至少能让你看到 Trace Viewer。我特意比平时更勤快地用 `jax.named_scope` 做了标注，方便你认出每段是什么。

{% include figure.liquid path="assets/img/transformer-xprof.png" class="img-fluid" %}

先自己把这份 profile 看明白——每一段在干什么。我们一点点拆，先从 FFW 块开始：

{% include figure.liquid path="assets/img/transformer-ffw.png" class="img-fluid" %}

这里放大的是 FFW 块。你会看到上投影算子是一个 fusion（矩阵乘）：输入 `bf16[8, 1024, 8192]` 和 `bf16[8192, 16384]`，输出 `bf16[8, 1024, 16384]`。因为代码是我写的，我知道这是一个“4 路数据并行 + 2 路模型并行”分片矩阵乘的局部视角，真实在做的是

**X:** `bf16[32, 1024, 8192]` \* **W<sub>in</sub>**: `bf16[8192, 32768]` -> **Tmp**: `bf16[32, 1024, 32768]`

**我们预期它该跑多久？**首先，每个数据并行分片的批是 `8 * 1024 = 8192`，所以稳处计算受限区。这在 8 个 TPU v2 核上跑，预期耗时约 `2 * 32 * 1024 * 8192 * 32768 / (23e12 * 8) = 95.6ms`——和实测（96ms）几乎分毫不差。漂亮！这说明 FLOPs 利用率非常高！

注意：Google Colab 现在不再发放 TPU v2-8 切片了。想亲手跟做一遍，可以去 [Kaggle](https://www.kaggle.com/)（目前还免费提供）或者在 GCP 上开一个 8 核切片。<d-footnote>如果只想在假问题上玩分片，也可以在 CPU 上“假装”有 8 台设备：`import jax; jax.config.update("jax_num_cpu_devices", 8)`（需要 jax >= 0.4.27 左右），然后 `print(jax.devices())`。这只适合玩具问题，性能不代表真实情况。</d-footnote>

**通信呢？**你会注意到第二个矩阵乘末尾藏着一个小 fusion。点开它：

```c
%fusion.1 = bf16[8,1024,4096]{2,1,0:T(8,128)(2,1)} fusion(bf16[8,1024,8192]{2,1,0:T(8,128)(2,1)} %fusion.31), kind=kCustom, calls=%all-reduce-scatter.1
```

这基本上就是一个 ReduceScatter（这是它的 Graph Viewer）：

{% include figure.liquid path="assets/img/reduce-scatter-xprof.png" class="img-fluid" %}

我们预期它跑多久？这是在 TPU v2 4x2 上做 ReduceScatter，只需一跳，双向带宽 1.2e11。数组大小 `2*32*1024*8192`，批维切了 4 份，每个分片是 `2*8*1024*8192=128MB`。所以大约 1.1ms。**实际呢？**profile 里报告 1.13ms。几乎是贴着屋顶线跑的！

**注意力也看一眼！**这是注意力部分的 profile：

{% include figure.liquid path="assets/img/attn-xprof.png" class="img-fluid" %}

我点选了 Q 投影算子，它用的矩阵是 $$W_Q$$，形状 [d<sub>model</sub> = 8192, n<sub>heads</sub> = 32, d<sub>qkv</sub> = 256]，沿头维做了 Megatron 分片。请照上面的方法也算一遍这些该花多久。

### Memory Profile

Memory Profile 能直观看到程序内存占用随时间的曲线，调试内存溢出（OOM）时很好用。这张图里可以看到约 7.5GB 被模型参数占用，还有约 8.5GB 空着——说明还能往里塞不少东西。

{% include figure.liquid path="assets/img/memory-viewer.png" class="img-fluid" %}

## 动手解题

**问题 1**：看看[这个](https://colab.research.google.com/drive/1LfLO3OTr-_MWFPxUN36KJ3cqH0BcAoli?usp=sharing) Colab/profile，找出哪里不对劲、到底发生了什么。你能说出这里具体在做哪些计算、每个操作在干什么吗？涉及的每个矩阵真实形状是什么、怎么分片的？*先只看 profile，别看代码。*

{% include figure.liquid path="assets/img/all-reduce-profile.png" class="img-fluid" %}

{% details 点此查看答案。 %}

这是两次矩阵乘，具体来说就是这段：

```py
def matmul(w1, w2, x):
  return jnp.einsum('wf,bf->bw', w2, jnp.einsum('fw,bw->bf', w1, x))
```

图上能看到一个 reduce、两个大 fusion、一次 all-reduce。第一个大 fusion 是：

```%fusion.1 = bf16[4096]{0:T(1024)(128)(2,1)} fusion(bf16[4096,8192]{1,0:T(8,128)(2,1)} %param.1, bf16[8192]{0:T(1024)(128)(2,1)} %reduce.6), kind=kLoop, calls=%fused_computation.1```

可见每分片形状是 `bf16[8192] * bf16[4096, 8192] -> bf16[4096]`（沿 8192 维收缩）。再看最后那个 AllReduce 的 {% raw %}`replica_groups={{0,16,32,48,64,80,96,112}, ...}`{% endraw %}，就知道这是 8 路模型并行。真实形状是 `bf16[8, 8192] * bf16[32768, 8192] -> bf16[8, 32768]`。

{% enddetails %}

**问题 2：**[前面那个 Transformer Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing)实现了一个简易的假 Transformer。由于 Colab 不再提供 TPU v2-8 切片，请在 [Kaggle](https://www.kaggle.com/) 或 GCP 的 8 核切片上跟做。按 Colab 里的说明，拿到“GSPMD 自动切分的朴素 Transformer”的基准：每部分各花多久？本该花多久？用的是哪种分片？试着把分片改对！*提示：用 `jax.lax.with_sharding_constraint` 来约束行为。改好之后，你能做到多好的 MFU？*

供参考：初始版本约 184ms/层，优化后的 profile 是 67ms/层。做完之后，再盯着 profile 看，试试纯凭 profile 回答：

- 这是什么分片策略？
- 批大小、$$d_\text{model}$$、$$d_\text{ff}$$ 各是多少？
- 注意力和 MLP 块各占多少时间？
- 按屋顶线，每个操作*应该*占多少时间？

**注**：这道题写好后，XLA 编译器又进步了。初始版本现在约 90ms/层，优化后的 profile 也只好了约 10ms/层（80ms/层）。即便如此，仍然值得一玩——看看你能不能做得更好。

<h3 markdown=1 class="next-section">第 9 部分到此结束！第 10 部分深入 JAX 的并行机制，[点击这里](../jax-stuff)。</h3>
