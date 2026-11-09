---
layout: distill
title: "How to Scale Your Model"
subtitle: "从系统视角看大模型如何在 TPU 上做大"
# permalink: /main/
description: "训练大模型常常像在“炼丹”，但想搞懂它、把模型性能调好，其实不必靠玄学。这本书想把“把语言模型做大”这件事讲明白：TPU（和 GPU）是怎么工作的、它们彼此之间怎么通信，大模型在真实硬件上到底是怎么跑起来的，以及在训练和推理时该怎么并行，才能在超大规模里跑得又稳又省。如果你也好奇过“这个模型训练得花多少钱”“我自己要跑这个模型得配多少内存”“AllGather 到底是个啥”，我们希望这本书能帮到你。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

giscus_comments: true
giscus_discussion: 2

section_number: 0

previous_section_url: ""
previous_section_name: "第 0 部分 · 引言"

next_section_url: roofline
next_section_name: "第 1 部分 · Roofline 入门"

bibliography: main.bib

citation: true

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
  - name: 内容大纲
  - name: 各节链接

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
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

{% include figure.liquid path="assets/img/dragon.png" class="img-fluid" %}

深度学习在很多情况下仍然像一门“玄学”，但想让模型跑得更快，其实不必靠玄学——哪怕规模再大也一样！道理其实很简单，而且到处都通用：不管你用的是一块加速卡，还是成千上万块。把这些道理搞明白，你就能做到很多有用的事：

- 估一估：你的模型各个部分，离“理论上能达到的最好效果”还差多远。
- 做选择：在不同的规模下，有底气地挑一种“并行方案”（也就是怎么把计算切开、分给许多台设备一起算）。
- 算成本：训练和运行一个大号 Transformer 模型，大概要花多少钱、多少时间。
- 设计算法：专门设计那些能发挥[特定](https://arxiv.org/abs/2205.14135) [硬件](https://arxiv.org/abs/1911.02150) [长处](https://arxiv.org/abs/2007.00072)的算法。
- 设计硬件：反过来，从“现在这些算法到底卡在哪里”出发，去设计硬件。

**需要哪些背景：** 我们假设你已经大致了解大语言模型（LLM）和 Transformer 架构，但不一定清楚它们在超大规模下是怎么跑起来的。你最好懂一点大模型训练的基础，并且对 JAX 有初步了解。如果想补补课，可以先读读这篇讲解 Transformer 架构的[博客](https://jalammar.github.io/illustrated-transformer/)，以及[最初的 Transformer 论文](https://arxiv.org/abs/1706.03762)。还可以看看[这份清单](conclusion#further-reading)，里面有更多有用的、正在写和将要写的阅读材料。

**这本书的目标，以及欢迎你反馈：** 读完之后，你应该能比较有信心地判断：在某个硬件平台上，给一个 Transformer 模型选哪种并行方案最好；训练和推理大概分别要花多久。如果你觉得还没看明白，欢迎给我们发邮件，或者在页面下面留言。我们很想知道怎样才能讲得更清楚。

<p markdown=1 class="announce">你可能也会喜欢我们新写的[第 12 节](gpus)——它讲的是 NVIDIA GPU！</p>

### 为什么要学这些？

三四年前，我觉得大多数机器学习研究者都用不着懂这本书里的东西。可到了今天，就算是“小”模型，也已经贴着硬件的极限在跑了；想做出新东西，你就必须认真考虑“在大规模下怎么跑得高效”。<d-footnote>长期以来，机器学习研究在“系统创新”和“软件改进”之间像钟摆一样来回摆。Alex Krizhevsky 当年为了把 CNN 跑快，不得不写极其难懂的 CUDA 代码；但没过两年，有了 Theano、TensorFlow 这些库，你就不用自己写了。也许这里也一样，几年之后这本书里的东西会被彻底封装掉、不用你操心。但“缩放定律（scaling laws）”总是把我们的模型一次次推到硬件的最前沿。所以可以预见，在可预见的未来，做最前沿的研究，和“怎么把模型高效地铺到大规模硬件上”这件事是分不开的。</d-footnote> **如果跑分赢了 20%，却为此让 roofline 效率也损失 20%，那点优势毫无意义。** 很多很有前途的新模型结构最后都失败了——要么因为它们根本没法高效地大规模运行，要么因为没人愿意花力气把它们跑高效。

**“模型扩展（model scaling）”的目标是：当训练或推理所用的芯片数量增加时，吞吐量（单位时间的处理量）也能按比例、线性地增长。** 这叫作“*强扩展（strong scaling）*”。多加芯片（也就是“并行”）通常能缩短计算时间，但代价是芯片之间要传更多数据。一旦“传数据”比“做计算”还慢，我们就变成了“通信受限”，也就没法继续强扩展了。<d-footnote>随着计算时间变短，你通常还会撞上“单块芯片内部”的瓶颈。你手里这块崭新的 TPU 或 GPU，标称可能每秒能做 500 万亿次运算；可如果它成天忙着在内存里搬来搬去参数、被拖住手脚，它可能只能跑出十分之一的速度。所以，单块芯片的算力、内存带宽、内存容量这三者如何配合，是决定能不能扩展的关键。</d-footnote> 只要我们对硬件了解得足够透，能提前预判这些瓶颈会出现在哪里，就能设计或重新调整模型来避开它们。<d-footnote>硬件设计者面对的正好相反：要在尽量省钱的前提下，做出“刚好够用”的硬件，给算法提供刚好的算力、带宽和内存。你能想象这种“共同设计（co-design）”有多难：你得赌上“等第一代芯片真正造出来时，算法会长什么样”——而那往往是两三年之后的事。TPU 的故事，在这场豪赌里算是大获全胜。矩阵乘法很特别：几乎没有什么算法像它那样，每搬运 1 字节内存要做这么多次运算（每字节 N 次运算），所以早期的 TPU 靠它的“脉动阵列（systolic array）”结构，性价比远高过当时同代的 GPU。TPU 一开始就是为机器学习设计的；而带 Tensor Core 的 GPU 也在飞速改造，努力抢下这块地盘。但你也大可想象，如果神经网络当年没有火起来，或者发生了某些 TPU 应对不了的根本变化（TPU 天生不如 GPU 灵活），那赌错的代价会有多大……</d-footnote>

*我们在这本书里的目标，是讲清楚 TPU（和 GPU）这类硬件是怎么工作的，以及 Transformer 架构是怎么一步步演化成如今这样、能在现有硬件上跑得很好的。我们希望，无论是设计新模型的研究者，还是想让这一代大模型跑得更快的工程师，都能从中受益。*

## 内容大纲

本书的整体结构如下：

[第 1 节](roofline) 会讲“roofline 分析”，以及是什么在限制我们扩展——通信、计算和内存。[第 2 节](tpus) 和 [第 3 节](sharding) 会详细讲 TPU 是怎么工作的：既讲单块芯片，更讲——这一点极其重要——很多块芯片连成一体、由带宽和延迟都有限的链路互联起来，构成的“系统”。我们会回答这些问题：

* 一个特定大小的矩阵乘法，该花多久？什么时候是卡在算力上，什么时候是卡在内存或通信带宽上？
* 很多块 TPU 是怎么连成一个训练集群的？系统的各个部分各有多大的带宽？
* 要把数据在多块 TPU 之间收集（gather）、分发（scatter）或重新排布（re-distribute），要花多久？
* 当矩阵在各台设备上的切法不一样时，我们怎么高效地把它们相乘？

{% include figure.liquid path="assets/img/pointwise-product.gif" class="img-small" caption="<b>图：</b>一张出自 <a href='tpus'>第 2 节</a> 的示意图，展示一块 TPU 是怎么做逐元素相乘的。取决于数组的大小、以及各种链路的带宽，我们可能变成计算受限（把硬件算力全用上了），也可能变成内存受限（被数据加载卡住）。" %}

五年前，机器学习里模型结构五花八门——ConvNet、LSTM、MLP、Transformer——但现在我们基本只用 Transformer<d-cite key="transformers"></d-cite>。我们强烈相信，Transformer 的每一块都值得彻底弄明白：每个矩阵到底多大、归一化（normalization）放在哪里、每个部分有多少参数、多少 FLOP<d-footnote>FLOP 即“浮点运算次数”，说白了就是要做的加法和乘法总共多少次。很多资料用 FLOPs 表示“每秒运算次数”；我们则专门用 FLOPs/s 来明确表示“每秒”。</d-footnote>。[第 4 节](transformers) 会仔细走一遍这套“Transformer 数学”，教你怎么数出训练和推理各自的参数量和 FLOPs。这会告诉我们：模型要吃掉多少内存、多少时间花在计算、多少花在通信，以及“注意力（attention）”什么时候会变得和“前馈模块（feed-forward）”一样重要。

{% include figure.liquid path="assets/img/transformer-diagram.png" class="img-fluid" caption="<b>图：</b>一个标准的 Transformer 层，每一次矩阵乘法（matmul）都画成圆圈里的一个小圆点。除归一化外的所有参数都标成紫色。<a href='transformers'>第 4 节</a> 会更详细地讲解这张图。" %}

[第 5 节：训练](training) 和 [第 7 节：推理](inference) 是本书的核心。我们会讨论那个最根本的问题：给定一个这么大的模型、这么多块芯片，我该怎么并行，才能一直待在“强扩展”的区间里？这个问题问得简单，答案却出奇地复杂。往大了说，把模型切到多块芯片上，主要有四种并行方法：**数据并行（data）**、**张量并行（tensor）**、**流水并行（pipeline）** 和**专家并行（expert）**；此外还有不少能降低内存占用的技巧：**重计算（rematerialization）**、**优化器/模型分片（也就是 ZeRO）**、**卸载到主机内存（host offload）**、**梯度累积（gradient accumulation）**。这些我们大部分都会讲到。

希望读完这些之后，面对新的模型或场景，你也能自己从中做出选择。[第 6 节](applied-training) 和 [第 8 节](applied-inference) 是实操教程，会把这些概念用到一个很流行的开源模型 LLaMA 3 上。

最后，[第 9 节](profiling) 和 [第 10 节](jax-stuff) 会讲怎么用 JAX 实现上面的一些想法，以及在程序出问题时怎么做性能剖析（profiling）、怎么调试。[第 12 节](gpus) 是新加的一节，专门讲 GPU。

我们会尽量在每个部分都留一些练习题，给你自己动手。请别有压力，不必把所有节都读完，也不必按顺序读。也欢迎你多提反馈。目前这本书还是草稿，会不断修订。谢谢！

*我们要感谢 James Bradbury 和 Blake Hechtman，书里很多想法都是他们推导出来的。*

<h3 markdown=1 class="next-section">事不宜迟，[这就去看第 1 节](roofline)，讲 TPU 的 roofline。</h3>

## 各节链接

*这套书也许比必要的篇幅要长一些，但我们希望这不会吓退你。开头三章是预备知识：如果你已经熟悉，可以跳过——不过它们会引入后面要用的记号。最后三部分也许最有用，因为它们讲的是怎么在真实模型上动手。*

**第一部分：预备知识**

* [**第 1 章：roofline 分析简明入门**](roofline)。一个算法会被三样东西卡住：计算、通信、内存。我们用这三样来估算算法能跑多快。

* [**第 2 章：怎么理解 TPU**](tpus)。TPU 是怎么工作的？这又如何决定了我们能训练、能部署什么样的模型？

* [**第 3 章：分片矩阵，以及怎么把它们相乘**](sharding)。我们用一个自己最爱的运算——（分片）矩阵乘法——来讲清楚什么是模型分片和多 TPU 并行。

**第二部分：Transformer**

* [**第 4 章：Transformer 数学，你需要的都在这里**](transformers)。Transformer 在前向和反向传播里各要花多少 FLOPs？你能算出它有多少参数吗？它的 KV 缓存有多大？这套数学我们在这里一步步算清楚。

* [**第 5 章：训练 Transformer 时怎么做并行**](training)。FSDP、Megatron 分片、流水并行。给定这么多块芯片，我该怎么用最省的方式、以给定的批大小，训练一个给定的模型？

* [**第 6 章：在 TPU 上训练 LLaMA 3**](applied-training)。如果要在 TPU 上训练 LLaMA 3，该怎么做？要多久？要花多少钱？

* [**第 7 章：Transformer 推理全解**](inference)。模型训练好了，就得把它跑起来对外服务。推理多了一个新考量——延迟（latency）——内存的账也要重新算。我们会讲“预填充和生成分离”的服务是怎么做的，以及该怎么理解 KV 缓存。

* [**第 8 章：在 TPU 上部署 LLaMA 3**](applied-inference)。在 TPU v5e 上部署 LLaMA 3 要花多少钱？延迟和吞吐之间该怎么权衡？

**第三部分：实操教程**

* [**第 9 章：怎么对 TPU 程序做性能剖析**](profiling)。真实的 LLM 从来不像上面理论那么干净。这里我们讲 JAX + XLA 这套技术栈，以及怎么用 JAX/TensorBoard 性能剖析工具去排查、修好真实的问题。

* [**第 10 章：用 JAX 给 TPU 编程**](jax-stuff)。JAX 提供了一堆把计算并行化的“魔法”接口，但你得会用。有意思的例子和动手练习。

**第四部分：结论与彩蛋**

* [**第 11 章：结论与延伸阅读**](conclusion)。收尾的思考，以及关于 TPU 和大模型的更多阅读材料。

* [**第 12 章：怎么理解 GPU**](gpus)。一个讲 GPU 的加餐：GPU 怎么工作、怎么连成网络，它的 roofline 和 TPU 有什么不同。
