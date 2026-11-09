---
layout: distill
title: "结语与延伸阅读"
# permalink: /main/
description: "谢谢阅读！这里再附上一些可供进一步学习的资料。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 11

previous_section_url: "../jax-stuff"
previous_section_name: "第 10 部分 · JAX 那些事"

next_section_url: "../gpus"
next_section_name: "第 12 部分 · GPU"

giscus_comments: true
giscus_discussion: 13

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
  - name: "致谢"
  - name: "延伸阅读"
  - name: "反馈"

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
  .algorithm {
    padding: 10px;
    margin-top: 5px;
    margin-bottom: 5px;
    border-style: dashed;
    background-color: #fffaf2;
  }

  .algorithm li {
    margin-bottom: 0px;
  }
---

**谢谢你一路读到这里，恭喜你走完了全程。**收尾之前，先说几句致谢：

## 致谢

这份文档是 Google DeepMind 许多人共同投入的心血，请允许我们在此简短地感谢他们！

- James Bradbury、Reiner Pope、Noam Shazeer 和 Blake Hechtman 最早推导出了本书的许多核心想法，也很早就理解了“用系统的眼光看 Transformer”。
- Sholto Douglas 写了本书的第一版，是项目的发起人。就全书的叙事主线而言，最大的功劳属于他。
- Jacob Austin 主导了把第一版从零散笔记打磨成更完善、更全面成品的工程。编辑、排版、发布的大量工作由他完成，其他作者的贡献也由他协调。
- 大部分图和动画出自 Anselm Levskaya 和 Charlie Chen。
- Charlie Chen 撰写了推理部分，并画了其中许多插图。
- Roy Frostig 在出版、校对和沿途许多环节都帮了大忙。

也要感谢在这个过程中提出宝贵意见的许多其他人，特别是 Zak Stone、Nikhil Sethi、Caitlin Stanton、Alek Dimitriev、Sridhar Lakshmanamurthy、Albert Magyar、Diwakar Gupta、Jeff Dean、Corry Wang、Matt Johnson、Peter Hawkins 等多位朋友。感谢 Ruiqi Gao 在 HTML 排版上的帮助。

**谢谢大家！**

<p markdown=1 class="announce">走之前，不妨再去读读讲 NVIDIA GPU 的新篇章[第 12 部分](../gpus)！</p>

## 延伸阅读

相关的优质材料不少，包括以下这些：

- [**TPU Deep Dive**](https://henryhmko.github.io/posts/tpu/tpu.html)：与本书精神相通，把 TPU 架构讲得深入又精彩。
- [**Domain specific architectures for AI inference**](https://fleetwood.dev/posts/domain-specific-architectures)：同样是硬件与模型双线深入的佳作。
- [**A Domain-Specific Supercomputer for Training Deep Neural Networks**](https://dl.acm.org/doi/pdf/10.1145/3360307)：最早一批 TPU 论文之一，讲了大量 Google TPU 计划的精彩细节，书中未曾覆盖。
- [**Making Deep Learning Go Brrrr From First Principles**](https://horace.io/brrr_intro.html)：更偏 GPU 和 PyTorch 的 LLM 屋顶线与性能工程入门。
- [**Writing TPU Kernels with Pallas**](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html)：如今写 TPU 代码越来越离不开用 Pallas 写自定义内核了。这篇讲了怎么写内核，以及许多本书没提的 TPU 底层细节。
- [**How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog**](https://siboehm.com/articles/22/CUDA-MMM)：虽是 GPU/CUDA 专属，但这是一篇出色的工作记录：看一个 CUDA 矩阵乘内核如何一步步优化到位。想深入理解 TPU 与 GPU 的差别，这是好材料。
- [**Distributed arrays and automatic parallelization**](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)：JAX 并行 API 的极佳指南，想动手实现本书讲的想法，从这里入手最好。
- [**Rafi Witten's High Performance LLMs 2024 Class**](https://github.com/rwitten/HighPerfLLMs2024)：我们的前同事 Rafi 开的一门 TPU 性能工程好课，讲义全在 GitHub 上，许多内容比本书讲得更深。
- [**\[2211.05102\] Efficiently Scaling Transformer Inference**](https://arxiv.org/abs/2211.05102)：讲 Transformer 推理数学的详实论文，本书的很多内容正由它启发。
- [**Huggingface Ultra-Scale Playbook**](https://huggingface.co/spaces/nanotron/ultrascale-playbook)：某种意义上 GPU 版的“本书”，更细致地讲 PyTorch 在训练时如何实现各种并行与省内存技术。
- [**Transformer Inference Arithmetic**](https://kipp.ly/transformer-inference-arithmetic/)：一篇与本书想法高度一致的博文，配图画得极好。
- [**Stanford CS336 Slides and Videos**](https://stanford-cs336.github.io/spring2025/index.html#coursework)：斯坦福的一门出色课程，覆盖 LLM 训练与服务的诸多细节，习题很有用——作业 1 和作业 2 尤其切题。
- [**Stas Bekman's ML Engineering Handbook**](https://github.com/stas00/ml-engineering)：一本极其实用的 ML 基础设施手册，讲的都是本书没顾上的话题：怎么和云厂商打交道、怎么管理集群、GPU 吞吐实测数字。
- [**ezyang's blog**](https://blog.ezyang.com/2026/01/computing-sharding-with-einsum/)：一篇围绕分片 + PyTorch 的博客，包括一篇 [PyTorch 内部机制指南](https://blog.ezyang.com/2019/05/pytorch-internals/)和一篇[分片矩阵乘笔记](https://blog.ezyang.com/2026/01/computing-sharding-with-einsum/)，此外还有许多好东西。
- [**The Anatomy of Collective Communication**](https://www.aleksagordic.com/blog/collective-operations)：与本书风格相通的 GPU/TPU 集合通信详解，其中 N 维网格与 GPU 集合通信部分讲得比本书更好。

这个领域依然缺少成体系的写作，希望本书能带动更多人动笔。我们也相信这是很值得钻研和研究的领域——很多时候，手头没有多少加速卡也照样能研究。

## 反馈

欢迎留言提问，帮助我们继续改进。你可以联系本文通讯作者 Jacob Austin，邮箱 jacobaustin123 [at] gmail [dot] com；也可以[在 GitHub 上](https://github.com/jax-ml/scaling-book)提 issue、发 pull request 或开讨论，直接建议修改。
