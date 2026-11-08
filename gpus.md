---
layout: distill
title: "怎么理解 GPU"
description: "Google 爱用 TPU，但 GPU 同样很棒。这一章带你深入 GPU 的世界——每块芯片怎么工作、它们如何连成网络、这对 LLM 意味着什么，尤其是和 TPU 相比会怎样。NVIDIA、AMD、Intel 等厂商的 GPU 架构多如牛毛，本章只谈 NVIDIA GPU。本节建立在<a href='https://jax-ml.github.io/scaling-book/tpus/'>第 2 章</a>和<a href='https://jax-ml.github.io/scaling-book/training'>第 5 章</a>之上，建议先读那两章。"
date: 2025-08-18
future: true
htmlwidgets: true
hidden: false

section_number: 12

previous_section_url: "../conclusion"
previous_section_name: "第 11 部分 · 结语"

next_section_url:
next_section_name: "全书完"

bibliography: main.bib

giscus_comments: true
giscus_discussion: 66

authors:
  - name: Jacob Austin<sup>†</sup>
    url: "https://www.jacobaustin.org/"
    affiliations:
      name: <sup>†</sup>Google DeepMind
  - name: Swapnil Patil<sup>†</sup>
    url: "https://www.linkedin.com/in/swapnil-patil-5b47a068"
  - name:  Adam Paszke<sup>†</sup>
    url: https://x.com/apaszke
  - name: Reiner Pope<sup>*</sup>
    url: https://x.com/reinerpope
    affiliations:
      name: <sup>*</sup>MatX

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "GPU 是什么？"
  - subsections:
    - name: "内存"
    - name: "GPU 规格总表"
    - name: "芯片层面的 GPU 与 TPU 对比"
    - name: "练习 1：GPU 硬件"
  - name: "组网"
  - subsections:
    - name: "节点内的组网"
    - name: "练习 2：GPU 节点"
    - name: "节点之外的组网"
    - name: "练习 3：节点之外"
  - name: "GPU 上的集合通信是怎么跑的？"
  - subsections:
    - name: "节点内集合通信"
    - name: "跨节点集合通信"
    - name: "练习 4：集合通信"
  - name: "LLM 在 GPU 上扩展的屋顶线"
  - subsections:
    - name: "数据并行"
    - name: "张量并行"
    - name: "专家并行"
    - name: "流水线并行"
    - name: "例子"
    - name: "一句话总结：LLM 在 GPU 上怎么扩"
    - name: "练习 5：LLM 屋顶线"
  - name: "致谢与延伸阅读"
  - name: "附录"
  - subsections:
    - name: "附录 A：换成 GB200 会怎样？"
    - name: "附录 B：更多组网细节"

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

## GPU 是什么？

一块现代 ML GPU（比如 H100、B200），说白了就是一堆专门做矩阵乘的计算核心（叫**流式多处理器**，Streaming Multiprocessor，简称 **SM**）接在一根快速内存（叫 **HBM**）上。图示如下：

{% include figure.liquid path="assets/gpu/gpu-diagram.png" class="img-fluid" link="true" caption="<b>图：</b>H100/B200 GPU 的抽象布局示意图。H100 有 132 个 SM，B200 有 148 个。我们宽泛地使用“Warp 调度器”一词，指代一组 32 个 CUDA SIMD 核心<i>加上</i>给它们派发任务的调度器。看看这像不像一块 TPU！" %}

每个 SM 和 TPU 的 Tensor Core 一样，有专属的矩阵乘核心（不巧也叫 **Tensor Core**<d-footnote>GPU 的 Tensor Core 是 SM 里的矩阵乘子单元；TPU 的 TensorCore 则是包含 MXU、VPU 等部件的大组件。名字撞了，认命吧。</d-footnote>）、一个向量运算单元（叫 **Warp 调度器**<d-footnote>NVIDIA 没给这东西起好名字，我们姑且用这个“矬子里拔将军”的叫法。Warp 调度器本义是给一组 CUDA 核心派活的单元，这里我们把控制单元连同它管的那组核心一起这么叫。</d-footnote>）、一块快速的片上缓存（叫 **SMEM**）。TPU 至多只有 2 个独立的 “Tensor Core”，而现代 GPU 有 100 多个 SM（H100 有 132 个）。单个 SM 远不如一个 TPU Tensor Core 能打，但整套系统更灵活。各 SM 之间几乎完全独立，所以一块 GPU 可以同时干几百件互不相干的事。<d-footnote>SM 虽然独立，但因为大家共享一块容量有限的 L2 缓存，为了跑满性能往往还是被迫“步调一致”。</d-footnote>

放大看看 H100 的 SM：

{% include figure.liquid path="assets/gpu/blackwell-sm.png" class="img-small" link="true" caption="<b>图：</b>H100 的 SM 示意图（<a href='https://wccftech.com/nvidia-hopper-gh100-gpu-official-5nm-process-worlds-fastest-hpc-chip-80-billion-transistors-hbm3-memory/'>来源</a>）：4 个相同的<b>子分区</b>，每个含一个 Tensor Core、一个 Warp 调度器、一个寄存器堆、若干不同精度的 CUDA 核心。下方标注 “L1 Data Cache” 的就是 256kB 的 SMEM。B200 长相类似，但为“喂饱”更大的 Tensor Core，额外加了一大块张量内存 TMEM。" %}

每个 SM 均分为 4 个象限——NVIDIA 叫 **SM 子分区**——每个含一个 Tensor Core、16k 个 32 位寄存器、以及一个 SIMD/SIMT 向量运算单元 Warp 调度器（它的计算车道即 ALU，NVIDIA 叫 **CUDA 核心**）。分区里最核心的部件当属 Tensor Core：矩阵乘全靠它，分区里绝大部分 FLOPs/s 也来自它——但它不是唯一值得注意的部件。

* **CUDA 核心：**每个子分区有一组做 SIMD/SIMT 向量运算的 ALU，即 CUDA 核心。每个 ALU 一般每周期做 1 次运算，比如 f32.add<d-footnote>新架构支持 FMA（乘加融合）指令，严格说每周期做 2 个 FLOPs——NVIDIA 毫不容情地靠这一点把标称规格直接翻倍。</d-footnote>。每个子分区有 32 个 fp32 核心（int32 和 fp64 核心少一些），同一周期所有核心执行同一条指令。和 TPU 的 VPU 一样，CUDA 核心负责 ReLU、逐元素向量运算和归约（求和）这类活。<d-footnote>早年间（Tensor Core 出现之前），CUDA 核心才是 GPU 的主角，干的是渲染：射线-三角形求交、着色……今天的游戏卡上，渲染主力依然是它们；Tensor Core 则用于超分（DLSS）——先用低分辨率渲染（像素少=活少），再用机器学习放大。</d-footnote>

* **Tensor Core（TC）：**每个子分区自带一个 Tensor Core，是和 TPU MXU 同类的专用矩阵乘单元。GPU 的 FLOPs/s 绝大部分出自 Tensor Core（比如 H100 上 Tensor Core 有 990 bf16 TFLOP/s，CUDA 核心只有 66 TFLOPs/s）。
  * [990 bf16 TFLOPs/s](https://www.nvidia.com/en-us/data-center/h100/)、132 个 SM、1.76GHz 主频，算下来每个 H100 Tensor Core 每周期能做 `7.5e12 / 1.76e9 / 4 ~ 1024` 个 bf16 FLOPs——大约一个 8x8x8 的矩阵乘<d-footnote>NVIDIA 不公开 Tensor Core 的硬件细节，所以这更像推测而非定论——至少说明不了 TC 是怎么实现的。已知 V100 是 256 FLOPs/TC/周期，A100 是 512，H100 是 1024；B200 虽未公布，`2250e12 / (148 * 4 * 1.86e9)` 约等于 2048，很可能就是 2048。另有若干旁证见<a href='https://forums.developer.nvidia.com/t/how-to-calculate-the-tensor-core-fp16-performance-of-h100/244727'>这里</a>。</d-footnote>。
  * 和 TPU 一样，GPU 做低精度矩阵乘更快（比如 H100 的 fp8 FLOPs/s 是 fp16 的 2 倍）。低精度训练或服务能快上很多。
  * 自 Volta 以来，每一代 GPU 都在把 TC 做大（[这篇讲得很好](https://semianalysis.com/2025-06-23/nvidia-tensor-core-evolution-from-volta-to-blackwell/)）。到 B200，TC 大到 SMEM 都装不下它的输入了——于是 B200 新增了一块叫 TMEM 的存储空间<d-footnote>Ampere 时代，喂饱一个 Tensor Core 只要一个 warp；Hopper 需要整个 SM（warpgroup）；Blackwell 要两个 SM 一起喂。Blackwell 的矩阵乘也大到了参数（尤其是累加器）放不进寄存器/SMEM 的地步，于是加了 TMEM。</d-footnote>。

**CUDA 核心比 TPU 的 VPU 更灵活：**自 V100 起，GPU 的 CUDA 核心使用所谓 SIMT（*Single Instruction Multiple Threads*，单指令多线程）编程模型，对比之下 TPU 用的是 SIMD（*Single Instruction Multiple Data*，单指令多数据）。和 TPU VPU 里的 ALU 一样，同一子分区内的 CUDA 核心每周期必须执行同一种操作（一个核心在加两个浮点数，同分区所有核心就都在加）。但与 VPU 不同，每个 CUDA 核心（按 CUDA 的说法叫每个“线程”）有自己的指令指针，可以*独立编程*。同一个 warp 里两个线程被要求干不同的活怎么办？结果是两个活都干了——不需要干其中一件的那些核心被掩码盖住空转。

{% include figure.liquid path="assets/gpu/warp-divergence.png" class="img-fluid" caption="<b>图：</b>一组线程内出现 warp 分叉的例子（<a href='https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf'>来源</a>）。白色空白处表示一部分物理 CUDA 核心在停滞。" %}

这让线程级的编程很灵活，代价是：warp 频繁分叉时，性能会悄悄变差。线程能碰的内存也更自由：VPU 只能对连续内存块下手，而 CUDA 核心可以按线程访问共享寄存器里的单个浮点数、还能保留线程私有状态。

**CUDA 核心的调度也更灵活：**SM 跑起来有点像多线程 CPU——它可以“挂起”许多程序（**warp**）同时驻留（每个 SM 最多 64 个），只是每个 _Warp 调度器_ 每个时钟周期只执行一个程序<d-footnote>驻留在某个 SM 上的 warp 就叫 “resident”（常驻）。</d-footnote>。Warp 调度器会在活跃的 warp 之间自动切换，把内存读取这类 I/O 藏起来。相比之下，TPU 基本是单线程的。

### 内存

计算单元之外，GPU 有一整套内存层级：最大的是 HBM（主内存），往下是一层层更小的缓存（L2、L1/SMEM、TMEM、寄存器内存）。

* **寄存器：**每个子分区有自己的寄存器堆，H100/B200 上是 16,384 个 32 位字（每个 SM `4 * 16384 * 4 = 256kiB`），供 CUDA 核心使用。
  * 每个 CUDA 核心一次最多只能访问 256 个寄存器。所以虽然每个 SM 最多能挂 64 个“常驻 warp”，若每个线程用满 256 个寄存器，实际同一时刻只装得下 8 个（`256 * 1024 / (4 * 32 * 256)`）。

* **SMEM（L1 缓存）：**每个 SM 有一块 256kB 片上缓存 SMEM，既可由程序员当“共享内存”用，也可交给硬件当片上缓存。喂给 Tensor Core 的矩阵乘的输入和激活都存在这里。

* **L2 缓存：**所有 SM<d-footnote>严格说 L2 一分为二：H100 上一半 SM 各接 25MB，两半之间有链路相连，但带宽较低。</d-footnote>共享一块约 50MB 的“大” L2 缓存，用来减少对主存的访问。
  * 容量和 TPU 的 VMEM 差不多，但**慢得多**，也不受程序员控制。这带来一点“鬼魅般的超距作用”：程序员得调整访存模式，才能保证 L2 被用好。<d-footnote>L2 全 SM 共享这一点，实际上等于逼着原本“各自为政”的 SM 们协同行事。</d-footnote>
  * NVIDIA 不公布 L2 带宽，有人[实测](https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth)约 5.5TB/s——约为 HBM 带宽的 1.6 倍；又因是全双工，双向合计有效带宽接近 3 倍。对比之下，TPU 的 VMEM 不但大一倍，带宽还高得多（约 40TB/s）。

* **HBM：**主内存，放模型权重、梯度、激活等等。
  * 容量一路狂飙：Volta 的 32GB 到 Blackwell（B200）的 192GB。
  * HBM 到计算核心的通道叫 HBM 带宽/内存带宽：H100 约 3.35TB/s，B200 约 9TB/s。

### GPU 规格总表

下面是近几代 GPU 的规格汇总。同一款 GPU 的不同变体在 SM 数、主频、FLOPs 上略有出入。先看内存容量：

|  GPU  | 代际 |     主频      | SM 数/芯片 | SMEM 容量/SM | L2 容量/芯片 | HBM 容量/芯片 |
| :---: | :--------: | :-------------: | :------: | :--------------: | :--------------: | :---------------: |
| V100  |   Volta    | 1.25GHz/1.38GHz |    80    |       96kB       |       6MB        |       32GB        |
| A100  |   Ampere   | 1.10GHz/1.41GHz |   108    |      192kB       |       40MB       |       80GB        |
| H100  |   Hopper   | 1.59GHz/1.98GHz |   132    |      256kB       |       50MB       |       80GB        |
| H200  |   Hopper   | 1.59GHz/1.98GHz |   132    |      256kB       |       50MB       |       141GB       |
| B200  | Blackwell  |        ?        |   148    |      256kB       |      126MB       |       192GB       |

各代每个 SM 都有 256kB 寄存器内存；Blackwell 还额外加了每 SM 256kB 的 TMEM。再看各芯片的算力与带宽：

|  GPU  | 代际 | HBM 带宽/芯片 | FLOPs/s/芯片 (bf16/fp16) | FLOPs/s/芯片 (fp8/int8) | FLOPs/s/芯片 (fp4) |
| :---: | :--------: | :---------: | :----------------------: | :---------------------: | :----------------: |
| V100  |   Volta    |   9.0e11    |            —             |            —            |         —          |
| A100  |   Ampere   |   2.0e12    |          3.1e14          |         6.2e14          |         —          |
| H100  |   Hopper   |   3.4e12    |          9.9e14          |         2.0e15          |         —          |
| H200  |   Hopper   |   4.8e12    |          9.9e14          |         2.0e15          |         —          |
| B200  | Blackwell  |   8.0e12    |          2.3e15          |         4.5e15          |       9.0e15       |

B100 没有量产，就不列了<d-footnote>NVIDIA 确实做过 B100，但只是短暂生产和销售过——据说是设计缺陷导致跑不到标称规格：散热和功耗一受限就撞不了顶。</d-footnote>另外，NVIDIA GPU 不像 TPU 那样标准划一，具体规格会随版本略有不同。

一张 GPU-TPU 部件对照速查表：

|              GPU              |     TPU     |              这是什么？              |
| :---------------------------: | :---------: | :-----------------------------------: |
| 流式多处理器（SM） | Tensor Core | 装着其他组件的核心“小间” |
|        Warp 调度器         |     VPU     |      SIMD 向量运算单元      |
|           CUDA 核心           |   VPU ALU   |               SIMD ALU                |
|        SMEM（L1 缓存）        |    VMEM     |       快速片上缓存       |
|          Tensor Core          |     MXU     |      矩阵乘单元      |
|        HBM（又叫 GMEM）         |     HBM     |  高带宽大容量内存  |

### 芯片层面的 GPU 与 TPU 对比

GPU 出道时是给电子游戏渲染画面的。可自从深度学习起飞，它越来越像一台专职的矩阵乘机器——换句话说，越来越像 TPU 了<d-footnote>深度学习爆发之前，GPU（Graphics Processing Unit，图形处理器）就干图形——主要是游戏。游戏用几百万个小三角形表示物体，再把三角形“光栅化”成屏幕上每秒显示 30-60 次的 2D 图像（这个频率叫帧率）。光栅化要做的是：把三角形投影到相机坐标系、算清哪个三角形压住哪个像素——每秒几十亿次。你可以想象这有多贵，而且这还只是开始：每个像素的颜色还得把可能好几个半透明三角形的颜色按射线叠加出来。GPU 就是为了把这些操作做得飞快而生的，同时兼顾通用性——你得同时跑许多种图形任务（“着色器”），不能谁一家独大。所以消费级图形卡能做矩阵乘，但那不是它的本职。</d-footnote>。某种程度上，这段历史解释了今天的 GPU 为什么长这样：它生来不是专为 LLM 或 ML 模型的，而是通用加速器；硬件追求的这种“通用”既可能是福气也可能是坑。GPU 在新任务上常常“拿来就能跑”，对编译器的依赖也远比 TPU 小。但代价是：想推算它的性能、想跑出屋顶线，难得多——任何一个编译器的“小动作”都可能成为瓶颈。

**GPU 更模块化。**TPU 是 1-2 个大 Tensor Core，GPU 是几百个小 SM。同样，TPU 的一个 TC 里只有一个大 VPU（由 4 个可独立编程的 8x128 单元组成，共 4096 个 ALU）；H100 则是 132 * 4 = 528 个独立的 32 宽 SIMD 单元（合计 16k 个 ALU）。一一对比如下：

|              GPU              |           TPU            | H100 数量 | TPU v5p 数量 |
| :---------------------------: | :----------------------: | :----: | :-------: |
| SM（流式多处理器） |       Tensor Core        |  132   |     2     |
|        Warp 调度器         |        VPU 槽位         |  528   |     8     |
|        SMEM（L1 缓存）        |           VMEM           |  32MB  |   128MB   |
|           寄存器           | 向量寄存器（VReg） |  32MB  |   256kB   |
|          Tensor Core          |           MXU            |  528   |     8     |

模块化程度的差异，一头让 TPU 造得更便宜、看得更明白；另一头把“做对的事”的压力全压给了编译器。TPU 只有一条控制流、只支持整 VPU 宽的向量化指令，编译器必须手动把所有内存装载和 MXU/VPU 工作流水化，才能不卡顿。GPU 程序员呢，尽管同时启动几十个互不相干的 kernel、各跑各的 SM。但反过来，这些 kernel 也可能因为把 L2 缓存冲得稀烂、或者访存没合并而跑得很惨——运行时的大量决策掌握在硬件手里，幕后发生了什么反而更难推理。结果就是：TPU 常常花更少的功夫，反而更接近峰值屋顶线。

**从历史上看，单张 GPU 比同代 TPU 更强（也更贵）：**单张 H200 的 FLOPs/s 接近 TPU v5p 的 2 倍、HBM 是 1.5 倍；同时 Google Cloud 标价约 $10/小时对 $4/小时。TPU 的路线更依赖“把很多芯片连起来”。

**TPU 的快速缓存大得多。**TPU 的 VMEM 远多于 GPU 的 SMEM（+TMEM），而且可以拿来回存权重和激活，取用极快。若能稳定地把模型权重放进/预取进 VMEM，LLM 推理可以更快。

### 练习 1：GPU 硬件

下面几道题考一考上面的内容。答案就在折叠里，但强烈建议纸笔在手先做再看。

**问题 1 [CUDA 核心]：**一张 H100 有多少个 fp32 CUDA 核心（ALU）？B200 呢？和 TPU v5p 的独立 ALU 数量比一比？

{% details 点此查看答案。 %}

**答案**：H100 有 132 个 SM，每个含 4 个子分区、每个子分区 32 个 fp32 CUDA 核心，共 `132 * 4 * 32 = 16896` 个。B200 有 `148` 个 SM，合计 `18944` 个。TPU v5p 有 2 个 TensorCore（通常经 Megacore 合为一体），各带一个 (8, 128) 车道、每车道 4 个独立 ALU 的 VPU，即 `2 * 4 * 8 * 128 = 8192` 个 ALU——大约是 H100 向量车道数的一半，主频倒是差不太多。

{% enddetails %}

**问题 2 [向量算力计算]**：单张 H100 有 132 个 SM，主频 1.59GHz（睿频至 1.98GHz）。假设每个 ALU 每周期一条向量指令，一共能出多少向量 fp32 FLOPs/s？睿频呢？和矩阵乘 FLOPs 比一比？

{% details 点此查看答案。 %}

**答案**：`132 * 4 * 32 * 1.59e9 = 26.9TFLOPs/s`；睿频下 33.5 TFLOPs/s。官方[规格表](https://www.nvidia.com/en-us/data-center/h100/)报的是这个的两倍——因为严格说 FMA（乘加融合）一周期算两个 FLOPs，尽管多数场合用不上。矩阵乘方面是 990 bfloat16 TFLOPs/s——不计 FMA 的水分，Tensor Core 的算力约是向量单元的 30 倍。

{% enddetails %}

**问题 3 [GPU 矩阵乘强度]**：H100 的 fp16 峰值矩阵乘强度是多少？B200 呢？fp8 呢？*强度 = 矩阵乘 FLOPs/s 除以内存带宽。*

{% details 点此查看答案。 %}

**答案**：H100 峰值 990e12 fp16 FLOPs、带宽 3.35e12 字节/秒，临界强度 `990e12 / 3.35e12 = 295`——和 TPU 的 240 相当接近。B200 是 `2250e12 / 8e12 = 281`，也很接近。也就是说，和 TPU 类似，矩阵乘要计算受限大约需要 280 左右的批大小。

H100 和 B200 的 fp8 算力都恰好是 2 倍，所以峰值强度也翻倍到 590 和 562——不过考虑到权重多半也会以 fp8 载入，某种意义上这个强度又没变。

{% enddetails %}

**问题 4 [矩阵乘耗时]**：用第 3 题的结论，估算 `fp16[64, 4096] * fp16[4096, 8192]` 在单张 B200 上要跑多久？`fp16[512, 4096] * fp16[4096, 8192]` 呢？

{% details 点此查看答案。 %}

由上可知，批小于 281 个 token 就是通信受限，所以第一个纯粹被带宽卡着。要读写 $2BD + 2DF + 2BF$ 个字节（`2*64*4096 + 2*4096*8192 + 2*64*8192=69e6`），带宽 `8e12` 字节/秒，约 `69e6 / 8e12 = 8.6us`。实际拿不到全部带宽，真实多半要 10-12us。批一大就是计算受限：`T=2*512*4096*8192/2.3e15=15us`，同样只能兑现部分算力，实际更接近 20us。

{% enddetails %}

**问题 5 [L1 缓存容量]**：一张 H100 的 L1/SMEM 总容量是多少？寄存器内存呢？和 TPU 的 VMEM 比一比？

{% details 点此查看答案。 %}

**答案**：每 SM 有 256kB SMEM 和 256kB 寄存器内存，各约 33MB（`132 * 256kB`），合计约 66MB——大约是当代 TPU 那 120MB VMEM 的一半。不过 TPU 的寄存器内存总共只有 256kB！TPU 的 VMEM 延迟比 SMEM 还低，这也是寄存器内存在 TPU 上不那么要紧的原因之一（VMEM 的溢出/回填很便宜）。

{% enddetails %}

**问题 6 [推算 B200 主频]**：NVIDIA 在[这里](https://resources.nvidia.com/en-us-blackwell-architecture)称 B200 能跑出 80TFLOPs/s 的向量 fp32 算力。已知每个 CUDA 核心在 FMA（乘加融合）里每周期可做 2 个 FLOPs，估算峰值主频。

{% details 点此查看答案。 %}

**答案**：共 148 * 4 * 32 = 18944 个 CUDA 核心，每周期 `18944 * 2 = 37888 FLOPs / 周期`。于是 `80e12 / 37888 = 2.1GHz`——偏高但在合理范围。B200 通常上液冷，这个主频就更说得通了。

{% enddetails %}

**问题 7 [估算 H100 加法耗时]**：用上面的数字，算算在单张 H100 上把两个 `fp32[N]` 向量相加该花多久。把 $T_\text{math}$ 和 $T_\text{comms}$ 都算出来。这个运算的算术强度是多少？有条件的话，再用 PyTorch 或 JAX 实际跑一遍 `N = 1024` 和 `N=1024 * 1024 * 1024`，对比如何？

{% details 点此查看答案。 %}

**答案**：首先，两个 `fp32[N]` 向量相加一共 N 个 FLOPs：要读 `4 * N * 2` 字节、写回 4 * N 字节，共 `3 * 4 * N = 12N`。比值是 `总 FLOPs / 总字节 = N / 12N = 1 / 12`——惨不忍睹。

上面算过，忽略 FMA 时向量算力约 33.5 TFLOPs/s，且这是所有 CUDA 核心全开才有。`N = 1024` 时，*最多*只能用到 1024 个 CUDA 核心、8 个 SM——若计算受限，大约要慢 16 倍。内存带宽 3.35e12 字节/秒。硬件的峰值强度是 `33.5e12 / 3.35e12 = 10`<d-footnote>有意思的是，这个强度在近年各代 GPU 上竟一成不变：H100 是 33.5 / 3.5，B200 是 80 / 8。原因不明，但值得一提。</d-footnote>，所以这个运算被通信按在地上摩擦。于是运行时间就是

$$T = \max(T_\text{comms}, T_\text{math}) = \frac{12 \cdot N}{\text{3.35e12}} = \frac{N}{\text{2.8e11}}$$

`N = 65,536` 时约 0.23us。实测 JAX 约 1.5us——正常，这种小活儿本来就重度延迟受限。`N = 1024 * 1024 * 1024` 时，屋顶线约 3.84ms，实测 4.1ms——很好！

{% enddetails %}

## 组网

组网是 GPU 和 TPU 差别最大的领域之一。前面讲过，TPU 连成 2D 或 3D 环面（torus）：每台 TPU 只和邻居相连。两台 TPU 之间传消息必须途经中间每一台，通信模式也被限制成网格上的“均匀”模式。这固然有诸多不便，好处是每台 TPU 的链路数恒定，“整机群”想扩多大都不掉带宽。

GPU 走的是更传统的分层交换树。8 张一组的 GPU 叫**节点**（GB200 可达 72 张<d-footnote>“节点”一词身兼两义：一是 NVLink 域，即全部由 NVLink 互联直连的那组 GPU；二是接在同一台 CPU 主机上的那组 GPU。B200 之前两者通常重合；GB200 NVL72 里 NVLink 域有 72 张，但每台主机仍只挂 8 张。本文用“节点”指 NVLink 域——这个用法有争议。</d-footnote>）：节点内用高带宽的 NVLink 一跳互达；节点之间再用低一档的 InfiniBand（IB）或以太网连成更大的单元（叫 **SU**，Scalable Unit，可扩展单元）；再往上，用更高层交换机把它们连成任意大的集群。

{% include figure.liquid path="assets/gpu/superpod-diagram.png" class="img-fluid" caption="<b>图：</b>典型 H100 组网示意。8 张 GPU 经 NVSwitch（也叫 NVLink 交换机）组成一个节点（NVLink 域），节点之间再连成交换式 InfiniBand 网络（fabric）。H100 在 NVLink 域内每出口方向约 450GB/s，每个节点通往 IB 网络的出口带宽为 400GB/s。" %}

### 节点内的组网

GPU 节点是很小的一档：通常 8 张（GB200 至多 72 张），用 NVLink 互联成“任意两张直连、满带宽、低延迟”的完整图<d-footnote>有人这样向我形容 NVLink：类似打了激素的 PCIe——低延迟、协议开销小，但不为扩展性和容错而设计；InfiniBand 则更像以太网，为更大规模、允许丢包的网络而生。</d-footnote>。每个节点装几台大带宽 NVSwitch，负责在本地各 GPU 之间转发包。节点级拓扑这些年变了不少（含每节点交换机数量）。H100 的画像是：每节点 4 台 NVSwitch，8 张 GPU 按 `5 + 4 + 4 + 5` 的链路模式接上去，如图：

{% include figure.liquid path="assets/gpu/nvlink-nodes.png" class="img-fluid" caption="<b>图：</b>从 Pascal（P100）到现在的节点（NVLink 域）示意图。自 Volta（V100）起，节点内经一组交换机实现任意两张直连。H100 节点用 4 台 NVSwitch 以 25GB/s 链路连齐 8 张 GPU。" %}

Hopper 一代（NVLink 4.0）每条 NVLink 链路有 25GB/s 全双工带宽<d-footnote>全双工指双向各 25GB/s、互不干扰：一条链路总共可送 50GB/s，但每个方向至多 25GB/s。</d-footnote>（B200 是 50GB/s），于是每张 GPU 到网络的出口是全双工 `18 * 25=450GB/s`。巨大的 NVSwitch 有至多 64 个 NVLink 端口——8xH100 节点配 4 台，交换层面可承载 `64 * 25e9 * 4=6.4TB/s`。各代参数一览：

| NVLink 代 | NVSwitch 代 | GPU 代际 | NVLink 带宽（GB/s，全双工） | 每 GPU 链路数 | 节点内 GPU-GPU 带宽（GB/s，全双工） | 节点规模（NVLink 域） | 每节点 NVSwitch 数 |
| :--------: | :----------: | :------------: | :----------------------------------: | :----------------: | :------------------------------------------: | :-----------------------: | :-----------------: |
|  **3.0**   |   **2.0**    |     Ampere     |                  25                  |         12         |                     300                      |             8             |          6          |
|  **4.0**   |   **3.0**    |     Hopper     |                  25                  |         18         |                     450                      |             8             |          4          |
|  **5.0**   |   **4.0**    |   Blackwell    |                  50                  |         18         |                     900                      |           8/72            |        2/18         |

Blackwell（B200）的节点是 8 卡；GB200NVL72 支持 72 卡的更大 NVLink 域。表中把 8 卡和 72 卡两种系统都列了出来。

### 练习 2：GPU 节点

再来几道组网问答。这些题特别值得亲手算，算一遍通信路径才真的长在手上。

**问题 1 [H100 节点总带宽]：**8xH100 节点、4 台交换机，整个节点一共多少带宽？*提示：* NVLink 和 NVSwitch 两头都要算。

{% details 点此查看答案。 %}

**答案**：4 台第四代 NVSwitch，每台单向 `64 * 25e9=1.6TB/s`，交换层面共 `4 * 1.6e12=6.4e12`。但每张 GPU 单向只出得了 450GB/s，全节点至多 `450e9 * 8 = 3.6TB/s`。取小的那个：峰值 3.6TB/s。

{% enddetails %}

**问题 2 [二分带宽]**：二分带宽的定义是：把网络对半切开，任意一种切法下两半之间最小的那个带宽。换句话说，网络一分为二时，中间能过多少量？能算出 8xH100 节点的二分带宽吗？*提示：*二分带宽通常把双向流量都算上。

{% details 点此查看答案。 %}

**答案**：任何一种对半分，都是每半 4 张卡；每张卡能朝对面送出 `4 * 450GB/s`。双向都算，穿过切面的量是 `8 * 450GB/s`，即 3.6TB/s。这正是 NVIDIA 报告的数字，比如[这里](https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf)。

{% enddetails %}

**问题 3 [AllGather 成本]**：给一个 B 字节的数组，在 8xH100 节点上一次（吞吐受限的）AllGather 要多久？请对 bf16[D<sub>X</sub>, F]、`D=4096`、`F=65,536` 算一遍。*建议先读 TPU 集合通信那一[节](https://jax-ml.github.io/scaling-book/sharding/)再作答。这里先自己想，后面还会大谈特谈集合通信。*

{% details 点此查看答案。 %}

**答案**：每张 GPU 单向出 450GB/s，每张手里有 $B / N$ 字节（`N=8`，节点规模）。可以想象每台依次把自己的那份发给其余 $N - 1$ 台，共 (N - 1) 轮，每轮 $T_\text{comms} = (B / (N * W_\text{unidirectional}))$，合计 $T_\text{comms} = (N - 1) * B / (N * W_\text{unidirectional})$，约等于 $B / W_\text{uni}$，即 $B / \text{450e9}$。

题给的数组 `B = 4096 * 65536 * 2 = 536e6` 字节，总时间 `536e6 * (8 - 1) / (8 * 450e9) = 1.04ms`（用近似式则 `536e6 / 450e9 = 1.19ms`）。这个量级可能吃到延迟受限，实际会更久（实测约 1.5ms）。

{% enddetails %}

## 节点之外的组网

节点之外，GPU 网络就没有那么标准的拓扑了。NVIDIA 发布了一份[参考 DGX SuperPod 架构](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/network-fabrics.html)：用 InfiniBand 把比单个节点大得多的 GPU 群连起来。但客户和数据中心提供商尽可以按需魔改<d-footnote>比如 Meta 训练 LLaMA-3 的数据中心网络就和这里描述的大不一样：用以太网、三层交换结构，顶层还是台 oversubscribed（超订）交换机。</d-footnote>

这是一张 1024 卡 H100 参考系统的图：底行每个方框是一个 8xH100 节点——8 张 GPU、8 块 400Gbps CX7 网卡（每卡配一块）、4 台 NVSwitch。

{% include figure.liquid path="assets/gpu/h100-superpod.png" class="img-fluid" caption="<b>图：</b>1024 卡 H100 DGX SuperPod 参考架构：128（有时 127）个节点、每节点 8 张 H100，接入 InfiniBand 扩容网络。每 32 个节点（256 卡）为一个“可扩展单元”（SU）。叶（leaf）与脊（spine）IB 交换机保证节点之间的全二分带宽。" %}

**可扩展单元（SU）：**每 32 个节点共用一组 8 台叶 IB 交换机，组成一个 “Scalable Unit”。一个 SU 含 256 张 GPU、每节点 4 台 NVSwitch、外加 8 台 IB 叶交换机。图中所有线缆都是 InfiniBand NDR（50GB/s 全双工），交换机是 64 口 NDR IB（每口也是 50GB/s）。*注意 IB 交换机的带宽是 NVSwitch 的 2 倍（64 个 400Gbps 端口）。*

**SuperPod：**整个 SuperPod 再把 4 个 SU 接到 16 台顶层“脊”IB 交换机上：1024 张 GPU、512 台节点级 NVSwitch、32 台叶 IB、16 台脊 IB，合计 512 + 32 + 16 = 560 台交换机。叶交换机按“每 32 节点一组”接入，每 256 卡对应 8 台叶；所有叶全部上联到所有脊。

**我们有多少带宽？**这张 InfiniBand 网络（叫“扩容网络”，scale-out network）的总拓扑是棵**胖树（fat tree）**：线缆和交换机保证节点层以上全二分带宽（这里是 400GB/s）。也就是说把节点对半分，每节点仍能同时向另一半的节点送出 400GB/s。更要紧的推论是：在扩容网络里，AllReduce 带宽大致恒定！哪怕实现未必如此，你尽可以想象在任意多个节点上做环形归约——总能构造一条串起所有节点的环。

| 层级 | GPU 数 | 每单元交换机数 | 交换机类型 | 每单元带宽（TB/s，全双工） | GPU-GPU 带宽（GB/s，全双工） | 胖树带宽（GB/s，全双工） |
| :---: | :------------: | :-------------------------: | :---------: | :------------------------------------------: | :--------------------------------------: | :---: |
| 节点  |       8        |              4              |     NVL     |                     3.6                      |                   450                    | 450
| 叶    |      256       |              8              |     IB      |                     12.8                     |                    50                    | 400 |
| 脊    |      1024      |             16              |     IB      |                     51.2                     |                    50                    | 400 |

对比一下：TPU v5p 每条链路出口约 90GB/s，三维环面全部轴向合计 540GB/s。它不是点对点网络，只适合受限的均匀通信模式，但 TPU-TPU 带宽就是高，而且能扩到任意大的拓扑（至少扩到 8960 台）。

GPU 交换网络理论上可以靠加交换机、加层级一直扩下去——代价是延迟更高、网络设备更贵。

<p markdown=1 class="takeaway">**要点**：H100 节点内，每张 GPU 有 450GB/s 的完整胖树带宽；出了节点，就掉到节点对节点 400GB/s。这一点马上就会在通信原语上见分晓。</p>

**GB200 NVL72：**NVIDIA 已开始出货新的 GB200 NVL72 集群：72 张 GPU 同处一个 NVLink 域，卡对卡满速 900GB/s。这些“机架”再连成更大的 SuperPod，IB 胖树带宽也按同比例（9 倍）水涨船高。拓扑如下：

{% include figure.liquid path="assets/gpu/gb200-superpod.png" class="img-fluid" caption="<b>图：</b>GB200 DGX SuperPod，共 576 张 GPU。最下层的每个机架装 72 张 GB200。" %}

数一数单个节点的出口（上图橙线）：`4 * 18 * 400 / 8 = 3.6TB/s`，通向叶层——正好是 H100 节点的 9 倍（节点里的卡也是 9 倍）。这就是说：节点出口带宽这个“命门”被抬高了许多_许多_，跨节点的集合通信带宽甚至可能比节点内还低。
更多讨论见[附录 A](#appendix-a-how-does-this-change-with-gb200)。

|   节点类型   | 每节点 GPU 数 | GPU 出口带宽 | 节点出口带宽 |
| :---------: | :-----------: | :------------------: | :-------------------: |
|    H100     |       8       |        450e9         |         400e9         |
|    B200     |       8       |        900e9         |         400e9         |
| GB200 NVL72 |      72       |        900e9         |        3600e9         |

<p markdown=1 class="takeaway">**要点**：GB200 NVL72 SuperPod 把节点规模和节点出口带宽同时拉高一大截，我们的屋顶线因此大不一样。</p>

### 练习 3：节点之外

**问题 1 [胖树拓扑]：**用上文那张 DGX H100 的图，算出整个 1024 卡 pod 在节点层的二分带宽，并验证每条链路、每台交换机的带宽恰好保证全二分。*提示：链路和交换机两头都要算。*

{% details 点此查看答案。 %}

**答案**：逐个部件来：

* 首先，每个节点用 8 条 400Gbps NDR IB 线缆接叶交换机，节点到叶有 `8 * 400 / 8 = 400 GB/s`。叶层 8 台、每台 3.2TB/s（64 条 400GBps 链路），但 SU 上联只能用其中 32 口：`32 * 400 / 8 = 12.8TB/s` 摊给 32 个节点，又正好是每节点 400GB/s。
* 再到脊层：每个 SU 用 `8 * 16 * 2` 条 400Gbps NDR 线缆上联，SU 到上层的带宽是 `8 * 16 * 2 * 400 / 8 = 12.8 TB/s`——还是每节点 400GB/s。脊层 16 台、每台 3.2TB/s，合计 `16 * 3.2 = 51.2 TB/s`，摊到 128 个节点，照旧 400GB/s。

所以无论怎么把节点对半分，每节点都有 400GB/s 跨切面带宽。每个部件的带宽都不多不少，恰好撑起这棵胖树。

{% enddetails %}

**问题 2 [扩到更大的 DGX pod]：**要改成在 2048 张（乃至 4096 张）GPU 上训练，上面这套 DGX 拓扑最省事/最好的改法是什么？*提示：没有唯一正确答案，但要控制成本，留意链路容量。[这份文档](https://docs.nvidia.com/dgx-superpod-reference-architecture-dgx-h100.pdf)也许有用。*

{% details 点此查看答案。 %}

**答案**：一种做法是保持 SU 结构原封不动（32 节点挂 8 台叶交换机），只是多造几个 SU、多加顶层交换机。脊交换机要翻倍：8 个 SU、32 台脊，带宽就够了。

有个问题：每台叶交换机只有 64 口，上面那张图已经全用满了。好在把“每台上联 2 条 400Gbps NDR”改成“每条 1 条”很容易——总带宽不变，还能省出一半端口。

到 4096 张时，端口就真的不够用了，得再加一层间接——也就是层级里再多一层。NVIDIA 管这层叫“核心（core）交换机”：一个 4096 卡的集群要配 128 台脊交换机加 64 台核心交换机。你可以自己算一遍，带宽正好够。

{% enddetails %}

## GPU 上的集合通信是怎么跑的？

GPU 能做和 TPU 一样的集合通信：ReduceScatter、AllGather、AllReduce、AllToAll。不同的地方在于：这些操作在节点内（走 NVLink）还是节点外（走 InfiniBand）做，实现方式完全两样。这些集合通信由 NVIDIA 实现在 [NVSHMEM](https://developer.nvidia.com/nvshmem) 和 [NCCL](https://developer.nvidia.com/nccl)（读音同 “nickel”）两个库里，NCCL 已开源（[仓库在这里](https://github.com/NVIDIA/nccl)）。NCCL 会按延迟要求/拓扑换不同实现（[细节](https://github.com/NVIDIA/nccl/issues/1415#issuecomment-2310650081)），下文我们只讨论交换树结构上的理论最优模型。

### 节点内集合通信

**AllGather 或 ReduceScatter：**节点内的 AllGather/ReduceScatter，可以和 TPU 一样绕环进行：把 GPU 随意排个序，让数组沿环逐跳转发，每跳都用满 GPU-GPU 带宽<d-footnote>也可以理解成：每台 GPU 把自己的 $\text{bytes} / N$ 发给其余 $N - 1$ 台，共通信 $(N - 1) \cdot N \cdot \text{bytes} / N$ 字节——殊途同归。</d-footnote>。每跳成本 $T_\text{hop} = \text{bytes} / (N \cdot \text{GPU 出口带宽})$，于是总成本

$$T_\text{AG or RS comms} = \frac{\text{bytes} \cdot (N - 1)}{N \cdot \text{GPU egress bandwidth}} \rightarrow \frac{\text{bytes}}{\text{GPU egress bandwidth}}$$

你会发现这和 TPU 上一模一样。AllReduce 照旧拆成 RS + AG，两倍代价。

{% include figure.liquid path="assets/gpu/all-gather.gif" class="img-fluid" caption="<b>图：</b>带宽最优的一维环形 AllGather 算法：B 字节要经顶层交换机转发 $X - 1$ 次、每次 B / X 字节。" %}

若你在意延迟（比如数组很小），可以改用树形归约：两两成对先合，再 4、再 8……总共 $\log(N)$ 跳而不是 $N - 1$ 跳——总代价不变，只是延迟更低。

<p markdown=1 class="takeaway">**要点**：单节点内对一个 B 字节数组做 AllGather/ReduceScatter，成本约为 $T_\text{comms} = B * (8 - 1) / (8 * W_\text{GPU egress}) \approx B / W_\text{GPU egress}$。理论上 H100 约 $B / \text{450e9}$、B200 约 $B / \text{900e9}$。AllReduce 是两倍，除非开了网内归约。</p>

<b markdown=1 style="color: #57cf57;">小测验 1 [AllGather 耗时]：</b>用 8xH100 节点（450 GB/s 全双工），AllGather(bf16[B<sub>X</sub>, F]) 要多久？设 $B=1024$、$F=16,384$。

{% details 点此查看答案。 %}

**答案**：共 $2 \cdot B \cdot F$ 字节，单向带宽 450e9。约 $T_\text{comms} = (2 \cdot B \cdot F) / \text{450e9}$，更精确是 $(2 \cdot B \cdot F \cdot (8 - 1)) / (8 \cdot \text{450e9})$。代入得约 $(2 \cdot 1024 \cdot 16384) / \text{450e9} = \text{75us}$，精确值是 $\text{65us}$。

{% enddetails %}

**AllToAll：**节点内 GPU 两两直连，AllToAll 自然顺手。每台直接把数据发到目的机器就行。节点内共 B 字节、每台持有 $B / N$，各自给其余 $N - 1$ 台各发 $(B / N^2)$，合计

$$T_\text{AllToAll comms} = \frac{B \cdot (N - 1)}{W \cdot N^2} \approx \frac{B}{W \cdot N}$$

对比 TPU 的成本 $B / (4W)$：单节点内我们理论上快了一倍（$B / 4W$ 对 $B / 8W$）。

对 MoE 模型，我们常做*稀疏/不规则（ragged）AllToAll*：保证输出维上 $N$ 份里至多 $k$ 份非零，即 $T_\text{AllToAll} \rightarrow K[B, N]$ 每轴至多 $k$ 个非零。成本按 $k/N$ 缩减，约 $\min(k/N, 1) \cdot B / (W \cdot N)$。MoE 里非零位置通常是独立随机的，“不足 $k$ 个非零”总有一定概率，于是再打个折，约
$(N-1)/N \cdot \min(k/N, 1) \cdot B / (W \cdot N)$。<d-footnote>真实成本其实是 $$(1 - \left(\frac{Z - 1}{Z}\right)^K) \cdot \frac{Z - 1}{Z}$$——掷 K 次骰子期望覆盖的不同点数——和这个近似式非常接近，细节见附录。</d-footnote>

<b markdown=1 style="color: #c55404ff;">小测验 2 [AllToAll 耗时]：</b>8xH100 节点、450 GB/s 单向带宽，AllToAll<sub>X->N</sub>(bf16[B<sub>X</sub>, N]) 要多久？若已知 8 份里只有 4 份非零呢？

{% details 点此查看答案。 %}

**答案**：注意这里 $B$ 是数组的批维，数组总大小 $V = 2 \cdot B \cdot N$ 字节。由上，稠密场景成本是 $V \cdot (N-1) / (W \cdot N^2)$，约 $V / (W \cdot N)$。若只有一半元素非 padding，发送量只剩 $V \cdot k/N$，成本 $V / (2 \cdot W \cdot N)$——约为总成本的一半。

{% enddetails %}

<p markdown=1 class="takeaway">**要点**：GPU 单节点内对 $B$ 字节数组做 AllToAll，成本约 $T_\text{comms} = (B \cdot (8 - 1)) / (8^2 \cdot W_\text{GPU egress}) \approx B / (8 \cdot W_\text{GPU egress})$。稀疏（top-$k$）AllToAll 再降到 $(B \cdot k) / (64 \cdot W_\text{GPU egress})$。</p>

**实测：**这是一份 8xH100 节点上 AllReduce 带宽的实测。Algo BW 是实测带宽（字节/耗时），Bus BW 按 $2 \cdot W \cdot (8 - 1) / 8$ 折算，理论上衡量的是真实链路带宽。可以看到我们确实能逼近 370GB/s——离 450GB/s 有差距但也不算远——不过要 10GB 的大数组才能兑现每设备约 10GB 的量。也就是说：这些理论估算没错，但只有消息足够大才跑得出来。

{% include figure.liquid path="assets/gpu/gpu-all-reduce-bw.png" class="img-fluid" caption="<b>图：</b>关闭 SHARP 的 8xH100 节点 AllReduce 吞吐。蓝线是按实测换算的链路带宽 $2 * \text{bytes} * (N - 1) / (N * \text{runtime})$。注意即使数组大到 10GB，也没有特别接近标称的 450GB/s。" %}

这是个大问题：它让我们所有基于理论的结论都要打折扣。比如对一个“合理大小”的数组——LLaMA-3 70B 的 MLP 权重（`bf16[8192, 28672]`；按 8 路模型分片是 `bf16[8192, 3584] = 58MB`）——AllReduce 只能跑到约 150GB/s，而峰值标称是 450GB/s。相比之下，TPU 在很小的消息上就能打满峰值带宽（见附录 B）。

<p markdown=1 class="takeaway">**要点**：NVIDIA 标称 H100 NVLink 有 450GB/s，但实践中很难超过 370GB/s——请据此调低上面的估算。</p>

**网内归约：**自 Hopper 起，NVIDIA 交换机支持 [“SHARP”（Scalable Hierarchical Aggregation and Reduction Protocol，可扩展层级聚合与归约协议）](https://developer.nvidia.com/blog/advancing-performance-with-nvidia-sharp-in-network-computing/)，即“网内归约”：**交换机本身**就能做归约运算，并把结果“广播（MultiCast）”给多台目标 GPU：

{% include figure.liquid path="assets/gpu/sharp-algorithm.png" class="img-fluid" caption="<b>图：</b>不开 SHARP 的 AllReduce 理论成本是两倍——数据要过每台 GPU 两遍。实践中提速只有约 30%（NCCL 2.27.5 实测）。" %}

理论上，这几乎把 AllReduce 的成本腰斩：每台 GPU 把自己的数据送到顶层交换机，由交换机完成归约并广播回各台，每块 GPU 不必“出口两次”，网络延迟也随之降低。

$$T_\text{SHARP AR comms} = \frac{\text{bytes}}{\text{GPU egress bandwidth}}$$

注意这个式子是精确的，没有 $1/N$ 的出入：每台 GPU 先送出 $B \cdot (N - 1) / N$，收到本片的半归约结果（入口 $B/N$）、补完归约，再送出 $B/N$，最后收到完全归约的结果（入口 $B \cdot (N - 1) / N$）——入口合计恰好 $B$ 字节。

然而实践中，开 SHARP 只带来约 30% 的带宽提升，而非预测的 75%。有效集合带宽只到约 480GB/s，远没有翻倍。

{% include figure.liquid path="assets/gpu/sharp-all-reduce-cost.png" class="img-fluid" caption="<b>图：</b>节点内 AllReduce Algo 带宽的实测对比（开/关 NVIDIA SHARP）。峰值收益约 30%，而按算法本应有接近 75% 的提升空间。" %}

<p markdown=1 class="takeaway">**要点**：理论上，NVIDIA SHARP（多数 NVIDIA 交换机支持）应把 $B$ 字节的 AllReduce 从约 $2 * B / W$ 降到 $B / W$；实测只有约 30% 提升。好在 LLM 里纯 AllReduce 本来不多，这条帮不上大忙。</p>

### 跨节点集合通信

出了节点，账就要算得细一点。在树上做归约，可以自底向上：先节点内、再叶层、再脊层，每层用常规算法。尤其对 AllReduce，这样能少传很多数据——节点内归约完，只需要向叶层送 $B$ 字节，而不是 $B * N$。

**这得花多少？**一级近似下，由于网络有完全二分带宽，AllGather/ReduceScatter 的成本就约等于字节数除以节点出口带宽（H100 上 400GB/s）——*与树形归约的任何细节无关。*

$$T_\text{AG or RS comms} = \frac{\text{bytes}}{W_\text{node egress}} \underset{H100}{=} \frac{\text{bytes}}{\text{400e9}}$$

其中 $W_\text{node}$ 在上面的 H100 网络里通常是 400GB/s（每节点 8 条 400Gbps IB 链路出口）。最直观的理解：想象对*集群里每个节点*做一次环形归约——胖树拓扑保证任意两节点间都构得出 $W_\text{node}$ 的环。节点内那一段（几乎）永远成不了瓶颈：它带宽更高、延迟更好。一般式是

$$T_\text{total} = \max(T_\text{comms at node}, T_\text{comms in scale-out network}) = \max\left[\frac{\text{bytes}}{W_\text{GPU egress}}, \frac{\text{bytes}}{W_\text{node egress}}\right]$$

{% details 更精确的推导收在这里。 %}

可以更精确些：我们实际上在网络每一层各做一次环形归约，且这些归约大多能相互重叠，于是

$$T_\text{AG or RS comms} = \text{bytes} \cdot max_\text{depth i}\left[\frac{D_i - 1}{D_i \cdot W_\text{link i}}\right]$$

其中 $D_i$ 是深度 $i$ 的度（该层的孩子数），$W_\text{link i}$ 是把每个孩子连向父节点的链路带宽。

由此，给定拓扑的 AllGather/AllReduce 可用带宽是 $min_\text{depth i}(D_i * W_\text{link i} / (D_i - 1))$。套用到上面的网络：

* **节点：**$D_\text{node}$ = 8（节点里 8 张 GPU），Wlink i = 450GB/s。AG 带宽 `450e9 * 8 / (8 - 1) = 514GB/s`。
* **叶：**$D_\text{leaf}$ = 32（SU 里 32 个节点），Wlink i = 400GB/s（8 条 400Gbps IB）。带宽 `400e9 * 32 / (32 - 1) = 413GB/s`。
* **脊：**$D_\text{spine}$ = 4（4 个 SU），$W_\text{link i}$ = 12.8TB/s（`8 * 16 * 2 * 400Gbps`）。带宽 `12.8e12 * 4 / (4 - 1) = 17.1TB/s`。

所以整体 AG/RS 带宽是 `min(514GB/s, 413GB/s, 17.1TB/s) = 413GB/s`，瓶颈在叶层。实际 $T_\text{AG or RS comms} = B / \text{413GB/s}$——哪怕扩到最大规模，也还有约 413GB/s。开 SHARP 的 AllReduce 会略低于这个数（约 400GB/s），因为少了 $(N - 1) / N$ 那个因子。不过 450GB/s 和 400GB/s 相差无几，估算时可以混用。

{% enddetails %}

**其他集合通信：**AllReduce 仍是上述两倍，除非开 SHARP。NVIDIA 也卖支持 SHARP 的 IB 交换机，但不是每个云厂商都装了。AllToAll 跨节点变化就大了——它不像 AllReduce 那样“分层”。要从每台 GPU 都发往每台 GPU，就用不上节点层的那个完全二分带宽。一个跨 $M = N / 8$ 个节点的 N 路 AllToAll，成本是

$$T_\text{AllToAll comms} = \frac{B \cdot (M - 1)}{M^2 \cdot W_\text{node egress}} \approx \frac{B}{M \cdot W_\text{node egress}}$$

等效带宽只剩 50GB/s 而不是 400GB/s。从单节点内的 $B / (8 * \text{450e9})$ 跌到跨两个节点的 $B / (2 \cdot \text{400e9})$——劣化超过 4 倍。

1024 卡 DGX H100 SuperPod 架构汇总：

|   层级   | GPU 数 | 度（孩子数） | 交换机带宽（全双工，TB/s） | 线缆带宽（全双工，TB/s） | 集合通信带宽（GB/s） |
| :-------: | :------------: | :-----------------: | :----------------------------------: | :---------------------------------: | :-------------------------: |
|   节点    |       8        |          8          |                 6.4                  |                 3.6                 |             450             |
| 叶（SU） |      256       |         32          |                 25.6                 |                12.8                 |             400             |
|   脊    |      1024      |          4          |                 51.2                 |                51.2                 |             400             |

这里“集合通信带宽”指的是：无论出口在 GPU 还是节点这一层，实际可用的出口带宽；它同时也等于 $\text{对分带宽} * 2 / N$。

<p markdown=1 class="takeaway">**要点**：出了节点层，对 B 字节做 AllGather 或 ReduceScatter 的成本约为 $B / W_\text{node egress}$——在 H100 DGX SuperPod 上就是 $B / \text{400e9}$；AllReduce 则是两倍，除非开 SHARP。整套拓扑是一棵胖树，目的就是让任意两个节点之间的带宽恒定。</p>

**数组已沿另一轴分片时的归约：**考虑这样一次归约的代价

$$\text{AllReduce}_X(A[I_Y, J]\ \{ U_X \})$$

——归约的对象本身还沿另一条轴 $Y$ 分着片。在 TPU 上，这条轴的每条链路只传 $1 / Y$ 的数据，总成本相比不分片版直接除以 $Y$。在 GPU 上，成本取决于哪条轴是“内”轴（节点内还是节点间）、以及每个分片是否横跨多个节点。假设 $Y$ 是内轴、数组共 $\text{bytes}$ 字节，那么总成本确实约除以 $Y$——但前提同样是 $Y$ 横跨了多个节点：

$$T_\text{comms at node} = \frac{\text{bytes}}{W_\text{GPU egress}} \cdot \frac{1}{\min(Y, D_\text{node})}$$

$$T_\text{comms in scale-out network} = \frac{\text{bytes}}{W_\text{node egress}} \cdot \frac{D_\text{node}}{\max(D_\text{node}, Y)}$$

$$T_\text{total} = \max(T_\text{comms at node}, T_\text{comms in scale-out network})$$

其中 N 是 GPU 总数，$D_\text{node}$ 照旧是节点内 GPU 数（节点的度）。看得出来：$Y < D_\text{node}$ 时只是节点内省了力气，总时长通常纹丝不动；$Y > D_\text{node}$ 时才有真加速，加速比正比于横跨的节点数。

想要精确到环形归约，树形 AllGather<sub>X</sub>(A<sub>Y</sub> { U<sub>X</sub> })（设 Y 为内轴）的通式是

$$T_\text{AR or RS comms} = \text{bytes} \cdot \max_{\text{depth } i}\left[\frac{D_i - 1}{D_i \cdot \max(Y, S_{i-1}) \cdot W_{\text{link } i}}\right]$$

其中 $S_i$ 是 M * N * …，即树中第 i 层以下各子节点的规模。这句话大意是：跨的 GPU/节点越多，可用带宽越大——但仅限节点内部如此。

**小测验 3 [沿两条轴分片]：**在单个 SU（256 张芯片）上做 $\text{AllGather}_X(\text{bf16}[D_X, F_Y])$，$Y$ 为内轴。以 $D$、$F$、$Y$ 为变量，这要多久？

{% details 点此查看答案。 %}

**答案**：分两种情况：Y <= 8 和 Y > 8。$Y <= 8$ 时瓶颈仍在叶交换机，照旧 $T_\text{comms} = 2 * D * F * (32 - 1) / (32 * 400e9)$。Y > 8 时，由上面的式子约为

$$T_\text{comms} = \frac{2 \cdot D \cdot F \cdot 256}{Y \cdot \text{12.8e12}} = \frac{2DF}{Y \cdot \text{50GB/s}}$$

取 `D = 8192`、`F = 32,768`：

{% include figure.liquid path="assets/gpu/sharded-all-gather-cost.png" class="img-fluid" caption="<b>图：</b>内轴跨更多节点时，分片 AllGather 的理论成本。" %}

注意看：恰好 8 路模型并行时，节点级归约的成本确实被除以 8——可总成本一点没变。免费，但对整体带宽无感。

{% enddetails %}

<p markdown=1 class="takeaway">**要点**：多轴分片并存时，外层归约的成本按“内轴横跨的节点数”成比例减少。</p>

### 练习 4：集合通信

**问题 1 [SU 上的 AllGather]：**只看单个 SU：M 个节点、每节点 N 张 GPU。一次 AllGather 中，节点层交换机精确地收/发多少字节？顶层交换机呢？

{% details 点此查看答案。 %}

**答案**：一步一步数：

1. 每张 GPU 向交换机送 $B / MN$ 字节，节点交换机入口合计 $NB / MN = B / M$ 字节。
2. 向脊交换机完整送出 $B / M$ 字节。
3. 从脊交换机收进 $B * (M - 1) / M$ 字节。
4. 向其余方向各送 $B - B / MN$ 字节、共 $N$ 次，合计 $N * (B - B / MN) = NB - B / M$。

总计入口 $B$、出口 $BN$——瓶颈在出口。总时长 $T_\text{AllGather} = BN / W_\text{node} = B / \text{450e9}$。

脊交换机那笔账反而更简单：入口是 $B / M$ 收 $M$ 次（合计 $B$ 字节），出口是 $B (M - 1) / M$ 发 $M$ 次（合计 $B * (M - 1)$）。后者大得多，所以成本由它定：$T_\text{AllGather} = B \cdot (M - 1) / (M \cdot W_\text{node}) = B \cdot (M - 1) / (M \cdot \text{400e9})$。

{% enddetails %}

**问题 2 [单节点 SHARP AllReduce]：**单节点 N 张 GPU，用 SHARP（网内归约）做一次 AllReduce，交换机精确地收/发多少字节？

{% details 点此查看答案。 %}

**答案**：照旧一步步数。

1. 每张 GPU 送出 $B * (N - 1) / N$，交换机入口 $N * B * (N - 1) / N = B * (N - 1)$。
2. 交换机累加出部分和，给每台回 $B / N$ 字节，出口 $N * B / N = B$。
3. 每台对自己那份余量做局部部分和再送回交换机：入口又是 $N * B / N = B$。
4. 交换机集齐各片后广播，给 $N$ 个目的地各发 $B * (N - 1) / N$，出口 $B * (N - 1)$。

于是入口总计 $B * (N - 1) + B = BN$，出口同样。这正支持“吞吐恰为 $B / W_\text{egress}$”的结论。

{% enddetails %}

**问题 3 [跨节点 SHARP AllReduce]：**bf16[D<sub>X</sub>, F<sub>Y</sub>] 分片在单节点 N 张 GPU 上。AllReduce(bf16[D, F<sub>Y</sub>] { U<sub>X</sub> }) 要多久？可以假设有网内归约。若不止一个节点，差别在哪？

{% details 点此查看答案。 %}

**答案**：在上题思路上改。大致是：每台先送 $B * (X - 1) / XY$，交换机回给每台 $B / XY$，原样送回交换机，最后再广播 $B * (X - 1) / XY$ 给每台。进出口都是 $NB / Y$，总时长 $T_\text{comms} = NB / (Y * N * W_\text{link}) = N * 2DF / (Y * N * W_\text{link}) = 2 * D * F / (Y * W_\text{link})$——时长确实随 $Y$ 变小。

若跨出单节点，归约大体照做，但出节点层交换机时必须整份 $B$ 全送、不能只送 $B / Y$——因为每一份分片都得保持独立。

{% enddetails %}

**问题 4 [脊层 AllReduce 成本]：**同上设置，但 $Y = 256$（AllReduce 打到脊层）。要多久？同样可假设有网内归约。

{% details 点此查看答案。 %}

**答案**：这回能吃上脊层那“离谱”的带宽了：脊层共 51.2TB/s、摊到 4 个 SU 各 12.8TB/s。开 SHARP，最快 `2 * D * F / 12.8e12` 秒就能完事。

{% enddetails %}

**问题 5 [两节点 AllGather 成本]：**算出恰好在 2 个节点上做 $B$ 字节 AllGather 的精确成本。*务必算精确值而不是近似式，节点内和跨节点两头都要算。*

{% details 点此查看答案。 %}

**答案**：节点层面 $T_\text{comms} = B * 7 / (8 * \text{450e9}) = B / \text{514e9}$；跨节点实际是 $T_\text{comms} = B * (2 - 1) / (2 * \text{400e9}) = B / \text{800e9}$。也就是说，瓶颈竟然在节点内，而不在叶层！这正是 DeepSeek v3 采用 2 路数据并行之类的动机。

{% enddetails %}

## LLM 在 GPU 上扩展的屋顶线

铺垫至此，正题来了：搞清楚 LLM 在 GPU 上扩展的屋顶线，与 TPU 训练那[一章](../training)互为补充。和那里一样，目标是算出各种并行方案的 $T_\text{math}$ 与 $T_\text{comms}$，找到 $T_\text{comms} > T_\text{math}$ 的那条线。照旧只看 MLP 块：

$$\text{MLP}(x) \equiv x[B, D] *_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D]$$

其中 $B$ 是全局批、**按 token 计**（即 $B = \text{批大小} \cdot \text{序列长度}$）。

先把上面那张“GPU 层与节点层有效带宽”表贴回来：

|   节点类型   | 每节点 GPU 数 | GPU 出口带宽 | 节点出口带宽 |
| :---------: | :-----------: | :------------------: | :-------------------: |
|    H100     |       8       |        450e9         |         400e9         |
|    B200     |       8       |        900e9         |         400e9         |
| GB200 NVL72 |      72       |        900e9         |        3600e9         |

**注意**：GPU 出口带宽和节点出口带宽都会决定 LLM 的屋顶线。下文用 $W_\text{collective}$ 统称二者——在节点内就是 GPU 带宽，出了节点就是节点带宽。

像对 TPU 那样，来看**数据并行、张量并行、流水线并行、专家并行**及其组合的计算-通信屋顶线。具体数字以 H100 为准。GB200-NVL72 的屋顶线形状相同，只是节点出口带宽大得多，有时瓶颈反而落在节点内。

### 数据并行

前面说过，DP 和 ZeRO 分片在反向要么来一次权重 AllReduce，要么 ReduceScatter + AllGather。两者代价相同。要让纯数据并行或 FSDP *在不开网内归约*时保持计算受限，设该轴大小为 X，反向逐层有：

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{X \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF}{W_\text{collective}}$$

要 $T_\text{math} > T_\text{comms}$，需要 $B / (XC) > 1 / W_\text{collective}$，即

$$\frac{B}{X} > \frac{C}{W_\text{collective}}$$

$W_\text{collective}$ 取 GPU 还是节点带宽，取决于分片在节点内还是跨节点。于是：

* **节点内**：每 GPU 的 **token** 批只需 > $\text{990e12} / \text{450e9} = 2200$。
* **SU 内或脊层**：BS > $\text{990e12} / \text{400e9} = 2475$。

这比 TPU 高不少——TPU 三条轴全开也只要 850。比如 LLaMA-3 用 16000 张 H100 训练，批至少要 4000 万 token（参考：他们实际用了 1600 万）。DeepSeek v3 在 2048 张 H800 上训练，带宽更低（300GB/s 而不是 H100 的 450GB/s），每 GPU 需要 $\text{990e12} / \text{300e9} = 3300$ 个 token，合计约 670 万（实际用了 400 万）。

若开网内归约做纯数据并行，理论上 AllReduce 带宽翻倍，上面两个数直接减半。但实测收益只有三成上下，差不多刚好补回“跑不满标称带宽”的窟窿。再加之纯数据并行本来就很少单用，这条实际意义不大。

**MoE 模型：**对专家混合（MoE）模型——共 E 个专家、每 token 激活 k 个——式子变成

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot k \cdot BDF}{X \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot EDF}{W_\text{collective}}$$

每 GPU 的 token 批被抬高 $E/k$ 倍，即

$$\frac{B}{X} > \frac{E}{k} \frac{C}{W_\text{collective}}$$

例如新版 OpenAI OSS 模型，$k=4$、$E=128$：跨节点时需要 `32 * 2475 = 79,200`——高得离谱。

**X 很小时会怎样？**若只做比如 2 节点的数据并行，能吃到 $(X - 1) / X$ 的折扣：

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{N * C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF \cdot (X-1)}{X \cdot W_\text{collective}}$$

X 是节点数、$N = 8 \cdot X$。稠密模型于是有 $B / N > \alpha \cdot (X - 1) / X$，例如 $B / N > \text{1237}$——上面数值的一半。这就是为什么 2 路数据并行意外地常见。

<p markdown=1 class="takeaway">**要点**：在 H100/B200 上，数据并行和 ZeRO 分片想计算受限，每 GPU 批大小约需 2500 个 token（假设完美重叠、算力拉满）。MoE 模型再乘 $E / k$（总参数/激活参数之比）。数据并行做得少时，临界批会变小。</p>

### 张量并行

张量并行要对激活做 AllGather 和 ReduceScatter，且必须和 MLP 的 FLOPs 重叠。换句话说，前向有

$$T_\text{math} = \frac{2\cdot 2 \cdot BDF}{Y \cdot C}$$

$$T_\text{comms} = \frac{2\cdot 2 \cdot BD}{W_\text{collective}}$$

要计算受限，法则为

$$Y < \frac{F \cdot W_\text{collective}}{C}$$

节点内约 $F / 2200$，跨节点约 $F / 2475$。LLaMA-3 这种 $F=\text{28000}$ 的模型，约 11 路 TP（向下取整就是 8 路——正好一个节点）。和前面一样，恰好跨 2 个节点时有 2 倍带宽，所以一般能做到 16 路张量并行（$F > 2475 \cdot (Y - 8)$），理论上顶到天也就 19 路模型并行。

<p markdown=1 class="takeaway">**要点**：对大小 Y 的轴做张量并行，当 $Y > F / 2475$ 时通信受限——一般只能节点内 TP，至多跨两个节点。</p>

### 专家并行

前面已提：MoE 模型权重多 E 倍、FLOPs 只多 k 倍，数据并行因此难得多。缓解办法之一：沿专家维切权重，即 W<sub>in</sub>[E<sub>Z</sub>, D, F]。做 MLP 块时，要引入 2 次 AllToAll，把激活送去对应的专家。

如前所述，跨多节点时 AllToAll<sub>Z->k</sub>([B, D, k]) 的成本约 $T_\text{AllToAll} = 2 \cdot B \cdot D \cdot (Z-8)/Z \min(8 * k / Z, 1)$。纯专家并行于是有

$$T_\text{math} = \frac{4 \cdot B \cdot k \cdot D \cdot F}{Z \cdot C}$$

$$T_\text{comms} = \frac{4 \cdot B \cdot D \cdot (Z-8)}{W \cdot Z} \cdot \min\left(\frac{8 \cdot k}{Z}, 1\right)$$

要么 $K > Z/8$ 且 $F > \alpha \cdot (Z - 8)/k$；要么 $Z \gg K$ 且 $F > 8 \cdot \alpha$，其中 $\alpha = C/W$。这给出专家并行可行的两个“领地”：要么小规模专家并行（约 2 个节点）配小 $F$；要么大 $F$，$Z$ 可以任意大（直到 E 路专家并行）。

两种在实践里都看得到：DeepSeek v3 的 F 很小，跨节点专家并行小而受限；而 F 大的模型可以边 TP 边做大规模跨节点 EP。

<p markdown=1 class="takeaway">**要点**：若 $F < 8 * C / W_\text{node}$，专家并行只宜跨 1-2 个节点，成本和 TP 相当（略低）；若 $F > 8 * C / W_\text{node}$，则可做大量（至 $E$ 个节点）、成本不高的专家并行。</p>

### 流水线并行

流水线把层分到各节点，通信成本低得几乎可以忽略——每隔几层才传一小份微批激活。历史上流水线被“流水线气泡”所困，但有了新的零气泡流水线方法后，通常可以没有气泡。

流水线的总通信成本微乎其微：$N_\text{MB}$ 个微批、$N_\text{stages}$ 个阶段时，$T_\text{comms per hop} = 2 \cdot B \cdot D / (W \cdot N_\text{MB})$，共 $N_\text{MB} + N_\text{stages} - 2$ 跳，于是约

$$T_\text{total PP comms} = \frac{2BD}{W \cdot N_\text{MB}} \cdot (N_\text{MB} + N_\text{stages} - 2)$$

$$T_\text{per-layer comms} \approx 1.5 \cdot \frac{2BD}{W \cdot N_\text{layers}}$$

除以了 $N_\text{layers}$，比其他任何成本都小得多。从通信角度看，流水线几乎是免费的。那为什么不全用流水线？原因有几条：

(1) **代码复杂度：**流水线和自动并行框架（如 XLA 的 GSPMD）的“合拍”程度不如别的方案。它为了藏气泡引入微批次，改变了程序结构；各种零气泡调度还要求前向反向精细交错，雪上加霜。

(2) **流水线和数据并行/FSDP 犯冲：**大概是不用流水线的最大理由：它跟 FSDP、数据并行配合很差。ZeRO-3 尤其糟——它要求每个微批都 AllGather 权重，而我们只有 $B / N_\text{microbatches}$ 个 token 来摊薄 AllGather 的开销。更糟的是反向里，*某个阶段的梯度必须等最后一个微批走过才能 AllReduce/ReduceScatter，这段通信没法重叠。*

{% include figure.liquid path="assets/gpu/pipeline-bubble.png" class="img-fluid" caption="<b>图：</b>一个 2 阶段 2 微批的流水线示例。F 是该阶段的前向，B 是反向（两倍代价），G 是数据并行的 AllReduce——它可能比一个微批的时间长得多。" %}

(3) **流水线气泡与步不均衡：**看上面那张（糟糕的）排程图，朴素排程很容易产生大量气泡（=白白浪费算力）：第二阶段的第 0 步闲着，第一阶段在第 2-3 步闲着，第二阶段最后一步又闲着。精心排程能减少，却不能根除。激活还得沿关键路径从一个阶段传给下一个，也添开销：

{% include figure.liquid path="assets/gpu/pipeline-transfer.png" class="img-fluid" caption="<b>图：</b>流水线中的传输开销（红色）。它把各阶段相互错开，气泡随之变大。" %}

这些难题各有绕法，但实现复杂、难维护。平心而论，流水线始终是通信成本相对最低的方法之一。

**关于延迟的提醒：**前面说过，GPU 即使消息不小也很难跑满 AllReduce 带宽。所以哪怕理论上能把专家并行的 AllToAll 铺到多节点，实际能兑现一半带宽就不错。因此实践中总尽量把 TP 或 EP 收在更少节点里，把延迟开销压到最低。

### 例子

**DeepSeek 是怎么做的？**作为参照，[DeepSeek V3](https://arxiv.org/abs/2412.19437)用 2048 张 H800 训练，配置是：

* 64 路专家并行（EP），跨 8 个节点
* 16 路流水线并行（PP）
* 2 路 ZeRO-1 数据并行（DP）

稳态批大小 `4096 * 15360 = 62,914,560` 个 token，即每 GPU 3 万。这个数字已经很大——但模型也真的很稀疏（k=8、E=256），批就是不得不大。64 路 EP + 16 路 PP 合起来是 1024 路模型并行，AllReduce 落在脊层；又因为数据并行只有 2 路，实际到手的带宽反而多 $2 / (2 - 1) = 2$ 倍。这还顺带让最后那次数据并行 AllReduce 能和流水线末段的计算重叠，进一步摊薄成本。

**LLaMA-3 是怎么做的？**LLaMA-3 在 1.6 万张 GPU 上用 1600 万 token 的批训练，即每 GPU 约 1000 个 token。他们的配置：

* 节点内 8 路张量并行（TP）
* 16 路流水线并行（PP）
* 128 路 ZeRO-1 数据并行

这也是稠密模型，各项都算轻松。16 路 PP 把数据并行 AllReduce 的成本除以 16，临界批随之降了下来。

### 一句话总结：LLM 在 GPU 上怎么扩

退后一步，把这章学到的串成一张总图：

* **数据并行或 FSDP（ZeRO-1/3）要求每 GPU 约 2500 个 token 的本地批**；理论上“网内归约 + 纯 DP”能压下一部分。
* **张量并行最多 8 路左右还是计算受限**，再往上就没有带宽可用了。这把模型并行基本锁死在一个 NVLink 域内（单节点，或动用最多 72 卡的 GB200NVL72）。
* **任何跨节点的模型并行都能进一步摊薄 FSDP 的成本**，所以常见的做法是 PP + EP + TP 混搭、跨很多节点，把 FSDP 的代价压下去。
* **流水线并行很好用——前提是你扛得住零气泡流水线的代码复杂度，并保持足够大的批以躲开数据并行瓶颈。**流水线通常让 ZeRO-3 没法用（每个流水线阶段都要 AllGather），可以改投 ZeRO-1。

**高屋建瓴，在 GPU 上切大模型的食谱是：**

* 不大的稠密模型：批够的话，激进 FSDP 就很好；必要时掺一点流水线或张量并行。
* 更大的稠密模型：1-2 节点 TP + 跨很多节点的 PP + 纯数据并行，这个组合就很好。
* MoE：上面通用；另外可以做专家并行，且优先于 TP。若 $F > 8 * C / W_\text{node}$，多节点专家并行可以放开了做；否则就只到 2 节点上下。

### 练习 5：LLM 屋顶线

**问题 1 [B200 屋顶线]：**B200 DGX SuperPod（**不是 GB200 NVL72**）节点内带宽翻倍（出口 900GB/s），扩容网络照旧（400GB/s）（[出处](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-b200/latest/network-fabrics.html)）。总算力见前文表格。模型并行和数据并行的屋顶线会怎么变？

{% details 点此查看答案。 %}

**答案**：bfloat16 算力从 990 涨到 2250 TFLOPs，2.25 倍。带宽只翻倍，节点内的屋顶线基本不动。以 TP 为例，临界强度升到 `2250e12 / 900e9 = 2500`，上限 $Y < F / 2500$——只高一点点（节点规模没变就帮不上忙）。

出了节点就不妙了：带宽没涨，计算受限反而更难！比如数据并行，临界批涨到 `2250e12 / 400e9 = 5625`——同样的带宽，GPU 却能干多得多的活。

72 卡节点的 GB200 SuperPod 另走一路：它直接加出口带宽（[出处](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html#compute-fabric-576)）。

{% enddetails %}

**问题 2 [怎么切 LLaMA-3 70B]：**考虑以 bfloat16 训练、fp32 Adam 优化器状态的 LLaMA-3 70B。

1. 只求放得下权重和优化器状态，至少要多少张 H100？
2. 若要在 4096 张 H100 上训练 15T token，达到 45% MFU（模型算力利用率），要训多久？
3. LLaMA-3 70B 的 `F = 28,672`，实际用约 400 万 token 的批训练。不做通信受限的前提下，最多能做几路模型并行？仅靠这个加纯数据并行，能在 4096 张卡上保持计算受限地训练吗？ZeRO-3 呢？再加 8 路流水线呢？*注：通信成本和显存占用都要考虑。*

{% details 点此查看答案。 %}

1. 权重 2 字节、优化器状态 8 字节，至少 700GB。80GB 显存至少需要 9 张卡，即（取整到节点）至少 2 个 8xH100 节点。这样训练天荒地老、梯度检查点也没地方放，但下界就是这个。
2. 总共 `6 * 70e9 * 15e12 = 6.3e24` bf16 FLOPs。每张 GPU `990e12` FLOPs，45% MFU 下全场 1.8e18 FLOPs/s，合计 3.5e6 秒，即 40 天。
3. 节点内带宽 450GB/s，上限约 `F / 1995 = 28672 / 1995 = 14.372`。因为跨不了两个节点，现实选择就是 8 路模型并行。
   1. 这就得做 512 路数据并行。先看内存：模型只切 8 份，意味着 `700GB / 8 = 87.5GB / GPU`——装不下，不行！
   2. 8 路 TP 之上做 512 路 ZeRO-3：内存没问题，什么都激进地切了。但每 GPU 批只有 `4e6 / 4096 = 976`——很低，连纯 DP 的下限都不到；ZeRO-3 要搬权重，限度还是它的两倍。不行。
   3. 加 8 路流水线后，每个模型并行分片横跨 8 个节点。前面讲过，这把叶层 AllGather 的成本除以 8：该层 AllReduce/AllGather 的有效带宽从 400GB/s 升到 `8 * 400GB/s = 3200GB/s`。屋顶线 `990e12 / 3200e9 = 309`，没问题！前提是流水线实现得足够利落。

{% enddetails %}

**问题 3 [Megatron-LM 的超参]：**看[这张图](https://github.com/NVIDIA/Megatron-LM)——出自 Megatron-LM 仓库，展示其引以为傲的高 MFU。

{% include figure.liquid path="assets/gpu/megatron-hparams.png" class="img-fluid" %}

注意序列长度处处是 4096。对 16B、70B、314B 三个模型，分别算出每 GPU 的 token 批。设数据并行在最外层、bfloat16 归约，判断各配置理论上是计算受限还是通信受限？有没有更优配置？

{% details 点此查看答案。 %}

**答案**：先算每 GPU 批大小。

* **16B**：`192 * 4096 / 192 = 4096` token/GPU
* **70B**：`384 * 4096 / 768 = 2048` token/GPU
* **314B**：`1536 * 4096 / 3072 = 2048` token/GPU

除第一个外，后两个都贴着 2k 上下——正卡在我们算出的 FSDP 临界线附近。按脊层归约算出的界限是 2472 token/GPU，这里大体适用。不过 70B 和 314B 分别有 16 路和 64 路模型并行（PP + TP），脊层吞吐因此好 2 倍和 8 倍，换算下来，大约 1k 和 300 token/步就是它们的计算受限线。

{% enddetails %}

## 致谢与延伸阅读

本章得到许多 GPU 行家的指点，谨此致谢：

* Adam Paszke：讲清了 GPU 上写内核的真实处境。
* Swapnil Patil：教我认识了 GPU 组网。
* Stas Bekman：提醒我 GPU 的实测表现常常和纸面规格对不上。
* Reiner Pope：厘清 GPU 与 TPU 硬件层面的对比。
* Frédéric Bastien：对芯片部分提了详尽意见。
* Nouamse Tazi：以在 GPU 上训练 LLM 的实战经验打磨了屋顶线部分。
* Sanford Miller：帮我理解 GPU 如何组网，以及 NVIDIA 的规格与实际部署的常见差距。

GPU 的好材料汗牛充栋，个人常翻的有：

* [SemiAnalysis' History of the NVIDIA Tensor Core](https://semianalysis.com/2025-06-23/nvidia-tensor-core-evolution-from-volta-to-blackwell/)：讲 GPU 如何从游戏引擎变身 ML 加速器，写得极好。
* [SemiAnalysis' Analysis of Blackwell Performance](https://semianalysis.com/2024-04-10/nvidia-blackwell-perf-tco-analysis/)：想看懂 NVIDIA 下一代，就读它。
* [H100 DGX SuperPod Reference](https://docs.nvidia.com/dgx-superpod-reference-architecture-dgx-h100.pdf)：枯燥但实用，大型 GPU 集群如何组网。[这份](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html#compute-fabric-576)是 GB200 的姊妹篇。
* [Hot Chips Talk about the NVLink Switch](https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf)：关于 NVLink 与 NCCL 集合通信的有趣报告，尤其包含网内归约。
* [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437)：大型半开放 LLM 训练报告的好范本，连切分方案怎么定的都写了。
* [How to Optimize a CUDA Matmul](https://siboehm.com/articles/22/CUDA-MMM)：博客佳作：用 CUDA 核心写高效矩阵乘，处处照顾 GPU 的缓存一致性。
* [HuggingFace Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)：GPU 上的 LLM 并行指南，本章部分灵感来自它。
* [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)：更偏 GPU 和 PyTorch 的屋顶线与性能工程入门。
* [Cornell Understanding GPU Architecture site](https://cvw.cac.cornell.edu/gpu-architecture)：气质与本书相近的指南，更具体地对比 GPU 与 CPU 的内部。

## 附录 A {#appendix-a-how-does-this-change-with-gb200}：换成 GB200 会怎样？

Blackwell 带来一堆组网大改，包括 NVLink 5——NVLink 总带宽直接翻倍（900GB/s）。B200 仍是 8 卡节点、和 H100 相同；但 GB200 系统（B200 GPU + Grace CPU 的组合）把 NVLink 域做得大得多（NVL72 为 72 卡，理论上可到 576）。NVLink 域变大，等效于节点出口带宽同步放大，节点以上的集合通信因此变便宜。

{% include figure.liquid path="assets/gpu/b200-node.png" class="img-small" caption="<b>图：</b>GB200 NVL72 单元的构造：18 台交换机、72 张 GPU。" %}

节点内，带宽从 450GB/s 到 900GB/s 并没有让屋顶线怎么变——毕竟每 GPU 的 FLOPs/s 也翻倍了。总体格局不变；只是 NVLink 带宽毕竟强得多，专家并行会更容易做。

出了节点，变化就大了。这是[这里](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html#compute-fabric-576)的 SuperPod 图。

{% include figure.liquid path="assets/gpu/gb200-superpod.png" class="img-fluid" caption="<b>图：</b>576 卡的 GB200 DGX SuperPod。" %}

如你所见，每节点出口带宽升到 `4 * 18 * 400 / 8 = 3.6TB/s`，而 H100 只有 400GB/s。每芯片 FLOPs 只是翻倍，跨节点屋顶线因此实打实好了约 4 倍。接下来要担心的，反倒变成节点内会不会成为瓶颈，而不是扩容网络。

**Grace Hopper：**NVIDIA 还卖 GH200 / GB200 这类“GPU + Grace CPU”一体机。比如 GH200 是 1 张 H200 配 1 颗 Grace CPU，GB200 是 2 张 B200 配 1 颗 Grace CPU。好处在于 CPU 和 GPU 之间是满速 NVLink 直连（NVLink C2C），CPU 到 GPU 带宽极高，把参数卸载到主机内存非常合算。换句话说，对任意一张 GPU 来说，读主机内存的带宽和读另一张 GPU 的 HBM 完全相同。

## 附录 B：更多组网细节

这是一台 NVLink 4 交换机的图：共 64 个 NVLink4 端口（每个占 2 条物理链路），加一个大的横排（crossbar）负责跨链路交换。TPU 则不然，用的是带镜片的可动态重配光交换机。

{% include figure.liquid path="assets/gpu/nvlink4.png" class="img-fluid" caption="<b>图：</b>单台 NVLink4 交换机的更底层视图。" %}

在每一层，瓶颈都可能在链路带宽或交换机总带宽之间产生。

* **节点层**：节点上有 4 * 1.6TB/s = 6.4TB/s 的 NVSwitch 带宽，但 8 张 GPU 各自只能向交换机送 450GB/s，于是节点内的峰值实际是 450e9 * 8 = 3.6TB/s（全双工）。
* **SU/叶层**：8 台交换机把 32 个节点以 1x400 Gbps InfiniBand 全互连。节点侧出口合计 8 * 32 * 400 / 8 = 12.8TB/s，交换机侧 8 * 1.6TB/s = 12.8TB/s——两边分毫不差。
* **脊层**：16 台交换机用 2x400 Gbps 链路连接 32 台叶交换机，出口合计 32 * 16 * 400 * 2 / 8 = 51.2TB/s。与叶交换机不同，脊交换机的 64 个端口全部朝下，每台可转发 64 * 400 / 8 = 3.2TB/s，全层 16 * 3.2TB/s = 51.2TB/s——又是分毫不差。

摊到每 GPU：节点层 GPU-GPU 带宽 450GB/s；SU 层和脊层都是 50GB/s。

**GPU 实测 AllReduce 带宽：**

{% include figure.liquid path="assets/gpu/gpu-all-reduce-bw.png" class="img-fluid" caption="<b>图：</b>8xH100 集群（节点内、禁用 SHARP）的 AllReduce 带宽。" %}

TPU v5p 带宽（沿一条轴）：

{% include figure.liquid path="assets/gpu/tpu-all-reduce-bw.png" class="img-fluid" caption="<b>图：</b>TPU v5p 4x4x4 集群（沿一条轴）的 AllReduce 带宽。" %}

AllGather 带宽同样奉上：

{% include figure.liquid path="assets/gpu/gpu-all-gather-bw.png" class="img-fluid" caption="<b>图：</b>8xH100 集群（节点内）的 AllGather 带宽。" %}

{% include figure.liquid path="assets/gpu/tpu-all-gather-bw.png" class="img-fluid" caption="<b>图：</b>TPU v5e 8x16 集群（沿一条轴）的 AllGather 带宽。" %}

**关于 AllToAll 成本的补充：**

这里把近似式 $\min(K / Z) * (Z - 1) / Z$ 与真实值 $(1 - ((Z - 1) / Z) ** K) * (Z - 1) / Z$ 放在一起对比。除了 $Z$ 很小的一段，两者非常接近。

{% include figure.liquid path="assets/gpu/all-to-all-approx.png" class="img-fluid" caption="<b>图：</b>随着分片数增加，不规则 AllToAll 的近似成本与真实成本对比。" %}
