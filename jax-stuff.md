---
layout: distill
title: "用 JAX 给 TPU 编程"
# permalink: /main/
description: "怎么用 JAX 高效地给 TPU 编程！本节大量内容取自<a href='https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html'>这里</a>。本节代码示例可以在<a href='https://colab.sandbox.google.com/'>Google Colab</a> 的免费 TPU 上直接运行。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 10

previous_section_url: "../profiling"
previous_section_name: "第 9 部分 · 性能剖析"

next_section_url: ../conclusion
next_section_name: "第 11 部分 · 结语"

giscus_comments: true
giscus_discussion: 12

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
  - name: Yash Katariya
    url: https://x.com/yashk2810
  - name: Reiner Pope<sup>*</sup>
    url: https://x.com/reinerpope

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "JAX 里的并行是怎么工作的？"
  - subsections:
    - name: "Auto 分片模式"
    - name: "Explicit 分片模式"
    - name: "Manual 分片模式：shard_map"
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

## JAX 里的并行是怎么工作的？

JAX 对多设备编程支持三种流派：

1. **编译器，你来开方向盘！**让 XLA 编译器自动切分数组、自动决定要加哪些通信。这样，一个单设备程序不改一行代码，就能自动跑到几千块芯片上。
2. **JAX，你来开方向盘！**自动并行虽好，可编译器时不时会犯疯。Explicit 分片让你照常写单设备代码，但由 JAX（而不是编译器）来做分片传播。遇到看不懂的意图，JAX 会反问你、要你讲清楚。
3. **别猜了，我想跑什么通信我自己写！**编译器虽好，但有时做傻事、加一些你根本没想要的通信。有些时候，我们就是想明明白白地写清楚“我只要这些通信”。

| 模式 | 视角？ | 显式分片？ | 显式集合通信？ |
|:---:|:---:|:---:|:---:|
| Auto | 全局 | ❌ | ❌ |
| Explicit | 全局 | ✅ | ❌ |
| Manual | 每设备 | ✅ | ✅ |

对应地，JAX 给每种模式都提供了 API：

1. `jax.jit`（配 `Auto` 网格轴）：任何现成的 JAX 函数，都可以直接用分片输入去调用。随后 JAX 会交给 XLA 的 [Shardy](https://openxla.org/shardy) 编译器自动并行。需要时（AllGather、ReduceScatter、AllReduce 等等）XLA 会自己补上通信。它并不完美，但大多数时候干得不错：不改代码，就能把你的程序扩到任意数量的芯片。
2. `jax.jit` 配 `Explicit` 网格轴：看起来和 (1) 差不多，但分片传播由 JAX 而不是 XLA 负责。这意味着数组的分片成了 JAX 类型系统的一部分：检测到含糊的通信，JAX 会直接报错，让你来定夺。
3. `jax.shard_map`：更手动的那一头。你拿到的是“单设备视角”的程序，想要什么通信都得自己写。手里有分片数组、想每台设备都拿全量？自己加一个 `jax.lax.all_gather`。想把数组跨设备求和？自己加一个 `jax.lax.psum`（就是 AllReduce）。写起来更累，但也绝不会做出你没要的事。

<h3 id="auto-分片模式">Auto 分片模式</h3>

`jax.jit` 在 JAX 里身兼两职。顾名思义，它“即时”把 Python 函数编译成字节码（经 XLA/HLO/LLO），跑得更快。此外，只要输入是分片的、或者你指定了 `in_sharding`/`out_sharding`，它还会让 XLA 把计算铺到多设备上、自动补通信。比如，用 `jax.jit` 写一个分片矩阵乘：

```py
import jax
import jax.numpy as jnp

Auto = jax.sharding.AxisType.Auto

# This creates a fake set of 8 CPU devices so you can run this on a CPU without TPUs.
jax.config.update("jax_num_cpu_devices", 8)

# This creates a 2D 4x2 mesh with axis names X and Y that JAX uses by default.
# We explicitly tell JAX to let the XLA compiler infer sharding along these axes.
mesh = jax.make_mesh(axis_sizes=(4, 2), axis_names=('X', 'Y'), axis_types=(Auto, Auto))
jax.set_mesh(mesh)

# We create a matrix W and input activations In sharded across our devices.
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, device=jax.NamedSharding(mesh, jax.P('X', 'Y')))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, device=jax.NamedSharding(mesh, jax.P('Y', None)))

def matmul_square(In, W):
  return jnp.einsum('bd,df->bf', jnp.square(In), W)

# We can explicitly compile the sharded matmul function here. This adds all the
# necessary comms (e.g. an AllReduce after the matmul).
jit_matmul = jax.jit(matmul_square, out_shardings=jax.P('X', None)).lower(In, W).compile()

out = jit_matmul(In, W)
```

它会自动按分片运行、把计算铺到设备上。**那硬件层面到底发生了什么？**

1. 我们先创建了跨设备分片的 In 和 W<d-footnote>注意我们是怎么创建的。这是给数组指定分片的一种方式（在创建函数里加 device 参数）。另一种是先用 `jnp.array(....)` 正常创建、再 `jax.device_put(..., jax.P('X', 'Y'))`。还有一种：写一个创建该数组的函数，用想要的 `out_shardings` 去 jit 编译它。</d-footnote>。W 沿收缩维切了 2 份；In 切了 8 份：输入维切 4、收缩维切 2。按记号就是 W[D<sub>Y</sub>, F]、In[B<sub>X</sub>, D<sub>Y</sub>]——等于模型并行和数据并行各来了一点。
2. 如果只在一台设备上跑，`matmul_square` 就是先平方再矩阵乘，平平无奇。但我们把 `out_shardings` 指定为 `P('X', None)`：输出沿批维分片、沿模型维复制——要算出来，就少不了一次 AllReduce。

用前几节的记号，它多半会做这样一件事：

1. Out[B<sub>X</sub>, F] { U<sub>Y</sub> } = In[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F]
2. Out[B<sub>X</sub>, F] = **AllReduce**(Out[B<sub>X</sub>, F] { U<sub>Y</sub> })

`jax.jit` 全自动帮我们加好了！用 `jit_matmul.as_text()` 还能把 HLO 打印出来（大幅节选）：

```py
# This fusion is the actual matmul of the sharded inputs and matrix
%fusion = bf16[2,8192]{1,0:T(4,128)(2,1)S(1)} fusion(bf16[2,1024]{1,0:T(4,128)(2,1)} %param, bf16[8192,1024]{1,0:T(8,128)(2,1)S(1)} %copy-done)

# We reduce the partially summed results across devices
ROOT %AllReduce = bf16[2,8192]{1,0:T(4,128)(2,1)} AllReduce(bf16[2,8192]{1,0:T(4,128)(2,1)S(1)} %fusion)
```

上面就是矩阵乘（fusion）和 AllReduce。特别留意形状：`bf16[2, 1024]` 是激活的**局部视图**——`batch_size=8` 切给了 4 台设备、`d_model=2048` 又切了 2 份。

**这很魔幻！**不管程序多复杂，[Shardy](https://openxla.org/shardy) 和 jit 都会尝试给所有中间激活找出分片、按需补通信。话虽如此，Shardy 也有毛病、也会犯错。有时你打开 profile 就会看到不对劲的地方：一个根本不需要的大 AllGather 占了 80% 的时长。这时候，我们可以出手“纠偏”——给中间张量显式标注 `jax.lax.with_sharding_constraint`。比如两次矩阵乘之间，我可以强行让中间激活沿 `y` 维分片（虽然这不是什么好主意）：

```py
import jax
import jax.numpy as jnp

Auto = jax.sharding.AxisType.Auto

mesh = jax.make_mesh((4, 2), ('X', 'Y'), (Auto, Auto))
jax.set_mesh(mesh)

def matmul(x, W_in, W_out):
  hidden = jnp.einsum('bd,df->bf', x, W_in)
  hidden = jax.lax.with_sharding_constraint(hidden, jax.P('X', 'Y'))
  return jnp.einsum('bf,df->bd', hidden, W_out)
```

在“用 `jax.lax.with_sharding_constraint` 控制中间分片”的自动切分世界里，这样的活儿大概占了 JAX 并行编程的 60%。可“挠编译器痒痒”是出了名的不好玩：你可以把每个中间变量都标一遍，还是不敢保证拿到想要的结果。那么，如果分片传播这件事本来就由 JAX 自己来管、来控，会怎样？

<h3 id="explicit-分片模式">Explicit 分片模式</h3>

Explicit 分片（又叫“分片进类型”）看起来和自动分片很像，只是分片传播发生在 JAX 这一层！每个 JAX 运算都有一条分片规则：读入参的分片，推出结果的分片。用 `jax.typeof` 就能看到结果分片：

```py
import jax
import jax.numpy as jnp
import numpy as np

Explicit = jax.sharding.AxisType.Explicit

# Running on a TPU v5e 2x2. This assigns names to the two physical axes of the hardware.
mesh = jax.make_mesh(axis_sizes=(2, 2), axis_names=('X', 'Y'), axis_types=(Explicit, Explicit))

# This tells JAX to use this mesh for all operations, so you can just specify the PartitionSpec P.
jax.set_mesh(mesh)

x = jax.device_put(np.arange(16, dtype=np.float32).reshape(8, 2), jax.P('X', 'Y'))

@jax.jit
def f(x):
  print(jax.typeof(x))  # float32[8@X,2@Y]
  out = x * 2
  print(jax.typeof(out))  # float32[8@X,2@Y]
  return out

f(x)
```

可以看到，JAX 把分片从输入 (`x`) 传播到了输出 (`out`)，跟踪期就能用 `jax.typeof` 查看。对多数运算，这些规则简单明了，因为只有一种合理选择（比如逐元素运算原样保留分片）。但有些运算的结果分片是有歧义的，这时 JAX 会在跟踪期直接报错，请程序员显式给出 `out_sharding` 参数（比如 jnp.einsum、jnp.reshape 等）。再看一个会起冲突的例子：

```py
# We create a matrix W and input activations In sharded across our devices.
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, out_sharding=jax.P('X', 'Y'))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, out_sharding=jax.P('Y', None))

@jax.jit
def matmul_square(In, W):
  print(jax.typeof(In))  # bfloat16[8@X, 2048@Y]
  print(jax.typeof(W))  # bfloat16[2048@Y, 8192]
  return jnp.einsum('bd,df->bf', jnp.square(In), W)

matmul_square(In, W)  # This will error
```

这段代码会报错：

```
Contracting dimensions are sharded and it is ambiguous how the output should be sharded.
Please specify the output sharding via the `out_sharding` parameter.
Got lhs_contracting_spec=('Y',) and rhs_contracting_spec=('Y',)
```

这非常好：einsum 的输出分片确实有歧义。输出既可以：
* P('X', 'Y')——会诱发一次 ReduceScatter；或者
* P('X', None)——会诱发一次 AllReduce

和 Auto 模式不同，Explicit 模式检测到歧义通信就报错，必须你来定。于是你可以写：

```py
@jax.jit
def matmul_square(In, W):
  return jnp.einsum('bd,df->bf', jnp.square(In), W, out_sharding=jax.P('X', 'Y'))

out = matmul_square(In, W)
print(jax.typeof(out))  # bfloat16[8@X,8192@Y]
```

Auto 模式和 Explicit 模式可以用 `jax.sharding.auto_axes` 与 `jax.sharding.explicit_axes` 组合使用。想深入了解，[这篇文档](https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html)非常值得一读。

<h3 id="manual-分片模式-shard-map">Manual 分片模式：shard_map</h3>

如果说 Shardy 是“编译器你来开方向盘”模式，那 jax [shard_map](https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html) 就是把方向盘整个交到你手里。输入的分片还是像 `jax.jit` 那样指定，但之后所有通信都得自己写。`jax.jit` 给你的是跨设备的全局视图，`shard_map` 给你的则是每台设备的局部视图。

看个例子。先试着想想这个函数在干什么：<d-footnote>想在自己的 Colab 里假装一张网格玩一玩，可以跑这个单元：`import jax; jax.config.update('jax_num_cpu_devices', 8)`</d-footnote>

```py
import jax
import jax.numpy as jnp

Explicit = jax.sharding.AxisType.Explicit

mesh = jax.make_mesh((2, 4), ('x', 'y'), (Explicit, Explicit))
jax.set_mesh(mesh)

x = jnp.arange(0, 512, dtype=jnp.int32, out_sharding=jax.P(('x', 'y')))

# This function will operate on 1/8th of the array.
@jax.shard_map(in_specs=jax.P(('x', 'y')), out_specs=jax.P())
def slice_and_average(x):
  assert x.shape == (512 // 8,)
  return jax.lax.pmean(x[:4], axis_name=('x', 'y'))

out = slice_and_average(x)
assert out.shape == (4,)
```

**它到底干了什么？**`slice_and_average` 会在每台 TPU 上各拿数组的 1/8 运行：从各自那一份里切出前 4 个元素，再在整张网格上求平均。等价于 `mean(x[:4], x[64:68], x[128:132], …)`。这很妙——因为这种操作在别的写法里根本不好表达。

**为什么不直接用 jax.jit？**若用 `jax.jit`，`slice_and_average` 看到的就是全局视图（完整的 `[512,]` 数组）。你得先把这段“不均匀的切片”抠出来、再求平均，还得指望 XLA 正确理解你的意图——它可能加错通信，也可能直接懵掉。而在局部视图里，我们看得清清楚楚，只写需要的那点通信。

**例子 [集合式矩阵乘]：**来看更真实的场景。假设要实现模型并行，激活一开始就是模型分片的，即 A[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D, F<sub>Y</sub>] -> Out[B<sub>X</sub>, F<sub>Y</sub>]。朴素做法是先对 A 做 AllGather，再本地矩阵乘：

1. A[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(A[B<sub>X</sub>, D<sub>Y</sub>])
2. Out[B<sub>X</sub>, F<sub>Y</sub>] = A[B<sub>X</sub>, D] *<sub>D</sub> W[D, F<sub>Y</sub>]

可惜这不好：通信没法和计算重叠。要做重叠，得用“集合式矩阵乘（collective matmul）”，可参考 [Wang et al. 2023](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959)。算法大致是：

* 对每个 Y 分片：用本地那块 A 乘本地那块 W，得到 `[B / X, F / Y]` 的结果；与此同时，把 A 做一次置换，让下一块“自己送上门”，算完后把结果累加起来。

用 `jax.shard_map` 实现它相当容易：

```py
import functools

import jax
import jax.numpy as jnp
import numpy as np

Explicit = jax.sharding.AxisType.Explicit

# This is intended to run on a TPU v5e-8 runtime. If you can't get this,
# try setting jax.config.update('jax_num_cpu_devices', 8).
#
mesh = jax.make_mesh(axis_sizes=(2, 4), axis_names=('X', 'Y'), axis_types=(Explicit, Explicit))
jax.set_mesh(mesh)

B, D, F = 1024, 2048, 8192
A = jnp.arange(np.prod((B, D))).reshape((B, D))
W = jnp.arange(np.prod((D, F))).reshape((D, F))

A = jax.device_put(A, jax.P('X', 'Y'))
W = jax.device_put(W, jax.P(None, 'Y'))

@functools.partial(jax.jit, out_shardings=jax.P('X', 'Y'))
def matmul(lhs, rhs):
  return lhs @ rhs

def collective_matmul_allgather_lhs_contracting(lhs, rhs):
  # lhs is the looped operand; rhs is the local operand
  axis_size = jax.lax.axis_size('Y')  # axis_size = 4 for this example
  idx = jax.lax.axis_index('Y')

  chunk_size = lhs.shape[1]
  assert rhs.shape[0] % chunk_size == 0

  def f(i, carrys):
    accum, lhs = carrys
    rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, (idx + i) % axis_size * chunk_size, chunk_size)
    # Matmul for a chunk
    update = lhs @ rhs_chunk
    # Circular shift to the left
    lhs = jax.lax.ppermute(
        lhs,
        axis_name='Y',
        perm=[(j, (j - 1) % axis_size) for j in range(axis_size)]
    )
    return accum + update, lhs

  accum = jnp.zeros((lhs.shape[0], rhs.shape[1]), dtype=lhs.dtype)
  accum = jax.lax.pcast(accum, ('X', 'Y'), to='varying')
  accum, lhs = jax.lax.fori_loop(0, axis_size - 1, f, (accum, lhs), unroll=True)

  # Compute the last chunk after the final permute to leave lhs in the state we found it
  i = axis_size - 1
  rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, (idx + i) % axis_size * chunk_size, chunk_size)
  update = lhs @ rhs_chunk
  return accum + update

jit_sharded_f = jax.jit(jax.shard_map(
  collective_matmul_allgather_lhs_contracting,
  in_specs=(jax.P('X', 'Y'), jax.P(None, 'Y')), out_specs=jax.P('X', 'Y')))

shmapped_out = jit_sharded_f(A, W)
expected_out = matmul(A, W)

np.testing.assert_array_equal(shmapped_out, expected_out)
```

漂亮！实测一跑，果然还快了不少！[这是](https://imgur.com/a/e9I6SrM)默认 jit 矩阵乘的 profile：311 微秒，开头杵着一个碍事的大 AllGather：

{% include figure.liquid path="assets/img/not-overlapped.png" class="img-fluid" %}

[这是](https://imgur.com/a/21iy0Sv)上面这份实现：244 微秒。profile 里再也看不到 AllGather——全是正经活！FLOPs 利用率也高得多。

{% include figure.liquid path="assets/img/overlapped.png" class="img-fluid" %}

还值得注意：收缩维不分片时的矩阵乘耗时是 [224 微秒](https://imgur.com/a/i3gNKfq)——我们已经非常接近“完全不分片”的基线了。想把 TPU 利用率榨上去，做的就是这样那种性能工程。更多 `shard_map` 例子，[这篇笔记](https://jax.readthedocs.io/en/latest/notebooks/shard_map.html#example-1-all-gather-on-one-side)写得很好。

下面来几道实用的题，用 `jax.jit` 或 `shard_map` 写写看！

## 动手解题

这里是一些零散的 JAX 相关题目，以后还会再加。所有题目都需要若干块 TPU。Colab 现在不发 TPU v2-8 切片了，请用 [Kaggle](https://www.kaggle.com/)（目前仍免费）或 GCP 的 8 核切片<d-footnote>若只想在假问题上模拟一张网格，也可以在 CPU 上假装 8 台设备：`import jax; jax.config.update('jax_num_cpu_devices', 8)`（需要 jax >= 0.4.27 左右），不过性能不代表真实情况。</d-footnote>以下都假设你手上有 N 台设备。

**问题 1：**设 **A** 是形状为 float32[S<sub>X</sub>, D<sub>Y</sub>] 的激活数组，`X * Y = N`。请：

1. 用 JAX 写一个函数，对每个 `(X, Y)` 分片内部求平均，即返回一个 `[X, Y]` 的数组，其中 `arr[i, j]` 是分片 `(i, j)` 的均值。分别用 `jax.jit` 和 `shard_map` 各写一遍，各自 profile 一下用了多久。有被偷偷加上通信吗？*提示：本不该有，但 XLA 有时就是会加。*

2. 用 JAX 写一个函数，在**每个沿 X 的分片内部**返回 `roll(x, shift, axis=0) - x`（某个 shift）。我不打算残忍到逼你用 jax.jit 写这个，用 `shard_map` 就好。

{% details 点此查看答案。 %}

**第 1 部分：**下面是第 1 问的一种解法。注意 `jax.jit` 那份要绕好几个挺复杂的 reshape。

```py
import numpy as np

import jax
import jax.numpy as jnp

Auto = jax.sharding.AxisType.Auto

mesh = jax.make_mesh((4, 2), ('X','Y'), (Auto, Auto))

average_shmap = jax.shard_map(
    lambda x: x.mean(keepdims=True),
    mesh=mesh,
    in_specs=jax.P('X','Y'), out_specs=jax.P('X','Y')
)

def average(x):
  X, Y = mesh.axis_sizes
  return x.reshape(X, x.shape[0] // X, Y, x.shape[1] // Y).mean(axis=(1, 3))

average_jit = jax.jit(average, out_shardings=jax.NamedSharding(mesh, jax.P('X','Y')))

x = jnp.arange(8 * 64 * 8, dtype=jnp.float32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, jax.P('X','Y')))

y1 = average_shmap(x)
y2 = average_jit(x)

np.testing.assert_array_equal(y1, y2)
```

**第 2 部分：**下面是第 2 问的类似解法。题目要的是 `shard_map` 版，`jax.jit` 版附上作对照。

```py
import numpy as np

import jax
import jax.numpy as jnp

import functools

Auto = jax.sharding.AxisType.Auto

mesh = jax.make_mesh((4, 2), ('X','Y'), (Auto, Auto))

def shift_shmap(x, shift: int):
  shmapped = jax.shard_map(
      lambda x: jnp.roll(x, shift, axis=0) - x,
      mesh=mesh,
      in_specs=jax.P('X','Y'), out_specs=jax.P('X','Y')
  )
  return shmapped(x)

@functools.partial(jax.jit, static_argnames=['shift'], out_shardings=jax.NamedSharding(mesh, jax.P('X','Y')))
def shift_jit(x, shift: int):
  X, Y = mesh.axis_sizes
  reshaped = x.reshape(X, x.shape[0] // X, -1)
  return (jnp.roll(reshaped, shift, axis=1) - reshaped).reshape(x.shape[0], x.shape[1])

x = jnp.arange(8 * 64 * 8, dtype=jnp.float32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, jax.P('X','Y')))

y1 = shift_shmap(x, 5)
y2 = shift_jit(x, 5)

np.testing.assert_array_equal(y1, y2)
```

{% enddetails %}

**问题 2：**我们一起做一个基础的“专家混合”（mixture of experts）模型。设 **W**：float32[E<sub>X</sub>, D, F] 是 E 个“专家”矩阵；**A**：float32[S<sub>X</sub>, D] 是激活；**B**：int32[S<sub>X</sub>] 是“路由分配”，B[i] 是 `[0, E)` 内的整数，表示这条激活该交给哪个矩阵处理。请实现 `Out[i] = A[i] @ W[B[i]]`。

1. 先完全不管分片，把这些张量都小到能塞进一台设备。写出本地实现。*注意别真的物化出一个 `[S, D, F]` 的数组！提示：办法不止一个。最省事的是：循环每个专家，把整个 **A** 乘 `W[e]`，再把没路由到专家 `e` 的 token 掩掉。或者：把 token 按专家排序、装进形状 `[E, S, D]` 的补齐缓冲区，做一次批量矩阵乘、掩掉 padding（为什么第二维必须也是 S？）。`jax.lax.ragged_dot` 做的正是“排好序、不补齐”的那个版本。*

2. 把上面的方法直接 `jax.jit`，会发生一些事。profile 一下：它自己决定了什么通信？花了多久？

3. 你大概已经注意到，上面多半会把整份激活 **A** 收集到本地，即 AllGather<sub>X</sub>([S<sub>X</sub>, D])。这不但通信贵，而且本地装不下全部激活时内存更贵。请用 `shard_map` 加显式通信重新实现。

      1. 第一版可以直接 `jax.lax.all_gather`，然后按第 1 问的方式重排。

      2. 第二版试着别再物化任何 `[E, S, D]` 大小的数组：在 `jax.lax.while_loop` 里用 `jax.lax.all_to_all` 以不规则（ragged）方式算。这样既不必物化全部激活，也不用给 padding 白算。和你最初的实现比，快了多少？

4. 多数 MoE 会路由到 k 个专家再对结果取平均。把上面的实现重构成这种。此时路由输入变成 **B**：int32[S<sub>X</sub>, k]，即每条激活对应 k 个专家。

{% details 点此查看（部分）答案。 %}

**第 1 部分：**选择很多。这里给最简版：只管按专家循环加掩码，token 永远不排序：

```py
def moe_local(W: jnp.ndarray, A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    S, _ = A.shape
    E, _, F = W.shape

    def expert_forward(carry, e):
        output = carry  # [S, F]
        mask = (B == e)[:, None]  # [S, 1]
        expert_result = A @ W[e]  # [S, F] - this expert's transform of ALL tokens
        output = output + expert_result * mask  # Only keep results for assigned tokens
        return output, None

    output = jnp.zeros((S, F))
    output, _ = jax.lax.scan(expert_forward, output, jnp.arange(E))

    return output
```

这浪费计算——每个专家都要处理所有 token。更省的做法是按专家排序后交给 `jax.lax.ragged_dot`：给它排好序的激活和每个专家的 token 数，它就能在无补齐、无掩码的情况下完成矩阵乘。

**第 3 部分：**这里只勾勒一下伪代码（你若写出了干净的解法，欢迎补上）：

```py
chunk_size = 128
def matmul(W, x, B):
  i = 0
  outs = []
  x = # sort x according to assignments
  while (chunk := x[i:i+chunk_size]).any():
     chunk = all_to_all(chunk)
     outs.append(matmul_local(W, chunk))
     i += chunk_size
  return concat(outs)
```

基本思路：一块一块地过数组——排序、all_to_all、然后干本地的活。

{% enddetails %}

**问题 3：**上面的集合式矩阵乘其实和真实 LLM 关系很大。我们把它改造成完整的 Transformer 栈。

1. 先练手实现 AllReduce 版集合式矩阵乘：A[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F] -> Out[B<sub>X</sub>, F]。注意输出只沿 X 分片，即沿 Y 复制。朴素算法上文讲过：本地矩阵乘 + AllReduce。请做一个通信可重叠的“集合式”版本。*提示：沿输出维分块，放心使用 `jax.lax.psum`（即 AllReduce）。* *注：受 XLA 处理方式的限制，它实际可能并不比基线快。*

2. 与 AllReduce 版相对的是 ReduceScatter 版：Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W2[F<sub>Y</sub>, D] -> Out[B<sub>X</sub>, D<sub>Y</sub>]。Transformer 的下投影矩阵正是这个形状。请在 JAX 里实现它的集合式、可重叠版本。注意只传非传不可的那点数据。*提示：边累加边置换结果。*

3. 把正文的 AllGather 版集合式矩阵乘和第 (2) 问的 ReduceScatter 版拼起来，做成一个端到端的 Transformer 块：In[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D] -> Out[B<sub>X</sub>, D<sub>Y</sub>]，通信全程重叠。<d-footnote>和之前一样，中间有我们省略的非线性，所以不能先算 $W_{in} \cdot W_{out}$。</d-footnote>比 `jax.jit` 实现快多少？

**问题 4：**上面所有集合式矩阵乘都是单向的：只朝一个方向置换。请把 AllReduce 版和 ReduceScatter 版都改写成双向通信。能快多少？

### 第 10 部分到此结束。全书正文也就到这里了！最终结论与延伸阅读，[点击这里](../conclusion)。
