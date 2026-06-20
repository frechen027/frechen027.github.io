# 大模型算法八股框架完整答案清单

---

## 一、基础

### 1. KL散度、交叉熵、两者关系

**交叉熵 (Cross-Entropy):**
$H(P, Q) = -\sum_{x} P(x) \log Q(x)$

**KL散度 (Kullback-Leibler Divergence):**
$D_{KL}(P \| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}$

**关系:**
$H(P, Q) = H(P) + D_{KL}(P \| Q)$

即交叉熵 = 信息熵 + KL散度。由于真实分布P的信息熵H(P)是常数，最小化交叉熵等价于最小化KL散度。

**关键性质:**

- KL散度非对称：$D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$
- KL散度非负：$D_{KL} \geq 0$，当且仅当P=Q时取等号
- 正向KL（forward KL）：$D_{KL}(P\|Q)$，P为真实分布，Q为模型分布，倾向于覆盖P的所有模式（mean-seeking）
- 反向KL（reverse KL）：$D_{KL}(Q\|P)$，倾向于让Q集中在P的某个模式上（mode-seeking）

### 2. 大模型中的幻觉、复读机等现象的成因与解决方法

**幻觉 (Hallucination):**

成因：

- **训练数据偏差**：训练数据中存在错误信息或噪声
- **自回归生成的累积误差**：每一步预测的微小偏差在长序列中累积
- **知识边界**：模型对训练数据中未覆盖的知识进行"编造"
- **过度自信**：softmax输出概率分布过于尖锐，模型对不确定的内容也给出高置信度
- **对齐税 (Alignment Tax)**：RLHF过程中模型可能学到"讨好"人类标注者的模式而非事实准确性

解决方法：

- **RAG (Retrieval-Augmented Generation)**：检索增强生成，引入外部知识
- **事实性RLHF**：在奖励模型中加入事实一致性奖励
- **Self-Consistency**：多次采样取一致性最高的答案
- **Contrastive Decoding**：对比解码，抑制幻觉token
- **知识编辑 (Knowledge Editing)**：直接修改模型内部知识
- **DoLa (Decoding by Contrasting Layers)**：利用不同层的logits差异来减少幻觉
- **引用生成**：要求模型生成答案时附带来源引用

**复读机 (Repetition):**

成因：

- **注意力退化 (Attention Fade)**：随着序列增长，attention权重分散，模型失去对上下文的追踪
- **KV Cache中的近邻相似性**：新生成的token与前面token的KV向量过于相似，导致不断重复
- **解码策略问题**：greedy decoding容易陷入循环
- **训练数据中的重复模式**：模型在训练中见过大量重复文本

解决方法：

- **Repetition Penalty**：对已出现的token施加惩罚
- **Frequency/Presence Penalty**：OpenAI使用的频率惩罚和存在惩罚
- **对比解码 (Contrastive Decoding)**：SimCTG等方法
- **增加温度**：提高采样随机性
- **Top-p/Top-k采样**：限制候选token集合
- **Min-p采样**：过滤掉概率过低的token
- **N-gram约束**：禁止生成已出现的n-gram

### 3. 为什么使用Decoder-only架构

- **通用性**：Decoder-only可以建模任意方向的依赖关系（通过因果掩码实现从左到右），通过适当的prompt设计可以完成各种任务（in-context learning）
- **缩放效率**：相比Encoder-Decoder，相同参数量下Decoder-only的FLOPs利用率更高，因为所有参数都参与自回归生成
- **统一范式**：一个模型可以同时处理理解（如分类、抽取）和生成任务，通过指令微调实现
- **预训练效率**：自回归语言建模目标（next token prediction）简单高效，可以利用海量无标注文本
- **规模化表现**：GPT系列证明了Decoder-only在规模增大后涌现出强大的能力（涌现能力）
- **工程简洁**：只需实现一种attention mask，推理时KV Cache机制天然适配
- **对比Encoder-only (BERT)**：只能做双向理解，无法直接生成；对比Encoder-Decoder (T5)：Encoder部分在推理时不参与生成，参数利用率低

### 4. 梯度爆炸、梯度消失、梯度饱和

**梯度消失 (Vanishing Gradient):**

成因：

- 反向传播时梯度连乘，如果每层的梯度值小于1，经过多层后会趋近于0
- 常见于Sigmoid、Tanh激活函数（导数最大值为0.25和1）
- 深层网络中尤为严重

解决办法：

- ReLU及其变体（LeakyReLU、GELU等），正区间梯度为1
- ResNet残差连接：梯度可以通过skip connection直接传播
- LayerNorm/RMSNorm：稳定前向传播的数值范围
- LSTM/GRU的门控机制
- 合理的权重初始化（Xavier、He初始化）

**梯度爆炸 (Exploding Gradient):**

成因：

- 反向传播时梯度连乘，如果每层的梯度值大于1，经过多层后会指数级增长
- 权重初始化过大
- 学习率过大

解决办法：

- **梯度裁剪 (Gradient Clipping)**：当梯度范数超过阈值时按比例缩放
- 合理的权重初始化
- BatchNorm/LayerNorm稳定中间层输出
- 降低学习率、使用warmup
- 使用门控机制（LSTM）

**梯度饱和 (Gradient Saturation):**

成因：

- 激活函数在输入值很大或很小时导数趋近于0（如Sigmoid两端）
- 模型输出过于自信（logits值很大），softmax后梯度很小

解决办法：

- 使用非饱和激活函数（ReLU、GELU）
- Label Smoothing：避免one-hot标签导致的过度自信
- 合理的损失函数设计
- 梯度裁剪

### 5. GPT、BERT、CLIP、Llama

**GPT (Generative Pre-trained Transformer):**

- Decoder-only架构
- 自回归语言建模（next token prediction）
- 因果掩码（causal mask），只能看到前文
- GPT-1→GPT-2→GPT-3→GPT-4：规模从1.17亿→15亿→1750亿→更大
- 关键创新：In-context learning、Chain-of-thought

**BERT (Bidirectional Encoder Representations from Transformers):**

- Encoder-only架构
- Masked Language Modeling (MLM)：随机mask 15%的token进行预测
- Next Sentence Prediction (NSP)：判断两个句子是否连续
- 双向注意力，可以看到完整上下文
- 适合理解类任务（分类、NER、问答抽取）

**CLIP (Contrastive Language-Image Pre-training):**

- 双塔架构：Image Encoder (ViT) + Text Encoder (Transformer)
- 对比学习：将图像和文本映射到同一嵌入空间
- 训练目标：最大化匹配图文对的相似度，最小化不匹配对的相似度
- 零样本分类能力：通过文本prompt实现图像分类
- 后续发展：OpenCLIP、Chinese-CLIP、SigLIP

**Llama (Large Language Model Meta AI):**

- Decoder-only架构
- 关键改进：
    - Pre-Norm：使用RMSNorm（而非LayerNorm）
    - SwiGLU激活函数（而非ReLU/GELU）
    - RoPE旋转位置编码
    - GQA（Grouped Query Attention），Llama-2 70B开始使用
- Llama-1：7B/13B/33B/65B，1T token训练
- Llama-2：7B/13B/70B，2T token，更长上下文4096
- Llama-3：8B/70B，15T token，128K上下文，GQA全系列使用
- 开源许可，推动了开源大模型生态

### 6. Python进程/线程/协程、GIL锁、异步计数器、async

**进程 (Process):**

- 操作系统资源分配的最小单位
- 每个进程有独立的内存空间
- 进程间通信(IPC)开销大：管道、消息队列、共享内存、socket
- 可以跨多核CPU并行执行
- `multiprocessing`模块

**线程 (Thread):**

- CPU调度的最小单位
- 同一进程内的线程共享内存空间
- 线程间通信方便（直接读写共享变量），但需要同步机制（锁、信号量）
- 创建和切换开销小于进程

**GIL (Global Interpreter Lock):**

- CPython解释器中的全局锁
- 同一时刻只有一个线程执行Python字节码
- 原因：CPython的内存管理（引用计数）不是线程安全的
- 影响：CPU密集型任务无法利用多线程并行
- 规避方法：
    - 使用`multiprocessing`多进程
    - 使用C扩展释放GIL（如NumPy）
    - Python 3.13+实验性支持free-threaded模式（no-GIL）

**协程 (Coroutine):**

- 用户态的轻量级线程
- 在单线程内通过协作式调度实现并发
- 切换开销极小（不需要内核态切换）
- 适合I/O密集型任务
- `asyncio`库实现

**async/await:**

```python
import asyncio

async def fetch_data():
    await asyncio.sleep(1)  # 非阻塞等待
    return "data"

async def main():
    # 并发执行多个协程
    results = await asyncio.gather(
        fetch_data(),
        fetch_data(),
        fetch_data()
    )
```

**异步计数器:**

```python
class AsyncCounter:
    def __init__(self):
        self._count = 0
        self._lock = asyncio.Lock()

    async def increment(self):
        async with self._lock:
            self._count += 1
            return self._count
```

注意：在单线程asyncio中，如果没有await，代码是原子的，不需要锁。但如果有await让出控制权，则需要锁保护共享状态。

### 7. SGD优化器、Momentum、Adam、AdamW、Muon

**SGD (Stochastic Gradient Descent):**
$\theta_{t+1} = \theta_t - \eta \cdot g_t$

- 最基础的优化器
- 收敛慢，容易陷入局部最优
- 需要精心调学习率

**SGD + Momentum:**
$v_t = \mu v_{t-1} + g_t$
$\theta_{t+1} = \theta_t - \eta \cdot v_t$

- 引入动量项，累积历史梯度方向
- 加速收敛，减少震荡
- μ通常设为0.9

**Adam (Adaptive Moment Estimation):**
$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(一阶矩估计)}$
$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(二阶矩估计)}$
$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \quad \text{(偏差修正)}$
$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$

- 自适应学习率，每个参数有独立的学习率
- β1=0.9, β2=0.999, ε=1e-8
- 收敛快，对超参数不敏感
- 广泛用于大模型预训练

**AdamW:**

- 在Adam基础上加入**解耦权重衰减** (Decoupled Weight Decay)
  $\theta_{t+1} = \theta_t - \eta \cdot \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)$
- 原始Adam中权重衰减被自适应学习率缩放，效果减弱
- AdamW将权重衰减与梯度更新解耦，正则化效果更好
- 大模型训练的标准优化器

**Muon (Momentum-based optimization with orthogonalization):**

- 2024-2025年提出的新优化器
- 核心思想：对动量矩阵进行正交化处理（通过Newton-Schulz迭代近似矩阵极分解）
- 使得更新方向在参数空间中更加均匀
- 在矩阵参数（如Linear层的权重矩阵）上使用Muon，向量参数（如bias、LayerNorm）上使用AdamW
- 优势：收敛更快，在大模型训练中展现出优于AdamW的性能
- 由 Keller Jordan 等人提出，在开源社区引起关注

### 8. 深拷贝浅拷贝

**浅拷贝 (Shallow Copy):**

- 创建新对象，但内部子对象仍然是引用（不拷贝子对象）
- 修改新对象的子对象会影响原对象

```python
import copy
a = [[1, 2], [3, 4]]
b = copy.copy(a)  # 或 a.copy()
b[0][0] = 99
print(a[0][0])  # 99，被修改了
```

**深拷贝 (Deep Copy):**

- 递归拷贝所有层级的对象
- 新对象与原对象完全独立

```python
c = copy.deepcopy(a)
c[0][0] = 100
print(a[0][0])  # 99，不受影响
```

**在深度学习中的意义:**

- 模型参数拷贝：`model.load_state_dict()`是深拷贝
- 数据增强：需要深拷贝避免修改原始数据
- 内存效率：大模型中尽量避免不必要的深拷贝

### 9. 智能指针

C++中的智能指针，用于自动管理内存：

**unique_ptr:**

- 独占所有权，不可复制，可以移动
- 零开销抽象，性能等同裸指针

```cpp
std::unique_ptr<Tensor> ptr(new Tensor());
auto ptr2 = std::make_unique<Tensor>();
```

**shared_ptr:**

- 共享所有权，引用计数
- 多个shared_ptr指向同一对象，最后一个销毁时释放内存
- 循环引用问题：需要`weak_ptr`打破循环

```cpp
std::shared_ptr<Tensor> ptr1 = std::make_shared<Tensor>();
std::shared_ptr<Tensor> ptr2 = ptr1; // 引用计数+1
```

**weak_ptr:**

- 不增加引用计数
- 观察shared_ptr管理的对象
- 使用前需要`lock()`转为shared_ptr
- 解决循环引用

**在深度学习框架中的应用:**

- PyTorch的C++后端(LibTorch)大量使用shared_ptr管理Tensor
- 自动微分引擎中用shared_ptr管理计算图节点
- 分布式训练中共享模型参数

### 10. 为什么大模型选用交叉熵损失

- **概率解释**：语言建模本质是估计下一个token的条件概率分布，交叉熵直接衡量预测分布与真实分布的差异
- **最大似然估计**：最小化交叉熵等价于最大化训练数据的对数似然
- **梯度特性**：交叉熵 + softmax的组合梯度形式简洁：$\frac{\partial L}{\partial z_i} = p_i - y_i$，预测越偏离真实，梯度越大
- **信息论基础**：交叉熵衡量编码效率，语言建模本质是信息压缩
- **与困惑度 (Perplexity) 的关系**：$PPL = e^{CE}$，交叉熵越低，困惑度越低，模型越好
- **对比MSE**：MSE假设高斯分布，不适合离散分类问题；MSE在概率接近0或1时梯度很小（饱和问题）

### 11. 为什么回归用MSE，分类用交叉熵

**回归用MSE:**

- 回归目标是连续值，MSE直接衡量预测值与真实值的距离
- 假设误差服从高斯分布时，MSE等价于最大似然估计
- 梯度线性：$\frac{\partial MSE}{\partial \hat{y}} = 2(\hat{y} - y)$，误差越大梯度越大
- 替代方案：MAE (L1 loss) 对异常值更鲁棒，Huber Loss兼顾两者

**分类用交叉熵:**

- 分类目标是离散类别，需要输出概率分布
- 交叉熵衡量两个概率分布的差异，天然适合分类
- 配合softmax使用，梯度形式简洁且不会出现梯度饱和
- MSE用于分类时：输出经过softmax后，MSE的梯度会因softmax的饱和而变得很小，导致学习缓慢
- 交叉熵 + softmax：梯度 = 预测概率 - 真实标签，误差越大梯度越大

---

## 二、训练推理加速

### 1. 推理优化

#### vLLM：PagedAttention、Continuous Batching

**PagedAttention:**

- 核心思想：借鉴操作系统的虚拟内存分页机制管理KV Cache
- 问题：传统KV Cache需要预分配连续内存，导致严重的内存碎片和浪费（内部碎片+外部碎片）
- 解决：将KV Cache分成固定大小的block（如16个token一块），通过block table映射逻辑块到物理块
- 优势：
    - 内存利用率接近100%（几乎无碎片）
    - 支持动态分配和释放
    - 支持copy-on-write实现beam search和parallel sampling的block共享
    - 可以处理更长的序列和更大的batch size

**Continuous Batching (连续批处理):**

- 问题：传统static batching中，一个batch必须等所有序列都生成完毕才能处理下一个batch。短序列生成完后GPU空闲等待长序列
- 解决：
    - 迭代级别调度（iteration-level scheduling）：每次forward后都可以加入新请求、移除已完成的请求
    - 不同长度的请求可以在同一个batch中混合处理
    - 显著提高GPU利用率和吞吐量
- vLLM实现：通过Scheduler管理请求队列，每次iteration动态组batch

#### KV Cache

- 自回归生成时，每生成一个新token需要用到之前所有token的Key和Value
- 不缓存：每步都要重新计算所有token的K、V，复杂度O(n²)
- 缓存：只计算新token的Q、K、V，与缓存的K、V做attention，复杂度O(n)
- 内存占用：$2 \times n_{layers} \times n_{heads} \times d_{head} \times seq\_len \times batch\_size \times dtype\_size$
- 优化方向：
    - MQA/GQA：减少KV head数量
    - 量化KV Cache：FP16→INT8/INT4
    - PagedAttention：高效内存管理
    - Prefix Caching：共享相同前缀的KV Cache
    - Sliding Window Attention：只缓存最近W个token的KV

#### Prefill & Decode

**Prefill阶段 (Prompt Processing):**

- 处理用户输入的prompt（所有token并行计算）
- 计算量大（矩阵乘法密集），计算密集型（compute-bound）
- 生成第一个token
- 填充KV Cache
- 关键指标：TTFT (Time To First Token)

**Decode阶段 (Token Generation):**

- 逐个生成后续token（自回归）
- 每步只处理1个token，矩阵运算退化为向量运算
- 内存带宽受限（memory-bound）：需要从显存读取整个KV Cache和模型权重，但计算量很小
- 关键指标：TPOT (Time Per Output Token) / 吞吐量(tokens/s)

**优化策略不同:**

- Prefill：提高计算利用率，tensor parallelism
- Decode：减少内存访问，量化、投机解码、batching

#### 梯度检查点 & 梯度累计

**梯度检查点 (Gradient Checkpointing):**

- 问题：训练时保存所有中间激活值用于反向传播，显存占用O(n)
- 解决：只保存部分关键层的激活值（checkpoint），反向传播时重新计算未保存的中间值
- 时间换空间：增加约20-30%计算量，显存从O(n)降到O(√n)
- 使用：`torch.utils.checkpoint.checkpoint`

**梯度累计 (Gradient Accumulation):**

- 问题：大batch size需要大量显存
- 解决：将大batch分成多个micro-batch，每个micro-batch计算梯度后累加，最后再更新参数
- 等效batch size = micro_batch_size × accumulation_steps
- 不改变模型行为（线性累加梯度），只减少显存峰值
- 分布式训练中常与DP结合使用

### 2. 训练/模型优化

#### Megatron-LM并行策略

**数据并行 (DP / DDP):**

- DP (Data Parallelism)：PyTorch原生，单进程多线程，GIL限制，效率低
- DDP (Distributed Data Parallel)：多进程，每个进程有完整模型副本，梯度AllReduce同步
- 适用场景：模型能放入单卡显存
- 通信：每次backward后AllReduce梯度

**张量并行 (Tensor Parallelism, TP):**

- 将单层内部的矩阵运算拆分到多个GPU
- 对Attention层：按head维度切分（每个GPU负责部分head）
- 对MLP层：按hidden维度切分（Column Parallel + Row Parallel）
- 适用场景：单层太大无法放入单卡（通常用于模型内部的大矩阵运算）
- 通信：每层forward/backward需要AllReduce或AllGather
- 通常在单机内使用（NVLink高带宽）

**流水线并行 (Pipeline Parallelism, PP):**

- 将模型按层切分到不同GPU（如GPU0: Layer 0-7, GPU1: Layer 8-15）
- 问题：气泡 (bubble) ——前面的GPU计算时后面的GPU空闲
- 微批次调度策略：
    - GPipe：所有micro-batch forward完再backward，气泡大
    - 1F1B (One Forward One Backward)：交替执行forward和backward，减少气泡
    - Interleaved 1F1B：每个GPU负责多个不连续的层块，进一步减少气泡
- 适用场景：模型层数很多，跨机部署
- 通信：相邻stage之间传递激活值

**混合精度训练 (Mixed Precision):**

- FP32 master weights + FP16/BF16 计算
- FP16：范围小（±65504），容易溢出，需要loss scaling
- BF16：范围大（同FP32），不需要loss scaling，精度略低
- 三种精度：
    - FP32：master weights、optimizer states
    - FP16/BF16：前向计算、反向计算
    - FP32：梯度累加（避免精度损失）
- 应用场景：几乎所有现代大模型训练都使用

**三种并行的嵌套逻辑:**

```
典型部署：8节点 × 8卡 = 64 GPU
- TP = 8（单机内，NVLink）
- PP = 8（跨机，节点间）
- DP = 1（64 / 8 / 8 = 1）

或：
- TP = 4
- PP = 4
- DP = 4（64 / 4 / 4 = 4）
```

- 原则：TP通信量最大，放在带宽最高的NVLink域内；PP通信量小但延迟敏感，放在同机或高速网络；DP通信量中等（AllReduce梯度），可以跨机

#### DeepSpeed ZeRO

**ZeRO (Zero Redundancy Optimizer):**

- 问题：DDP中每张卡都保存完整的模型参数、梯度、优化器状态，浪费显存
- 核心思想：将优化器状态、梯度、参数分片（partition）到不同GPU，消除冗余

**ZeRO-1 (Optimizer State Partitioning):**

- 将优化器状态（如Adam的m、v）分片到各GPU
- 显存节省4倍（Adam有2个状态 + FP32参数副本）
- 通信量与DDP相同（梯度Reduce-Scatter + 参数AllGather）

**ZeRO-2 (Optimizer State + Gradient Partitioning):**

- 额外将梯度也分片
- 显存节省8倍
- 通信：Reduce-Scatter梯度

**ZeRO-3 (Optimizer State + Gradient + Parameter Partitioning):**

- 参数也分片，每张卡只保存1/N的参数
- 显存节省与GPU数量成线性关系
- 代价：forward和backward时需要AllGather参数，通信量增大
- 支持offload：将参数/优化器状态卸载到CPU内存或NVMe

**ZeRO-Offload / ZeRO-Infinity:**

- 将优化器状态、参数offload到CPU内存甚至NVMe SSD
- 用单卡训练超大模型
- 代价：CPU-GPU数据传输延迟

#### 并行节点通信方式

**Broadcast:** 一个节点发送数据到所有节点（一对多）

**Reduce:** 所有节点发送数据到一个节点，进行归约操作（如sum）（多对一）

**AllReduce:** 所有节点发送数据到所有节点，每个节点都得到归约结果

- 实现：Ring AllReduce、Tree AllReduce
- 用于DDP中同步梯度

**AllGather:** 所有节点发送数据到所有节点，每个节点得到完整数据

- 用于ZeRO-3中收集完整参数

**Reduce-Scatter:** 先归约，再将结果分片到各节点

- 用于ZeRO中梯度同步

**通信原语组合:**

- AllReduce = ReduceScatter + AllGather
- ZeRO使用Reduce-Scatter（梯度分片）+ AllGather（参数收集）

#### FlashAttention

**核心思想:**

- 标准Attention：计算完整的N×N attention矩阵，写入HBM（高带宽内存），再读回计算，IO成本高
- FlashAttention：使用tiling技术，将Q、K、V分成小块，在SRAM（片上高速缓存）中完成attention计算，避免将N×N矩阵写入HBM
- IO-aware：优化HBM↔SRAM的数据传输

**具体实现:**

1. 将Q、K、V分成block（如Br×d, Bc×d）
2. 在SRAM中加载Q block、K block、V block
3. 计算局部attention score和softmax统计量（online softmax）
4. 累加输出，更新softmax归一化因子
5. 写回输出到HBM

**Online Softmax:**

- 传统softmax需要两遍扫描：第一遍求max，第二遍求exp和sum
- Online softmax：一遍扫描，维护running max和running sum
  $m_i = \max(m_{i-1}, \max(x_i))$
  $l_i = e^{m_{i-1}-m_i} l_{i-1} + \sum e^{x_i - m_i}$

**优势:**

- 显存：O(N) → 不需要存储N×N attention矩阵
- 速度：2-4x加速（减少HBM读写）
- 精确：数学上等价于标准attention（不是近似）

**FlashAttention-2:**

- 改进并行策略：减少非矩阵乘法运算
- 改进warp间的工作分配
- A100上达到50-73%的理论FLOPs峰值

**FlashAttention-3 (2024-2025):**

- 利用H100的FP8精度
- 异步执行：warp-level的异步数据加载和计算重叠
- 进一步优化H100架构特性
- 支持稀疏attention模式

#### MQA、GQA、MLA

**MHA (Multi-Head Attention):**

- 每个head有独立的Q、K、V投影
- n_heads个Q head，n_heads个K head，n_heads个V head
- KV Cache大小：$2 \times n_{layers} \times n_{heads} \times d_{head} \times seq\_len$

**MQA (Multi-Query Attention):**

- 所有Q head共享同一组K、V head
- n_heads个Q head，1个K head，1个V head
- KV Cache缩小n_heads倍
- 缺点：模型质量略有下降

**GQA (Grouped Query Attention):**

- 折中方案：将Q head分成若干组，每组共享一组K、V
- n_heads个Q head，n_kv_heads个K/V head（如8组）
- KV Cache缩小n_heads/n_kv_heads倍
- 质量接近MHA，速度接近MQA
- Llama-2 70B、Llama-3全系列使用

**MLA (Multi-head Latent Attention) — DeepSeek-V2/V3:**

- 核心思想：将K、V通过低秩投影压缩到低维latent space，推理时只缓存压缩后的latent
- 具体实现：
    - $c_t = W_{DKV} h_t$（压缩到低维 $d_c \ll d_{model}$）
    - 从$c_t$解压缩出K、V：$K_t = W_{UK} c_t$, $V_t = W_{UV} c_t$
    - Q使用独立的投影保持表达力
- KV Cache压缩比极大（DeepSeek-V2报告压缩93.3%）
- 代价：forward时需要额外的解压缩计算，但推理时内存带宽大幅节省

### 3. 模型训练和推理显存需求分析

**训练显存组成:**

1. **模型参数 (Model Parameters):**
    - FP16/BF16参数：$2 \times P$ bytes（P为参数量）

2. **优化器状态 (Optimizer States):**
    - Adam：FP32参数副本(4P) + 一阶矩m(4P) + 二阶矩v(4P) = 12P bytes
    - 加上FP16梯度(2P) = 共20 bytes/parameter

3. **激活值 (Activations):**
    - 与batch size、序列长度、模型结构相关
    - 粗略估计：$O(L \times d \times s \times b)$
    - 使用gradient checkpointing可大幅减少

4. **临时缓冲区 (Fragments/Temp buffers):**
    - 通信buffer、对齐填充等

**以Llama-7B为例 (AdamW, BF16):**

- 参数(BF16): 7B × 2 = 14 GB
- 优化器状态: 7B × 12 = 84 GB
- 梯度(BF16): 7B × 2 = 14 GB
- 总计: ~112 GB（不含激活值）
- 至少需要8×A100-80G或2×A100-80G + ZeRO-3

**推理显存组成:**

1. **模型参数:**
    - FP16: 2 bytes/parameter
    - INT8: 1 byte/parameter
    - INT4: 0.5 byte/parameter

2. **KV Cache:**
    - $2 \times n_{layers} \times n_{kv\_heads} \times d_{head} \times seq\_len \times batch\_size \times \text{dtype\_size}$
    - Llama-70B (GQA, 8 KV heads, d_head=128, 80 layers):
        - Per token per sample: 2 × 80 × 8 × 128 × 2 = 327,680 bytes ≈ 0.31 MB
        - 128K context: ~40 GB per sample (FP16)

3. **中间激活值:**
    - 推理时只需当前步的激活，通常较小

**经验法则:**

- 训练：~20 bytes/parameter（AdamW, BF16, 不含激活）
- 推理FP16：~2 bytes/parameter + KV Cache
- 推理INT4：~0.5 bytes/parameter + KV Cache

---

## 三、Transformer内部

### 1. Norm归一化

#### 内部协变量偏移 (Internal Covariate Shift)

- 定义：训练过程中，每一层输入的分布随着前面层的参数更新而不断变化
- 影响：导致训练不稳定，需要较小的学习率，收敛慢
- BatchNorm的提出动机就是通过归一化稳定每层输入的分布

#### Pre-norm、Post-norm、Deep-norm

**Post-norm (原始Transformer):**
$x_{l+1} = \text{Norm}(x_l + \text{Sublayer}(x_l))$

- 先做sublayer（attention/FFN），再加残差，最后归一化
- 问题：深层网络中梯度不稳定，难以训练

**Pre-norm (GPT-2, 现代大模型主流):**
$x_{l+1} = x_l + \text{Sublayer}(\text{Norm}(x_l))$

- 先归一化，再做sublayer，最后加残差
- 优势：训练更稳定，梯度流更平滑
- 缺点：最终输出未经归一化，需要在最后加一层LayerNorm

**DeepNorm (2022, 超深Transformer):**
$x_{l+1} = \text{Norm}(\alpha \cdot x_l + \text{Sublayer}(x_l))$

- 对残差连接乘以系数α > 1（如α = (2N)^{1/4}）
- 配合特定的权重初始化策略
- 可以稳定训练非常深的Transformer（如1000层）

#### BatchNorm、LayerNorm、RMSNorm

**BatchNorm:**

- 在batch维度做归一化
- $\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$，其中μ_B和σ_B在batch维度计算
- 训练时：使用当前batch的统计量，同时维护running mean/var
- 推理时：使用训练期间累积的running mean/var
- 问题：
    - 依赖batch size，小batch效果差
    - 对变长序列（如NLP）不适用（不同样本长度不同）
    - 在Transformer中几乎不使用

**LayerNorm:**

- 在feature维度做归一化
- $\hat{x} = \frac{x - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}} \cdot \gamma + \beta$
- μ_L和σ_L在每个样本的feature维度上计算
- 可学习参数γ(scale)和β(shift)
- 不依赖batch，适合NLP和变长序列
- 训练和推理行为相同（没有running statistics）

**RMSNorm (Root Mean Square Layer Normalization):**
$\hat{x} = \frac{x}{\text{RMS}(x)} \cdot \gamma = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \cdot \gamma$

- 去掉了mean subtraction（减均值）步骤
- 只有scale参数γ，没有shift参数β
- 假设：mean subtraction不是必须的，re-scaling才是关键
- 计算量更小（不需要计算均值和方差）
- 实验证明效果与LayerNorm相当甚至更好

**为什么现在大模型都使用RMSNorm:**

1. **降低计算量**：不需要计算均值，减少了约50%的归一化计算
2. **偏移自带隐含信息**：mean subtraction移除的信息可能不是关键的，re-scaling已经足够
3. **经过softmax近似归一化**：attention中的softmax本身就有归一化效果
4. **更简洁**：少一个可学习参数β，减少过拟合风险
5. **实验验证**：Llama、PaLM、Gemini等主流大模型都使用RMSNorm

**LayerNorm vs RMSNorm的归一化维度:**

- 都在最后一个维度（hidden dimension）做归一化
- 对shape为(batch, seq_len, hidden_dim)的张量，在hidden_dim维度计算统计量

### 2. 激活函数

**Sigmoid:** $\sigma(x) = \frac{1}{1+e^{-x}}$

- 输出(0,1)，适合二分类输出
- 问题：梯度消失（两端饱和）、输出非零均值

**Tanh:** $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

- 输出(-1,1)，零均值
- 问题：仍然有梯度消失

**ReLU:** $\text{ReLU}(x) = \max(0, x)$

- 正区间梯度为1，解决梯度消失
- 问题：Dead ReLU（负区间永久死亡）、输出非零均值

**LeakyReLU:** $\text{LeakyReLU}(x) = \max(\alpha x, x)$，α通常为0.01

- 解决Dead ReLU问题
- 负区间有小梯度

**GELU (Gaussian Error Linear Unit):**
$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$

- BERT、GPT使用
- 平滑的ReLU变体，概率解释
- 近似计算：$0.5x(1 + \tanh[\sqrt{2/\pi}(x + 0.044715x^3)])$

**GLU (Gated Linear Unit):**
$\text{GLU}(x) = \sigma(xW_1) \otimes (xW_2)$

- 门控机制，一部分做sigmoid gate，另一部分做变换后相乘
- 信息选择性地通过

**Swish / SiLU:** $\text{Swish}(x) = x \cdot \sigma(x)$

- Google发现，效果优于ReLU
- 平滑、非单调

**SwiGLU:**
$\text{SwiGLU}(x) = \text{Swish}(xW_1) \otimes (xW_2)$

- Swish + GLU的结合
- PaLM、Llama系列使用
- FFN层变为：$\text{SwiGLU}(x) = \text{Swish}(xW_1) \otimes (xW_3) \cdot W_2$
- 引入额外的参数矩阵W3，增加模型容量
- 实验证明比GELU效果更好，成为大模型FFN的标准选择

**为什么SwiGLU能取代GELU:**

- 门控机制提供了更好的信息过滤能力
- 非单调性允许更复杂的特征学习
- 实验上在相同FLOPs下表现更好
- 代价：FFN层多一个权重矩阵，但通常通过缩小hidden_dim来保持总FLOPs不变

### 3. 位置编码

**正余弦位置编码 (Sinusoidal PE, 原始Transformer):**
$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$
$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$

- 固定编码，不需要学习
- 可以表示相对位置关系（线性组合）
- 外推性有限

**可学习的位置编码 (Learned PE):**

- 直接学习每个位置的embedding向量
- GPT-2、BERT使用
- 限制：最大长度固定，无法外推到更长的序列

**RoPE (Rotary Position Embedding):**

- 核心思想：通过旋转矩阵将位置信息编码到Q和K中
- 对Q和K的每两个相邻维度施加一个旋转角度，角度与位置成正比
  $f(q, pos) = q \cdot e^{i \cdot pos \cdot \theta}$
- 优势：
    - 编码了相对位置信息（两个位置的内积只与相对距离有关）
    - 可以外推到更长序列（配合NTK-aware scaling、YaRN等）
    - 不需要额外参数
- Llama、Qwen、PaLM等主流模型使用
- 长文本外推方法：
    - Position Interpolation (PI)：压缩位置索引
    - NTK-aware Scaling：调整频率基底
    - YaRN：结合多种方法的综合方案
    - Dynamic NTK：动态调整

**RoPE长文本外推失效的根本原因:**

- 训练时模型只见过一定范围内的位置编码频率
- 超出训练长度时，高频分量的旋转角度超出训练分布
- Attention score计算中出现未见过的模式，导致注意力分布异常
- 本质是频率外推问题：高频分量在训练时未被充分学习

**ALiBi (Attention with Linear Biases):**

- 不使用位置embedding，直接在attention score上加线性偏置
- $\text{score}_{ij} = q_i \cdot k_j - m \cdot |i - j|$
- m是每个head不同的斜率（预先设定，越远的head斜率越小）
- 优势：无需额外计算，天然支持任意长度外推
- 缺点：偏置是固定的线性函数，表达力有限

### 4. 其他

#### 为什么除以根号dk

$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

- Q和K的每个元素假设是均值0、方差1的独立随机变量
- $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$，方差为$d_k$
- 当$d_k$较大时，点积的绝对值会很大，导致softmax进入梯度极小的饱和区
- 除以$\sqrt{d_k}$将方差归一化为1，使softmax保持在梯度合理的区域
- 实验验证：不缩放时大$d_k$下训练不稳定

#### Encoder-Decoder注意力结构

**Encoder中的多头注意力:**

- Self-Attention：每个token可以attend到所有其他token
- 双向注意力，无mask

**Decoder中的注意力:**

- **Causal Masked Self-Attention (因果掩码自注意力):**
    - 每个token只能attend到它自己和之前的token
    - 使用下三角mask矩阵，上三角部分设为-∞
    - 保证自回归特性（生成时不泄露未来信息）

- **Cross-Attention (交叉注意力):**
    - Decoder提供Q，Encoder提供K和V
    - Decoder的每个位置可以attend到Encoder的所有位置
    - 用于Encoder-Decoder架构（如T5、原始Transformer）
    - Decoder-only架构（GPT、Llama）不使用Cross-Attention

#### 为什么要分为Q、K、V三个矩阵

- **Q (Query)**：当前token想要查询什么信息
- **K (Key)**：每个token提供的"索引"，用于被查询
- **V (Value)**：每个token的实际内容，被提取的信息

**为什么不能只用两个矩阵（如Q和V）:**

- 如果K=Q：自己与自己的点积总是最大的，attention会退化为恒等映射
- 如果K=V：attention score和提取的内容耦合，表达力受限
- 三个独立的投影矩阵提供了最大的灵活性：
    - Q和K的投影决定"匹配函数"（哪些token相关）
    - V的投影决定"提取什么信息"
    - 解耦了"查询什么"和"返回什么"

#### Dropout训练和推理时的行为区别

**训练时:**

- 以概率p随机将某些神经元的输出置为0
- 剩余神经元的输出除以(1-p)进行缩放（inverted dropout）
- 目的：防止过拟合，增强泛化能力
- 每次forward使用不同的dropout mask

**推理时:**

- 不使用dropout，所有神经元都参与计算
- 不需要缩放（因为训练时已经通过inverted dropout处理了）
- 确定性输出

**大模型中的Dropout:**

- 大多数现代大模型（GPT-3、Llama等）不使用dropout
- 原因：模型规模巨大，数据量巨大，过拟合不是主要问题
- 正则化主要通过weight decay（AdamW）实现

---

## 四、强化学习

### Value-based Methods

#### 蒙特卡洛法 (Monte Carlo)

- 通过完整的episode回报来估计状态价值
- $V(s) = \mathbb{E}[G_t | S_t = s]$，其中$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$
- 优点：无偏估计
- 缺点：需要等到episode结束，方差大
- 不适合 continuing task（无终止的任务）

#### SARSA (State-Action-Reward-State-Action)

- On-policy TD控制方法
- 更新规则：$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$
- 使用实际执行的下一个action A\_{t+1}
- 相对保守，考虑了exploration的风险

#### 时序差分法 (Temporal Difference, TD)

- 结合蒙特卡洛和动态规划的思想
- 用估计的值来更新估计的值（bootstrapping）
- TD(0)：单步更新
- TD(λ)：多步更新，结合n-step returns
- 优点：可以在线学习，不需要等到episode结束
- 缺点：引入偏差（bootstrapping）

#### Q-learning

- Off-policy TD控制方法
- 更新规则：$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]$
- 使用贪心策略选择下一个action（max操作）
- 直接学习最优action-value函数
- 与行为策略无关（off-policy）

#### DQN (Deep Q-Network)

- 用神经网络近似Q函数
- 关键技术：
    - **Experience Replay**：存储经验样本，随机采样打破相关性
    - **Target Network**：使用单独的目标网络计算TD target，稳定训练
    - **Reward Clipping**：将奖励裁剪到[-1, 1]
- 更新：$L = \mathbb{E}[(R + \gamma \max_{a'} Q_{target}(S', a') - Q(S, A))^2]$

### Policy-based Methods

#### REINFORCE

- 最基础的策略梯度方法
- 目标：$\nabla J(\theta) = \mathbb{E}[\nabla \log \pi_\theta(a|s) \cdot G_t]$
- 使用完整的episode回报G_t作为信号
- 优点：简单，直接优化策略
- 缺点：方差大，需要baseline（如V函数）来减少方差

#### Actor-Critic

- 结合Policy-based和Value-based
- Actor：策略网络$\pi_\theta(a|s)$，负责选择action
- Critic：价值网络$V_\phi(s)$，负责评估状态价值
- Actor更新：$\nabla J(\theta) = \mathbb{E}[\nabla \log \pi_\theta(a|s) \cdot A(s,a)]$
- Critic更新：最小化$V_\phi(s)$与target的MSE
- Advantage $A(s,a) = R + \gamma V(s') - V(s)$，减少方差

#### TRPO (Trust Region Policy Optimization)

- 限制策略更新幅度，保证单调改进
- 约束：$\mathbb{E}[D_{KL}(\pi_{old} \| \pi_{new})] \leq \delta$
- 使用共轭梯度法求解约束优化
- 理论保证：每次更新策略性能不会下降太多
- 缺点：计算复杂，需要二阶优化

#### PPO (Proximal Policy Optimization)

- TRPO的简化版本，一阶优化即可
- 核心：clip机制限制策略更新幅度
  $L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$，概率比
- 当advantage > 0时：限制r不超过1+ε（防止过度增加概率）
- 当advantage < 0时：限制r不低于1-ε（防止过度减少概率）

**PPO的clip解决什么问题:**

- 防止策略更新过大，导致性能崩溃
- 不clip的话：如果某个action获得高reward，概率比可能变得非常大，策略过度偏向这个action，导致exploration不足和训练不稳定
- Clip提供了简单有效的trust region近似

**PPO loss组成:**
$L = L^{CLIP} - c_1 L^{VF} + c_2 S[\pi_\theta]$

- $L^{CLIP}$：策略优化目标（clip surrogate）
- $L^{VF}$：价值函数损失（MSE）
- $S[\pi_\theta]$：熵奖励（鼓励exploration）
- $c_1, c_2$：权重系数

**PPO的clip ratio ε:**

- 通常设为0.1-0.2
- 太小：更新过于保守，学习慢
- 太大：失去clip的保护作用

#### GRPO (Group Relative Policy Optimization)

**核心思想:**

- 不需要单独的Critic/Value网络
- 对同一个prompt采样一组（group）回答，用组内的相对排名作为advantage
- 奖励来源：规则模型（rule-based reward）或奖励模型（RM）

**具体流程:**

1. 对每个prompt，采样G个回答 $\{o_1, o_2, ..., o_G\}$
2. 对每个回答计算奖励 $\{r_1, r_2, ..., r_G\}$
3. 计算组内归一化的advantage：$\hat{A}_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$
4. 使用PPO-style clip更新策略

**GRPO loss:**
$L_{GRPO} = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{\min\left[\rho_{i,t} \hat{A}_i, \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_i\right] - \beta D_{KL}(\pi_\theta \| \pi_{ref})\right\}\right]$

**GRPO为什么不需要单独训练RM:**

- 对于有明确规则的任务（如数学、代码），可以使用规则模型直接打分
- 对于需要RM的任务，GRPO仍然可以使用RM，但不需要训练Critic网络
- 关键创新：用组内相对排名替代绝对价值估计，省去了Critic的训练

**GRPO的超参调整:**

- Group size G：越大advantage估计越稳定，但计算成本增加（通常8-64）
- KL系数β：控制与reference model的偏离程度
- Clip ratio ε：通常0.1-0.2
- 温度：影响采样多样性

**GRPO在dense和MoE上的表现:**

- Dense模型：GRPO效果良好
- MoE模型：由于MoE的稀疏激活特性，GRPO可能需要调整（如更大的group size）
- GSPO是对GRPO在MoE上的改进

#### GSPO (Group-level Sequence Policy Optimization)

**GSPO vs GRPO的区别:**

- GRPO在token level计算advantage（每个token共享同一个sequence-level的advantage）
- GSPO在sequence level计算advantage，并考虑整个序列的联合优化
- GSPO的gap理解：GRPO假设组内每个样本的advantage是独立的，但实际序列之间存在相关性

**GSPO为什么效果更优:**

- 更好地处理序列级别的全局优化
- 在MoE模型上表现更好，因为MoE的路由机制使得token-level的优化不够稳定
- 减少了GRPO中的方差

**GSPO和GRPO的显存占用:**

- 两者都需要存储group内所有样本的logits和梯度
- GSPO可能需要额外的显存用于sequence-level的计算
- 实际差异取决于实现细节

#### DAPO (Decoupled Alignment Policy Optimization)

**DAPO的六个改进点:**

1. **Dynamic Sampling**：动态调整采样策略，避免重复采样相似的prompt
2. **Adaptive KL**：自适应调整KL系数，根据训练阶段动态变化
3. **Decoupled Reward**：将不同来源的奖励解耦，分别处理
4. **Overlong Reward Penalty**：对超长回答施加惩罚，避免模型通过生成冗长文本来获取奖励
5. **Token-level Loss**：在token level而非sequence level计算loss，更精细
6. **Dual-Clip**：使用双重clip机制，更好地控制策略更新

#### 其他Policy方法

**DPO (Direct Preference Optimization):**

- 不需要训练RM和RL过程
- 直接从偏好数据优化策略
- 目标：$\max_\theta \mathbb{E}[\log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)})]$
- 优点：简单，不需要RM
- 缺点：offline方法，受限于偏好数据质量

**KTO (Kahneman-Tversky Optimization):**

- 不需要成对偏好数据，只需要good/bad标注
- 基于前景理论（Prospect Theory）
- 损失函数考虑了人类对损失和收益的非对称敏感性

**IPO (Identity Preference Optimization):**

- DPO的改进版本
- 解决了DPO在确定性偏好下的过拟合问题
- 使用更平滑的优化目标

**Reference Free Alignment:**

- 不需要reference model的对齐方法
- 如KTO、IPO的某些变体
- 减少了推理和训练时的计算开销

### 其他关键概念

#### 模态/熵坍塌问题

**Entropy Collapse:**

- 策略的熵持续下降，最终变得确定性（只输出少数几个token）
- 导致exploration不足，模型陷入局部最优
- 表现：生成多样性急剧下降，模式崩溃

**如何避免:**

- 熵奖励：在loss中加入熵项，鼓励高熵策略
- Clamped entropy：设置熵的下界，当熵低于阈值时增加熵奖励权重
- 温度调节：动态调整采样温度
- KL散度约束：限制与reference model的偏离

**Clamped Entropy:**

- 设置熵的目标范围$[H_{min}, H_{max}]$
- 当$H < H_{min}$时，增加熵奖励系数
- 当$H > H_{max}$时，减少熵奖励系数
- 通常$H_{min}$设为初始熵的某个比例

**Policy entropy的重要性:**

- 衡量策略的随机性/exploration程度
- 高熵：充分exploration，但可能效率低
- 低熵：exploitation为主，可能陷入局部最优
- 理想的训练过程：熵逐渐下降但不过快，保持足够的exploration

**KL散度在RL中的作用:**

- 约束策略更新幅度，防止偏离reference model太远
- 防止reward hacking（模型找到RM的漏洞而非真正改进）
- 保持生成多样性

**KL估计器K1, K2, K3:**

- K1（非偏估计）：$D_{KL} = \mathbb{E}_{\pi_\theta}[\log \pi_\theta - \log \pi_{ref}]$
- K2（Schulman估计）：$D_{KL} \approx \mathbb{E}[\frac{\pi_\theta}{\pi_{ref}} - 1 - \log \frac{\pi_\theta}{\pi_{ref}}]$，更稳定
- K3（另一种近似）：$D_{KL} \approx \mathbb{E}[\frac{1}{2}(\log \frac{\pi_\theta}{\pi_{ref}})^2]$

**正向KL vs 反向KL在大模型RL中:**

- 正向KL：$D_{KL}(\pi_{ref} \| \pi_\theta)$，reference→policy
    - Mean-seeking：policy倾向于覆盖reference的所有模式
    - 保持多样性，但可能生成低质量样本
- 反向KL：$D_{KL}(\pi_\theta \| \pi_{ref})$，policy→reference
    - Mode-seeking：policy倾向于集中在reference的某个高质量模式
    - 生成质量高，但多样性下降
- 对齐几乎都用反向KL：因为目标是提高生成质量，而非保持多样性

**Clip-Cov, KL-Cov:**

- 针对协方差高、优势高的token做特殊处理
- Clip-Cov：对高协方差的token进行更激进的clip
- KL-Cov：用KL散度约束高方差的token更新
- 目的：减少训练中的方差，提高稳定性

**Clip-Higher:**

- 提升概率小但分数高的回答的概率
- 解决GRPO/PPO中低概率高质量样本被忽略的问题
- 实现：对advantage > 0且概率 < 阈值的token，使用更宽松的clip上界

#### Reward Hacking

**定义:**

- 模型找到了奖励函数的漏洞，获得了高奖励但实际质量并未提升
- 例：模型学会生成冗长但空洞的文本来获得长度奖励

**检测方法:**

- 监控reward和实际质量指标（如人工评估、其他metric）的关系
- 如果reward持续上升但质量指标停滞或下降，可能存在reward hacking
- 观察生成样本的多样性变化

**解决方法:**

- KL散度约束：限制策略偏离reference model
- 多奖励融合：使用多个不同的奖励模型
- 奖励归一化：减少极端奖励值的影响
- 定期更新RM：避免模型适应固定的RM
- Overlong penalty：惩罚过长的回答

#### DPO为什么会出现损失下降但性能没有提升

- **过拟合到偏好数据**：DPO可能过拟合到训练偏好数据的特定模式，而非真正学习人类偏好
- **分布偏移**：DPO是offline方法，训练分布和推理分布可能不一致
- **隐式RM的局限**：DPO等价于优化一个隐式RM，但这个隐式RM可能不准确
- **长度偏差**：DPO倾向于选择更长的回答（如果偏好数据中存在长度偏差）
- **缺乏exploration**：DPO不生成新样本，只在已有数据上优化

#### Online vs Offline强化学习

**Online RL:**

- 在训练过程中不断生成新数据
- PPO、GRPO属于online RL
- 优点：数据分布与当前策略匹配
- 缺点：需要不断采样，计算成本高

**Offline RL:**

- 使用预先收集好的固定数据集
- DPO属于offline RL
- 优点：不需要在线采样
- 缺点：分布偏移问题，受限于数据集质量

#### On-policy vs Off-policy

**On-policy:**

- 用于更新策略的数据由当前策略生成
- PPO是on-policy（虽然实践中会用多个epoch，变成近似on-policy）
- GRPO是on-policy

**Off-policy:**

- 用于更新策略的数据可以由其他策略生成
- Q-learning是off-policy
- DPO可以视为off-policy（使用固定的偏好数据）

#### RL训练为什么会让seq_len变长

- RL优化目标是最大化奖励，模型可能发现"生成更长回答"能获得更高奖励
- 特别是当RM对长度有偏好时（如详细回答得分更高）
- 解决方法：
    - 长度惩罚：在reward中减去长度相关项
    - Overlong penalty：对超过一定长度的回答施加惩罚
    - 截断：限制最大生成长度
    - 使用长度无关的RM

#### 什么是on-policy distillation

- 使用当前策略生成的数据来进行知识蒸馏
- 与offline distillation（使用固定数据集）相对
- 优点：蒸馏数据与当前策略分布匹配
- 应用：在大模型RL训练中，用更强的模型指导当前模型

#### 训练中batch size和minibatch

**Batch size:**

- 一次迭代中使用的总样本数
- 在RL中：一次收集的经验数量

**Minibatch:**

- 将batch分成多个minibatch进行多次梯度更新
- 目的：在有限的计算资源下增加更新次数
- 搭配seq_len：总token数 = batch_size × seq_len
- PPO中：一个batch的数据会被使用多个epoch，每个epoch分成多个minibatch

**计算示例:**

- batch_size = 512, seq_len = 2048
- 总token = 512 × 2048 = 1,048,576
- minibatch_size = 64
- 每个minibatch = 64 × 2048 = 131,072 tokens
- 一个batch有 512/64 = 8 个minibatch
- 如果n_epochs = 4，总共更新 8 × 4 = 32 次

#### RL训练你一般看哪些指标

- **Reward**：平均奖励，应该上升
- **KL散度**：与reference model的KL，应该保持在合理范围
- **Policy Entropy**：策略熵，应该缓慢下降但不坍塌
- **Clip Ratio**：被clip的比例，过高说明更新幅度过大
- **Approx KL**：实际策略更新的KL散度
- **Response Length**：平均生成长度，异常增长可能表示reward hacking
- **Accuracy/Quality Metrics**：实际任务性能指标
- **Loss components**：policy loss, value loss, entropy的各自变化

#### 什么是IR，为什么RL要引入IR

**IR (Importance Resampling / Importance Ratio):**

- 重要性采样/重要性比率
- 在RL中用于off-policy学习或多次epoch更新时校正分布偏移
- $r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$
- 目的：用旧策略的数据估计新策略的梯度

#### GUI Agent，Query对应的不同镜像不同拉取

- 在GUI agent训练中，不同的query可能需要不同的环境镜像
- 问题：如何高效管理多个镜像
- 解决方案：
    - 容器化：使用Docker/Kubernetes管理不同环境
    - 镜像缓存：共享基础层，只拉取差异部分
    - 按需加载：根据query类型动态选择镜像
    - 并行环境：使用多个虚拟机/容器并行运行不同环境

#### GRPO在dense和MoE上表现

**Dense模型:**

- GRPO效果良好，组内相对排名提供了稳定的advantage估计
- 不需要Critic网络，节省显存

**MoE模型:**

- MoE的稀疏激活导致不同token经过不同的expert
- 组内样本可能激活不同的expert路径，导致advantage估计不稳定
- GSPO通过sequence-level的优化缓解了这个问题

#### 为什么Clip-Higher有效

- 标准PPO/GRPO的clip是对称的：$[1-\epsilon, 1+\epsilon]$
- 问题：对于概率很低但质量很高的回答，clip上界限制了其概率的增长
- Clip-Higher：对advantage > 0的token，使用更大的上界（如$1+2\epsilon$）
- 效果：允许低概率高质量样本的概率更快增长
- 类似于exploration bonus，鼓励模型发现新的优质模式

#### Policy里的top_p, top_k

**Top-k采样:**

- 只从概率最高的k个token中采样
- 固定候选集大小

**Top-p (Nucleus) 采样:**

- 从累积概率达到p的最小token集合中采样
- 动态候选集大小
- 更灵活：高置信度时候选少，低置信度时候选多

**在RL Policy中的使用:**

- 训练时：通常使用全vocab采样（不截断），保证exploration
- 推理时：使用top-p/top-k控制生成质量
- 有些方法在训练时也使用截断，减少无效token的计算

---

## 五、面经中的具体问题补充

### 1. verl中的hybridengine、auto mapping原理

**verl (Volcano Engine RL):**

- 字节跳动开源的大模型RL训练框架

**HybridEngine:**

- 核心创新：将RL训练和推理引擎融合
- 传统方法：训练和推理使用不同的引擎，需要频繁同步模型参数
- HybridEngine：在同一个引擎中同时支持训练和推理
- 实现：
    - 共享模型权重，避免参数拷贝
    - 动态切换训练模式和推理模式
    - 推理时使用vLLM等高效引擎
    - 训练时使用Megatron/DeepSpeed等分布式训练框架
- 优势：减少参数同步开销，提高整体训练效率

**Auto Mapping:**

- 自动将模型层映射到不同的并行策略
- 根据模型结构和硬件拓扑，自动决定TP/PP/DP的配置
- 减少人工调优成本

### 2. Qwen2.5-VL到Qwen3.5架构变化

**Qwen2.5-VL:**

- 视觉-语言模型
- Vision Encoder (ViT) + Language Model (Transformer)
- Cross-attention或projection layer连接视觉和语言

**Qwen3.5 (假设的原生多模态):**

- "原生多模态"意味着从预训练开始就同时处理多种模态
- 不是先训练语言模型再微调视觉，而是联合训练
- 架构可能变化：
    - 统一的tokenizer处理文本和视觉token
    - 更紧密的模态融合（不是简单的concatenation）
    - 可能使用early fusion而非late fusion
- 为什么称为原生多模态：
    - 模型从一开始就学习跨模态的表示
    - 不是"语言模型 + 视觉适配器"的拼凑
    - 更好的跨模态理解和推理能力

### 3. 线性注意力为什么能把复杂度降成O(n)

**标准Attention:**

- 计算$QK^T$：$O(n^2 d)$
- Softmax后再乘V：$O(n^2 d)$
- 总复杂度：$O(n^2 d)$

**线性Attention:**

- 核心思想：避免显式计算$n \times n$的attention矩阵
- 利用矩阵乘法结合律：$\text{softmax}(QK^T)V \approx \phi(Q)(\phi(K)^T V)$
- 其中$\phi$是核函数映射
- 计算顺序改变：
    - 先计算$K^T V$：$O(n d^2)$（与n线性相关）
    - 再计算$\phi(Q) \cdot (K^T V)$：$O(n d^2)$
- 总复杂度：$O(n d^2)$，对n是线性的
- 代价：失去了精确的softmax attention，是近似方法
- 代表：Linear Transformer、Performer、RetNet等

### 4. 推理时出现重复生成、断句、卡顿的排查

**重复生成:**

- KV Cache问题：检查KV Cache是否正确更新
- Attention问题：检查attention mask是否正确（因果mask）
- Norm问题：检查LayerNorm/RMSNorm的参数是否正确
- 解码策略：检查repetition penalty、temperature等参数

**断句:**

- 可能是EOS token预测过早
- 检查tokenizer的EOS设置
- 可能是模型训练数据中短句子过多

**卡顿:**

- KV Cache内存不足，触发swap或重新分配
- PagedAttention的block分配问题
- Batch size过大导致OOM
- 检查GPU利用率，是否memory-bound

### 5. per-channel和per-token量化

**Per-channel量化:**

- 对每个output channel（或input channel）使用独立的量化参数（scale, zero-point）
- 权重量化常用：每个filter有自己的scale
- 精度较高，但参数量增加

**Per-token量化:**

- 对每个token使用独立的量化参数
- 激活值量化常用：每个token的激活分布不同
- 更好地捕捉token间的差异

**Per-tensor量化:**

- 整个tensor使用一组量化参数
- 最简单，但精度最低

**应用:**

- W8A8量化：权重per-channel，激活per-token
- INT4量化：权重per-channel（group-wise），激活per-token

---

## 六、手撕代码题

### 1. Multi-Head Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q, K, V: (batch, n_heads, seq_len, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Attention
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final linear
        output = self.W_o(attn_output)
        return output

# Causal mask for decoder
def create_causal_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
```

### 2. Grouped Query Attention

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_k = d_model // n_heads

        assert n_heads % n_kv_heads == 0
        self.n_groups = n_heads // n_kv_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)

        Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)

        # Expand K, V to match n_heads
        K = K.repeat_interleave(self.n_groups, dim=1)
        V = V.repeat_interleave(self.n_groups, dim=1)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(output)
```

### 3. Softmax (safe softmax)

```python
def safe_softmax(x, dim=-1):
    # 防止上溢：减去最大值
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

# 带温度的softmax
def softmax_with_temperature(logits, temperature=1.0):
    return safe_softmax(logits / temperature, dim=-1)
```

### 4. 交叉熵损失

```python
def cross_entropy_loss(logits, targets, ignore_index=-100):
    """
    logits: (batch_size, seq_len, vocab_size)
    targets: (batch_size, seq_len)
    """
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_targets = targets[..., 1:].contiguous()

    # Flatten
    shift_logits = shift_logits.view(-1, logits.size(-1))
    shift_targets = shift_targets.view(-1)

    # Safe softmax + log + NLL
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Ignore padding
    mask = shift_targets != ignore_index
    log_probs = log_probs[mask]
    shift_targets = shift_targets[mask]

    loss = F.nll_loss(log_probs, shift_targets)
    return loss
```

### 5. GRPO Loss

```python
def grpo_loss(
    logprobs,        # (batch, group_size, seq_len)
    ref_logprobs,    # (batch, group_size, seq_len)
    rewards,         # (batch, group_size)
    mask,            # (batch, group_size, seq_len)
    epsilon=0.2,
    beta=0.01
):
    """
    GRPO loss implementation
    """
    # Compute advantages (group-relative)
    rewards_mean = rewards.mean(dim=1, keepdim=True)
    rewards_std = rewards.std(dim=1, keepdim=True)
    advantages = (rewards - rewards_mean) / (rewards_std + 1e-8)

    # Expand advantages to token level
    advantages = advantages.unsqueeze(-1).expand_as(logprobs)

    # Compute ratio
    ratio = torch.exp(logprobs - ref_logprobs.detach())

    # Clipped surrogate loss
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2)

    # KL penalty
    kl_penalty = beta * (logprobs - ref_logprobs)

    # Total loss
    loss = (policy_loss + kl_penalty) * mask

    # Average over valid tokens
    loss = loss.sum() / mask.sum()

    return loss
```

### 6. Hot 100 原题（高频算法题）

面试中常考的LeetCode Hot 100题目（与大模型相关的高频题）:

- **Two Sum** (1)
- **Valid Parentheses** (20)
- **Merge Two Sorted Lists** (21)
- **Maximum Subarray** (53)
- **Binary Tree Level Order Traversal** (102)
- **Clone Graph** (133)
- **Course Schedule** (207)
- **LRU Cache** (146) — 非常高频
- **Min Stack** (155)
- **Word Search** (79)
- **Top K Frequent Elements** (347)

---

## 七、总结与面试建议

### 知识体系梳理

1. **基础理论**：概率论、信息论、优化理论
2. **模型架构**：Transformer及其变体、Decoder-only设计
3. **训练技术**：分布式训练、混合精度、高效微调
4. **推理优化**：KV Cache、量化、投机解码
5. **对齐技术**：RLHF、DPO、GRPO等
6. **工程实践**：框架使用、性能调优、问题排查

### 面试准备建议

1. **深度优先**：选择2-3个方向深入准备（如RL + 推理优化 + 分布式训练）
2. **项目驱动**：准备2-3个有深度的项目，能够串联八股知识
3. **手撕代码**：熟练掌握MHA、GQA、softmax、cross-entropy的实现
4. **最新进展**：关注FlashAttention-3、MLA、GRPO/GSPO、DAPO等最新技术
5. **系统设计**：准备大模型训练/推理系统的设计题

### 常见面试流程

1. **项目介绍**（10-15分钟）：深入讨论项目细节、技术选型、遇到的问题
2. **八股问答**（20-30分钟）：从项目出发，串联基础知识
3. **手撕代码**（20-30分钟）：1-2道算法题 + 1道模型相关实现
4. **系统设计**（可选，15-20分钟）：设计大模型训练/推理系统
5. **反问环节**（5-10分钟）

---

这份八股清单覆盖了大模型算法面试的核心知识点，从基础理论到工程实践，从模型架构到训练推理优化。每个知识点都提供了详细的解释和关键细节，适合系统性复习和查漏补缺。建议结合自己的项目经验，将知识点串联起来，形成完整的知识体系。
