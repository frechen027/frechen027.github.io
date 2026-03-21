# Python 深度学习笔记

!!! info "简介"
    本笔记主要参考自 B 站 **爆肝杰哥** 的深度学习视频教程。这是本人首次正式系统性地学习深度学习，旨在掌握这一核心技能。
    
    注: 本章环境搭建过程仅作记录，后续实战将主要利用 **Kaggle** 的免费 GPU 资源完成。

## 环境搭建与配置

### 1. Conda 虚拟环境管理

在开始深度学习之前，建议为每个项目创建独立的虚拟环境，以避免依赖冲突。以下是常用的 Conda 命令速查：

=== "基础操作"
    ```bash
    # 清屏 (Windows)
    cls
    
    # 列出所有环境
    conda env list
    
    # 激活虚拟环境
    conda activate <环境名>
    
    # 退出当前虚拟环境
    conda deactivate
    ```

=== "环境创建与删除"
    ```bash
    # 创建名为 "env_name" 的环境，指定 Python 版本 (默认在 conda 安装目录下的 envs 中)
    conda create -n env_name python=3.9
    
    # 指定路径创建环境 (例如在当前目录下)
    conda create --prefix=./env_name python=3.9
    
    # 删除环境 (及其所有包)
    conda remove -n env_name --all
    ```

### 2. 常用库管理 (Pip)

进入虚拟环境后，使用 `pip` 安装所需的科学计算库。

!!! tip "加速安装技巧"
    使用国内镜像源（如清华源）可以显著提升下载速度：
    `-i https://pypi.tuna.tsinghua.edu.cn/simple`

```bash
# 查看当前环境下的所有库 (Conda)
conda list

# 查看特定库的版本
pip show numpy

# 安装基础科学计算库 (指定版本示例)
pip install numpy==1.21.5 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pandas==1.2.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install matplotlib==3.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. GPU 与 CUDA 基础

深度学习通常需要 GPU 加速。理解显卡驱动、CUDA 与 PyTorch 的关系非常重要。

!!! abstract "核心概念"
    *   **NVIDIA 显卡**: 硬件基础。
    *   **CUDA (System)**: 显卡驱动层面的运算平台。即使有显卡，也需要安装 CUDA Toolkit。
    *   **cuda (PyTorch)**: PyTorch 库内部集成的 cuda 组件（通常较小）。
    
    **版本兼容原则**：
    > System CUDA Version $\ge$ PyTorch Internal cuda Version

!!! warning "注意"
    如果运行报错，可能需要更新显卡驱动或重新安装匹配版本的 CUDA Toolkit。

### 4. PyTorch 安装指南

PyTorch 主要包含三个模块：`torch` (核心，约 2G)、`torchvision` (视觉)、`torchaudio` (音频)。

#### 方法一：官方命令安装（推荐）

访问 [PyTorch 官网](https://pytorch.org/get-started/previous-versions/)，选择对应的 OS、Package (Pip)、Language (Python) 和 Compute Platform (CUDA 版本)，获取安装命令。

#### 方法二：离线 Wheel 安装（网络不佳时）

如果直接 pip 安装速度过慢或失败，可以下载 `.whl` 文件进行离线安装。

1.  访问 [PyTorch Wheel 下载站](https://download.pytorch.org/whl/torch_stable.html)。
2.  下载对应版本 (注意核对 `cu` 版本, `cp` python版本, `os` 系统)。
3.  在虚拟环境中执行安装：

```bash
# 示例：安装 PyTorch 1.12.0 (CUDA 11.3, Python 3.9, Windows)
pip install D:\whl\torch-1.12.0+cu113-cp39-cp39-win_amd64.whl 
pip install D:\whl\torchvision-0.13.0+cu113-cp39-cp39-win_amd64.whl 
pip install D:\whl\torchaudio-0.12.0+cu113-cp39-cp39-win_amd64.whl
```

!!! success "提示"
    安装完毕后，即可删除 `D:\whl` 文件夹（但建议留着备份，之后可能还要安装）。

### 5. Jupyter Notebook 配置

#### 关联虚拟环境

默认情况下，Jupyter Notebook 可能只能识别 base 环境。为了在 Notebook 中使用我们创建的 `DL` 环境，需要安装 `ipykernel` 并注册内核。

**步骤如下：**

1.  **激活环境**: `conda activate DL`
2.  **安装 ipykernel**:
    ```bash
    pip install ipykernel -i https://pypi.tuna.tsinghua.edu.cn/simple 
    ```
3.  **写入 Kernel**:
    ```bash
    # --name 指定内核名称，--user 针对当前用户安装
    python -m ipykernel install --user --name=环境名
    ```

这样，在启动 Jupyter Notebook 后，就可以在 "Kernel" -> "Change kernel" 中找到 `DL` 环境了。

## Python 基础

本部分梳理了 Python 语言的核心特性，重点关注深度学习常用的基础语法。

### 1. 变量与数据类型

Python 是动态类型语言，无需显式声明变量类型。

=== "基础类型"
    *   **整数 (int)**: 如 `10`, `-5`
    *   **浮点数 (float)**: 如 `3.14`, `1.2e-3`
    *   **布尔值 (bool)**: `True`, `False`
    *   **字符串 (str)**:用单/双引号包裹，如 `'hello'`, `"world"`

=== "类型转换"
    ```python
    x = 10
    y = float(x)  # 10.0
    z = str(y)    # "10.0"
    ```

### 2. 核心数据结构

掌握列表和字典对于数据处理至关重要。

!!! abstract "常用容器"
    *   **列表 (list)**: 有序、可变。`li = [1, 2, 3]`
    *   **元组 (tuple)**: 有序、不可变。`tu = (1, 2)`
    *   **字典 (dict)**: 键值对映射。`di = {'name': 'AI', 'ver': 2.0}`
    *   **集合 (set)**: 无序、去重。`se = {1, 2, 3}`

```python
# 列表推导式 (List Comprehension) - 高频用法
squares = [x**2 for x in range(5)]  # [0, 1, 4, 9, 16]
```

### 3. 程序控制流

#### 条件判断与循环

Python 使用缩进来表示代码块。

```python
# if-elif-else
score = 85
if score >= 90:
    print("A")
elif score >= 60:
    print("B")
else:
    print("C")

# for 循环
for i in range(3):
    print(i)  # 输出 0, 1, 2

# while 循环
n = 0
while n < 3:
    print(n)
    n += 1
```

### 4. 函数与模块

函数用于封装代码逻辑，模块用于组织代码结构。

=== "函数定义"
    ```python
    def add(a, b=1):
        """返回 a + b，b 默认为 1"""
        return a + b
    
    result = add(5)  # 6
    ```

=== "模块导入"
    ```python
    import math
    print(math.pi)
    
    # 常用别名导入
    import numpy as np
    import pandas as pd
    ```

!!! tip "学习建议"
    对于深度学习，切勿在此阶段深究 Python 的所有细节。掌握上述基础后，即可直接进入库的学习（如 Numpy, PyTorch），在实践中补充高级特性。

## NumPy 科学计算库

NumPy 是 Python 进行科学计算的基础包，提供高性能的多维数组对象（ndarray）及相关操作，是几乎所有深度学习框架（如 PyTorch, TensorFlow）的数据处理基石。

常用导入惯例：
```python
import numpy as np
```

### 1. 数组创建 (Creation)

深度学习中常需要初始化特定的数组作为输入数据或模型参数。

=== "基础创建"
    ```python
    # 从列表创建
    arr = np.array([1, 2, 3])
    
    # 指定数据类型（重要：深度学习常默认 float32）
    arr_f = np.array([1, 2, 3], dtype=np.float32)
    ```

=== "常用生成函数"
    ```python
    # 全 0 数组 (常用于初始化)
    z = np.zeros((3, 4))  # 3行4列
    
    # 全 1 数组
    o = np.ones((2, 3))
    
    # 等差数列 [0, 2, 4, 6, 8]
    a = np.arange(0, 10, 2)
    
    # 线性等分 (0到1之间5个点)
    l = np.linspace(0, 1, 5)
    ```

=== "随机数生成"
    **参数初始化**常用。
    ```python
    # 均匀分布 [0, 1)
    r1 = np.random.rand(3, 3)
    
    # 标准正态分布 (均值0, 方差1) - 权重初始化常用
    r2 = np.random.randn(3, 3)
    
    # 随机整数 [low, high)
    r3 = np.random.randint(0, 10, size=(2, 2))
    
    # 固定随机种子（复现实验结果必备）
    np.random.seed(42)
    ```

### 2. 数组属性 (Attributes)

调试神经网络时，经常需要检查张量的形状。

| 属性 | 说明 | 示例 |
| :--- | :--- | :--- |
| `arr.ndim` | 维度数 (秩) | `2` (矩阵), `3` (张量) |
| `arr.shape` | 形状 (各维度大小) | `(3, 4)` (3行4列) |
| `arr.size` | 元素总数 | `12` |
| `arr.dtype` | 元素数据类型 | `float32`, `int64` |

### 3. 索引与切片 (Indexing)

NumPy 的切片操作比 Python 列表更强大，支持高维操作。

```python
data = np.random.randn(5, 4)  # 5行4列的数据

# 基础切片 [行, 列]
row_1 = data[0, :]    # 第0行所有元素
col_1 = data[:, 1]    # 第1列所有元素
part = data[1:3, 0:2] # 第1~2行，第0~1列

# 布尔索引 (条件筛选) - 非常实用
mask = data > 0
positive = data[mask] # 选出所有大于0的元素（结果展平为1维）
```

!!! warning "视图 vs 副本"
    NumPy 切片通常返回**视图（View）**，修改切片会直接修改原数组。
    如果需要独立副本，请使用 `.copy()` 方法。

### 4. 维度变换 (Reshape)

深度学习中经常需要在不同层之间转换数据形状（如 Flatten、Batch处理）。

=== "形状改变"
    ```python
    arr = np.arange(12)
    
    # 改变形状 (不复制数据)
    new_arr = arr.reshape(3, 4)
    
    # 自动推导维度 (-1)
    # 系统会自动计算列数为4 (12 / 3 = 4)
    auto_arr = arr.reshape(3, -1) 
    ```

=== "维度展平与转置"
    ```python
    # 展平为一维 (常用在全连接层前)
    flat = arr.flatten()
    
    # 转置 (矩阵操作)
    t = new_arr.T 
    # 或者指定轴交换 (Transposing axes)
    t2 = new_arr.transpose(1, 0)
    ```

### 5. 运算与广播 (Broadcasting)

NumPy 支持向量化运算，速度远快于 Python 循环。

**广播机制 (Broadcasting)** 是 NumPy 的核心特性，允许不同形状的数组在算术运算期间进行处理（如矩阵加向量）。

**规则**：从后缘维度（最后一维）开始比较，如果维度相等或其中一个为 1，则可以广播。

```python
A = np.ones((3, 4))
b = np.arange(4)     # shape (4,) -> Broadcasats to (1, 4) -> (3, 4)

C = A + b  # b 会被加到 A 的每一行上
```

### 6. 常用数学与统计函数

```python
arr = np.random.randn(3, 4)

# 聚合运算
s = np.sum(arr)          # 所有元素求和
m = np.mean(arr, axis=0) # 按列求均值 (压缩行，保留列特征)
mx = np.max(arr, axis=1) # 按行求最大值

# 矩阵乘法 (注意区分元素乘法 *)
A = np.ones((2, 3))
B = np.ones((3, 2))

C = np.dot(A, B) # 还有更现代的写法: C = A @ B
```

