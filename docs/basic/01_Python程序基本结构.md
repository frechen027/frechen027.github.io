# Python 程序基本结构

## 什么是 Python？

Python 是一种高级、解释型、通用的编程语言。由 Guido van Rossum 于 1989 年发明，以简洁优雅的语法著称。

### Python 的核心特点

| 特点         | 说明                         |
| ------------ | ---------------------------- |
| **简洁易读** | 语法接近自然语言，代码即文档 |
| **解释执行** | 无需编译，直接运行           |
| **动态类型** | 变量类型在运行时确定         |
| **跨平台**   | 支持 Windows、Linux、macOS   |
| **丰富生态** | 拥有海量的第三方库           |

## Python 程序的执行过程

```
源代码 (.py) → Python 解释器 → 字节码 (.pyc) → Python 虚拟机 (PVM) → 执行结果
```

!!! info "关于字节码"
Python 会将源代码编译为字节码缓存在 `__pycache__` 目录中，加速后续执行。

## 第一个 Python 程序

### 传统方式：Hello World

```python
print("Hello, World!")
```

**运行方式：**

=== "命令行"
`bash
    python hello.py
    `

=== "交互式"
`python
    >>> print("Hello, World!")
    Hello, World!
    `

### 进阶示例：带输入的程序

```python
# 获取用户输入
name = input("请输入你的名字: ")

# 格式化输出
print(f"你好, {name}! 欢迎学习 Python!")
```

## Python 程序的基本结构

一个完整的 Python 程序通常包含以下部分：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模块文档字符串 (Docstring)
描述这个模块的功能和用途
"""

# 1. 导入模块
import os
import sys
from datetime import datetime

# 2. 常量定义
VERSION = "1.0.0"
AUTHOR = "Your Name"

# 3. 函数定义
def greet(name: str) -> str:
    """返回问候语"""
    return f"Hello, {name}!"

# 4. 类定义
class Greeter:
    """问候器类"""
    def __init__(self, prefix: str = "Hi"):
        self.prefix = prefix

    def greet(self, name: str) -> str:
        return f"{self.prefix}, {name}!"

# 5. 主程序入口
def main():
    """主函数"""
    print(greet("Python"))

    greeter = Greeter("Welcome")
    print(greeter.greet("Developer"))

# 6. 模块入口检查
if __name__ == "__main__":
    main()
```

### 结构说明

| 组成部分                             | 作用                        | 是否必须                  |
| ------------------------------------ | --------------------------- | ------------------------- |
| Shebang (`#!/usr/bin/env python3`)   | 指定解释器路径 (Unix/Linux) | 可选                      |
| 编码声明 (`# -*- coding: utf-8 -*-`) | 声明文件编码                | Python 3 默认 UTF-8，可选 |
| 模块文档字符串                       | 描述模块功能                | 推荐                      |
| 导入语句                             | 引入外部模块                | 按需                      |
| 常量/变量定义                        | 定义全局常量                | 按需                      |
| 函数/类定义                          | 核心逻辑封装                | 核心                      |
| `if __name__ == "__main__":`         | 模块入口控制                | **强烈推荐**              |

## `__name__` 的奥秘

这是 Python 最重要的惯用法之一：

```python
if __name__ == "__main__":
    # 只有直接运行此文件时才执行
    main()
```

### 工作原理

| 场景                           | `__name__` 的值     |
| ------------------------------ | ------------------- |
| 直接运行 `python script.py`    | `"__main__"`        |
| 被其他模块导入 `import script` | `"script"` (模块名) |

### 实际演示

创建两个文件来理解：

**mymodule.py:**

```python
def say_hello():
    print("Hello from mymodule!")

print(f"mymodule 的 __name__ = {__name__}")

if __name__ == "__main__":
    print("mymodule 被直接运行")
    say_hello()
```

**main.py:**

```python
import mymodule

print(f"main 的 __name__ = {__name__}")
mymodule.say_hello()
```

**运行结果：**

```bash
# 直接运行 mymodule.py
$ python mymodule.py
mymodule 的 __name__ = __main__
mymodule 被直接运行
Hello from mymodule!

# 运行 main.py (导入 mymodule)
$ python main.py
mymodule 的 __name__ = mymodule
main 的 __name__ = __main__
Hello from mymodule!
```

## 代码注释

### 单行注释

```python
# 这是单行注释
x = 10  # 行尾注释
```

### 多行注释

```python
"""
这是多行注释
可以跨越多行
常用于模块、类、函数的文档说明
"""

'''
单引号也可以
但推荐使用双引号
'''
```

### 文档字符串 (Docstring)

```python
def calculate_area(radius: float) -> float:
    """
    计算圆的面积

    Args:
        radius: 圆的半径，必须为正数

    Returns:
        圆的面积

    Raises:
        ValueError: 当半径为负数时

    Examples:
        >>> calculate_area(1)
        3.141592653589793
        >>> calculate_area(2)
        12.566370614359172
    """
    if radius < 0:
        raise ValueError("半径不能为负数")
    import math
    return math.pi * radius ** 2
```

!!! tip "Docstring 规范"
推荐使用 Google 风格或 NumPy 风格的 Docstring，便于自动生成文档。

## 代码块与缩进

Python 使用**缩进**来定义代码块，而不是花括号 `{}`。

### 缩进规则

```python
# ✅ 正确：统一使用 4 个空格
if True:
    print("缩进正确")
    if True:
        print("嵌套缩进")

# ❌ 错误：混用 Tab 和空格
if True:
	print("使用 Tab")  # 可能导致 IndentationError
```

!!! warning "缩进最佳实践" 
- **始终使用 4 个空格**进行缩进 
- **不要混用 Tab 和空格** 
- 配置编辑器将 Tab 自动转换为空格

### 常见缩进错误

```python
# IndentationError: expected an indented block
if True:
print("错误")  # 缺少缩进

# IndentationError: unexpected indent
x = 1
    y = 2  # 意外的缩进

# TabError: inconsistent use of tabs and spaces
if True:
    x = 1    # 4 空格
	y = 2    # 1 Tab (混用导致错误)
```

## 代码行与续行

### 物理行与逻辑行

```python
# 一个物理行 = 一个逻辑行
x = 1

# 多个物理行 = 一个逻辑行 (显式续行)
total = 1 + 2 + 3 + \
        4 + 5 + 6

# 括号内自动续行 (推荐)
numbers = [
    1, 2, 3,
    4, 5, 6,
]

result = (
    first_value
    + second_value
    - third_value
)
```

### 多语句一行（不推荐）

```python
# 技术上可行，但不推荐
x = 1; y = 2; z = 3
```

## Python 交互式环境

### 标准 REPL

```python
$ python
Python 3.11.0 (main, Oct 24 2022, 18:26:48)
>>> 2 + 2
4
>>> exit()  # 或 Ctrl+D (Unix) / Ctrl+Z (Windows)
```

### IPython (推荐)

```python
$ ipython
In [1]: 2 + 2
Out[1]: 4

In [2]: %timeit sum(range(1000))  # 魔术命令
1.5 µs ± 10 ns per loop

In [3]: !ls  # 执行 Shell 命令
```

### Jupyter Notebook

适合数据分析和机器学习的交互式环境，支持：

-   代码单元格执行
-   Markdown 文档
-   可视化输出
-   实时交互

## 编码规范 (PEP 8)

Python 有官方的编码风格指南 [PEP 8](https://pep8.org/)：

### 命名规范

| 类型      | 规范               | 示例                           |
| --------- | ------------------ | ------------------------------ |
| 变量/函数 | `snake_case`       | `my_variable`, `calculate_sum` |
| 常量      | `UPPER_SNAKE_CASE` | `MAX_SIZE`, `PI`               |
| 类名      | `PascalCase`       | `MyClass`, `HttpRequest`       |
| 模块      | `lowercase`        | `mymodule`, `utils`            |
| 私有      | `_prefix`          | `_internal_var`                |
| 强私有    | `__prefix`         | `__private_method`             |

### 空格规范

```python
# ✅ 正确
x = 1
y = x + 1
my_list = [1, 2, 3]
my_dict = {"key": "value"}
func(arg1, arg2)

# ❌ 错误
x=1
y = x+1
my_list = [1,2,3]
my_dict = {"key":"value"}
func( arg1, arg2 )
```

### 导入规范

```python
# ✅ 正确的导入顺序
# 1. 标准库
import os
import sys

# 2. 第三方库
import numpy as np
import pandas as pd

# 3. 本地模块
from mypackage import mymodule
```

## 本章小结

| 概念                         | 要点                           |
| ---------------------------- | ------------------------------ |
| Python 特点                  | 简洁、解释型、动态类型、跨平台 |
| 程序结构                     | 导入 → 定义 → 主程序入口       |
| `if __name__ == "__main__":` | 区分直接运行和被导入           |
| 注释                         | `#` 单行，`"""` 多行/文档      |
| 缩进                         | 4 空格，不混用 Tab             |
| PEP 8                        | Python 编码风格指南            |

## 练习题

1. **基础练习**：编写一个程序，输入你的名字和年龄，输出一句自我介绍。

2. **结构练习**：创建一个模块 `greetings.py`，包含 `morning()`, `afternoon()`, `evening()` 三个函数，然后在另一个文件中导入并使用。

3. **理解 `__name__`**：修改 `greetings.py`，使其直接运行时打印测试信息，被导入时不打印。

---

_下一章：[基础语法与变量](02_基础语法与变量.md)_
