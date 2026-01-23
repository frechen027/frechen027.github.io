# Markdown 写作指南

Markdown 是一种轻量级标记语言，它允许人们使用易读易写的纯文本格式编写文档。目前它是技术文档、博客（包括本网站）、README 文件的行业标准。

**核心理念：专注于内容，而不是排版。**

## 1. 标题 (Headers)

使用 `#` 号表示标题，`#` 的数量代表标题级别。

```markdown
# 一级标题 (H1)

## 二级标题 (H2)

### 三级标题 (H3)

#### 四级标题 (H4)
```

## 2. 文本样式

| 语法         | 效果       | 说明           |
| :----------- | :--------- | :------------- |
| `**加粗**`   | **加粗**   | 强调重要内容   |
| `*斜体*`     | _斜体_     | 强调或专有名词 |
| `~~删除线~~` | ~~删除线~~ | 表示过时或修正 |
| `` `代码` `` | `code`     | 行内代码引用   |

## 3. 列表 (Lists)

### 无序列表

使用 `-`, `*` 或 `+` 加空格。

```markdown
-   Python
-   Java
-   Go
```

**效果:**

-   Python
-   Java
-   Go

### 有序列表

使用数字加点 `1. `。

```markdown
1. 第一步
2. 第二步
3. 第三步
```

**效果:**

1. 第一步
2. 第二步
3. 第三步

### 嵌套列表

缩进 4 个空格或 1 个 Tab。

```markdown
-   编程语言
    -   Python
    -   C++
```

## 4. 引用 (Blockquotes)

使用 `>` 符号。

```markdown
> 知识就是力量。
>
> —— 培根
```

**效果:**

> 知识就是力量。
>
> —— 培根

## 5. 代码块 (Code Blocks)

使用三个反引号 ` ``` ` 包裹代码，并指定语言（以实现语法高亮）。

````markdown
```python
def hello():
    print("Hello, Markdown!")
```
````

**效果:**

```python
def hello():
    print("Hello, Markdown!")
```

## 6. 链接与图片

### 链接

```markdown
[显示文本](链接地址)
例如：[Baidu](https://www.baidu.com)
```

效果：[Baidu](https://www.baidu.com)

### 图片

在链接前加一个感叹号 `!`。

```markdown
![图片描述/Alt文本](图片地址)
![Python Logo](https://www.python.org/static/img/python-logo.png)
```

## 7. 表格 (Tables)

使用 `|` 分隔列，使用 `-` 分隔表头。

```markdown
| 姓名  | 年龄 |   职业 |
| :---- | :--: | -----: |
| Alice |  25  | 工程师 |
| Bob   |  30  | 设计师 |
```

**对齐方式：**

-   `:---` 左对齐
-   `:--:` 居中
-   `---:` 右对齐

**效果:**

| 姓名  | 年龄 |   职业 |
| :---- | :--: | -----: |
| Alice |  25  | 工程师 |
| Bob   |  30  | 设计师 |

## 8. 分割线

使用三个以上的 `-` 或 `*`。

```markdown
---

---
```

## 9. 任务列表 (Task Lists)

-   [x] 已完成任务
-   [ ] 待办任务

```markdown
-   [x] 已完成任务
-   [ ] 待办任务
```

## 10. 数学公式 (LaTeX)

本站使用 MathJax/KaTeX 渲染数学公式。

### 行内公式

使用单美元符号 `$`。
`$E = mc^2$` 的效果是 $E = mc^2$。

### 块级公式

使用双美元符号 `$$`。

```markdown
$$
\sum_{i=1}^{n} i = \frac{n(n+1)}{2}
$$
```

**效果:**

$$
\sum_{i=1}^{n} i = \frac{n(n+1)}{2}
$$

## 11. 扩展功能 (MkDocs Material 特有)

由于本站使用 MkDocs Material 主题，支持一些高级扩展。

### 提示块 (Admonitions)

```markdown
!!! note "提示"
这是一个提示信息。

!!! warning "警告"
这是一个警告信息。

!!! danger "危险"
这是一个危险操作。
```

**效果:**

!!! note "提示"
这是一个提示信息。

!!! warning "警告"
这是一个警告信息。

### 选项卡 (Tabs)

````markdown
=== "Python"
    ```python
    print("Hello")
    ```

=== "C++"
    ```cpp
    cout << "Hello";
    ```
````

**效果:**

=== "Python"
    ```python
    print("Hello")
    ```

=== "C++"
    ```cpp
    cout << "Hello";
    ```

## 推荐工具

-   **VS Code**: 配合 "Markdown All in One" 插件，体验极佳。
-   **Typora**: 真正所见即所得的 Markdown 编辑器（收费）。
-   **Obsidian**: 强大的双向链接笔记工具。

---

_学习完 Markdown，你就可以优雅地编写文档和博客了。_

_下一章：[Vibe Coding](03_Vibe_Coding.md)_
