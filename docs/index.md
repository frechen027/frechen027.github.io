# Python 全栈 AI 工程师一学期特训计划 (20 周)

## 网站简介

本计划旨在 5 个月内打通 **"应用开发 -> 数据处理 -> 模型训练 -> AI 部署"** 的全链路。

### 核心语言选择评估：为什么选择 Python？

**结论：非常合理，且是唯一的最佳选择。**

选择 Python 作为核心语言来构建 "开发 + 数据 + AI + Infra" 的知识体系是完全正确的。

-   **开发 (Backend):** FastAPI 等现代异步框架让 Python 在高性能微服务中占据一席之地。
-   **数据 (Data):** Pandas, NumPy 是数据处理的绝对标准。
-   **机器学习 (ML):** PyTorch, TensorFlow, Scikit-learn 均以 Python 为首选接口。
-   **大模型 (LLMs):** HuggingFace, LangChain, OpenAI 等生态几乎是 Python Native 的。
-   **AI Infra:** Ray, Kubeflow, MLflow 等基础设施工具的胶水层全是 Python。

**注意：** 虽然 Python 是核心，但在 Infra 阶段你需要接触 Linux Shell 和 Dockerfile，在高性能计算底层可能涉及 C++/CUDA（但通常作为 Python 库调用，无需深究）。

## 关键学习资源清单

### 书籍

-   **《Fluent Python (流畅的 Python)》** (进阶必读)
-   **《Python Crash Course》** (如果基础不牢可快速过一遍)
-   **《Designing Machine Learning Systems》** (Chip Huyen 著，MLOps 圣经)

### 在线课程

-   **Fast.ai:** 深度学习最佳入门。
-   **DeepLearning.AI:** Andrew Ng 的系列课程，紧跟 LLM 前沿。
-   **Full Stack Deep Learning (FSDL):** 弥补学校教育与工业界 AI 落地的差距。
-   **李宏毅机器学习:** Pytorch 部分不太适合，其余部分极佳。
-   **吴恩达机器学习:** 机器学习和深度学习都有涉及，讲解有趣，初学者友好。
-   **李沐动手学习深度学习:** 手把手用深度学习，需要 Python 和深度学习基础，更推荐看书的版本。 
-   **爆肝杰哥的深度学习:** 鱼书的优化版，适合入门观看。
-   **邱锡鹏神经网络与深度学习:** 国内课程，难度较高。
-   **小土堆Pytorch:** 需要有基础，可以搭配爆肝杰哥的课程。
-   **机器学习白板推导系列:** 数学原理角度理解，难度高。
-   **Udacity:** 课程质量优秀，但内容偏基础。 
-   **CS61A (UC Berkeley):** 适合巩固计算机科学基础。

### 练习平台

-   **LeetCode:** 保持算法手感 (主要刷 Array, String, Logic 类题目)。
-   **Kaggle:** 学习优秀 Notebook 的数据处理思路。

## 学习时间规划 (总计 20 周)

### 第一阶段：Python 核心与高阶编程 (周 1-3)

**目标：** 超越语法层面，掌握能够阅读复杂框架源码的高级特性。

-   **周 1: 扎实基础与工程规范**
    -   **重点:** 类型提示 (Type Hints)、虚拟环境 (uv/poetry)、包管理、Git 工作流。
    -   **内容:** 列表推导式、生成器、异常处理、文件操作 (pathlib)。
    -   **实战:** 编写一个 CLI 工具 (使用 Typer 或 Click)。
-   **周 2: 面向对象与函数式编程**
    -   **重点:** 类与对象、继承、多态、魔术方法 (`__call__`, `__init__`, `__enter__`)。
    -   **高级:** 装饰器 (Decorators) 原理、闭包、迭代器模式。
-   **周 3: 并发编程与元编程**
    -   **重点:** `asyncio` 深度理解 (AI 服务高并发核心)、多线程 vs 多进程 (GIL 锁)、元类 (Metaclasses)。

### 第二阶段：现代后端与微服务开发 (周 4-6)

**目标:** 放弃传统的 Django/Flask 全栈模式，直接切入适合 AI 服务的 FastAPI + 异步架构。

-   **周 4: Web 框架与 API 设计**
    -   **框架:** **FastAPI** (目前 AI 领域标配)。
    -   **内容:** Pydantic 数据验证、Dependency Injection (依赖注入)、OpenAPI 文档自动生成。
    -   **实战:** 开发一个只有 API 的后端服务 (ToDo List 或 博客 API)。
-   **周 5: 数据库与持久化**
    -   **SQL:** PostgreSQL (生产环境首选)、SQLAlchemy 2.0 (现代 ORM)。
    -   **NoSQL:** Redis (缓存与消息队列)、Vector DB (向量数据库基础，如 ChromaDB/Qdrant)。
    -   **实战:** 将数据库集成到 FastAPI，实现 CRUD 和简单的缓存。
-   **周 6: 容器化与基础部署**
    -   **工具:** Docker, Docker Compose。
    -   **内容:** 编写优化的 `Dockerfile` (多阶段构建)、`docker-compose.yaml` 编排服务。
    -   **实战:** 将 Web 服务 + DB + Redis 一键启动。

### 第三阶段：数据工程与机器学习基础 (周 7-10)

**目标:** 掌握数据的清洗、流转以及经典机器学习算法。

-   **周 7: 科学计算与数据清洗**
    -   **库:** NumPy (矩阵运算基础), Pandas (数据处理)。
    -   **内容:** DataFrame 操作、数据清洗、分组聚合、时间序列处理。
    -   **实战:** 分析 Kaggle 上的 Titanic 或 房价预测数据集。
-   **周 8: 经典机器学习 (Scikit-Learn)**
    -   **内容:** 监督学习 (回归、分类)、无监督学习 (聚类、降维)、特征工程。
    -   **实战:** 训练一个简单的预测模型并评估 (Precision/Recall)。
-   **周 9-10: 深度学习基础 (PyTorch)**
    -   **框架:** Python 深度学习事实标准 **PyTorch**。
    -   **内容:** Tensor 操作、Autograd (自动求导)、神经网络构建 (`nn.Module`)、DataLoader。
    -   **实战:** 使用 CNN 做 MNIST 手写数字识别。

### 第四阶段：大模型 (LLMs) 与 GenAI 开发 (周 11-15)

**目标:** 紧跟时代前沿，掌握大模型应用开发与调优。

-   **周 11: Transformer 与 HuggingFace 生态**
    -   **原理:** Attention 机制直观理解。
    -   **工具:** `transformers` 库、`datasets` 库。
    -   **实战:** 调用开源模型进行文本与情感分析。
-   **周 12: Prompt Engineering 与 LangChain**
    -   **内容:** 提示词工程、LangChain/LangGraph 框架、Chains, Agents, Memories。
    -   **实战:** 构建一个简单的 PDF 聊天助手。
-   **周 13: RAG (检索增强生成)**
    -   **技术:** 向量嵌入 (Embeddings)、向量数据库 (Pinecone/Milvus)、RAG 流程。
    -   **实战:** 搭建本地知识库问答系统。
-   **周 14-15: 模型微调 (Fine-tuning)**
    -   **技术:** LoRA, QLoRA, PEFT (高效参数微调)。
    -   **实战:** 在 Colab 上微调一个 Llama 3 或 Mistral 模型以适应特定风格。

### 第五阶段：AI Infra 与 MLOps (周 16-18)

**目标:** 将 AI 模型从 Notebook 搬到生产环境，解决规模化与稳定性问题。

-   **周 16: MLOps 基础与实验追踪**
    -   **工具:** MLflow 或 Weights & Biases (W&B)。
    -   **内容:** 模型版本管理、参数记录、Artifacts 管理。
-   **周 17: 模型服务化 (Model Serving)**
    -   **工具:** TorchServe, Triton Inference Server, 或简单的 FastAPI 封装。
    -   **挑战:** 批处理 (Batching)、低延迟优化、GPU 资源调度。
-   **周 18: 分布式与编排 (AI Infra)**
    -   **核心:** **Ray** 框架 (分布式计算标准)。
    -   **内容:** Ray Train, Ray Serve。了解 Kubernetes 对 AI 的支持 (KubeFlow 概念即可)。

### 第六阶段：综合实战与总结 (周 19-20)

**目标:** 融会贯通，产出一个高含金量的简历项目。

-   **项目选题推荐：** "构建一个企业级垂直领域 AI 助手"
    -   **后端:** FastAPI 提供高性能 API。
    -   **数据:** 编写爬虫或 ETL 脚本抓取行业数据 (Pandas 处理)。
    -   **AI:** 使用 RAG 技术检索私有数据，或微调开源 LLM。
    -   **部署:** Docker 容器化，使用 Ray Serve 或 K8s 部署，接入 Prometheus 监控。
    -   **Frontend:** 使用 Streamlit 或 Gradio 快速搭建演示界面。
