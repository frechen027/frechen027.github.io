# Git 与版本控制

Git 是目前世界上最先进的分布式版本控制系统（Distributed Version Control System，DVCS）。无论你是个人开发者还是团队成员，Git 都是必备技能。

## 为什么需要 Git？

试想以下场景：

-   **后悔药**：代码写炸了，想回到昨天的状态？
-   **多版本管理**：老板要一个"正式版"，又要一个"春节特别版"？
-   **协同工作**：多个人同时修改一个文件，怎么合并？

Git 就是为了解决这些问题而生的。

## 1. 安装与配置

### 安装 Git

-   **Windows**: 下载 [Git for Windows](https://git-scm.com/download/win) 并安装。
-   **Mac**: 使用 Homebrew (`brew install git`) 或下载安装包。
-   **Linux**: `sudo apt install git` (Ubuntu/Debian)。

### 全局配置 (首次使用必做)

安装后，告诉 Git 你是谁：

```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱@example.com"
```

查看配置信息：

```bash
git config --list
```

## 2. 核心概念

理解 Git 的三个区至关重要：

1.  **工作区 (Working Directory)**: 你电脑里能看到的目录，平时写代码的地方。
2.  **暂存区 (Staging Area/Index)**: 临时存储即将提交的修改 (类似于购物车的“待结算”状态)。
3.  **本地仓库 (Local Repository)**: 只有 `commit` 之后，修改才算真正进入了版本库（安全了）。

```mermaid
graph LR
    A[工作区] -- git add --> B[暂存区]
    B -- git commit --> C[本地仓库]
    C -- git push --> D[远程仓库]
```

## 3. 基础工作流

### 初始化仓库

```bash
# 在当前目录创建一个 Git 仓库
git init
```

这会生成一个隐藏的 `.git` 文件夹，不要删除它。

新建文件时是未被跟踪的，需要 `git add` 才能进入暂存区。

### 日常操作三板斧

1.  **添加 (Add)**: 将文件修改添加到暂存区。

    ```bash
    git add filename.py   # 添加指定文件
    git add .             # 添加当前目录所有文件 (最常用)
    ```

2.  **提交 (Commit)**: 将暂存区的内容提交到本地仓库。

    ```bash
    git commit -m "完成登录功能"  # -m 后面是提交说明，必须写清楚
    ```

3.  **推送 (Push)**: 将本地仓库推送到远程仓库 (如 GitHub, GitLab)。
    ```bash
    git push origin main
    ```

### 删除与移动文件

并不是直接在文件管理器删除那么简单，建议使用 git 命令操作：

1.  **删除文件 (Remove)**

    ```bash
    # 从版本库和工作区中同时删除 (最彻底)
    git rm filename.py

    # 仅从版本库删除，但在工作区保留 (比如误提交了 .env 或 __pycache__)
    git rm --cached filename.py
    ```

2.  **移动/重命名 (Move)**
    ```bash
    # 相当于：mv old.py new.py + git add new.py + git rm old.py
    git mv old.py new.py
    ```

### 查看状态与历史

```bash
# 查看当前状态 (有哪些文件修改了？哪些在暂存区？)
git status

# 查看提交历史
git log
git log --oneline  # 简洁模式
```

## 4. 分支管理 (Branching)

分支是 Git 的杀手锏。开发新功能时，通常新建一个分支，不影响主分支 (main/master)。

### 常用分支命令

| 命令                        | 作用                              |
| :-------------------------- | :-------------------------------- |
| `git branch`                | 列出所有本地分支                  |
| `git branch -a`             | 列出所有分支 (含远程)             |
| `git branch feature-A`      | 创建名为 feature-A 的分支         |
| `git checkout feature-A`    | 切换到 feature-A 分支             |
| `git checkout -b feature-A` | **[常用]** 创建并切换到 feature-A |
| `git merge feature-A`       | 将 feature-A 合并到当前分支       |
| `git branch -d feature-A`   | 删除分支                          |

### 合并流程示例

```bash
# 1. 在 main 分支上创建并切换到新分支 dev
git checkout -b dev

# 2. 在 dev 分支上写代码、提交
touch new_feature.py
git add .
git commit -m "开发新功能"

# 3. 切换回 main 分支
git checkout main

# 4. 将 dev 分支合并到 main
git merge dev

# 5. 删除 dev 分支 (可选)
git branch -d dev
```

## 5. 远程仓库 (Remote)

### 克隆远程仓库

如果你是加入一个已有的项目：

```bash
git clone https://github.com/username/repo.git
```

### 关联远程仓库

如果你是本地新建的项目，想推送到 GitHub：

1.  在 GitHub 上新建一个空仓库。
2.  在本地运行：

    ```bash
    # 添加远程仓库别名为 origin
    git remote add origin https://github.com/username/repo.git

    # 将本地 main 分支推送到 origin (初次推送加 -u)
    git push -u origin main
    ```

    !!! tip "为什么加 -u 参数？"
    `-u` 是 `--set-upstream` 的缩写。它的作用是将本地分支与远程分支**绑定**（建立追踪关系）。绑定后，以后只需要运行简短的 `git push` 或 `git pull`，Git 就会自动知道要往 `origin` 的 `main` 分支同步。

### 配置 SSH 密钥 (免密登录)

使用 HTTPS 协议每次推送都需要输入账号密码（或 Token），推荐使用 SSH 协议。

1.  **生成 SSH 密钥**:

    ```bash
    ssh-keygen -t rsa -C "你的邮箱@example.com"
    ```

    一路回车即可（默认路径和空密码）。

2.  **获取公钥**:
    密钥生成在用户主目录的 `.ssh` 文件夹下。

    ```bash
    # 查看公钥内容
    cat ~/.ssh/id_rsa.pub
    ```

3.  **配置 GitHub**:
    复制上一步输出的公钥内容（以 `ssh-rsa` 开头），打开 **GitHub Settings** -> **SSH and GPG keys** -> **New SSH key**，粘贴即可。

4.  **测试连接**:
    ```bash
    ssh -T git@github.com
    # 看到 "Hi username! You've successfully authenticated..." 即成功
    ```

### 拉取远程代码 (Pull vs Fetch)

1.  **直接拉取 (Pull)**:
    `git pull` 相当于 **下载代码** + **自动合并**。

    ```bash
    # 拉取并自动合并 (最常用)
    git pull origin main
    ```

2.  **仅获取更新 (Fetch)**:
    如果你只想把远程代码下载下来，但不合并到你的分支（想看看别人改了什么），使用 `git fetch`。

    ```bash
    # 下载远程更新，但不改变当前工作区
    git fetch origin main

    # 之后可以查看 diff，确认没问题再手动 merge
    git diff main origin/main
    git merge origin/main
    ```

    !!! info "Fetch vs Pull" - `git fetch`: **安全**，只是下载更新到本地仓库，不会弄乱你正在写代码的工作区。 - `git pull`: **快捷**，但如果远程代码和你本地有冲突，会直接触发合并流程，可能产生冲突文件。

## 6. `.gitignore` 文件

有些文件不应该提交到仓库中（如密码、虚拟环境、编译产生的临时文件）。
在项目根目录创建一个名为 `.gitignore` 的文件，写入要忽略的规则：

```text
# 忽略 Python 编译文件
__pycache__/
*.pyc

# 忽略虚拟环境
venv/
.env

# 忽略 IDE 配置
.vscode/
.idea/
```

## 7. 常见问题与回滚

### 临时保存工作现场 (Stash)

场景：代码写了一半需要切换分支去修紧急 Bug，但不想提交依然 Broken 的代码。

```bash
# 1. 保存当前进度到堆栈 (工作区瞬间变干净)
git stash
# 或者带个备注
git stash save "暂存未完成的登录功能"

# 2. 此时可以安全切换分支了
git checkout main

# ... (修完 Bug 后切回来) ...

# 3. 恢复最近一次保存的现场 (恢复并从堆栈删除)
git stash pop
```

**其他常用命令：**

-   `git stash list`: 查看所有暂存记录
-   `git stash apply`: 恢复但不删除记录
-   `git stash clear`: 清空所有记录

### 撤销工作区的修改

```bash
# 丢弃文件在工作区的修改 (慎用！无法找回)
git checkout -- filename.py
# 或 (新版 Git)
git restore filename.py
```

### 撤销暂存区的修改

```bash
# 将文件从暂存区撤回工作区 (unstage)
git reset HEAD filename.py

# 将所有的修改从暂存区撤回 (Undo `git add .`)
git reset HEAD

# 或 (新版 Git)
git restore --staged filename.py
```

### 回退版本

```bash
# 回退到上一个版本 (保留工作区修改)
git reset --soft HEAD^
# 或者 (HEAD^ 和 HEAD~ 是一样的，都是指上一个版本)
git reset --soft HEAD~

# 彻底回退到指定 commit (危险操作，修改全丢)
git reset --hard <commit_id>
```

!!! question "HEAD^ 与 HEAD~ 的区别" - `git reset --soft HEAD^` 和 `git reset --soft HEAD~` **功能完全一样**，都是回退到上一个版本。 - `HEAD^` (Caret) 主要用于合并节点，选择第几个父提交。 - `HEAD~` (Tilde) 主要用于线性回退，`HEAD~2` 代表回退两步。 - 在 Windows 命令行 (CMD/PowerShell) 中，`^` 是转义字符，所以**推荐使用 `HEAD~`**，否则你需要写成 `HEAD^^` 或 `"HEAD^"`。

## 8. GitHub/GitLab 工作流 (Pull Request)

团队开发的标准流程：

1.  **Fork** 项目到自己仓库 (若是开源项目) 或 **Clone** 项目。
2.  新建分支 `feature-xxx` 开发。
3.  开发完成，`Push` 到远程。
4.  在网页端发起 **Pull Request (PR)** 或 **Merge Request (MR)**。
5.  团队成员 **Code Review**。
6.  代码没问题，管理员点击 **Merge** 合并进主分支。

---

_下一章：[Markdown 写作指南](02_Markdown.md)_
