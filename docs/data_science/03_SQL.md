# 7. 使用 SQL 处理关系数据 (Working with Relations Using SQL)

在上一章《使用 pandas 处理 Dataframe》中，我们使用 Dataframe 来表示数据表。本章将介绍 **关系 (Relations)**，这是另一种广泛使用的数据表表示方法。我们还将介绍 **SQL**，这是用于处理关系的各种标准编程语言。

这里有一个包含流行犬种信息的关系示例：

*(此处应有图表或表格，类似于上一章的狗的数据)*

像 Dataframe 一样，关系中的每一行代表一条记录——在这个例子中，是一个特定的犬种。每一列代表记录的一个特征——例如，`grooming` 列表示每种犬种需要梳理毛发的频率。

关系和 Dataframe 都有列标签。然而，一个关键的区别是，关系中的行没有标签 (Index)，而 Dataframe 中的行有。

在本章中，我们将使用 SQL 演示常见的关系操作。我们将首先解释 SQL 查询的结构。然后我们将展示如何使用 SQL 执行常见的数据操作任务，如切片 (slicing)、过滤 (filtering)、排序 (sorting)、分组 (grouping) 和 连接 (joining)。

!!! note "注意"
    本章使用关系和 SQL 复制了上一章使用 Dataframe 和 Python 进行的数据分析。为了方便比较在 pandas 和 SQL 中执行数据操作，两章中使用的数据集、数据操作和结论几乎是相同的。

## 1. 子集化 (Subsetting)

为了处理关系，我们将介绍一种名为 **SQL (Structured Query Language，结构化查询语言)** 的特定领域编程语言。我们通常将“SQL”发音为“sequel”，而不是拼出首字母缩写。SQL 是处理关系的专用语言——因此，它在操作关系数据方面的语法与 Python 不同。

在本章中，我们将在 Python 程序中使用 SQL 查询。这说明了一个常见的工作流程——数据科学家通常先在 SQL 中处理和子集化数据，然后再将数据加载到 Python 中进行进一步分析。与 pandas 程序相比，SQL 数据库更容易处理大量数据。但是，将数据加载到 pandas 中可以更轻松地可视化数据和构建统计模型。

!!! note "为什么 SQL 系统往往能更好地处理大型数据集？"
    简而言之，SQL 系统拥有复杂的算法来管理存储在磁盘上的数据。例如，当处理大型数据集时，SQL 系统会透明地一次加载和操作一小部分数据；相比之下，在 pandas 中这样做可能会非常困难。我们将在“文件处理 (Wrangling Files)”章节更详细地讨论这个话题。

### 1.1 SQL 基础：SELECT 和 FROM

我们将使用 `pd.read_sql` 函数，它运行 SQL 查询并将输出存储在 pandas dataframe 中。使用此函数需要一些设置。我们要先导入 `pandas` 和 `sqlalchemy` Python 包。

```python
import pandas as pd
import sqlalchemy
```

我们的数据库存储在一个名为 `babynames.db` 的文件中。这个文件是一个 SQLite 数据库，所以我们将设置一个可以处理这种格式的 sqlalchemy 对象：

```python
db = sqlalchemy.create_engine('sqlite:///babynames.db')
```

!!! note "关于数据库的选择"
    在本书中，我们使用 SQLite，这是一个用于处理本地存储数据的极其有用的数据库系统。其他系统会在不同领域做出不同的权衡。例如，PostgreSQL 和 MySQL 是更复杂的系统，适用于许多最终用户同时写入数据的大型 Web 应用程序。虽然每个 SQL 系统都有细微的差别，但它们提供相同的核心 SQL 功能。读者可能也知道 Python 在其标准库 `sqlite3` 中提供了 SQLite 支持。我们选择使用 `sqlalchemy`，因为它可以更容易地将代码重用于 SQLite 以外的其他 SQL 系统。

现在我们可以使用 `pd.read_sql` 在此数据库上运行 SQL 查询。此数据库有两个关系：`baby` 和 `nyt`。这是一个读取整个 `baby` 关系的简单示例：

```python
# SQL 查询保存在 Python 字符串中
query = ''' 
SELECT *
FROM baby;
'''

pd.read_sql(query, db)
```

| | Name | Sex | Count | Year |
| :--- | :--- | :--- | :--- | :--- |
| 0 | Liam | M | 19659 | 2020 |
| 1 | Noah | M | 18252 | 2020 |
| ... | ... | ... | ... | ... |
| 2020722 | Wilma | F | 5 | 1880 |

`query` 变量中的文本包含 SQL 代码。`SELECT` 和 `FROM` 是 SQL 关键字。我们将前面的查询解读为：

```sql
SELECT *    -- 获取所有列...
FROM baby;  -- ...从 baby 关系中
```

`baby` 关系包含与上一章《使用 pandas 处理 Dataframe》中的 `baby` dataframe 相同的数据：美国社会保障局注册的所有婴儿的名字。

### 1.2 什么是关系？

让我们更详细地检查 `baby` 关系。关系有行和列。每一列都有一个标签。然而，与 dataframe 不同，关系中的**单个行没有标签 (Index)**。同样与 dataframe 不同的是，关系的行是**无序**的。

关系有着悠久的历史。更正式的关系处理使用术语 **元组 (tuple)** 来指代关系的行，用 **属性 (attribute)** 来指代列。还有一种使用关系代数定义数据操作的严格方法，它源自数学集合代数。

### 1.3 切片 (Slicing)

切片是从另一个关系中提取行或列的子集并创建一个新关系的操作。这就像切西红柿——切片既可以是垂直的也可以是水平的。要切分关系的列，我们在 `SELECT` 语句中给出我们将想要的列名：

```python
query = ''' 
SELECT Name
FROM baby;
''' 

pd.read_sql(query, db)
```

| | Name |
| :--- | :--- |
| 0 | Liam |
| 1 | Noah |
| ... | ... |

```python
query = ''' 
SELECT Name, Count
FROM baby;
''' 

pd.read_sql(query, db)
```

| | Name | Count |
| :--- | :--- | :--- |
| 0 | Liam | 19659 |
| 1 | Noah | 18252 |
| ... | ... | ... |

要切分出特定数量的行，使用 `LIMIT` 关键字：

```python
query = ''' 
SELECT Name
FROM baby
LIMIT 10;
''' 

pd.read_sql(query, db)
```

| | Name |
| :--- | :--- |
| 0 | Liam |
| 1 | Noah |
| ... | ... |
| 9 | Alexander |

总之，我们使用 `SELECT` 和 `LIMIT` 关键字来切分关系的列和行。

### 1.4 过滤行 (Filtering Rows)

现在我们转向过滤行——使用一个或多个条件获取行的子集。在 pandas 中，我们使用布尔 Series对象切分 dataframe。在 SQL 中，我们改用带有谓词 (predicate) 的 `WHERE` 关键字。以下查询过滤 `baby` 关系，只保留 2020 年的婴儿名字：

```python
query = ''' 
SELECT *
FROM baby
WHERE Year = 2020;
'''

pd.read_sql(query, db)
```

| | Name | Sex | Count | Year |
| :--- | :--- | :--- | :--- | :--- |
| 0 | Liam | M | 19659 | 2020 |
| ... | ... | ... | ... | ... |
| 31270 | Zynlee | F | 5 | 2020 |

!!! warning "注意：SQL 中的等于号"
    请注意，在比较相等性时，SQL 使用**单个等号**：

    ```sql
    SELECT *
    FROM baby
    WHERE Year = 2020;
    --         ↑
    --         单个等号
    ```

    而在 Python 中，单个等号用于变量赋值。语句 `Year = 2020` 将值 2020 赋给变量 Year。为了比较相等性，Python 代码使用**双等号**：

    ```python
    # 赋值
    my_year = 2021

    # 比较，结果为 False
    my_year == 2020
    ```

要向过滤器添加更多谓词，可以使用 `AND` 和 `OR` 关键字。例如，要查找 2020 年或 2019 年婴儿数量超过 10,000 的名字，我们写道：

```python
query = ''' 
SELECT *
FROM baby
WHERE Count > 10000
  AND (Year = 2020
       OR Year = 2019);
-- 注意我们使用括号来强制执行运算顺序
'''

pd.read_sql(query, db)
```

最后，为了找出 2020 年最常见的 10 个名字，我们可以使用 `ORDER BY` 关键字和 `DESC` 选项（DESCending 的缩写）按 `Count` 降序对 dataframe 进行排序：

```python
query = ''' 
SELECT *
FROM baby
WHERE Year = 2020
ORDER BY Count DESC
LIMIT 10;
'''

pd.read_sql(query, db)
```

| | Name | Sex | Count | Year |
| :--- | :--- | :--- | :--- | :--- |
| 0 | Liam | M | 19659 | 2020 |
| 1 | Noah | M | 18252 | 2020 |
| 2 | Emma | F | 15581 | 2020 |
| ... | ... | ... | ... | ... |

我们看到 Liam、Noah 和 Emma 是 2020 年最受欢迎的婴儿名字。

### 1.5 案例：Luna 这个名字是什么时候变流行的？ (Example: How Recently Has Luna Become a Popular Name?)

我们在第 6 章介绍 pandas 时提到，一篇《纽约时报》文章指出 Luna 这个名字在 2000 年之前几乎不存在，但后来逐渐成为非常受女孩欢迎的名字。Luna 到底是什么时候开始流行的？我们可以使用 SQL 的切片和过滤来检查这一点：

```python
query = ''' 
SELECT *
FROM baby
WHERE Name = "Luna"
  AND Sex = "F";
'''

luna = pd.read_sql(query, db)
luna
```

| | Name | Sex | Count | Year |
| :--- | :--- | :--- | :--- | :--- |
| 0 | Luna | F | 7770 | 2020 |
| 1 | Luna | F | 7772 | 2019 |
| ... | ... | ... | ... | ... |
| 128 | Luna | F | 15 | 1880 |

`pd.read_sql` 返回一个 `pandas.DataFrame` 对象，我们可以用它来绘图。这说明了一个常见的工作流程——**使用 SQL 处理数据，将其加载到 pandas dataframe 中，然后可视化结果**：

```python
import plotly.express as px
px.line(luna, x='Year', y='Count', width=350, height=250)
```

在本节中，我们介绍了数据科学家对关系进行子集化的常见方法——使用列标签切片和使用布尔条件过滤。在下一节中，我们将解释如何聚合行。

---

## 2. 聚合行 (Aggregating)

本节介绍 SQL 中的分组和聚合。我们将使用与上一节相同的婴儿名字数据：

```python
import sqlalchemy
db = sqlalchemy.create_engine('sqlite:///babynames.db')
query = ''' 
SELECT *
FROM baby
LIMIT 10;
'''

pd.read_sql(query, db)
```

*(显示前 10 行数据)*

### 2.1 使用 GROUP BY 进行基础分组聚合 (Basic Group-Aggregate Using GROUP BY)

假设我们要找出此数据中记录的所有出生婴儿总数。这只是 `Count` 列的总和。SQL 提供了我们在 `SELECT` 语句中使用的函数，如 `SUM`：

```python
query = ''' 
SELECT SUM(Count)
FROM baby;
'''

pd.read_sql(query, db)
```

| | SUM(Count) |
| :--- | :--- |
| 0 | 352554503 |

在后面章节中，我们使用分组和聚合来弄清楚美国出生人数是否随时间呈上升趋势。我们使用 `.groupby()` 按年份对数据集进行分组，然后使用 `.sum()` 对每组内的计数求和。

在 SQL 中，我们改用 `GROUP BY` 子句进行分组，然后在 `SELECT` 中调用聚合函数：

```python
query = ''' 
SELECT Year, SUM(Count)
FROM baby
GROUP BY Year;
'''

pd.read_sql(query, db)
```

| | Year | SUM(Count) |
| :--- | :--- | :--- |
| 0 | 1880 | 194419 |
| 1 | 1881 | 185772 |
| ... | ... | ... |
| 141 | 2020 | 3287724 |

与 dataframe 分组一样，请注意 `Year` 列包含唯一的 `Year` 值——由于我们将它们分组在一起，因此不再有重复的 `Year` 值。在 pandas 中分组时，分组列成为结果 dataframe 的索引。然而，关系没有行标签，因此 `Year` 值只是结果关系中的一列。

!!! note "Dataframe 索引 vs SQL 列"
    初学者常困惑的一点是：**Pandas Dataframe 的索引 (Index)** 与 **SQL 结果中的列** 有何不同？

    *   **Pandas 中**：当你执行 `baby.groupby('Year')` 时，`Year` 列会被提升为 **索引**。索引就像是每一行的“身份证号”或“标签”。Pandas 会将索引及其对应的数据分开存储和展示（通常索引在左侧加粗显示）。你可以通过这个索引标签快速找到某一行（例如 `df.loc[1880]`）。
    *   **SQL 中**：数据库表（关系）是“扁平”的。即使你按 `Year` 分组，`Year` 在结果中也只是一列普普通通的数据，和 `Count` 列没有什么地位上的区别。SQL 中没有“行标签”的概念，所有的筛选都必须通过 `WHERE` 子句（例如 `WHERE Year = 1880`）来完成，而不能像 Pandas 那样直接通过标签定位。

这是在 SQL 中进行分组的基本配方：

```sql
SELECT
  col1,           -- 用于分组的列
  SUM(col2)       -- 另一列的聚合
FROM table_name   -- 使用的关系
GROUP BY col1;    -- 要分组的列
```

注意 SQL 语句中子句的顺序很重要。为了避免语法错误，`SELECT` 需要首先出现，然后是 `FROM`，然后是 `WHERE`，然后是 `GROUP BY`。

使用 `GROUP BY` 时，我们需要小心提供给 `SELECT` 的列。一般而言，我们在使用列进行分组时，只能包含没有聚合的列。例如，在前面的例子中，我们按 `Year` 列分组，所以我们可以在 `SELECT` 子句中包含 `Year`。`SELECT` 中包含的所有其他列都应该像我们在前面的 `SUM(Count)` 中所做的那样进行聚合。如果我们包含一个像 `Name` 这样没有用于分组的“裸”列，那么在组内应该返回哪个名字是不明确的。尽管裸列在 SQLite 中不会导致错误，但它们会导致其他 SQL 引擎报错，因此我们建议避免使用它们。

### 2.2 多列分组 (Grouping on Multiple Columns)

我们将多个列传递给 `GROUP BY` 以便同时按多个列分组。当我们需要进一步细分我们的组时，这很有用。例如，我们可以按年份和性别分组，看看随着时间的推移有多少男婴和女婴出生：

```python
query = ''' 
SELECT Year, Sex, SUM(Count)
FROM baby
GROUP BY Year, Sex;
'''

pd.read_sql(query, db)
```

| | Year | Sex | SUM(Count) |
| :--- | :--- | :--- | :--- |
| 0 | 1880 | F | 83929 |
| 1 | 1880 | M | 110490 |
| ... | ... | ... | ... |
| 281 | 2020 | M | 1706423 |

请注意，前面的代码与单列分组非常相似，只是在 `GROUP BY` 中给出了多个列，以便同时按 `Year` 和 `Sex` 分组。

!!! note "注意"
    与 pandas 不同，SQLite 没有提供简单的方法来透视关系。相反，我们可以在 SQL 中对两列使用 `GROUP BY`，将结果读入 dataframe，然后使用 `unstack()` dataframe 方法。

### 2.3 其他聚合函数 (Other Aggregation Functions)

除了 `SUM`，SQLite 还有其他几个内置聚合函数，如 `COUNT`、`AVG`、`MIN` 和 `MAX`。有关函数的完整列表，请查阅 SQLite 网站。

要使用其他聚合函数，我们在 `SELECT` 子句中调用它。例如，我们可以使用 `MAX` 代替 `SUM`：

```python
query = ''' 
SELECT Year, MAX(Count)
FROM baby
GROUP BY Year;
'''

pd.read_sql(query, db)
```

| | Year | MAX(Count) |
| :--- | :--- | :--- |
| 0 | 1880 | 9655 |
| 1 | 1881 | 8769 |
| ... | ... | ... |
| 141 | 2020 | 19659 |

!!! note "注意"
    内置聚合函数是数据科学家可能会遇到 SQL 实现差异的首要地方之一。例如，SQLite 的聚合函数集相对较少，而 PostgreSQL 的聚合函数要多得多。也就是说，几乎所有的 SQL 实现都提供 `SUM`、`COUNT`、`MIN`、`MAX` 和 `AVG`。

本节介绍了使用 `GROUP BY` 关键字对一列或多列进行数据聚合的常见方法。在下一节中，我们将解释如何连接关系。

---

## 3. 连接关系 (Joining)

为了连接两个数据表中的记录，SQL 关系可以像 dataframe 一样连接在一起。在本节中，我们将介绍 SQL 连接，以复制我们对婴儿名字数据的分析。回想一下，第 6 章曾提到一篇《纽约时报》的文章，其中讨论了某些名字类别（如神话和婴儿潮一代的名字）随着时间的推移变得越来越流行或越来越不流行。

我们将《纽约时报》文章中的名字和类别放入了一个名为 `nyt` 的小关系中：

```python
# 设置数据库连接
import sqlalchemy
db = sqlalchemy.create_engine('sqlite:///babynames.db')
query = ''' 
SELECT *
FROM nyt;
'''

pd.read_sql(query, db)
```

| | nyt_name | category |
| :--- | :--- | :--- |
| 0 | Lucifer | forbidden |
| 1 | Lilith | forbidden |
| ... | ... | ... |
| 20 | Venus | celestial |

!!! note "注意"
    请注意，前面的代码在 `babynames.db` 上运行查询，该数据库包含与其包含前几节中较大的 `baby` 关系的同一个数据库。SQL 数据库可以容纳多个关系，这使得它们在我们需要同时处理多个数据表时非常有用。另一方面，CSV 文件通常每个文件包含一个数据表——如果我们执行使用 20 个数据表的分析，我们可能需要跟踪 20 个 CSV 文件的名称、位置和版本。相反，将所有数据表存储在单个文件中存储的 SQLite 数据库中可能更简单。

为了了解这些名字类别的流行程度，我们将 `nyt` 关系与 `baby` 关系连接起来，以获取 `baby` 表中的名字计数。

### 3.1 内连接 (Inner Joins)

正如第 6 章所述，我们制作了 `baby` 和 `nyt` 表的较小版本，以便更容易看到连接表时发生的情况。这些关系称为 `baby_small` 和 `nyt_small`：

```python
query = ''' 
SELECT *
FROM baby_small;
'''

pd.read_sql(query, db)
```

| | Name | Sex | Count | Year |
| :--- | :--- | :--- | :--- | :--- |
| 0 | Noah | M | 18252 | 2020 |
| 1 | Julius | M | 960 | 2020 |
| 2 | Karen | M | 6 | 2020 |
| 3 | Karen | F | 325 | 2020 |
| 4 | Noah | F | 305 | 2020 |

```python
query = ''' 
SELECT *
FROM nyt_small;
'''

pd.read_sql(query, db)
```

| | nyt_name | category |
| :--- | :--- | :--- |
| 0 | Karen | boomer |
| 1 | Julius | mythology |
| 2 | Freya | mythology |

要在 SQL 中连接关系，我们使用 `INNER JOIN` 子句来说明我们要连接哪些表，并使用 `ON` 子句来指定用于连接表的谓词。这是一个例子：

```python
query = ''' 
SELECT *
FROM baby_small INNER JOIN nyt_small
  ON baby_small.Name = nyt_small.nyt_name
'''

pd.read_sql(query, db)
```

| Name | Sex | Count | Year | nyt_name | category |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Julius | M | 960 | 2020 | Julius | mythology |
| Karen | M | 6 | 2020 | Karen | boomer |
| Karen | F | 325 | 2020 | Karen | boomer |

请注意，此结果与在 pandas 中进行内连接的结果相同：新表包含 `baby_small` 和 `nyt_small` 表的所有列。名为 Noah 的行消失了，剩下的行都有来自 `nyt_small` 的匹配类别。

为了将两个表连接在一起，我们使用带有 `ON` 关键字的谓词告诉 SQL 每个表中要用于进行连接的列。当连接列中的值满足谓词时，SQL 将行匹配在一起。

!!! note "注意"
    与 pandas 不同，SQL 在如何连接行方面提供了更大的灵活性。`pd.merge()` 方法只能使用简单的相等性进行连接，但 `ON` 子句中的谓词可以是任意复杂的。作为示例，我们在第 12 章第 12.2 节利用了这种额外的多功能性。

### 3.2 左连接和右连接 (Left and Right Joins)

像 pandas 一样，SQL 也支持左连接。我们不使用 `INNER JOIN`，而是使用 `LEFT JOIN`：

```python
query = ''' 
SELECT *
FROM baby_small LEFT JOIN nyt_small
  ON baby_small.Name = nyt_small.nyt_name
'''

pd.read_sql(query, db)
```

| Name | Sex | Count | Year | nyt_name | category |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Noah | M | 18252 | 2020 | None | None |
| Julius | M | 960 | 2020 | Julius | mythology |
| Karen | M | 6 | 2020 | Karen | boomer |
| Karen | F | 325 | 2020 | Karen | boomer |
| Noah | F | 305 | 2020 | None | None |

正如我们所期望的那样，连接的“左”侧是指出现在 `LEFT JOIN` 关键字左侧的表。我们可以看到，即使 Noah 行在右侧关系中没有匹配项，它们也会保留在结果关系中。

请注意，SQLite 直接**不支持右连接 (RIGHT JOIN)**，但我们可以通过交换关系的顺序，然后使用 `LEFT JOIN` 来执行相同的连接：

```python
query = ''' 
SELECT *
FROM nyt_small LEFT JOIN baby_small
  ON baby_small.Name = nyt_small.nyt_name
'''

pd.read_sql(query, db)
```

| nyt_name | category | Name | Sex | Count | Year |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Karen | boomer | Karen | F | 325.0 | 2020.0 |
| Karen | boomer | Karen | M | 6.0 | 2020.0 |
| Julius | mythology | Julius | M | 960.0 | 2020.0 |
| Freya | mythology | None | None | NaN | NaN |

SQLite 也没有外连接 (Outer Join) 的内置关键字。在需要外连接的情况下，我们必须使用不同的 SQL 引擎或通过 pandas 执行外连接。然而，根据我们的经验，与内连接和左连接相比，外连接在实践中很少使用。

### 3.3 案例：纽约时报名字类别的流行度 (Example: Popularity of NYT Name Categories)

现在让我们回到完整的 `baby` 和 `nyt` 关系。

我们要了解 `nyt` 中名字类别的流行度随时间的变化。为了回答这个问题，我们需要：

1.  将 `baby` 与 `nyt` 进行**内连接**，匹配名字相等的行。
2.  按 `category` 和 `Year` 对表进行**分组**。
3.  使用总和 (sum) **聚合**计数：

```python
query = ''' 
SELECT
  category,
  Year,
  SUM(Count) AS count           -- [3]
FROM baby INNER JOIN nyt        -- [1]
  ON baby.Name = nyt.nyt_name   -- [1]
GROUP BY category, Year         -- [2]
'''

cate_counts = pd.read_sql(query, db)
cate_counts
```

| category | Year | count |
| :--- | :--- | :--- |
| boomer | 1880 | 292 |
| boomer | 1881 | 298 |
| ... | ... | ... |
| mythology | 2020 | 3489 |

前面查询中的方括号中的数字 ([1], [2], [3]) 显示了我们计划中的每个步骤如何映射到 SQL 查询的各个部分。该代码重新创建了前一章节中的 dataframe，我们在那里创建了图表来验证《纽约时报》文章的说法。为了简洁起见，我们省略了在此处重复绘图。

!!! note "SQL 执行顺序与书写顺序"
    请注意，在此示例的 SQL 代码中，数字出现的顺序也是乱序的——[3] -> [1] -> [2]。作为 SQL 初学者的经验法则，我们通常可以将 `SELECT` 语句视为查询中**最后执行**的部分，即使它出现在最前面。

在本节中，我们介绍了关系的连接。将关系连接在一起时，我们使用 `INNER JOIN` 或 `LEFT JOIN` 关键字和布尔谓词来匹配行。在下一节中，我们将解释如何变换关系中的值。

---

## 4. 变换与通用表表达式 (Transforming and CTEs)

在本节中，我们展示如何调用内置 SQL 函数来变换数据列。我们还演示了如何使用通用表表达式 (Common Table Expressions, CTEs) 从简单的查询构建复杂的查询。像往常一样，我们首先加载数据库：

```python
# 设置数据库连接
import sqlalchemy
db = sqlalchemy.create_engine('sqlite:///babynames.db')
```

### 4.1 SQL 函数

SQLite 提供了各种**标量函数 (Scalar Functions)**，即变换单个数据值的函数。当在数据列上调用时，SQLite 会将这些函数应用于列中的每个值。相比之下，像 `SUM` 和 `COUNT` 这样的聚合函数将一列值作为输入并计算单个值作为输出。

SQLite 在其在线文档中提供了内置标量函数的完整列表。例如，要查找每个名字的字符数，我们使用 `LENGTH` 函数：

```python
query = ''' 
SELECT Name, LENGTH(Name)
FROM baby
LIMIT 10;
'''

pd.read_sql(query, db)
```

| | Name | LENGTH(Name) |
| :--- | :--- | :--- |
| 0 | Liam | 4 |
| 1 | Noah | 4 |
| ... | ... | ... |
| 9 | Alexander | 9 |

请注意，`LENGTH` 函数应用于 `Name` 列中的每个值。

!!! note "注意"
    与聚合函数一样，每个 SQL 实现都提供一组不同的标量函数。SQLite 具有相对较少的函数集，而 PostgreSQL 还有更多。也就是说，几乎所有的 SQL 实现都提供了一些相当于 SQLite 的 `LENGTH`、`ROUND`、`SUBSTR` 和 `LIKE` 函数。

尽管标量函数使用与聚合函数相同的语法，但它们的行为不同。如果在单个查询中将两者混合在一起，可能会导致令人困惑的输出：

```python
query = ''' 
SELECT Name, LENGTH(Name), AVG(Count)
FROM baby
LIMIT 10;
'''

pd.read_sql(query, db)
```

| Name | LENGTH(Name) | AVG(Count) |
| :--- | :--- | :--- |
| Liam | 4 | 174.47 |

在这里，`AVG(Count)` 计算整个 `Count` 列的平均值，但输出令人困惑——读者很容易认为平均值与名字 Liam 有关。因此，当标量函数和聚合函数一起出现在 `SELECT` 语句中时，我们必须小心。

要提取每个名字的首字母，我们可以使用 `SUBSTR` 函数（substring 的缩写）。如文档所述，`SUBSTR` 函数接受三个参数。第一个是输入字符串，第二个是开始子字符串的位置（**从1开始索引**），第三个是子字符串的长度：

```python
query = ''' 
SELECT Name, SUBSTR(Name, 1, 1)
FROM baby
LIMIT 10;
'''

pd.read_sql(query, db)
```

| | Name | SUBSTR(Name, 1, 1) |
| :--- | :--- | :--- |
| 0 | Liam | L |
| 1 | Noah | N |
| ... | ... | ... |
| 9 | Alexander | A |

我们可以使用 `AS` 关键字重命名列：

```python
query = ''' 
SELECT *, SUBSTR(Name, 1, 1) AS Firsts
FROM baby
LIMIT 10;
'''

pd.read_sql(query, db)
```

| Name | Sex | Count | Year | Firsts |
| :--- | :--- | :--- | :--- | :--- |
| Liam | M | 19659 | 2020 | L |
| Noah | M | 18252 | 2020 | N |
| ... | ... | ... | ... | ... |

在计算出每个名字的首字母后，我们的分析旨在了解首字母随时间的流行度。为此，我们希望获取此 SQL 查询的输出，并将其用作更长操作链中的一个步骤。

SQL 提供了多种选项将查询分解为更小的步骤，这在像这样的更复杂的分析中很有帮助。执行此操作的最常见选项是使用 `CREATE TABLE` 语句创建新关系，使用 `CREATE VIEW` 创建新视图，或使用 `WITH` 创建临时关系。这些方法中的每一个都有不同的用例。为简单起见，我们在本节中仅描述 `WITH` 语句，并建议读者查看 SQLite 文档以获取详细信息。

### 4.2 使用 WITH 子句的多步查询

`WITH` 子句允许我们将名称分配给任何 `SELECT` 查询。然后，我们可以将该查询视为仅在查询持续时间内存在于数据库中的关系。SQLite 称这些临时关系为 **通用表表达式 (Common Table Expressions, CTEs)**。例如，我们可以获取前面计算每个名字首字母的查询，并将其命名为 `letters`：

```python
query = ''' 
-- 创建一个名为 letters 的临时关系，计算 baby 中每个名字的首字母
WITH letters AS (
  SELECT *, SUBSTR(Name, 1, 1) AS Firsts
  FROM baby
)
-- 然后，从 letters 中选择前十行
SELECT *
FROM letters
LIMIT 10;
'''

pd.read_sql(query, db)
```

`WITH` 语句非常有用，因为它们可以链接在一起。我们可以在 `WITH` 语句中创建多个临时关系，每个关系都对前一个结果执行一点工作，这使我们能够逐步构建复杂的查询。

### 4.3 案例："L" 开头名字的流行度 (Example: Popularity of “L” Names)

我们可以使用 `WITH` 语句来查看以字母 L 开头的名字随时间的流行度。我们将按首字母和年份对临时 `letters` 关系进行分组，然后使用总和聚合 `Count` 列，然后过滤以仅获取以字母 L 开头的名字：

```python
query = ''' 
WITH letters AS (
  SELECT *, SUBSTR(Name, 1, 1) AS Firsts
  FROM baby
)
SELECT Firsts, Year, SUM(Count) AS Count
FROM letters
WHERE Firsts = "L"
GROUP BY Firsts, Year;
'''

letter_counts = pd.read_sql(query, db)
letter_counts
```

| | Firsts | Year | Count |
| :--- | :--- | :--- | :--- |
| 0 | L | 1880 | 12799 |
| 1 | L | 1881 | 12770 |
| ... | ... | ... | ... |
| 141 | L | 2020 | 239760 |

此关系包含与先前章节中的关系相同的数据。在那一章中，我们绘制了随时间变化的 `Count` 列图表，为了简洁起见，如果你需要你可以会看那一章的内容。

在本节中，我们介绍了数据变换。为了变换关系中的值，我们通常使用像 `LENGTH()` 或 `SUBSTR()` 这样的 SQL 函数。我们还解释了如何使用 `WITH` 子句构建复杂的查询。

---

[下一章：文件处理 (Wrangling Files)](./04_文件处理.md)
