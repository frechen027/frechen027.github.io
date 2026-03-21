# Pandas

Pandas 包在 Numpy 数组的基础上添加了与 Excel 类似的行列标签。

## 对象的创建

- 导入 Pandas 时，通常给其一个别名 "pd"，即`import pandas as pd`
- 作为标签库，Pandas 对象在 Numpy 数组基础上给予其行列标签，可以说，列表之于字典，就如 Numpy 之于 Pandas。
- Pandas 中，所有数组特性仍在，Pandas 的数据以 Numpy 数组的方式存储。

### 一维对象的创建

#### 字典创建法

NumPy中，可以通过 np.array()函数，将Python列表转化为NumPy数组；同样，Pandas中，可以通过pd.Series()函数，将Python字典转化为Series对象。


```python
import pandas as pd 
```


```python
# 创建字典 
dict_v = { 'a':0, 'b':0.25, 'c':0.5, 'd':0.75, 'e':1 } 
```


```python
# 用字典创建对象 
sr = pd.Series( dict_v ) 
sr 
```




    a    0.00
    b    0.25
    c    0.50
    d    0.75
    e    1.00
    dtype: float64



#### 数组创建法

最直接的创建方法即直接给pd.Series()函数参数，其需要两个参数。第一个参数是值values(列表、数组、张量均可)，第二个参数是键index(索引)。


```python
# 先定义键与值 
v = [0, 0.25, 0.5, 0.75, 1] 
k = ['a', 'b', 'c', 'd', 'e']
```


```python
# 用列表创建对象 
sr = pd.Series( v, index=k ) 
sr
```




    a    0.00
    b    0.25
    c    0.50
    d    0.75
    e    1.00
    dtype: float64




```python
sr.values
```




    array([0.  , 0.25, 0.5 , 0.75, 1.  ])




```python
# 其中，参数 index 可以省略，省略后索引即从 0 开始的顺序数字。 
sr = pd.Series( v ) 
sr
```




    0    0.00
    1    0.25
    2    0.50
    3    0.75
    4    1.00
    dtype: float64



### 一维对象的属性

Series对象有两个属性：values与index。


```python
import numpy as np 
import pandas as pd
```


```python
# 用数组创建 sr 
v = np.array( [ 53, 64, 72, 82 ] ) 
k = ['1 号', '2 号', '3 号', '4 号'] 
sr = pd.Series( v, index=k ) 
sr
```




    1 号    53
    2 号    64
    3 号    72
    4 号    82
    dtype: int64




```python
# 查看 values 属性 
sr.values
```




    array([53, 64, 72, 82])




```python
# 查看 index 属性 
sr.index
```




    Index(['1 号', '2 号', '3 号', '4 号'], dtype='object')



事实上，无论是用列表、数组还是张量来创建对象，**最终valus均为数组**。


```python
import pandas as pd 
import torch
```


```python
# 用张量创建 sr 
v = torch.tensor( [ 53, 64, 72, 82 ] ) 
k = ['1 号', '2 号', '3 号', '4 号'] 
sr = pd.Series( v, index=k ) 
sr
```




    1 号    53
    2 号    64
    3 号    72
    4 号    82
    dtype: int64




```python
# 查看 values 的属性 
sr.values 
```




    array([53, 64, 72, 82])



可见，虽然 Pandas 对象的第一个参数 values 可以传入列表、数组与张量，但传进去后默认的存储方式是 NumPy 数组。这一点更加提醒我们，Pandas 是建立在 NumPy 基础上的库，没有 NumPy 数组库就没有 Pandas 数据处理库。 

当想要 Pandas 退化为 NumPy 时，查看其 values 属性即可。

### 二维对象的创建

二维对象将面向矩阵，其不仅有行标签 index，还有列标签 columns。

#### 字典创建法

用字典创建二维对象时，必须基于多个Series对象，每一个Series就是一列数据，相当于对一列一列的数据进行拼接。

- 创建Series对象时，字典的键是index，其延展方向是竖直的；
- 创建DataFrame对象时，字典的键是columns，其延展方向是水平的。


```python
import pandas as pd 
```


```python
# 创建 sr1：各个病人的年龄 
v1 = [ 53, 64, 72, 82 ] 
i = [ '1 号', '2 号', '3 号', '4 号' ] 
sr1 = pd.Series( v1, index=i ) 
sr1 

```




    1 号    53
    2 号    64
    3 号    72
    4 号    82
    dtype: int64




```python
# 创建 sr2：各个病人的性别 
v2 = [ '女', '男', '男', '女' ] 
i = [ '1 号', '2 号', '3 号', '4 号' ] 
sr2 = pd.Series( v2, index=i ) 
sr2
```




    1 号    女
    2 号    男
    3 号    男
    4 号    女
    dtype: object




```python
# 创建 df 对象 
df = pd.DataFrame( { '年龄':sr1, '性别':sr2 } ) 
df 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>性别</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>53</td>
      <td>女</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>64</td>
      <td>男</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>72</td>
      <td>男</td>
    </tr>
    <tr>
      <th>4 号</th>
      <td>82</td>
      <td>女</td>
    </tr>
  </tbody>
</table>
</div>



如果 sr1 和 sr2 的index不完全一致，那么二维对象的 index 会取 sr1 与 sr2 的所有 index，相应的，该对象就会产生一定数量的缺失值（NaN）。


```python
import pandas as pd 

# 创建 sr1：各个病人的年龄 
v1 = [ 53, 64, 72, 82 ] 
i = [ '1 号', '2 号', '3 号', '4 号' ] 
sr1 = pd.Series( v1, index=i ) 

# 创建 sr2：各个病人的性别 
v2 = [ '女', '男', '男', '女' ] 
i = [ '1 号', '2 号', '3 号', '5 号' ] 
sr2 = pd.Series( v2, index=i ) 

# 创建 df 对象 
df = pd.DataFrame( { '年龄':sr1, '性别':sr2 } ) 
df 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>性别</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>53.0</td>
      <td>女</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>64.0</td>
      <td>男</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>72.0</td>
      <td>男</td>
    </tr>
    <tr>
      <th>4 号</th>
      <td>82.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5 号</th>
      <td>NaN</td>
      <td>女</td>
    </tr>
  </tbody>
</table>
</div>



#### 数组创建法

最直接的创建方法即直接给pd.DataFrame函数参数，其需要**三个参数**。第一个参数是值values(数组)，第二个参数是行标签index，第三个参数是列标签columns。其中，index和columns参数可以忽略，省略后即从0开始的顺序数字。


```python
import numpy as np 
import pandas as pd 

# 设定键值 
v = np.array( [ [53, '女'], [64, '男'], [72, '男'], [82, '女'] ] ) 
i = [ '1 号', '2 号', '3 号', '4 号' ] 
c = [ '年龄', '性别' ] 

# 数组创建法 
df = pd.DataFrame( v, index=i, columns=c ) 
df 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>性别</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>53</td>
      <td>女</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>64</td>
      <td>男</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>72</td>
      <td>男</td>
    </tr>
    <tr>
      <th>4 号</th>
      <td>82</td>
      <td>女</td>
    </tr>
  </tbody>
</table>
</div>



细心的同学可能会发现端倪，NumPy 数组居然又含数字又含字符串，上次课中明明讲过**数组只能容纳一种变量类型**。这里的原理是，数组**默默把数字转为了字符串**，于是 v 就是一个字符串型数组。 


### 二维对象的属性

DataFrame 对象有三个属性：values、index 与 columns。


```python
import pandas as pd 
```


```python
# 设定键值 
v = [ [53, '女'], [64, '男'], [72, '男'], [82, '女'] ] 
i = [ '1 号', '2 号', '3 号', '4 号' ] 
c = [ '年龄', '性别' ] 

```


```python
# 数组创建法 
df = pd.DataFrame( v, index=i, columns=c ) 
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>性别</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>53</td>
      <td>女</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>64</td>
      <td>男</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>72</td>
      <td>男</td>
    </tr>
    <tr>
      <th>4 号</th>
      <td>82</td>
      <td>女</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 查看 values 属性 
df.values 

```




    array([[53, '女'],
           [64, '男'],
           [72, '男'],
           [82, '女']], dtype=object)




```python
# 查看 index 属性 
df.index
```




    Index(['1 号', '2 号', '3 号', '4 号'], dtype='object')




```python
# 查看 columns 属性 
df.columns
```




    Index(['年龄', '性别'], dtype='object')



**当想要 Pandas 退化为 NumPy 时，查看其 values 属性即可。**



```python
# 提取完整的数组 
arr = df.values 
print(arr) 
```

    [[53 '女']
     [64 '男']
     [72 '男']
     [82 '女']]
    


```python
# 提取第[0]列，并转化为一个整数型数组 
arr = arr[:,0].astype(int)
print(arr) 
```

    [53 64 72 82]
    

## 对象的索引

在学习 Pandas 的索引之前，需要知道 
- Pandas 的索引分为显式索引与隐式索引。显式索引是使用 Pandas 对象提供的索引，而隐式索引是使用数组本身自带的从 0 开始的索引。 
- 现假设某演示代码中的索引是整数，这个时候显式索引和隐式索引可能会出乱子。于是，Pandas 作者发明了索引器 loc（显式）与 iloc（隐式），手动告诉程序自己这句话是显式索引还是隐式索引。 
- 本章示例中，若代码块出现两栏，则左侧为显式索引，右侧为隐式索引，左右两列属于平行关系，任选其一均可。

### 一维对象的索引

#### 访问元素


```python
import pandas as pd 

# 创建 sr 
v = [ 53, 64, 72, 82 ] 
k = ['1 号', '2 号', '3 号', '4 号'] 
sr = pd.Series( v, index=k ) 
sr 
```




    1 号    53
    2 号    64
    3 号    72
    4 号    82
    dtype: int64




```python
# 通过显式索引访问元素 
sr.loc[ '3 号' ] 
```




    np.int64(72)




```python
# 通过隐式索引访问元素 
sr.iloc[ 2 ]
```




    np.int64(72)




```python
# 通过显式索引进行花式索引
sr.loc[['1 号', '3 号']]
```




    1 号    53
    3 号    72
    dtype: int64




```python
# 通过隐式索引进行花式索引
sr.iloc[[0, 2]]
```




    1 号    53
    3 号    72
    dtype: int64




```python
# 通过显式索引修改元素
sr.loc['3 号'] = 100
sr
```




    1 号     53
    2 号     64
    3 号    100
    4 号     82
    dtype: int64




```python
# 通过隐式索引修改元素
sr.iloc[2] = 200
sr
```




    1 号     53
    2 号     64
    3 号    200
    4 号     82
    dtype: int64



#### 访问切片

使用显式索引时，'1 号':'3 号'可以覆盖最后一个'3 号'，但隐式与之前一样。


```python
import pandas as pd
```


```python
# 创建 sr 
v = [ 53, 64, 72, 82 ] 
k = ['1 号', '2 号', '3 号', '4 号'] 
sr = pd.Series( v, index=k ) 
sr 
```




    1 号    53
    2 号    64
    3 号    72
    4 号    82
    dtype: int64




```python
# 访问切片 
sr.loc[ '1 号':'3 号' ]
```




    1 号    53
    2 号    64
    3 号    72
    dtype: int64




```python
# 访问切片，可以看到还是左闭右开的
sr.iloc[ 0:3 ] 
```




    1 号    53
    2 号    64
    3 号    72
    dtype: int64




```python
# 切片仅是视图，改变切片实际上改变的是原数据
cut = sr.loc[ '1 号':'3 号' ] 
cut.loc['1 号'] = 100 
sr 

```




    1 号    100
    2 号     64
    3 号     72
    4 号     82
    dtype: int64




```python
# 切片仅是视图 
cut = sr.iloc[ 0:3 ] 
cut.iloc[0] = 200 
sr
```




    1 号    200
    2 号     64
    3 号     72
    4 号     82
    dtype: int64




```python
# 对象赋值仅是绑定，并非拷贝
cut = sr 
cut.loc['3 号'] = 200 
sr 
```




    1 号    200
    2 号     64
    3 号    200
    4 号     82
    dtype: int64




```python
# 对象赋值仅是绑定 
cut = sr 
cut.iloc[2] = 300 
sr
```




    1 号    200
    2 号     64
    3 号    300
    4 号     82
    dtype: int64



若想创建新变量，与 NumPy 一样，使用.copy()方法即可。

如果去掉 .loc 和 .iloc ，此时与 NumPy 中的索引语法完全一致。 

### 二维对象的索引

#### 访问元素


```python
import pandas as pd 
```


```python
# 字典创建法 
i = [ '1 号', '2 号', '3 号', '4 号' ] 
v1 = [ 53, 64, 72, 82 ] 
v2 = [ '女', '男', '男', '女' ] 
sr1 = pd.Series( v1, index=i ) 
sr2 = pd.Series( v2, index=i ) 
df = pd.DataFrame( { '年龄':sr1, '性别':sr2 } ) 
df 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>性别</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>53</td>
      <td>女</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>64</td>
      <td>男</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>72</td>
      <td>男</td>
    </tr>
    <tr>
      <th>4 号</th>
      <td>82</td>
      <td>女</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 访问元素 
df.loc[ '1 号', '年龄' ] 
```




    np.int64(53)




```python
# 访问元素 
df.iloc[ 0, 0 ] 
```




    np.int64(53)




```python
# 花式索引 
df.loc[ ['1 号', '3 号'] , ['性别','年龄'] ]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>性别</th>
      <th>年龄</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>女</td>
      <td>53</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>男</td>
      <td>72</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 花式索引 
df.iloc[ [0,2], [1,0] ] 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>性别</th>
      <th>年龄</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>女</td>
      <td>53</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>男</td>
      <td>72</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 修改元素 
df.loc[ '3 号', '年龄' ] = 100 
df 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>性别</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>53</td>
      <td>女</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>64</td>
      <td>男</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>100</td>
      <td>男</td>
    </tr>
    <tr>
      <th>4 号</th>
      <td>82</td>
      <td>女</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 修改元素 
df.iloc[ 2, 0 ] = 200 
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>性别</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>53</td>
      <td>女</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>64</td>
      <td>男</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>200</td>
      <td>男</td>
    </tr>
    <tr>
      <th>4 号</th>
      <td>82</td>
      <td>女</td>
    </tr>
  </tbody>
</table>
</div>



在 NumPy 数组中，花式索引输出的是一个向量。但在 Pandas 对象中，考虑到其行列标签的信息不能丢失，所以输出一个向量就不行了，所以这里才输出一个二维对象。

#### 访问切片


```python
# 数组创建法 
v = [ [53, '女'], [64, '男'], [72, '男'], [82, '女'] ] 
i = [ '1 号', '2 号', '3 号', '4 号' ] 
c = [ '年龄', '性别' ] 
df = pd.DataFrame( v, index=i, columns=c ) 
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>性别</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>53</td>
      <td>女</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>64</td>
      <td>男</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>72</td>
      <td>男</td>
    </tr>
    <tr>
      <th>4 号</th>
      <td>82</td>
      <td>女</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 切片 
df.loc[ '1 号':'3 号' , '年龄' ] 
```




    1 号    53
    2 号    64
    3 号    72
    Name: 年龄, dtype: int64




```python
# 切片 
df.iloc[ 0:3 , 0 ] 

```




    1 号    53
    2 号    64
    3 号    72
    Name: 年龄, dtype: int64




```python
# 提取二维对象的行 
df.loc[ '3 号' , : ]
```




    年龄    72
    性别     男
    Name: 3 号, dtype: object




```python
# 提取二维对象的行 
df.iloc[ 2 , : ] 
```




    年龄    72
    性别     男
    Name: 3 号, dtype: object




```python
# 提取矩阵对象的列 
df.loc[ : , '年龄' ]
```




    1 号    53
    2 号    64
    3 号    72
    4 号    82
    Name: 年龄, dtype: int64




```python
# 提取矩阵对象的列 
df.iloc[ : , 0 ] 
```




    1 号    53
    2 号    64
    3 号    72
    4 号    82
    Name: 年龄, dtype: int64



在显示索引中，提取矩阵的行或列还有一种简便写法，即 
- 提取二维对象的行：`df.loc['3 号']` （原理是省略后面的冒号，隐式也可以） 
- 提取二维对象的列：`df ['年龄']`（原理是列标签本身就是二维对象的键）

## 对象的变形

### 对象的转置

有时候提供的大数据很畸形，行是特征，列是个体，这必须要先进行转置。


```python
import pandas as pd
```


```python
# 创建畸形 df 
v = [ [53, 64, 72, 82], [ '女', '男', '男', '女' ] ] 
i = [ '年龄', '性别' ] 
c = [ '1 号', '2 号', '3 号', '4 号' ] 
df = pd.DataFrame( v, index=i, columns=c ) 
df 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1 号</th>
      <th>2 号</th>
      <th>3 号</th>
      <th>4 号</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>年龄</th>
      <td>53</td>
      <td>64</td>
      <td>72</td>
      <td>82</td>
    </tr>
    <tr>
      <th>性别</th>
      <td>女</td>
      <td>男</td>
      <td>男</td>
      <td>女</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 转置 
df = df.T 
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>性别</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>53</td>
      <td>女</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>64</td>
      <td>男</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>72</td>
      <td>男</td>
    </tr>
    <tr>
      <th>4 号</th>
      <td>82</td>
      <td>女</td>
    </tr>
  </tbody>
</table>
</div>



### 对象的翻转

紧接上面的例子，对Pandas对象进行左右翻转与上下翻转。


```python
# 左右翻转 
df = df.iloc[ : , : : -1 ] 
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>性别</th>
      <th>年龄</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>女</td>
      <td>53</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>男</td>
      <td>64</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>男</td>
      <td>72</td>
    </tr>
    <tr>
      <th>4 号</th>
      <td>女</td>
      <td>82</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 上下翻转 
df = df.iloc[ : : -1 , : ] 
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>性别</th>
      <th>年龄</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4 号</th>
      <td>女</td>
      <td>82</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>男</td>
      <td>72</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>男</td>
      <td>64</td>
    </tr>
    <tr>
      <th>1 号</th>
      <td>女</td>
      <td>53</td>
    </tr>
  </tbody>
</table>
</div>



### 对象的重塑

考虑到对象是含有行列标签的，`.reshape()`已经不再适用，因此对象的重塑没有那么灵活。但可以做到将 sr 并入 df，也可以将 df 割出 sr。


```python
# 数组法创建 sr 
i = [ '1 号', '2 号', '3 号', '4 号' ] 
v1 = [ 10, 20, 30, 40 ] 
v2 = [ '女', '男', '男', '女' ] 
v3 = [ 1, 2, 3, 4 ] 
sr1 = pd.Series( v1, index=i ) 
sr2 = pd.Series( v2, index=i ) 
sr3 = pd.Series( v3, index=i ) 
sr1, sr2, sr3 

```




    (1 号    10
     2 号    20
     3 号    30
     4 号    40
     dtype: int64,
     1 号    女
     2 号    男
     3 号    男
     4 号    女
     dtype: object,
     1 号    1
     2 号    2
     3 号    3
     4 号    4
     dtype: int64)




```python
# 字典法创建 df 
df = pd.DataFrame( { '年龄':sr1, '性别':sr2 } ) 
df 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>性别</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>10</td>
      <td>女</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>20</td>
      <td>男</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>30</td>
      <td>男</td>
    </tr>
    <tr>
      <th>4 号</th>
      <td>40</td>
      <td>女</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 把 sr 并入 df 中 
df['牌照'] = sr3 
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>性别</th>
      <th>牌照</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>10</td>
      <td>女</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>20</td>
      <td>男</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>30</td>
      <td>男</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4 号</th>
      <td>40</td>
      <td>女</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 把 df['年龄']分离成 sr4 
sr4 = df['年龄'] 
sr4
```




    1 号    10
    2 号    20
    3 号    30
    4 号    40
    Name: 年龄, dtype: int64



### 对象的拼接

Pandas中有一个pd.concat()函数，与np.concatenate()函数语法类似。

#### 一维对象的合并


```python
import pandas as pd
```


```python
# 创建 sr1 和 sr2 
v1 = [10, 20, 30, 40] 
v2 = [40, 50, 60] 
k1 = [ '1 号', '2 号', '3 号', '4 号' ] 
k2 = [ '4 号', '5 号', '6 号' ] 
sr1 = pd.Series( v1, index= k1 ) 
sr2 = pd.Series( v2, index= k2 ) 
sr1, sr2
```




    (1 号    10
     2 号    20
     3 号    30
     4 号    40
     dtype: int64,
     4 号    40
     5 号    50
     6 号    60
     dtype: int64)




```python
# 合并 
pd.concat( [sr1, sr2] ) 
```




    1 号    10
    2 号    20
    3 号    30
    4 号    40
    4 号    40
    5 号    50
    6 号    60
    dtype: int64



值得注意的是，上面输出的键中出现了两个“4 号”，这是因为 Pandas 对象的属性，放弃了集合与字典索引中“不可重复”的特性，实际中，这可以拓展大数据分析与处理的应用场景。那么，如何保证索引是不重复的呢？对对象的属性 .index 或 .columns 使用 .is_unique 即可检查，返回 True表示行或列不重复，False 表示有重复。

#### 一维对象与二维对象的合并

一维对象与二维对象的合并，即可理解为：给二维对象加上一列或者一行。因此，不必使用pd.concat()函数，只需要借助千问”二维对象的索引“语法。



```python
# 首先，创建好二维对象

import pandas as pd

# 创建 sr1 与 sr2 
v1 = [ 10, 20, 30] 
v2 = [ '女', '男', '男'] 
sr1 = pd.Series( v1, index=[ '1 号', '2 号', '3 号'] ) 
sr2 = pd.Series( v2, index=[ '1 号', '2 号', '3 号'] ) 
sr1, sr2
```




    (1 号    10
     2 号    20
     3 号    30
     dtype: int64,
     1 号    女
     2 号    男
     3 号    男
     dtype: object)




```python
# 创建 df 
df = pd.DataFrame( { '年龄':sr1, '性别':sr2 } ) 
df 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>性别</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>10</td>
      <td>女</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>20</td>
      <td>男</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>30</td>
      <td>男</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 其次，为二维对象加上一列特征，可以是列表、数组、张量或一维对象。
# 加上一列 
df['牌照'] = [1, 2, 3] 
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>性别</th>
      <th>牌照</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>10</td>
      <td>女</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>20</td>
      <td>男</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>30</td>
      <td>男</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 最后，为二维对象加上一行个体，可以是列表、数组、张量或一维对象。
# 加上一行 
df.loc['4 号'] = [40, '女', 4] 
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>性别</th>
      <th>牌照</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>10</td>
      <td>女</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>20</td>
      <td>男</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>30</td>
      <td>男</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4 号</th>
      <td>40</td>
      <td>女</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



#### 二维对象的合并

二维对象合并仍然用 pd.concat()函数，不过其多了一个 axis 参数。 


```python
# 设定 df1、df2、df3 
v1 = [ [10, '女'], [20, '男'], [30, '男'], [40, '女'] ] 
v2 = [ [1, '是'], [2, '是'], [3, '是'], [4, '否'] ] 
v3 = [ [50, '男', 5, '是'], [60, '女', 6, '是'] ] 
i1 = [ '1 号', '2 号', '3 号', '4 号' ] 
i2 = [ '1 号', '2 号', '3 号', '4 号' ] 
i3 = [ '5 号', '6 号' ] 
c1 = [ '年龄', '性别' ] 
c2 = [ '牌照', 'ikun' ] 
c3 = [ '年龄', '性别', '牌照', 'ikun' ] 
df1 = pd.DataFrame( v1, index=i1, columns=c1 ) 
df2 = pd.DataFrame( v2, index=i2, columns=c2 ) 
df3 = pd.DataFrame( v3, index=i3, columns=c3 ) 
df1, df2, df3 
```




    (     年龄 性别
     1 号  10  女
     2 号  20  男
     3 号  30  男
     4 号  40  女,
          牌照 ikun
     1 号   1    是
     2 号   2    是
     3 号   3    是
     4 号   4    否,
          年龄 性别  牌照 ikun
     5 号  50  男   5    是
     6 号  60  女   6    是)




```python
# 合并列对象（添加列特征） 
df = pd.concat( [df1,df2], axis=1 ) 
df 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>性别</th>
      <th>牌照</th>
      <th>ikun</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>10</td>
      <td>女</td>
      <td>1</td>
      <td>是</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>20</td>
      <td>男</td>
      <td>2</td>
      <td>是</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>30</td>
      <td>男</td>
      <td>3</td>
      <td>是</td>
    </tr>
    <tr>
      <th>4 号</th>
      <td>40</td>
      <td>女</td>
      <td>4</td>
      <td>否</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 合并行对象（添加行个体） 
df = pd.concat( [df,df3] ) 
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>性别</th>
      <th>牌照</th>
      <th>ikun</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>10</td>
      <td>女</td>
      <td>1</td>
      <td>是</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>20</td>
      <td>男</td>
      <td>2</td>
      <td>是</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>30</td>
      <td>男</td>
      <td>3</td>
      <td>是</td>
    </tr>
    <tr>
      <th>4 号</th>
      <td>40</td>
      <td>女</td>
      <td>4</td>
      <td>否</td>
    </tr>
    <tr>
      <th>5 号</th>
      <td>50</td>
      <td>男</td>
      <td>5</td>
      <td>是</td>
    </tr>
    <tr>
      <th>6 号</th>
      <td>60</td>
      <td>女</td>
      <td>6</td>
      <td>是</td>
    </tr>
  </tbody>
</table>
</div>



## 对象的运算

### 对象与系数之间的运算

下列演示代码中，左侧为一维对象，右侧为二维对象。


```python
import pandas as pd
```


```python
# 创建 sr 
sr = pd.Series( [ 53, 64, 72 ] , index=['1 号', '2 号', '3 号'] ) 
sr
```




    1 号    53
    2 号    64
    3 号    72
    dtype: int64




```python
# 创建 df 
v = [ [53, '女'], [64, '男'], [72, '男'] ] 
df = pd.DataFrame( v, index=[ '1 号', '2 号', '3 号' ], columns=[ '年龄', '性别' ] ) 
df 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>性别</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>53</td>
      <td>女</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>64</td>
      <td>男</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>72</td>
      <td>男</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 对一维对象进行操作
sr = sr + 10 
sr 
```




    1 号    63
    2 号    74
    3 号    82
    dtype: int64




```python
# 对二维对象进行操作
df['年龄'] = df['年龄'] + 10 
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>性别</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>63</td>
      <td>女</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>74</td>
      <td>男</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>82</td>
      <td>男</td>
    </tr>
  </tbody>
</table>
</div>




```python
sr = sr * 10 
sr 
```




    1 号    630
    2 号    740
    3 号    820
    dtype: int64




```python
df['年龄'] = df['年龄'] * 10 
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>性别</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>630</td>
      <td>女</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>740</td>
      <td>男</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>820</td>
      <td>男</td>
    </tr>
  </tbody>
</table>
</div>




```python
sr = sr ** 2 
sr
```




    1 号    396900
    2 号    547600
    3 号    672400
    dtype: int64




```python
df['年龄'] = df['年龄'] ** 2 
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>性别</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>396900</td>
      <td>女</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>547600</td>
      <td>男</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>672400</td>
      <td>男</td>
    </tr>
  </tbody>
</table>
</div>



### 对象与对象之间的运算

对象做运算，必须保证其都是数字型对象，两个对象之间的维度可以不同。

#### 一维对象之间的运算


```python
import pandas as pd
```


```python
# 创建 sr1 
v1 = [10, 20, 30, 40] 
k1 = [ '1 号', '2 号', '3 号', '4 号' ] 
sr1 = pd.Series( v1, index= k1 ) 
sr1
```




    1 号    10
    2 号    20
    3 号    30
    4 号    40
    dtype: int64




```python
# 创建 sr2 
v2 = [1, 2, 3 ] 
k2 = [ '1 号', '2 号', '3 号' ] 
sr2 = pd.Series( v2, index= k2 ) 
sr2
```




    1 号    1
    2 号    2
    3 号    3
    dtype: int64




```python
sr1 + sr2 # 加法 
```




    1 号    11.0
    2 号    22.0
    3 号    33.0
    4 号     NaN
    dtype: float64




```python
sr1 - sr2 # 减法 
```




    1 号     9.0
    2 号    18.0
    3 号    27.0
    4 号     NaN
    dtype: float64




```python
sr1 * sr2 # 乘法 
```




    1 号    10.0
    2 号    40.0
    3 号    90.0
    4 号     NaN
    dtype: float64




```python
sr1 / sr2 # 除法 
```




    1 号    10.0
    2 号    10.0
    3 号    10.0
    4 号     NaN
    dtype: float64




```python
sr1 ** sr2 # 幂方 
```




    1 号       10.0
    2 号      400.0
    3 号    27000.0
    4 号        NaN
    dtype: float64



#### 二维对象之间的运算


```python
import pandas as pd 
```


```python
# 设定 df1 和 df2 
v1 = [ [10, '女'], [20, '男'], [30, '男'], [40, '女'] ] 
v2 = [ 1, 2 ,3, 6 ] 
i1 = [ '1 号', '2 号', '3 号', '4 号' ]; c1 = [ '年龄', '性别' ] 
i2 = [ '1 号', '2 号', '3 号', '6 号' ]; c2 = [ '牌照' ] 
df1 = pd.DataFrame( v1, index=i1, columns=c1 ) 
df2 = pd.DataFrame( v2, index=i2, columns=c2 ) 
df1, df2
```




    (     年龄 性别
     1 号  10  女
     2 号  20  男
     3 号  30  男
     4 号  40  女,
          牌照
     1 号   1
     2 号   2
     3 号   3
     6 号   6)




```python
# 加法 
df1['加法'] = df1['年龄'] + df2['牌照'] 
df1 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>性别</th>
      <th>加法</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>10</td>
      <td>女</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>20</td>
      <td>男</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>30</td>
      <td>男</td>
      <td>33.0</td>
    </tr>
    <tr>
      <th>4 号</th>
      <td>40</td>
      <td>女</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 减法、乘法、除法、幂方 
df1['减法'] = df1['年龄'] - df2['牌照'] 
df1['乘法'] = df1['年龄'] * df2['牌照'] 
df1['除法'] = df1['年龄'] / df2['牌照'] 
df1['幂方'] = df1['年龄'] ** df2['牌照'] 
df1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>性别</th>
      <th>加法</th>
      <th>减法</th>
      <th>乘法</th>
      <th>除法</th>
      <th>幂方</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>10</td>
      <td>女</td>
      <td>11.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>20</td>
      <td>男</td>
      <td>22.0</td>
      <td>18.0</td>
      <td>40.0</td>
      <td>10.0</td>
      <td>400.0</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>30</td>
      <td>男</td>
      <td>33.0</td>
      <td>27.0</td>
      <td>90.0</td>
      <td>10.0</td>
      <td>27000.0</td>
    </tr>
    <tr>
      <th>4 号</th>
      <td>40</td>
      <td>女</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



本章的最后，补充两点内容，读者可自行尝试。 
- 使用 np.abs()、np.cos()、np.exp()、np.log() 等数学函数时，会保留索引； 
- Pandas 中仍然存在布尔型对象，用法与 NumPy 无异，会保留索引。


```python
import pandas as pd
import numpy as np

# 创建一个带有自定义索引的 Series
data = pd.Series([-1.5, 0, 2.5, -3.0], index=['a', 'b', 'c', 'd'])

print("原始数据 (Original Data):")
print(data)
print("-" * 30)

# 验证 np.abs() - 绝对值
print("使用 np.abs() 后 (索引保留):")
print(np.abs(data))
print("-" * 30)

# 验证 np.exp() - 指数
print("使用 np.exp() 后 (索引保留):")
print(np.exp(data))
print("-" * 30)

# 验证 np.log() - 对数 (注意：log(0) 和 log(负数) 会产生 inf 或 nan，但不影响索引保留)
print("使用 np.log() 后 (索引保留):")
# 为了演示 log，我们换一组正数数据
data_pos = pd.Series([1, 2.718, 10], index=['x', 'y', 'z'])
print(np.log(data_pos))
```

    原始数据 (Original Data):
    a   -1.5
    b    0.0
    c    2.5
    d   -3.0
    dtype: float64
    ------------------------------
    使用 np.abs() 后 (索引保留):
    a    1.5
    b    0.0
    c    2.5
    d    3.0
    dtype: float64
    ------------------------------
    使用 np.exp() 后 (索引保留):
    a     0.223130
    b     1.000000
    c    12.182494
    d     0.049787
    dtype: float64
    ------------------------------
    使用 np.log() 后 (索引保留):
    x    0.000000
    y    0.999896
    z    2.302585
    dtype: float64
    


```python
import pandas as pd
import numpy as np

# 创建一个 Series
s = pd.Series([10, -5, 0, 20, -15], index=['A', 'B', 'C', 'D', 'E'])

print("原始数据:")
print(s)
print("-" * 30)

# 1. 生成布尔对象 (Boolean Object)
# 用法与 NumPy 无异：直接比较
bool_mask = s > 0 
print("布尔对象 (s > 0):")
print(bool_mask) 
# 注意看：结果是一个 Series，且索引 ['A', 'B', ...] 被完整保留了
print("-" * 30)

# 2. 布尔运算 (Logical Operations)
# 用法与 NumPy 无异：使用 & (与), | (或), ~ (非)
# 比如：找出 大于0 且 小于15 的数
complex_mask = (s > 0) & (s < 15)
print("复杂布尔运算 ((s > 0) & (s < 15)):")
print(complex_mask)
print("-" * 30)

# 3. 布尔索引 (Boolean Indexing)
# 使用上面的布尔对象来筛选数据
filtered_data = s[bool_mask]
print("筛选后的数据 (s[s > 0]):")
print(filtered_data)
# 注意看：筛选出来的结果，其索引依然是原始的 'A' 和 'D'，而不是重置为 0, 1
```

    原始数据:
    A    10
    B    -5
    C     0
    D    20
    E   -15
    dtype: int64
    ------------------------------
    布尔对象 (s > 0):
    A     True
    B    False
    C    False
    D     True
    E    False
    dtype: bool
    ------------------------------
    复杂布尔运算 ((s > 0) & (s < 15)):
    A     True
    B    False
    C    False
    D    False
    E    False
    dtype: bool
    ------------------------------
    筛选后的数据 (s[s > 0]):
    A    10
    D    20
    dtype: int64
    

## 对象的缺失值

### 发现缺失值

发现缺失值使用`.isnull()`方法。


```python
import pandas as pd 
```


```python
# 创建 sr 
v = [ 53, None, 72, 82 ] 
k = ['1 号', '2 号', '3 号', '4 号'] 
sr = pd.Series( v, index=k ) 
sr
```




    1 号    53.0
    2 号     NaN
    3 号    72.0
    4 号    82.0
    dtype: float64




```python
# 创建 df 
v = [ [None, 1], [64, None], [72, 3], [82, 4] ] 
i = [ '1 号', '2 号', '3 号', '4 号' ] 
c = [ '年龄', '牌照' ] 
df = pd.DataFrame( v, index=i, columns=c ) 
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>牌照</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>64.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>72.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4 号</th>
      <td>82.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 发现 sr 的缺失值 
sr.isnull()
```




    1 号    False
    2 号     True
    3 号    False
    4 号    False
    dtype: bool




```python
# 发现 df 的缺失值 
df.isnull()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>牌照</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4 号</th>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 发现 df 的非缺失值 
~df.isnull()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>牌照</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4 号</th>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.notnull()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>牌照</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4 号</th>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



除了`.isnull()` 方法，还有一个与之相反的 `.notnull()` 方法，但不如在开头加一个非号`“~”`即可。 

### 剔除缺失值

剔除缺失值使用 .dropna() 方法，一维对象很好剔除；二维对象比较复杂，要么单独剔除 df 中含有缺失值的行，要么剔除 df 中含有缺失值的列。


```python
import pandas as pd 

```


```python
# 创建 sr 
v = [ 53, None, 72, 82 ] 
k = ['1 号', '2 号', '3 号', '4 号'] 
sr = pd.Series( v, index=k ) 
sr
```




    1 号    53.0
    2 号     NaN
    3 号    72.0
    4 号    82.0
    dtype: float64




```python
# 创建 df 
v = [ [None, None], [64, None], [72, 3], [82, 4] ] 
i = [ '1 号', '2 号', '3 号', '4 号' ] 
c = [ '年龄', '牌照' ] 
df = pd.DataFrame( v, index=i, columns=c ) 
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>牌照</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>64.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>72.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4 号</th>
      <td>82.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 剔除 sr 的缺失值 
sr.dropna() 

```




    1 号    53.0
    3 号    72.0
    4 号    82.0
    dtype: float64




```python
# 剔除 df 含缺失值的行 
df.dropna() 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>牌照</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3 号</th>
      <td>72.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4 号</th>
      <td>82.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



把含有 NaN 的行剔除掉了，你也可以通过 `df.dropna(axis='columns')`的方式剔除列。但请警惕，**一般都是剔除行**，只因大数据中行是个体，列是特征。 

有些同学认为，只要某行含有一个 NaN 就剔除该个体太过残忍，我们可以设定一个参数，只有当该行全部是 NaN，才剔除该列特征。 


```python
# 剔除 df 全是 NaN 的个体 
df.dropna(how='all') 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>牌照</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2 号</th>
      <td>64.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>72.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4 号</th>
      <td>82.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



### 填补缺失值

填充缺失值使用 `.fillna()` 方法，实际的数据填充没有统一的方法，很灵活。 

#### 一维对象


```python
import pandas as pd
```


```python
# 创建 sr 
v = [ 53, None, 72, 82 ] 
sr = pd.Series( v, index=['1 号', '2 号', '3 号', '4 号'] ) 
sr
```




    1 号    53.0
    2 号     NaN
    3 号    72.0
    4 号    82.0
    dtype: float64




```python
# 用常数（0）填充 
sr.fillna(0)
```




    1 号    53.0
    2 号     0.0
    3 号    72.0
    4 号    82.0
    dtype: float64




```python
# 用常数（均值）填充 
import numpy as np 
sr.fillna(np.mean(sr)) 

```




    1 号    53.0
    2 号    69.0
    3 号    72.0
    4 号    82.0
    dtype: float64




```python
# 用前值填充 
sr.fillna(method='ffill') 

```

    /tmp/ipykernel_55/89089568.py:2: FutureWarning: Series.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
      sr.fillna(method='ffill')
    




    1 号    53.0
    2 号    53.0
    3 号    72.0
    4 号    82.0
    dtype: float64




```python
# 用后值填充 
sr.fillna(method='bfill') 

```

    /tmp/ipykernel_55/568087098.py:2: FutureWarning: Series.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
      sr.fillna(method='bfill')
    




    1 号    53.0
    2 号    72.0
    3 号    72.0
    4 号    82.0
    dtype: float64



#### 二维对象


```python
import pandas as pd 
```


```python
# 设定 df 
v = [ [None, None], [64, None], [72, 3], [82, 4] ] 
i = [ '1 号', '2 号', '3 号', '4 号' ]; c = [ '年龄', '牌照' ] 
df = pd.DataFrame( v, index=i, columns=c ) 
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>牌照</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>64.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>72.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4 号</th>
      <td>82.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 用常数（0）填充 
df.fillna(0) 

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>牌照</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>64.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>72.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4 号</th>
      <td>82.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 用常数（均值）填充 
import numpy as np 
df.fillna( np.mean(df) ) 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>牌照</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>45.0</td>
      <td>45.0</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>64.0</td>
      <td>45.0</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>72.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4 号</th>
      <td>82.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



似乎此处的结果并不像教程中所示，`np.mean(df)`计算了所有数值的平均值并以此填充了所有缺失字段，而没有按照每一列计算均值并填充。

使用`df.mean()`反而能实现我们预期的效果。


```python
# 用常数（均值）填充 
df.fillna( df.mean() ) 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>牌照</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>72.666667</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>64.000000</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>72.000000</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4 号</th>
      <td>82.000000</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 用前值填充 
df.ffill() 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>牌照</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>64.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>72.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4 号</th>
      <td>82.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 用后值填充
df.bfill()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>牌照</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 号</th>
      <td>64.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2 号</th>
      <td>64.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3 号</th>
      <td>72.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4 号</th>
      <td>82.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



## 导入Excel文件

### 创建Excel文件

导入信息的命令如示例所示。


```python
import pandas as pd
```


```python
import pandas as pd

# 读取 Excel 文件
df_1 = pd.read_excel('/kaggle/input/datasets/frechen026/jie-pandas/Data.xlsx')
df_2 = pd.read_excel('/kaggle/input/datasets/frechen026/jie-pandas/Data_1773733480020.xlsx')
df_3 = pd.read_excel('/kaggle/input/datasets/frechen026/jie-pandas/test1.xlsx')
df_4 = pd.read_excel('/kaggle/input/datasets/frechen026/jie-pandas/test2.xlsx')


# 保存为 CSV 文件
df_1.to_csv('/kaggle/working/Data1.csv', index=False, encoding='utf-8-sig')
df_2.to_csv('/kaggle/working/Data2.csv', index=False, encoding='utf-8-sig')
df_3.to_csv('/kaggle/working/test1.csv', index=False, encoding='utf-8-sig')
df_4.to_csv('/kaggle/working/test2.csv', index=False, encoding='utf-8-sig')
```


```python
# 导入 Pandas 对象 
df = pd.read_csv('/kaggle/working/Data1.csv', index_col=0) 
df 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>gender</th>
      <th>num</th>
      <th>kun</th>
    </tr>
    <tr>
      <th>Unnamed: 0</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1号</th>
      <td>10</td>
      <td>女</td>
      <td>1</td>
      <td>是</td>
    </tr>
    <tr>
      <th>2号</th>
      <td>20</td>
      <td>男</td>
      <td>2</td>
      <td>是</td>
    </tr>
    <tr>
      <th>3号</th>
      <td>30</td>
      <td>男</td>
      <td>3</td>
      <td>是</td>
    </tr>
    <tr>
      <th>4号</th>
      <td>40</td>
      <td>女</td>
      <td>4</td>
      <td>否</td>
    </tr>
    <tr>
      <th>5号</th>
      <td>50</td>
      <td>男</td>
      <td>5</td>
      <td>是</td>
    </tr>
    <tr>
      <th>6号</th>
      <td>60</td>
      <td>女</td>
      <td>6</td>
      <td>是</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 提取纯数组 
arr = df.values 
arr 
```




    array([[10, '女', 1, '是'],
           [20, '男', 2, '是'],
           [30, '男', 3, '是'],
           [40, '女', 4, '否'],
           [50, '男', 5, '是'],
           [60, '女', 6, '是']], dtype=object)



## 数据分析

### 导入信息

首先，准备好Excel数据表格。

发现没有行标签，因此需要在最左侧快速填充一列顺序数字。

接着，按照第六章的方法，将其另存为 CSV 文件，并放置项目文件夹中。

最后，导入该表格至 Jupyter 中。


```python
import pandas as pd 
```


```python
# 导入 Pandas 对象 
pd.read_csv('/kaggle/working/test2.csv', index_col=0) 

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>发现时间</th>
      <th>发现数量</th>
      <th>观测方法</th>
      <th>行星质量</th>
      <th>距地距离</th>
      <th>轨道周期</th>
    </tr>
    <tr>
      <th>Unnamed: 0</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1989</td>
      <td>1</td>
      <td>径向速度</td>
      <td>11.680</td>
      <td>40.57</td>
      <td>83.888000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1992</td>
      <td>3</td>
      <td>脉冲星计时</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>25.262000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1992</td>
      <td>3</td>
      <td>脉冲星计时</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>66.541900</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1994</td>
      <td>3</td>
      <td>脉冲星计时</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>98.211400</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1995</td>
      <td>1</td>
      <td>径向速度</td>
      <td>0.472</td>
      <td>15.36</td>
      <td>4.230785</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1030</th>
      <td>2014</td>
      <td>1</td>
      <td>凌日</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.465020</td>
    </tr>
    <tr>
      <th>1031</th>
      <td>2014</td>
      <td>1</td>
      <td>凌日</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>68.958400</td>
    </tr>
    <tr>
      <th>1032</th>
      <td>2014</td>
      <td>1</td>
      <td>凌日</td>
      <td>NaN</td>
      <td>1056.00</td>
      <td>1.720861</td>
    </tr>
    <tr>
      <th>1033</th>
      <td>2014</td>
      <td>1</td>
      <td>凌日</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>66.262000</td>
    </tr>
    <tr>
      <th>1034</th>
      <td>2014</td>
      <td>1</td>
      <td>凌日</td>
      <td>NaN</td>
      <td>470.00</td>
      <td>0.925542</td>
    </tr>
  </tbody>
</table>
<p>1035 rows × 6 columns</p>
</div>



### 聚合方法

可在输出df时，对其使用`.head()`方法，使其仅仅输出前5行。


```python
import pandas as pd 

```


```python
# 导入 Pandas 对象 
df = pd.read_csv('/kaggle/working/test2.csv', index_col=0) 
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>发现时间</th>
      <th>发现数量</th>
      <th>观测方法</th>
      <th>行星质量</th>
      <th>距地距离</th>
      <th>轨道周期</th>
    </tr>
    <tr>
      <th>Unnamed: 0</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1989</td>
      <td>1</td>
      <td>径向速度</td>
      <td>11.680</td>
      <td>40.57</td>
      <td>83.888000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1992</td>
      <td>3</td>
      <td>脉冲星计时</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>25.262000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1992</td>
      <td>3</td>
      <td>脉冲星计时</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>66.541900</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1994</td>
      <td>3</td>
      <td>脉冲星计时</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>98.211400</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1995</td>
      <td>1</td>
      <td>径向速度</td>
      <td>0.472</td>
      <td>15.36</td>
      <td>4.230785</td>
    </tr>
  </tbody>
</table>
</div>



NumPy 中所有的聚合函数对 Pandas 对象均适用。此外，Pandas 将这些函数变为对象的方法，这样，不导入 NumPy 也可使用。


```python
# 最大值函数 np.max( ) 
df.max()
```




    发现时间        2014
    发现数量           7
    观测方法      轨道亮度调制
    行星质量        25.0
    距地距离      8500.0
    轨道周期    730000.0
    dtype: object




```python
# 最小值函数 np.min( ) 
df.min()
```




    发现时间        1989
    发现数量           1
    观测方法          凌日
    行星质量      0.0036
    距地距离        1.35
    轨道周期    0.090706
    dtype: object




```python
# 均值函数 np.mean( ) 
df.mean(numeric_only=True)
# 在 mean() 函数中添加参数 numeric_only=True。这会告诉 Pandas 自动忽略所有非数值类型的列，只对数字列计算均值。

# 可能是版本原因我这里需要添加该参数。
```




    发现时间    2009.070531
    发现数量       1.785507
    行星质量       2.638161
    距地距离     264.069282
    轨道周期    2002.917596
    dtype: float64




```python
# 求和函数 np.sum( ) 
df.sum(numeric_only=True) 

```




    发现时间    2.079388e+06
    发现数量    1.848000e+03
    行星质量    1.353376e+03
    距地距离    2.133680e+05
    轨道周期    1.986894e+06
    dtype: float64



### 描述方法

在数据分析中，用以上方法挨个查看未免太过麻烦，可以使用 `.describe()` 方法直接查看所有聚合函数的信息。 



```python
# 导入 Pandas 对象 
df = pd.read_csv('/kaggle/working/test2.csv', index_col=0) 
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>发现时间</th>
      <th>发现数量</th>
      <th>观测方法</th>
      <th>行星质量</th>
      <th>距地距离</th>
      <th>轨道周期</th>
    </tr>
    <tr>
      <th>Unnamed: 0</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1989</td>
      <td>1</td>
      <td>径向速度</td>
      <td>11.680</td>
      <td>40.57</td>
      <td>83.888000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1992</td>
      <td>3</td>
      <td>脉冲星计时</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>25.262000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1992</td>
      <td>3</td>
      <td>脉冲星计时</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>66.541900</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1994</td>
      <td>3</td>
      <td>脉冲星计时</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>98.211400</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1995</td>
      <td>1</td>
      <td>径向速度</td>
      <td>0.472</td>
      <td>15.36</td>
      <td>4.230785</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe() 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>发现时间</th>
      <th>发现数量</th>
      <th>行星质量</th>
      <th>距地距离</th>
      <th>轨道周期</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1035.000000</td>
      <td>1035.000000</td>
      <td>513.000000</td>
      <td>808.000000</td>
      <td>992.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2009.070531</td>
      <td>1.785507</td>
      <td>2.638161</td>
      <td>264.069282</td>
      <td>2002.917596</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.972567</td>
      <td>1.240976</td>
      <td>3.818617</td>
      <td>733.116493</td>
      <td>26014.728304</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1989.000000</td>
      <td>1.000000</td>
      <td>0.003600</td>
      <td>1.350000</td>
      <td>0.090706</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2007.000000</td>
      <td>1.000000</td>
      <td>0.229000</td>
      <td>32.560000</td>
      <td>5.442540</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2010.000000</td>
      <td>1.000000</td>
      <td>1.260000</td>
      <td>55.250000</td>
      <td>39.979500</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2012.000000</td>
      <td>2.000000</td>
      <td>3.040000</td>
      <td>178.500000</td>
      <td>526.005000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2014.000000</td>
      <td>7.000000</td>
      <td>25.000000</td>
      <td>8500.000000</td>
      <td>730000.000000</td>
    </tr>
  </tbody>
</table>
</div>



### 数据透视

#### 两个特征内的数据透视

数据透视，对数据分析来讲十分重要。 

现以泰坦尼克号的生还数据为例，以“是否生还”特征为考察的核心（或者说是神经网络的输出），研究其它特征（输入）与之的关系，如示例所示。


```python
import pandas as pd 
```


```python
# 导入 Pandas 对象 
df = pd.read_csv('/kaggle/working/test1.csv', index_col=0) 
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>性别</th>
      <th>年龄</th>
      <th>船舱等级</th>
      <th>费用</th>
      <th>是否生还</th>
    </tr>
    <tr>
      <th>Unnamed: 0</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>男</td>
      <td>22.0</td>
      <td>三等</td>
      <td>7.2500</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>女</td>
      <td>38.0</td>
      <td>一等</td>
      <td>71.2833</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>女</td>
      <td>26.0</td>
      <td>三等</td>
      <td>7.9250</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>女</td>
      <td>35.0</td>
      <td>一等</td>
      <td>53.1000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>男</td>
      <td>35.0</td>
      <td>三等</td>
      <td>8.0500</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 一个特征：性别 
df.pivot_table('是否生还', index='性别') 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>是否生还</th>
    </tr>
    <tr>
      <th>性别</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>女</th>
      <td>0.742038</td>
    </tr>
    <tr>
      <th>男</th>
      <td>0.188908</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 两个特征：性别、船舱等级 
df.pivot_table('是否生还', index='性别', columns='船舱等级') 

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>船舱等级</th>
      <th>一等</th>
      <th>三等</th>
      <th>二等</th>
    </tr>
    <tr>
      <th>性别</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>女</th>
      <td>0.968085</td>
      <td>0.500000</td>
      <td>0.921053</td>
    </tr>
    <tr>
      <th>男</th>
      <td>0.368852</td>
      <td>0.135447</td>
      <td>0.157407</td>
    </tr>
  </tbody>
</table>
</div>



在上述示例中，数据透视表中的数值默认是输出特征“是否生还”的均值（mean），行标签和列标签变成了其它的输入特征。可以看出，女性整体的生还概率是 74%，男性整体的生还概率为 18%。

按照船舱等级的下降，生还率逐步下降。 值得注意的是，pivot_table() 方法有一个很重要的参数：aggfunc，其默认值是'mean'，除此以外，所有的聚合函数'max'、'min'、'sum'、'count' 均可使用。

显然，对于这里的“是否生还”来说，'mean'就是最好的选择，其刚好为概率。 

#### 多个特征的数据透视

前面的示例只涉及到两个特征，有时需要考察更多特征与输出特征的关系。 

这里，将年龄和费用都加进去。但是，这两个特征的数值很分散，之前的性别和船舱等级都可以按照类别分，现在已经不能再按类别分了。因此，需要涉及****到数据透视表配套的两个重要函数：pd.cut()与 pd.qcut()。


```python
# 导入 Pandas 对象 
df = pd.read_csv('/kaggle/working/test1.csv', index_col=0) 
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>性别</th>
      <th>年龄</th>
      <th>船舱等级</th>
      <th>费用</th>
      <th>是否生还</th>
    </tr>
    <tr>
      <th>Unnamed: 0</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>男</td>
      <td>22.0</td>
      <td>三等</td>
      <td>7.2500</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>女</td>
      <td>38.0</td>
      <td>一等</td>
      <td>71.2833</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>女</td>
      <td>26.0</td>
      <td>三等</td>
      <td>7.9250</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>女</td>
      <td>35.0</td>
      <td>一等</td>
      <td>53.1000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>男</td>
      <td>35.0</td>
      <td>三等</td>
      <td>8.0500</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 三个特征：性别、船舱等级、年龄 
age = pd.cut( df['年龄'], [0,18,120] ) # 以 18 岁为分水岭 
df.pivot_table('是否生还', index= ['性别', age], columns='船舱等级') 
```

    /tmp/ipykernel_55/1361481379.py:3: FutureWarning: The default value of observed=False is deprecated and will change to observed=True in a future version of pandas. Specify observed=False to silence this warning and retain the current behavior
      df.pivot_table('是否生还', index= ['性别', age], columns='船舱等级')
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>船舱等级</th>
      <th>一等</th>
      <th>三等</th>
      <th>二等</th>
    </tr>
    <tr>
      <th>性别</th>
      <th>年龄</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">女</th>
      <th>(0, 18]</th>
      <td>0.909091</td>
      <td>0.511628</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>(18, 120]</th>
      <td>0.972973</td>
      <td>0.423729</td>
      <td>0.900000</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">男</th>
      <th>(0, 18]</th>
      <td>0.800000</td>
      <td>0.215686</td>
      <td>0.600000</td>
    </tr>
    <tr>
      <th>(18, 120]</th>
      <td>0.375000</td>
      <td>0.133663</td>
      <td>0.071429</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 四个特征：性别、船舱等级、年龄、费用 
fare = pd.qcut( df['费用'], 2 ) # 将费用自动分为两部分 
df.pivot_table('是否生还', index= ['船舱等级',fare], columns=['性别', age]) 

```

    /tmp/ipykernel_55/2785335320.py:3: FutureWarning: The default value of observed=False is deprecated and will change to observed=True in a future version of pandas. Specify observed=False to silence this warning and retain the current behavior
      df.pivot_table('是否生还', index= ['船舱等级',fare], columns=['性别', age])
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>性别</th>
      <th colspan="2" halign="left">女</th>
      <th colspan="2" halign="left">男</th>
    </tr>
    <tr>
      <th></th>
      <th>年龄</th>
      <th>(0, 18]</th>
      <th>(18, 120]</th>
      <th>(0, 18]</th>
      <th>(18, 120]</th>
    </tr>
    <tr>
      <th>船舱等级</th>
      <th>费用</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">一等</th>
      <th>(-0.001, 14.454]</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>(14.454, 512.329]</th>
      <td>0.909091</td>
      <td>0.972973</td>
      <td>0.800000</td>
      <td>0.391304</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">三等</th>
      <th>(-0.001, 14.454]</th>
      <td>0.714286</td>
      <td>0.444444</td>
      <td>0.260870</td>
      <td>0.125000</td>
    </tr>
    <tr>
      <th>(14.454, 512.329]</th>
      <td>0.318182</td>
      <td>0.391304</td>
      <td>0.178571</td>
      <td>0.192308</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">二等</th>
      <th>(-0.001, 14.454]</th>
      <td>1.000000</td>
      <td>0.880000</td>
      <td>0.000000</td>
      <td>0.098039</td>
    </tr>
    <tr>
      <th>(14.454, 512.329]</th>
      <td>1.000000</td>
      <td>0.914286</td>
      <td>0.818182</td>
      <td>0.030303</td>
    </tr>
  </tbody>
</table>
</div>



- pd.cut()函数需要手动设置分割点，也可以设置为`[0,18,60,120]`。
- pd.qcut()函数可自动分割，如果需要分割成 3 部分，可以设置为 3。 
