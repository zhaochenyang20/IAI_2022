---
title: 统计机器学习
date: 2022-05-30 16:40:51
tags: 炼丹
description: 统计机器学习方法，包括朴素贝叶斯方法、决策树、支持向量机（SVM）的原理
keywords: 机器学习入门
categories: 机器学习
katex: true
cover: /img/chen8.jpg
top_img: /img/chen6.jpg
password: IAI
---

# 朴素贝叶斯法（naive Bayes）

{% note red no-icon %}

 {% label 朴素贝叶斯(naive-Bayes) red %}朴素贝叶斯(naive Bayes) 法是基于**贝叶斯定理**与**特征条件独立假设**的分类方法.对于给定的训练数据集，首先基于*特征条件独立假设*学习**输入/输出**的联合概率分布；然后基于此模型，对给定的输入x，利用贝叶斯定理求出{% label 后验概率 red %}最大的输出y.朴素贝叶斯法实现简单，学习与预测的效率都很高，是一种常用的方法.

{% endnote %}

## 学习与分类

### 基本原理

设输入空间 $\mathcal{X} \subseteq \mathbf{R}^{n}$ 为 n 维向量的集合, 输出空间为类标记集合 $\mathcal{Y}=\left\{c_{1}\right., \left.c_{2}, \cdots, c_{K}\right\}$. 输入为特征向量 $x \in \mathcal{X}$, 输出为类标记 (class label) $y \in \mathcal{Y}$ . X 是定义 在输入空间 $\mathcal{X}$ 上的随机向量, Y 是定义在输出空间 $\mathcal{Y}$ 上的随机变量. $P(X, Y)$ 是 X 和 Y 的联合概率分布. 训练数据集
$$
T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\}
$$
**由 P(X, Y) 独立同分布产生.**

朴素贝叶斯法通过训练数据集学习联合概率分布 P(X, Y). 具体地, 学习以下先验概率分布及条件概率分布. 先验概率分布
$$
P\left(Y=c_{k}\right), \quad k=1,2, \cdots, K
$$
条件概率分布
$$
P\left(X=x \mid Y=c_{k}\right)=P\left(X^{(1)}=x^{(1)}, \cdots, X^{(n)}=x^{(n)} \mid Y=c_{k}\right), \quad k=1,2, \cdots, K
$$
于是学习到联合概率分布 P(X, Y).

条件概率分布 $P\left(X=x \mid Y=c_{k}\right)$ 有指数级数量的参数, 其估计实际是不可行 的. 

朴素贝叶斯法对条件概率分布作了**条件独立性**的假设. 由于这是一个较强的假设, 朴素贝叶斯法也由此得名. 具体地, 条件独立性假设是
$$
\begin{aligned}\\P\left(X=x \mid Y=c_{k}\right) &=P\left(X^{(1)}=x^{(1)}, \cdots, X^{(n)}=x^{(n)} \mid Y=c_{k}\right) \\\\&=\prod_{j=1}^{n} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)\\\end{aligned}
$$
朴素贝叶斯法实际上学习到生成数据的机制, 所以属于生成模型. `条件独立假设等于是说用于分类的特征在类确定的条件下都是条件独立的`. 这一假设使朴素贝叶斯法变得简单, 但有时会牺牲一定的分类准确率.

朴素贝叶斯法分类时, 对给定的输入 x, 通过学习到的模型计算后验概率分布 $P\left(Y=c_{k} \mid X=x\right)$, 将后验概率最大的类作为 x 的类输出. 后验概率计算根据贝叶斯定理进行:
$$
P\left(Y=c_{k} \mid X=x\right)=\frac{P\left(X=x \mid Y=c_{k}\right) P\left(Y=c_{k}\right)}{\sum_{k} P\left(X=x \mid Y=c_{k}\right) P\left(Y=c_{k}\right)}
$$
再由独立性假设：
$$
P\left(Y=c_{k} \mid X=x\right)=\frac{P\left(Y=c_{k}\right) \prod_{j} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)}{\sum_{k} P\left(Y=c_{k}\right) \prod_{j} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)}, \quad k=1,2, \cdots, K
$$
这是朴素贝叶斯法分类的基本公式. 于是, 朴素贝叶斯分类器可表示为
$$
y=f(x)=\arg \max _{c_{k}} \frac{P\left(Y=c_{k}\right) \prod_{j} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)}{\sum_{k} P\left(Y=c_{k}\right) \prod_{j} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)}
$$
注意到对所有y，分母都是相同的，因此不用计算：
$$
y=\arg \max _{c_{k}} P\left(Y=c_{k}\right) \prod_{j} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)
$$

## 参数估计

### 极大似然估计

在朴素贝叶斯法中, 学习意味着估计 $P\left(Y=c_{k}\right)$ 和 $P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right).$ 可以 应用极大似然估计法估计相应的概率. 先验概率 $P\left(Y=c_{k}\right)$ 的极大似然估计是
$$
P\left(Y=c_{k}\right)=\frac{\sum_{i=1}^{N} I\left(y_{i}=c_{k}\right)}{N}, k=1,2, \cdots, K
$$
设第 $j$ 个特征 $x^{(j)}$ 可能取值的集合为 $\left\{a_{j 1}, a_{j 2}, \cdots, a_{j s_{j}}\right\}$, 条件概率 $P\left(X^{(j)}=a_{j l} \mid Y=c_{k}\right)$ 的极大似然估计是
$$
\begin{array}{l}\\P\left(X^{(j)}=a_{j l} \mid Y=c_{k}\right)=\frac{\sum_{i=1}^{N} I\left(x_{i}^{(j)}=a_{j l}, y_{i}=c_{k}\right)}{\sum_{i=1}^{N} I\left(y_{i}=c_{k}\right)} \\\\j=1,2, \cdots, n ; \quad l=1,2, \cdots, S_{j} ; k=1,2, \cdots, K\end{array}
$$
式中, x_{i}^{(j)} 是第 i 个样本的第 j 个特征; a_{j l} 是第 j 个特征可能取的第 l 个值; I 为 指示函数.

### 贝叶斯估计

用极大似然估计可能会出现**所要估计的概率值为 0** 的情况. 这时会影响到后验概率的计算结果, 使分类产生偏差. 解决这一问题的方法是采用贝叶斯估计. 具体地, 条件概率的贝叶斯估计是
$$
P_{\lambda}\left(X^{(j)}=a_{j l} \mid Y=c_{k}\right)=\frac{\sum_{i=1}^{N} I\left(x_{i}^{(j)}=a_{j l}, y_{i}=c_{k}\right)+\lambda}{\sum_{i=1}^{N} I\left(y_{i}=c_{k}\right)+S_{j} \lambda}
$$
式中 $\lambda \geqslant 0$. 等价于在随机变量各个取值的频数上赋予一个正数 $\lambda>0$. 当 $\lambda=0$ 时 就是极大似然估计. 常取 $\lambda=1$, 这时称为拉普拉斯平滑 (Laplace smoothing). 显 然, 对任何 $l=1,2, \cdots, S_{j}, k=1,2, \cdots, K$, 有
$$
\begin{array}{l}\\P_{\lambda}\left(X^{(j)}=a_{j l} \mid Y=c_{k}\right)>0 \\\sum_{l=1}^{s_{j}} P\left(X^{(j)}=a_{j} \mid Y=c_{k}\right)=1\\\end{array}
$$
表明式 (4.10) 确为一种概率分布. 同样, 先验概率的贝叶斯估计
$$
P_{\lambda}\left(Y=c_{k}\right)=\frac{\sum_{i=1}^{N} I\left(y_{i}=c_{k}\right)+\lambda}{N+K \lambda}
$$

## 算法流程

输入: 训练数据 $T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\}$, 其中 $x_{i}=\left(x_{i}^{(1)}, x_{i}^{(2)}, \cdots, x_{i}^{(n)}\right)^{\mathrm{T}}, x_{i}^{(j)}$ 是第 i 个样本的第 j 个特征, $x_{i}^{(j)} \in\left\{a_{j 1}, a_{j 2}, \cdots, a_{j s_{j}}\right\}, a_{j 1}$ 是第 j 个特征可能取 的第 l 个值, $j=1,2, \cdots, n, l=1,2, \cdots, S_{j}, y_{i} \in\left\{c_{1}, c_{2}, \cdots, c_{K}\right\}$ ； 实例 x;

输出：实例 x 的分类.

（1）计算先验概率及条件概率
$$
\begin{array}{l}\\P\left(Y=c_{k}\right)=\frac{\sum_{i=1}^{N} I\left(y_{i}=c_{k}\right)}{N}, \quad k=1,2, \cdots, K \\P\left(X^{(j)}=a_{j 1} \mid Y=c_{k}\right)=\frac{\sum_{i=1}^{N} I\left(x_{i}^{(j)}=a_{j}, y_{i}=c_{k}\right)}{\sum_{i=1}^{N} I\left(y_{i}=c_{k}\right)} \\j=1,2, \cdots, n ; \quad l=1,2, \cdots, S_{j} ; \quad k=1,2, \cdots, K\end{array}
$$
（2）对于给定的实例 $x=\left(x^{(1)}, x^{(2)}, \cdots, x^{(n)}\right)^{\mathrm{T}}$, 计算
$$
P\left(Y=c_{k}\right) \prod_{j=1}^{n} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right), \quad k=1,2, \cdots, K
$$
(3) 确定实例 x 的类
$$
y=\arg \max _{c_{k}} P\left(Y=c_{k}\right) \prod_{j=1}^{n} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)
$$

# 支持向量机（SVM）

## 线性可分支持向量机

{% note blue no-icon %}

给定训练样本集 $D=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right). \ldots,\left(x_{m}, y_{m}\right)\right\}, y_{i} \in\{-1,+1\}$, 分 类学习最基本的想法就是基于训练集 D 在样本空间中找到一个划分超平面, 将 不同类别的样本分开. 但能将训练样本分开的划分超平面可能有很多,, 我们应该努力去找到哪一个呢?

{% endnote %}



![](/img/统计机器学习/image-20220531104453242.png)

直观上看，应该去找位于两类训练样本“正中间”的划分超平面，因为该划分超平面对训练样本局部扰动的{% label “容忍”性 blue %}最好.例如，由于训练集的局限性或噪声的因素，训练集外的样本可能比图中的训练 样本更接近两个类的分隔界，这将使许多划分超平面出现错误，而超平 面受影响最小.换言之，这个划分超平面所产生的分类结果是最{% label 鲁棒的 blue %}，对未见 示例的{% label 泛化能力 blue %}最强.

在样本空间中, 划分{% label 超平面 red %}可通过如下线性方程来描述:
$$
\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}+b=0
$$
其中$\boldsymbol{w}=\left(w_{1} ,w_{2}  \cdots  w_{d}\right)$为法向量, 决定了超平面的方向; b 为位移项, 决定 了超平面与原点之间的距离. 显然, 划分超平面可被法向量 $\boldsymbol{w}$ 和位移 b 确定,下面我们将其记为 $(\boldsymbol{w}, b)$. 样本空间中任意点 $\boldsymbol{x}$ 到超平面 $(\boldsymbol{w}, b)$ 的距离可写为:
$$
r=\frac{\left|\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}+b\right|}{\|\boldsymbol{w}\|}
$$
假设超平面 $(\boldsymbol{w}, b)$ 能将训练样本正确分类, 即对于 $\left(\boldsymbol{x}_{i}, y_{i}\right) \in D$, 若 $y_{i}= +1$, 则有 $\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b>0$; 若 $y_{i}=-1$, 则有 $\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b<0$. 令
$$
\left\{\begin{array}{ll}\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b \geqslant+1, & y_{i}=+1 \\\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b \leqslant-1, & y_{i}=-1\\\end{array}\right.
$$
如图所示, **距离超平面最近的这几个训练样本点使式的等号成立**, 它们被称为 {% label 支持向量 (support vector) blue  %}, 两个异类支持向量到超平面的距离之和为
$$
\gamma=\frac{2}{\|w\|}
$$
它被称为 {% label 间隔 blue %} **(margin)**.

![image-20220531163501041](/img/统计机器学习/image-20220531163501041.png)

欲找到具有 “最大间隔” (maximum margin)的划分超平面, 也就是要找到能满足式中约束的参数 $\boldsymbol{w}$ 和 b, 使得 $\gamma$ 最大, 即
$$
\begin{aligned}\max _{\boldsymbol{w}, b} & \frac{2}{\|\boldsymbol{w}\|} \\\text { s.t. } & y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right) \geqslant 1, \quad i=1,2, \ldots, m .\\\end{aligned}
$$
此即支持向量机的基本型。

{% note purple no-icon %}

在决定分离超平面时只有支持向量起作用，而其他实例点并不起作用.如果 移动支持向量将改变所求的解；但是如果在间隔边界以外移动其他实例点，甚至 去掉这些点，则解是不会改变的.由于支持向量在确定分离超平面中起着决定性 作用，所以将这种分类模型称为支持向量机.支持向量的个数一般很少，所以支 持向量机由很少的**“重要的”**训练样本确定.

{% endnote %}

### 对偶方法

为了求解线性可分支持向量机的最优化问题，将它作为原始 最优化问题， 应用拉格朗日对偶性(参阅附录C) ， 通过求解对偶问题(dual problem) 得到原始问题(primal problem) 的最优解， 这就是线性可分支持向量 机的对偶算法(dual algorithm) .这样做的优点， 一是对偶问题往往更容易求解 二是自然引入核函数，进而推广到非线性分类问题.

首先构建拉格朗日函数 (Lagrange function). 为此, 对每一个不等式约束 引进拉格朗日乘子 $(Lagrange multiplier) \alpha_{i} \geqslant 0, i=1,2, \cdots, N$ 定义拉格朗日函数:
$$
L(w, b, \alpha)=\frac{1}{2}\|w\|^{2}-\sum_{i=1}^{N} \alpha_{i} y_{i}\left(w \cdot x_{i}+b\right)+\sum_{i=1}^{N} \alpha_{i}
$$
其中, $\alpha=\left(\alpha_{1},\alpha_{2}, \cdots, \alpha_{N}\right)^{\mathrm{T}}$ 为拉格朗日乘子向量.

根据拉格朗日对偶性, 原始问题的对偶问题是极大极小问题:
$$
\max _{\alpha} \min _{w, b} L(w, b, \alpha)
$$
所以, 为了得到对偶问题的解, 需要先求 $L(w, b, \alpha)$ 对 w, b 的极小, **再求对 $\alpha$ 的极大**.

由偏导为0可以得到如下条件
$$
\min _{w, b} L(w, b, \alpha)=-\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)+\sum_{i=1}^{N} \alpha_{i}
$$
再求对$\alpha$的极大，有：
$$
\begin{array}{ll}\min _{\alpha} & \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)-\sum_{i=1}^{N} \alpha_{i} \\ \text { s.t. } & \sum_{i=1}^{N} \alpha_{i} y_{i}=0 \\ & \alpha_{i} \geqslant 0, \quad i=1,2, \cdots, N\end{array}
$$
**重要定理：**

设 $\alpha^{*}=\left(\alpha_{1}^{*}, \alpha_{2}^{*}, \cdots, \alpha_{l}^{*}\right)^{\mathrm{T}}$是**对偶最优化**问题的解, 则 存在下标 j, 使得 $\alpha_{j}^{*}>0,$ 并可按下式求得**原始最优化**问题的解 $w^{*}, b^{*}$ :
$$
w^{*}=\sum_{i=1}^{N} \alpha_{i}^{*} y_{i} x_{i}\\b^{*}=y_{j}-\sum_{i=1}^{N} \alpha_{i}^{*} y_{i}\left(x_{i} \cdot x_{j}\right)
$$

### 线性可分支持向量机——算法

输入: 线性可分训练集 $T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\}$, 其中 $x_{i} \in \mathcal{X}=\mathbf{R}^{n}, y_{i} \in \mathcal{Y}=\{-1,+1\}, i=1,2, \cdots, N$

输出: 分离超平面和分类决策函数.

{% timeline 算法分析 %}
<!-- timeline 列出对偶问题 -->

（1）构造并求解约束最优化问题
$$
\begin{array}{ll}\min _{\alpha} & \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)-\sum_{i=1}^{N} \alpha_{i} \\\text { s.t. } \quad & \sum_{i=1}^{N} \alpha_{i} y_{i}=0 \qquad\alpha_{i} \geqslant 0, \quad i=1,2, \cdots, N\\\end{array}
$$
求得最优解 $\alpha^{*}=\left(\alpha_{1}^{*}, \alpha_{2}^{*}, \cdots, \alpha_{N}^{*}\right)^{\mathrm{T}}.$

<!-- endtimeline -->

<!-- timeline 计算原始问题解 -->

(2) 计算
$$
w^{*}=\sum_{i=1}^{N} \alpha_{i}^{*} y_{i} x_{i}
$$
并选择 $\alpha^{*}$的一个正分量 $\alpha_{j}^{*}>0$, 计算
$$
b^{*}=y_{j}-\sum_{i=1}^{N} \alpha_{i}^{*} y_{i}\left(x_{i} \cdot x_{j}\right)
$$


<!-- endtimeline -->

<!-- timeline 获得答案 -->

（3）求得分离超平面
$$
w^{*} \cdot x+b^{*}=0
$$
分类决策函数:
$$
f(x)=\operatorname{sign}\left(w^{*} \cdot x+b^{*}\right)
$$


<!-- endtimeline -->

{% endtimeline %}

{% note red no-icon %}

考试的话就是考这种题，务必熟悉公式和流程，手算对偶算法

{% endnote %}

## 线性支持向量机与软间隔最大化

> 线性可分问题的支持向量机学习方法，对线性不可分训练数据是不适用的，因 为这时上述方法中的不等式约束并不能都成立.怎么才能将它扩展到线性不可分 问题呢?这就需要修改硬间隔最大化， 使其成为软间隔最大化.

线性不可分意味着某些样本点 $\left(x_{i}, y_{i}\right)$ 不能满足**函数间隔大于等于 1** 的约束条件. 为了解决这个问题, 可以对每个样本点 $\left(x_{i}, y_{i}\right)$ 引进一个松肔变量 $\xi_{i} \geqslant 0,$使函数间隔加上**松弛变量**大于等于 1. 这样, 约束条件变为
$$
y_{i}\left(w \cdot x_{i}+b\right) \geqslant 1-\xi_{i}
$$
同时, 对每个松弛变量 $\xi_{i}$, 支付一个代价 $\xi_{i}$. 目标函数由原来的 $\frac{1}{2}\|w\|^{2}$ 变成
$$
\frac{1}{2}\|w\|^{2}+C \sum_{i=1}^{N} \xi_{i}
$$
这里, C>0 称为惩罚参数, 一般由应用问题决定, C 值大时对误分类的惩罚增 大, C 值小时对误分类的惩罚减小. 最小化目标函数包含两层含义: 使 \frac{1}{2}\|w\|^{2} 尽量小即间隔尽量大, 同时使误分类点的个数尽量小, C 是调和二者的系数.

因此，问题转化为
$$
\begin{array}{ll}\min _{w, b, \xi} & \frac{1}{2}\|w\|^{2}+C \sum_{i=1}^{N} \xi_{i} \\ \text { s.t. } & y_{i}\left(w \cdot x_{i}+b\right) \geqslant 1-\xi_{i}, \quad i=1,2, \cdots, N \\ & \xi_{i} \geqslant 0, \quad i=1,2, \cdots, N\end{array}
$$

### 算法

输入: 训练数据集 $T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\}$, 其中, $x_{i} \in \mathcal{X}=\mathbf{R}^{n}, y_{i} \in \mathcal{Y}=\{-1,+1\}, i=1,2, \cdots, N$;

输出: 分离超平面和分类决策函数.

{% timeline 算法分析 %}
<!-- timeline 列出对偶问题 -->

（1）选择惩罚参数 C>0, 构造并求解凸二次规划问题
$$
\min _{\alpha} \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)-\sum_{i=1}^{N} \alpha_{i}\\\\s.t. \sum_{i=1}^{N} \alpha_{i} y_{i}=0\\\\0 \leqslant \alpha_{i} \leqslant C, \quad i=1,2, \cdots, N
$$
求得最优解 $\alpha^{*}=\left(\alpha_{1}^{*}, \alpha_{2}^{*}, \cdots, \alpha_{N}^{*}\right)^{\mathrm{T}}$.

<!-- endtimeline -->

<!-- timeline 计算原始问题解 -->

(2) 计算 $w^{*}=\sum_{i=1}^{N} \alpha_{i}^{*} y_{i} x_{i}$

选择 $\alpha^{*}$ 的一个分量 $\alpha_{j}^{*}$ 适合条件 $0<\alpha_{j}^{*}<C$, 计算
$$
b^{*}=y_{j}-\sum_{i=1}^{N} y_{i} \alpha_{i}^{*}\left(x_{i} \cdot x_{j}\right)
$$
<!-- endtimeline -->

<!-- timeline 获得答案 -->

（3）求得分离超平面
$$
w^{*} \cdot x+b^{*}=0
$$
分类决策函数:
$$
f(x)=\operatorname{sign}\left(w^{*} \cdot x+b^{*}\right)
$$
<!-- endtimeline -->

{% endtimeline %}

步骤 (2) 中, 对任一适合条件 $0<\alpha_{j}^{*}<C 的 \alpha_{j}^{*}$, 按式都可求出 $b^{*}$, 但是由 于原始问题对 b 的解并不唯一, 所以实际计算时可以取在所有符合条件的样本点上的平均值.

{% note red no-icon %}

注意与前者的区别，计算过程大体相同

{% endnote %}

### 支持向量

在线性不可分的情况下, 将对偶问题的解 $\alpha^{*}=\left(\alpha_{1}^{*}, \alpha_{2}^{*}, \cdots, \alpha_{N}^{*}\right)^{\mathrm{T}}$ 中对应于 $\alpha_{i}^{*}>0$的样本点 $\left(x_{i}, y_{i}\right)$ 的实例 $x_{i}$ 称为支持向量 (软间隔的支持向量). 如图 7.5 所示, 这时的支持向量要比线性可分时的情况复杂一些. 图中, 分离超平面由实线表示, 间隔边界由虚线表示,正例点由“$\cdot$”表示,负例点由 “ $\times$ ” 表示. 图中还标出了实例 $x_{i}$ 到间隔边界的距离 $\frac{\xi_{i}}{\|w\|}$.

软间隔的支持向量 $x_{i}$ 或者在间隔边界上, 或者在间隔边界与分离超平面之 间, 或者在分离超平面误分一侧. 

- 若 $\alpha_{i}^{*}<C$, 则 $\xi_{i}=0$, 支持向量 $x_{i}$ 恰好落在间 隔边界上; 
- 若 $\alpha_{i}^{*}=C, 0<\xi_{i}<1$, 则分类正确, x_{i} 在间隔边界与分离超平面之间; 
- 若 $\alpha_{i}^{*}=C, \xi_{i}=1$, 则 $x_{i}$ 在分离超平面上; 若 $\alpha_{i}^{*}=C, \xi_{i}>1$, 则 $x_{i}$ 位于分离超 平面误分一侧.

![image-20220531201721808](/img/统计机器学习/image-20220531201721808.png)

## 非线性可分支持向量机与核函数

非线性问题往往不好求解，所以希望能用解线性分类问题的方法解决这个问 题.所采取的方法是进行一个非线性变换，将非线性问题变换为线性问题，通过解 变换后的线性问题的方法求解原来的非线性问题、例如：通过变换，将左图中椭圆变换成右图中的直线，将非线性分类问题变换为线性分类问题.

![image-20220531202802352](/img/统计机器学习/image-20220531202802352.png)

### 核函数

设 $\mathcal{X}$ 是输入空间（欧氏空间 $\mathbf{R}^{n}$ 的子集或离散集合),设 $\mathcal{H}$为特征空间（希尔伯特空间）, 如果存在一个从 $\mathcal{X}$ 到 $\mathcal{H}$ 的映射
$$
\phi(x): \mathcal{X} \rightarrow \mathcal{H}
$$
使得对所有 $x, z \in \mathcal{X}$, 函数 $K(x, z)$ 满足条件
$$
K(x, z)=\phi(x) \cdot \phi(z)
$$
则称 $K(x, z)$ 为核函数, $\phi(x)$ 为映射函数, 式中 $\phi(x) \cdot \phi(z)$ 为 $\phi(x)$ 和 $\phi(z)$ 的内积.

#### 应用于支持向量机中：

在线性支持向量机的对偶问题中, 无论是目标函数还是决策函数 (分离超平面) 都只涉及输入实例与实例之间的内积. 在对偶问题的目标函数中的内积 $x_{i} \cdot x_{j}$ 可以用核函数 $K\left(x_{i}, x_{j}\right)=\phi\left(x_{i}\right) \cdot \phi\left(x_{j}\right)$ 来代替. 此时对偶问题的目标函数成
$$
W(\alpha)=\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j} K\left(x_{i}, x_{j}\right)-\sum_{i=1}^{N} \alpha_{i}
$$
同样, 分类决策函数中的内积也可以用核函数代替, 而分类决策函数式成
$$
f(x)=\operatorname{sign}\left(\sum_{i=1}^{N_{s}} a_{i}^{*} y_{i} \phi\left(x_{i}\right) \cdot \phi(x)+b^{*}\right)=\operatorname{sign}\left(\sum_{i=1}^{N_{1}} a_{i}^{*} y_{i} K\left(x_{i}, x\right)+b^{*}\right)
$$
这等价于经过映射函数 $\phi$ 将原来的输入空间变换到一个新的特征空间, 将输入空间中的内积 $x_{i} \cdot x_{j}$ 变换为特征空间中的内积 $\phi\left(x_{i}\right) \cdot \phi\left(x_{j}\right)$, 在新的特征空间里从训练样本中学习线性支持向量机. 当映射函数是非线性函数时, 学习到的含有核函数的支持向量机是非线性分类模型.

#### 常用的核函数

- 多项式核函数

  - $$
    K(x, z)=(x \cdot z+1)^{p}
    $$

- 高斯核函数

  - $$
    K(x, z)=\exp \left(-\frac{\|x-z\|^{2}}{2 \sigma^{2}}\right)
    $$

### 算法

输入: 训练数据集 $T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\}$, 其中 $x_{i} \in \mathcal{X}=\mathbf{R}^{n}, y_{i} \in \mathcal{Y}=\{-1,+1\} ， i=1,2, \cdots, N$;

输出: 分类决策函数.

{% timeline 算法分析 %}
<!-- timeline 列出对偶问题 -->

（1）选取适当的核函数 K(x, z) 和适当的参数 C, 构造并求解最优化问题
$$
\begin{array}{ll}\min _{\alpha} & \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j} K\left(x_{i}, x_{j}\right)-\sum_{i=1}^{N} \alpha_{i} \\\\\text { s.t. } & \sum_{i=1}^{N} \alpha_{i} y_{i}=0 \\& 0 \leqslant \alpha_{i} \leqslant C, \quad i=1,2, \cdots, N\\\end{array}
$$
求得最优解 $\alpha^{*}=\left(\alpha_{1}^{*}, \alpha_{2}^{*}, \cdots, \alpha_{N}^{*}\right)^{\mathrm{T}}.$

<!-- endtimeline -->

<!-- timeline 计算原始问题解 -->

(2) 选择 $\alpha^{*}$ 的一个正分量 $0<\alpha_{j}^{*}<C$, 计算
$$
b^{*}=y_{j}-\sum_{i=l}^{N} \alpha_{i}^{*} y_{i} K\left(x_{i} \cdot x_{j}\right)
$$
<!-- endtimeline -->

<!-- timeline 获得答案 -->

(3）构造决策函数:
$$
f(x)=\operatorname{sign}\left(\sum_{i=1}^{N} \alpha_{i}^{*} y_{i} K\left(x \cdot x_{i}\right)+b^{*}\right)
$$
<!-- endtimeline -->

{% endtimeline %}

{% note red no-icon %}

注意{% label 决策函数 red %}的区别

{% endnote %}