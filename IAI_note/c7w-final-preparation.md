一天刷完了录屏，明天刷题，后天考试，就这样，，，

## Ch01

+ A 算法
  + OPEN=(s), CLOSED=()
  + while OPEN 不空，取其中第一个结点 n，如果是目标就返回，不然把 n 移动到 CLOSED 表，接着考虑其所有的子节点，计算 $f(n, m_i) = g(n, m_i) + h(m_i)$，然后：
    + 如果是 $m_j$ 类型的结点，那么标记 $m_j$ 到 $n$ 的指针，加入 OPEN 表
    + 如果是 $m_k$ 类型的结点且当前值更小，那么更新 $f(m_k)$ 与相应指针
    + 如果是 $m_l$ 类型结点且当前值更小，那么更新 $f(m_l)$ 与相应指针，然后加入 OPEN 表
+ A* 算法：满足条件 $h(n) \le h^\star (n)$ 的 A 算法
  + 若存在从初始节点s到目标节点t有路径，则A*必能找到最佳解结束。
  + 如果h2(n) > h1(n) (目标节点除外)，则A1扩展的节点数≥A2扩展的节点数
+ A* 算法的改进
  + 对 h 加以限制（称 h 是单调的）
    + $h(n_i) - h(n_j) \le c(n_i, n_j), \ h(t) = 0$
    + 结论是当扩展到结点 $n$ 时就有 $g(n) = g^\star (n)$
    + 满足单调条件的一定满足 A* 条件
  + 对算法加以改进
    + 记 $f_m$ 为到目前为止已扩展结点的最大 $f$ 值，以 $f_m$ 为分界
    + 在每个迭代步的最开始，构造 NEST 表为 OPEN 表中所有满足 $f(n_i) < f_m$ 的 $n_i$
    + 如果 NEST 不空，取 NEST 中 g 最小的结点；不然取 OPEN 中 f 最小的结点

例题：A$^\star$算法，改进的 A$^\star$ 算法，s 赋值成 9 再做

![image-20220613211238566](https://s2.loli.net/2022/06/13/sIwa1enklVKtf5F.png)

## Ch02

### BP

![image-20220606232948144](https://s2.loli.net/2022/06/06/JOfDHEewM15S6VA.png)
![image-20220606233252946](https://s2.loli.net/2022/06/06/5hglL4VWJwqzPCa.png)

### Models

+ TextCNN
+ RNN
  + ![image-20220613214155079](https://s2.loli.net/2022/06/13/RYH9sgobyXxntUu.png)
  + 应用例子
    + 中文分词
    + 看图说话
      + ![image-20220613214812583](https://s2.loli.net/2022/06/13/cu4hDf7yGt3dMH9.png)
  + 双向循环神经网络
  + Seq2seq Encoder Decoder
  + LSTM
    + ![img](https://tangshusen.me/Dive-into-DL-PyTorch/img/chapter06/6.8_lstm_3.svg)

$$
\begin{aligned}
\boldsymbol{I}_t &= \sigma(\boldsymbol{W}_{xi} \boldsymbol{X}_t + \boldsymbol{W}_{hi} \boldsymbol{H}_{t-1} + \boldsymbol{b}_i),\\
\boldsymbol{F}_t &= \sigma(\boldsymbol{W}_{xf} \boldsymbol{X}_t + \boldsymbol{W}_{hf}\boldsymbol{H}_{t-1}  + \boldsymbol{b}_f),\\
\boldsymbol{O}_t &= \sigma(\boldsymbol{W}_{xo}\boldsymbol{X}_t  + \boldsymbol{W}_{ho}\boldsymbol{H}_{t-1}  + \boldsymbol{b}_o), \\ 
\tilde{\boldsymbol{C}}_t &= \text{tanh}(\boldsymbol{W}_{xc}\boldsymbol{X}_t  + \boldsymbol{W}_{hc}\boldsymbol{H}_{t-1}  + \boldsymbol{b}_c), \\ 
\boldsymbol{C}_t & = \boldsymbol{F}_t \odot \boldsymbol{C}_{t-1} + \boldsymbol{I}_t \odot \tilde{\boldsymbol{C}}_t, \\ 
\boldsymbol{H}_t &= \boldsymbol{O}_t \odot \text{tanh}(\boldsymbol{C}_t).
\end{aligned}
$$



###　Example Models

```python
LeNet = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(400, 120),
    nn.Linear(120, 84),
    nn.Linear(84, 10)
)
```

![image-20220606235802057](https://s2.loli.net/2022/06/06/1he6JDRCVA85QZK.png)

+ GoogLeNet
+ ResNet

![image-20220609220459061](https://s2.loli.net/2022/06/09/ChpO6BMrmDsPYyK.png)

### 机器学习中出现的问题

+ 梯度消失问题
  + 使用 ReLU 激活函数，多输出，设计残差模块
+ 过拟合：减少过拟合的方法
  + 正则项（L1 Loss 与 L2 Loss 的效果）、Dropout，数据增强

### 语言模型

+ NNLM
  + 词向量的训练：对输入也回传梯度
  + ![image-20220613211806202](https://s2.loli.net/2022/06/13/1puL9jtnJ6P8Ka3.png)
+ Word2vec
  + CBOW
    + ![image-20220613212818481](https://s2.loli.net/2022/06/13/3gfI7CRNxqTEKA4.png)
    + ![image-20220613213343788](https://s2.loli.net/2022/06/13/IwQnBuYaADfthZK.png)
    + ![image-20220613213354610](https://s2.loli.net/2022/06/13/SR8McAYkyW7mthx.png)
  + Skip-Gram Model

## Ch03

博弈问题：双人，一人一步，双方信息完备，零和

### Minimax / alpha-beta

极大极小模型：

![image-20220611233736400](https://s2.loli.net/2022/06/11/Htui2g5sTbKBVvM.png)

alpha-beta 剪枝：

![image-20220611234211509](https://s2.loli.net/2022/06/11/GuthCNl6frR1BgA.png)

![image-20220611234057407](https://s2.loli.net/2022/06/11/cuBahoYDTR4IMqd.png)

```
procedure search(node)
	while (n has children to extend) do
		if branch cut is possible
			cut branches
			break
		endif
		search(first child node not extended)
	endwhile
	update parent node's value
```





### 蒙特卡洛方法

+ 选择（选第一个有子节点没扩展的节点）
  + 对尚未充分了解的节点的探索
  + 对当前具有较大希望节点的利用
+ 扩展（生成子节点）
+ 模拟（获得收益）
+ 回传（胜负交替）

+ ![image-20220612133239012](https://s2.loli.net/2022/06/12/3yKBt2J6cDSFWLw.png)

选择：

![image-20220612133556296](https://s2.loli.net/2022/06/12/r3fGwCausO9IoLx.png)

![image-20220612133606921](https://s2.loli.net/2022/06/12/ifFPc6A435GhlS9.png)

### AlphaGo

策略网络（输入当前棋局，输出在某个点的落子概率）、估值网络（输入当前棋局，输出估计的收益 [-1, 1]）

![image-20220612135139674](https://s2.loli.net/2022/06/12/uhAioY2XP8ncqpL.png)

![image-20220612135509411](https://s2.loli.net/2022/06/12/S2tW7HljI94oJhi.png)

![image-20220612145236888](https://s2.loli.net/2022/06/12/X81M6WgGudmpjlQ.png)

### RL

+ 基于策略梯度的 RL
+ 基于价值评估的 RL
+ 演员-评价方法
  + ![image-20220612142123266](https://s2.loli.net/2022/06/12/O3lN4hqDzMxcnLo.png)

### AlphaGo Zero

+ 修改模拟阶段的收益 => 仅采用估值网络
+ 引入多样性：对策略网络的输出增加噪声



## Ch04

### SVM

#### 线性可分支持向量机

线性可分支持向量机（二分类）、函数间隔（超平面关于样本点的，超平面关于训练集 T 的）、几何间隔（欧氏距离）、求解的转化、学习的对偶算法

![image-20220530161313195](https://s2.loli.net/2022/05/30/bR3A857dXVDeMxI.png)

![image-20220530154546933](https://s2.loli.net/2022/05/30/tQpgT5Jj2LhvxwX.png)

![image-20220530154742491](https://s2.loli.net/2022/05/30/MDuhakFd3Xb4cyR.png)

支持向量的数目


#### 线性支持向量机 

![image-20220530162255542](https://s2.loli.net/2022/05/30/QEaFYkqWPKve4Vp.png)

![image-20220530162344229](https://s2.loli.net/2022/05/30/RSPtCgBvlEe7nD8.png)

![image-20220530162502134](https://s2.loli.net/2022/05/30/j2txhWbpSPsoZgB.png)

#### 非线性支持向量机

用某个变换将原空间数据映射到新空间，在新空间用线性分类方法学习。

核函数，用核函数 $K$ 找映射 $\phi$，

#### 应用

tf-idf 的算法？i 是词项，j 是文档

### 决策树

+ 信息增益的计算公式
+ 生成决策树的算法
  + ID3
  + C4.5

![image-20220612151544073](https://s2.loli.net/2022/06/12/UlMoh134KsAGCyV.png)

#### ID3

输入：训练集D，特征集A，阈值ε>0

输出：决策树T

1，若D中所有实例属于同一类Ck，则T为单节点树，将Ck作为该节点的类标记，返回T

2，若A为空，则T为单节点树，将D中实例数最大的类Ck作为该节点的类标记，返回T

3，否则计算A中各特征对D的信息增益，选择信息最大的特征Ag

4，如果Ag的信息增益小于阈值ε，则置T为单节点树，将D中实例数最大的类Ck作为该节点的类标记，返回T

5，否则对Ag的每一可能值ai，依Ag=ai将D分割为若干子集Di，作为D的子节点

6，对于D的每个子节点Di，如果Di为空，则将D中实例最大的类作为标记，构建子节点

7，否则以Di为训练集，以A-{Ag}为特征集，递归地调用步1~步6，得到子树Ti，返回Ti

![image-20220612151036336](https://s2.loli.net/2022/06/12/6ByI2ETwrQPM8kv.png)

![image-20220612151047204](https://s2.loli.net/2022/06/12/Z4w6VrS5fIyYBn2.png)

![image-20220612151100678](https://s2.loli.net/2022/06/12/J4hli5t7DPqBd6Z.png)

![image-20220612151106474](https://s2.loli.net/2022/06/12/bodISukNfHUBpYy.png)

#### C4.5

![image-20220612155858744](https://s2.loli.net/2022/06/12/IW4HSB2Ynet3gU6.png)
