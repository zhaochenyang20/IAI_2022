# 人工智障导论 2021

1. alpha-beta 剪枝。要求：1）标出每个节点的值；2）表明在何处剪枝；3）画出最终选择的走步。（P.S. 给的树是“不齐”的。）

2. 修正的 A\* 算法。给出求解过程、依次拓展的节点、最终选择的路径。

3. 非线性可分的 SVM 。

    样本：正例： $\boldsymbol{x_1}=(0, 0)^T$ ；负例： $\boldsymbol{x_2}=(1, 1)^T$ ， $\boldsymbol{x_3}=(-1, -1)^T$ 。

    核函数： $K(\boldsymbol{x}, \boldsymbol{y}) = (1 + \boldsymbol{x}^T \boldsymbol{y})^2$ 。

    请求解 SVM 参数，并给出 $(0, 1)^T$ 的分类。

<img src="./pic/exam/answer 1.jpg" style="zoom:50%;" />

4. （1）模拟退火。（温度固定，类似往年题。）

    （2）遗传算法。（类似 PPT 例题。）

5. 决策树。用 ID3 算法建立决策树，只需给出根节点及其子节点，表明叶节点的类别。

6. 设计智障神经网络：输入单个数字的图片，输出对应的英文单词（one, two, ...）。要求使用 MLP、CNN、RNN 。画示意图，简要说明。

# 口头考纲

> credit to 计 06 班学习委员，salute

- CH1

> A 和 A* 算法
>
> 易错：A* 算法结束条件 (目标点在OPEN表中第一个)
>
> <img src="./pic/exam/1-1.PNG" style="zoom:50%;" />
>
> <img src="./pic/exam/1-2.PNG" style="zoom:50%;"/>

- CH2

> 基本CNN RNN LSTM 及典型应用
>
> BP 算法的推导
>
> 典型神经网络系统
>
> ResNet、GoogleNet
>
> 给出图，能说出每部分（？？）
>
> 词向量
>
> 掌握实现方法



- CH3

>  alpha-beta剪枝
>
>  易错：注意找全所有祖先节点（不只是父节点），最佳走步只能标一步
>
>  蒙特卡洛树搜索
>
>  重点是选择过程
>
>  AlphaGo / AlphaGo Zero
>
>  大概懂就行：两个网络、结合蒙特卡洛
>
>  <img src="./pic/exam/3-1.png" style="zoom:50%;"/>
>
>  <img src="./pic/exam/3-2.png" style="zoom:50%;"/>
>
>  <img src="./pic/exam/3-3.PNG" style="zoom:50%;"/>

- CH4

>  SVM
>
>  会用对偶的方法求就行（ppt上例题）
>
>  决策树
>
>  给定数据，会用ID3/C4.5建树
>
>  **SVM**例题见最上方2021回忆版
>
>  **ID3决策树**
>
>  <img src="./pic/exam/4-1.PNG" style="zoom:50%;"/>
>
>  <img src="./pic/exam/4-2.png" style="zoom:50%;"/>
>
>  <img src="./pic/exam/4-3.png" style="zoom:50%;"/>
>
>  <img src="./pic/exam/4-4.png" style="zoom:50%;"/>



==注：==

==按马老师上课所说，第二章和第三章设计神经网络的部分不会单独出算法题，而应该是类似2021回忆版的最后一题，老师会出一个带有点创新的思考题，用到神经网络的知识。==

