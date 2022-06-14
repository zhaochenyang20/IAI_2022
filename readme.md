## 2042 IAI Review

## readme

**Credit to Eren, Saltyp0rridge, Monoxide Chen, tjy20, ylc20**

贵系 2042 年春季人智导复习仓库，~~都 2042 年了贵系还给咱讲 $\alpha-\beta$ 剪枝呢！~~本学期人智导课程相对 2021 年少了部分内容，加之线上考试充满不确定性，（马老师曾经说可能会更改考试风格，从考基于手跑算法的考大题改成基于考小题），遂决定组队复习人智导。（后来通知还是只考大题）

参考资料主要为计 8 年级白钰卓学姐和邱可玥学姐的笔记、马老师的课堂 PPT、动手学深度学习、tjy & Eren 的课堂笔记等等，其余参考资料会在每一章复习的具体段落给出

友情 repo：lambda 的[模拟题生成器](https://github.com/Btlmd/IAI_Gen)，涵盖 `svm` 与 $\alpha-\beta$

## 文档分工

- `Readme.md`—— Eren
- `deep learning.md` —— Eren（完结）
- `adversarial search part1.md` —— Eren（完结）
- `machine learning.md`—— Saltyp0rridge（完结），Eren 负责修订（完成）
- `graph search.md` —— Monoxide（完结）, Eren 负责修订（完成）
- `exam.md` —— tjy20（已摆烂）
- `adversarial search part2.md`—— tjy20（手写），ylc20（手动转 markdown，完结）
- `IAI_note/c7w_final_preparation.md`—— c7w（已完结）

# 编写须知

- 尽量开新的分支（~~规范参考软工~~）
- 图片请一定使用相对路径或者图床

+ 部分常用函数，如三角函数、对数函数等的 Roman 体符号已经内置于 LaTeX 中。如余弦函数应当使用 `$\cos$` 表示，渲染为 $\cos$。而不应当使用 `$cos$` 表示，渲染为 $cos$。
+ 渲染 Roman 体（正体）粗体请使用 `$\mathbf{}$`，如 $\mathbf{a}$。渲染 Italic 体（斜体）粗体请使用 `$\boldsymbol{a}$`，如 $\boldsymbol{a}$。请不要混淆两者。
+ 行内公式渲染分数时，请使用 `$\dfrac{}{}$`，如 $\dfrac{1}{2}$。而不应使用 `$\frac{}{}$`，如 $\frac{1}{2}$。
+ 行内公式应尽力避免求和、求积、积分等巨运算符，即 $\sum$、$\int$ 等运算符。
+ 省略号请使用 `$\cdots$`，如 $a_1, a_2, \cdots, a_n$。而不应使用 `$...$`，如 $a_1, a_2, ..., a_n$。
+ 注意区分数学语言与程序设计语言。请不要在公式块内使用双等号 $==$ 乃至三等号 $===$ 表示两个量相等，请使用数学语言中的单等号 $=$，如 $a = b$。此外，请不要使用中括号连缀表达高维数组，如 $a[i][j][k]$，请使用函数表示，如 $f(i, j, k)$。
+ 特殊集合（如自然数集、实数集）的字体请使用 `$\mathbb{}$`，如 $\mathbb{R}$、$\mathbb{N}$。
+ 括号内为分式等内容时，请使用自适应高度括号 `$\left(\right)$`，如 $\left(\dfrac{1}{2}\right)^n$。请不要直接使用 `$()$`，如 $(\dfrac{1}{2})^n$。
+ 公式内请不要出现汉字，出现英文单词时请使用正体。
+ 数字、字母、公式、英文与前后的汉字之前各留一个空格。例如，在第 k 个根节点。
+ 中文之间请使用中文标点。
+ 请使用粗体，不使用斜体。
