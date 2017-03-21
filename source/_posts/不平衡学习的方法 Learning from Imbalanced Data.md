title: 不平衡学习的方法 Learning from Imbalanced Data #文章页面上的显示名称，可以任意修改，不会出现在URL中
date: 2016-02-31 15:30:16 #文章生成时间，一般不改，当然也可以任意修改
categories: 机器学习 #分类
tags: [不平衡学习,数据挖掘,算法] #文章标签，可空，多标签请用格式，注意:后面有个空格
description: 不平衡学习的方法 Learning from Imbalanced Data
---
之前做二分类预测的时候，遇到了正负样本比例严重不平衡的情况，甚至有些比例达到了50:1，如果直接在此基础上做预测，对于样本量较小的类的召回率会极低，这类不平衡数据该如何处理呢？


# 不平衡数据的定义
----------
顾名思义即我们的数据集样本类别极不均衡，以二分类问题为例，数据集中的多数类 为$S_{max}$，少数类为$S_{min}$，通常情况下把多数类样本的比例为$100:1$、$1000:1$，甚至是$10000:1$这种情况下为不平衡数据。

# 为什么不平衡学习
----------
 因为传统的学习方法以降低总体分类精度为目标，将所有样本一视同仁，同等对待，造成了分类器在多数类的分类精度较高而在少数类的分类精 度很低。例如上面正负样本50:1的例子，算法就算全部预测为另一样本，准确率也会达到98%(50/51)，因此传统的学习算法在不平衡数据集中具有较大的局限性。

# 不平衡学习的方法
----------
解决方法主要分为两个方面。

 - 第一种方案主要从**数据**的角度出发，主要方法为抽样，既然我们的样本是不平衡的，那么可以通过某种策略进行抽样，从而让我们的数据相对均衡一些；
 - 第二种方案从**算法**的角度出发， 考虑不同误分类情况代价的差异性对算法进行优化，使得我们的算法在不平衡数据下也能有较好的效果。

## 采样


----------
### 随机采样


----------


采样算法通过某一种策略改变样本的类别分布，以达到将不平衡分布的样本转化为相对平衡分布的样本的目的，而随机采样是采样算法中最简单也最直观易 懂的一种方法。随机采样主要分为两种类型，分别为随机欠采样和随机过采样两种。
随机欠采样顾名思义即从多数类$S_{max}$中随机选择少量样本$E$再合 并原有少数类样本作为新的训练数据集，新数据集为$S_{min}+E$，随机欠采样有两种类型分别为有放回和无放回两种，无放回欠采样在对多数类某样本被采 样后不会再被重复采样，有放回采样则有可能。
随机过采样则正好相反，即通过多次有放回随机采样从少数类$S_{min}$中抽取数据集$E$，采样的数量要大 于原有少数类的数量，最终的训练集为$S_{max}+E$。
    显然，随机采样是通过改变多数类或者少数类的样本比例达到修改样本分类分布的目的，其中也存在着诸多的问题，例如随机欠采样，由于丢失了一些样本，造成一些信息的缺失，如果未被采样的样本具有重要的信息呢？而过采样扩大了数据集，训练模型的复杂度会加大，而且有可能造成过拟合的情况。


----------


## SMOTE算法
SMOTE全称是Synthetic Minority Oversampling Technique即合成少数类过采样技术，SMOTE算法的基本思想SMOTE算法的基本思想是对少数类样本进行分 析并根据少数类样本人工合成新样本添加到数据集中，具体如图2所示，算法流程如下。

 1. 对于少数类中每一个样本$x$，以欧氏距离为标准计算它到少数类样本集$S_{min}$中所有样本的距离，得到其k近邻。 
 2. 根据样本不平衡比例设置一个采样比例以确定采样倍率$N$，对于每一个少数类样本$x$，从其k近邻中随机选择若干个样本，假设选择的近邻为$\hat{x}$。 
 3. 对于每一个随机选出的近邻$\hat{x}$，分别与原样本按照如下的公式构建新的样本。 
$$x_{new} = x + rand(0,1)*(\hat{x}-x)$$
<center>![SMOTE算法](http://img.blog.csdn.net/20170106221646788?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2hpbmUxOTkzMDgyMA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
图2 SMOTE算法</center>
SMOTE算法摈弃了随机采样复制样本的做法，使得算法的性能有所提升，但由于每个少数样本都会产生新样本，也会产生样本重叠的问题，下面介绍其改进算法：





## Borderline-SMOTE算法


----------


在Borderline-SMOTE中，若少数类样本的每个样本$x_i$求k近邻，记作$S_i-knn$，且$S_i-knn$属于整个样本集合$S$而不再是少数类样本，若满足
$$\frac{k}{2}<|s_{i-knn}\cap s_{max}|<k$$
即k近邻中超过一半是多数样本。
则将样本$x_i$加入DANGER集合，显然DANGER集合代表了接近分类边界的样本，将DANGER当作SMOTE种子样本的输入生成新样本。特别地，当上述条件取右边界，即k近邻中全部样本都是多数类时此样本不会被选择为种样本生成新样本，此情况下的样本为噪音。
<center>![图3 Borderline-SMOTE算法](http://img.blog.csdn.net/20170106223631471?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2hpbmUxOTkzMDgyMA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
图3 Borderline-SMOTE算法</center>



## Informed Undersampling
前面讲了关于过采样的的算法，那么下面就是欠采样算法informed undersampling，informed undersampling采样技术主要有两种方法分别是EasyEnsemble算法和BalanceCascade算法。
    EasyEnsemble算法如下图4所示，此算法类似于随机森林的Bagging方法，它把数据划分为两部分，分别是多数类样本和少数类样 本，对于多数类样本$S_maj$，通过**n次有放回抽样**生成n份子集，少数类样本分别和这n份样本合并训练一个模型，这样可以得到n个模型，最终的模型是 这**n个模型预测结果的平均值**。
<center>![Informed Undersampling](http://img.blog.csdn.net/20170106230518290?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2hpbmUxOTkzMDgyMA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
 
BalanceCascade算法是一种级联算法，BalanceCascade从多数类$S_{max}$中有效地选择N且满 足$\mid N \mid=\mid S_{min} \mid$，将N和$S_{min}$合并为新的数据集进行训练，新训练集对每个多数类样本$x_i$进行预测 若预测对则$S_{max}=S_{maj}-x_i$。依次迭代直到满足某一停止条件，最终的模型是多次迭代模型的组合。
    核心思想：使用之前已形成的集成分类器来为下一次训练选择多类样本，然后再进行欠抽样。


## 代价敏感学习
----------
代价敏感学习算法(Cost-Sensitive Learning)主要从算法层面上解决不平衡数据学习，代价敏感学习方法的核心要素是代价矩阵，我们注意到在实际的应用中不同类型的误分类情况导致的代价是不一样的，例如在医疗中，“将病 人误疹为健康人”和“将健康人误疹为病人”的代价不同，因此 我们定义代价矩阵如下图所示。
<center>![代价矩阵](http://img.blog.csdn.net/20170106225945454?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2hpbmUxOTkzMDgyMA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
代价矩阵

### 代价敏感学习方法


----------


基于以上代价矩阵的分析，代价敏感学习方法主要有以下三种实现方式

 1. 从学习模型出发，着眼于对某一具体学习方法的改造，使之能适应不平衡数据下的学习，研究者们针对不同的学习模型如感知机，支持向量机，决策 树，神经网络等分别提出了其代价敏感的版本。以代价敏感的决策树为例，可从三个方面对其进行改进以适应不平衡数据的学习，这三个方面分别是决策阈值的选择 方面、分裂标准的选择方面、剪枝方面，这三个方面中都可以将代价矩阵引入，具体实现算法可参考参考文献中的相关文章。
 2. 从贝叶斯风险理论出发，把代价敏感学习看成是分类结果的一种后处理，按照传统方法学习到一个模型，以实现损失最小为目标对结果进行调整，优化公式如下所示。此方法的优点在于它可以不依赖所用具体的分类器，但是缺点也很明显它要求分类器输出值为概率。
	 $$H(x) = \arg \min_{i}(\sum_{j\in \{+,-\}}P(j|xc(i,j))$$
 3. 从预处理的角度出发，将代价用于权重的调整，使得分类器满足代价敏感的特性，下面讲解一种基于Adaboost的权重更新策略。

### AdaCost算法
让我们先来简单回顾一下Adaboost算法，如下图6所示。Adaboost算法通过反复迭代，每一轮迭代学习到一个分类器，并根据当前分类器 的表现更新样本的权重，如图中红框所示，其更新策略为正确分类样本权重降低，错误分类样本权重加大，最终的模型是多次迭代模型的一个加权线性组合，分类越 准确的分类器将会获得越大的权重。

![Adaboost算法](http://img.blog.csdn.net/20170106230232930?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2hpbmUxOTkzMDgyMA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

AdaCost算法修改了Adaboost算法的权重更新策略，其基本思想是对于代价高的误分类样本大大地提高其权重，而对于代价高的正确分类样 本适当地降低其权重，使其权重降低相对较小。总体思想是代价高样本权重增加得大降低得慢。其样本权重按照如下公式进行更新。其 中$\beta_+$和$\beta_-$分别表示样本被正确和错误分类情况下$\beta$的取值。
$$\frac{D_{t+1}(i) = D_t(i)\exp (-\alpha_t h_t(x_i)y_i\beta_i)}{Z_t}$$
$$\beta_+ = -0.5C_i+0.5$$
$$\beta_- = 0.5C_i+0.5$$


# 不平衡学习的评价方法


----------


## 正确率和F值


----------
![这里写图片描述](http://img.blog.csdn.net/20170106230450040?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2hpbmUxOTkzMDgyMA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
Precidsion = tp/(tp+fn)
Recall =  tp/(tp+fn)
F-Measure = (1+β）²*Recall*Precision / (β²*Recall+Precision）
β取值一般为1；
Accuracy = (tp+tn)/(pc+nc);
ErrorRate = 1- accuracy

正确率和F值的计算都是基于混淆矩阵(Confusion Matrix)的，混淆矩阵如下图7所示，每行代表预测情况，每列代表实际类别，TP,FP,FN,TN分别代表正类正确分类数量，预测为正类但是真实为负类，预测为负类但是真实为正类，负类正确分类数量。

## G-Mean
G-Mean是另外一个指标，也能评价不平衡数据的模型表现，其计算公式如下。
$$G-mean = \sqrt {\frac{TP}{TP+FN}* \frac{TN}{TN+FP}}$$
下面将会介绍TP、TN等

## ROC曲线和AUC
为了介绍ROC曲线首先引入两个是，分别是FP_rate和TP_rate，它们分别表示1-负类召回率和正类召回率，显然模型表示最好的时候 FP_rate=0且TP_rate=1，我们以FP_rate为横坐标，TP_rate为纵坐标可以得到点(FP_rate,TP_rate)，通过调 整模型预测的阈值可以得到不同的点，将这些点可以连成一条曲线，这条曲线叫做接受者工作特征曲线(Receiver Operating Characteristic Curve，简称ROC曲线）如下图8所示。显然A点为最优点，ROC曲线越靠近A点代表模型表现越好，曲线下面积（Area Under Curve, AUC）越大，AUC是衡量模型表现好坏的一个重要指标。
<center>![ROC](http://img.blog.csdn.net/20170106232102016?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2hpbmUxOTkzMDgyMA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
ROC曲线


Ps：为啥每个公式后面会有一个竖线？？？

参考文献：[http://www.cs.utah.edu/~piyush/teaching/ImbalancedLearning.pdf](http://www.cs.utah.edu/~piyush/teaching/ImbalancedLearning.pdf)

python sklearn数据预处理：
[http://blog.csdn.net/shine19930820/article/details/50915361](http://blog.csdn.net/shine19930820/article/details/50915361)

