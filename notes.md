# Hands on Machine Learning with Scikit-learn and TensorFlow

标签（空格分隔）： sklearn ml

---

## Part I, The Fundamentals of Machine Learning

### Chapter 1. The Machine Learning Lanscape

 - What is Machine Learning（什么是ML）
 - Why use Machine Learning（为什么使用ML）
 - Types of Machine Learning Systems（ML系统的类型）
    - Supervised/Unsupervised Learning（监督/无监督学习）
        - supervised learning（监督学习-- training data is labeled）:classification（分类） & regression（回归）
            -  k-Nearest Neighbors
            -  Linear Regression
            -  Logistic Regression
            -  Support Vector Machines (SVMs)
            -  Decision Trees and Random Forests
            -  Neural networks2
        - unsupervised learning（无监督学习 -- training data is unlabeled）:Clustering（聚类） & Visualization and dimensionality reduction（可视化和降维） & Association rule learning（关联规则）
            - Clustering（k-Means,  Hierarchical Cluster Analysis (HCA), Expectation Maximization）
            - Visualization and dimensionality reduction(Principal Component Analysis (PCA), Kernel PCA, Locally-Linear Embedding (LLE),  t-distributed Stochastic Neighbor Embedding (t-SNE))
            -  Association rule learning（Apriori, Eclat）
        - semisupervised learning（半监督学习）:a lot of unlabeled data and a little bit of labeled data.
        - Reinforcement Learning（强化学习）:agent observe the environment,select and perform actions,and get rewards in returns.
    - Batch and Online Learning（批量和在线学习）
        - batch learning（批量学习）:不能逐步学习，只能一次性训练全部可用的数据，通常是离线完成。
        - online learning（在线学习）：序列化接收数据训练，高效，廉价，
    - Instance-Based Versus Model-Based Learning（基于实例和基于模型的学习）
        - Instance-based learning（基于实例学习）：判定样本之间的相似程度，类似于 kNN 。
        - Model-based learning（基于模型学习）：根据已有数据集创建模型，然后在新数据集上进行预测。

#### example 1-1 是不是钱（GDP）能够使人们幸福？（单变量回归）

```python
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
# Load the data
oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("gdp_per_capita.csv",thousands=',',delimiter='\t',
 encoding='latin1', na_values="n/a")
# Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]
# Visualize the data
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.show()
# Select a linear model
lin_reg_model = sklearn.linear_model.LinearRegression()
# Train the model
lin_reg_model.fit(X, y)
# Make a prediction for Cyprus
X_new = [[22587]] # Cyprus' GDP per capita
print(lin_reg_model.predict(X_new)) # outputs [[ 5.96242338]]
```

如果想要使用 基于实例学习的算法，可以使用下面的算法：

```python
clf = sklearn.linear_model.LinearRegression()
替换为：
clf = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
```

 - ML 的主要挑战
    - 训练数据量不足
    - 数据的有效性不合理
    - 没有代表性的训练数据
    - 质量很差的数据
    - 不相关的 features
    - 过拟合训练数据集
    - 欠拟合训练数据集

 - Testing and Validating（测试和验证）
 - Exercise（练习题）


### Chapter 2. End-to-End Machine Learning Project

several open data repositories:

 1. Popular open data repositories:
    - UC Irvine Machine Learning Repository(http://archive.ics.uci.edu/ml/index.php)
    - Kaggle datasets(https://www.kaggle.com/datasets)
    - Amazon's AWS datasets(https://aws.amazon.com/fr/datasets/)
 2. Meta portals (they list open data repositories):
    - http://dataportals.org/
    - http://opendatamonitor.eu/
    - http://quandl.com/
 3. Other pages listing many popular open data repositories:
    - Wikipedia’s list of Machine Learning datase(https://goo.gl/SJHN2k)
    - Quora.com question(http://goo.gl/zDR78y)
    - Datasets subredd(https://www.reddit.com/r/datasets/)

main steps you will go through：

#### 2.1、Look at the big picture（从大局考虑）

 - Frame the problem（解决什么问题）
 - Select a Performance Measure（选择一个性能指标）
 - Check the Assumptions（验证我们的假设）

#### 2.2、Get the data

 - Create the Workspace（创建工作区间）
 - Download the Data（下载数据）
 - take a quick look at the Data Structure（简略看一下数据结构）
 - Create a Test Set（创建一个测试数据集）

#### 2.3、Discover and visualize the data to gain insights（发现并可视化数据以获取灵感）

 - 可视化数据中的特征
 - 寻找相关性
 - 试着组合 features 进行实验，观察效果

#### 2.4、Prepare the data for Machine Learning algorithms

 - Data Cleaning
    - 处理缺失值（可以使用 dropna(), drop(), fillna() 方法，sklearn 提供了一个 Imputer 类来方便处理缺失值）
        - drop 相关字段
        - drop 整个 feature
        - 将缺失值用其他值来填充（0,平均值，中位数等）
 - Handling Text and Categorical Attribut（处理文本和类别数据）
    - 使用 sklearn 提供的 LabelEncoder ,将 text 文件转换成为对应的数字类别。
    - 使用 sklearn 提供的 OneHotEncoder ,将类别数据转换成为对应的 N 维数组（只包含 0/1）
    - 使用 sklearn 提供的 LabelBinarizer ,将标签进行转换为二值化（如 yes/no 转化为 1/0）
 - Custom Transformers（自定义转换）
    - 只需要创建一个类，并实现三种方法：
        - fit() 返回 self
        - transform()
        - fit_transform()，这一个方法只需要简单添加 TransformerMixin 作为基类就可以。
        - 如果将 BaseEstimator 作为基类，将获得两个额外的方法（get_params(） 和 set_params()）
 - Feature Scaling（特征缩放）
    - min-max scaling（减去最小值然后除以最大值减去最小值的差值），大多数人叫它 - normalization（正则化），sklearn中是MinMaxScaler 。
    - standardization（标准化）：减去方差，除以方差使得结果分布有单位方差。sklearn中是 StandardScaler 。
 - Transformation Pipelines（转换流水线）
    - sklearn 提供了 Pipeline 类来实现对应的序列化转换。
    - sklearn 中使用 Pipeline 的一个例子

#### 2.5、Select a model and train it

 - Training and Evaluating on the Training Set（在训练数据集上训练并进行评估）
    - 在房价数据集上训练一个 Linear Regression model 的例子
 - Better Evaluation Using Cross-Validation（使用交叉验证进行更好的评估）
    - sklearn 提供了 K-fold cross-validation

#### 2.6、Fine-tune your model（调试好你的模型）

 - Grid Search：使用 sklearn 中的 GridSearchCV 来帮助你，你只需要告诉它你需要实验哪个超参数和这个超参数的测试值。
 - Randomized Search：使用 sklearn 中的 RandomizedSearchCV ，当我们需要 search 的 space 范围超大的时候，可以使用 RandomizedSearchCV ，只需要设置 迭代次数就可以了。
 - Ensemble Methods（集成方法）：比如 RF 比单个 decision tree 效果更好
 - Analyze the Best Models and Their Errors（分析最好模型和误差）：比如  RandomForestRegressor 可以列出每个 feature 的相对重要性，我们根据这些重要性，可以选择 drop 哪些重要性不高的 features 。
 - Evaluate Your System on the Test Set（在测试数据集上进行评估你的系统）：在测试数据集上进行评估模型的准确度。

#### 2.7、Launch, monitor, and maintain your system（启动，监视和维护你的系统）

#### 2.8、Try it Out！（小结）

这个章节已经告诉你，一个 ML project 是什么样的，并且展示了好几种比较好用的 tools 。但是会发现，大多数的任务还是在 data preparation 步骤，创建 监控工具，设置人为的评估 pipeline，并自动化常规模型训练。

#### 2.9、Exercise（训练题）

### Chapter 3、Classification（分类）

#### 3.1、MNIST

在这一章节中，我们将使用 MNIST 的数据集，其中包含了 70000 张手写数字的小图片。每个图片都已经有了 label，代表它是哪个数字。这个数据集被很广泛的使用以至于它被称为机器学习的 "Hello World" 。

#### 3.2、Training a Binary Classifier（训练一个二分分类器）

二分分类器，比如，我们要训练一个区分数字 5 的分类器，这个分类器的工作就是区分：是 5 还是不是 5 。

我们首先使用 Stochastic Gradient Descent (SGD) classifier, 使用的是 sklearn 中的 SGDClassifier 类。

#### 3.3、Performance Measures（性能度量）

 1. Measuring Accuracy Using Cross-Validation（使用交叉验证验证测量准确性）
    - Implementing Cross-Validati（实现交叉验证）：使用 sklearn 的 StratifiedKFold 。如：from sklearn.model_selection import StratifiedKFold
 2. Confusion Matrix（混淆矩阵）：总的思路是要看 数字5 被错分成 3 的次数，就查看混淆矩阵的 5行3列就好了。调用如： from sklearn.metrics import confusion_matrix
 3. Precision and Recall（精确度和召回）：如： from sklearn.metrics import precision_score, recall_score
 4. Precision/Recall Tradeof（精确度/召回 权衡）：提高精确度会降低召回率，而提高召回率会降低精确度。
 5. The ROC Curve（ROC 曲线）：receiver operating characteristic (ROC) curve 与二分分类器一起使用的另一种常用的工具。

#### 3.4、Multiclass Classification

一些算法是针对二分分类的（如：SVM 和一些线性分类器），而另外一些是可以处理多分类的问题的（如：随机森林分类器和朴素贝叶斯分类器）。

 - one-versus-all(OvA) strategy(也叫 one-versus-the-rest)。创建专门针对一个的分类器，比如0-9的分类器，我们创建 10 个分类器。每个分类器针对特定的数字识别。
 - one-versus-one（OvO）strategy。创建 N X (N-1)/2 个分类器，比如0-9的分类器，训练0-1,0-2,1-2 等分类器。

#### 3.5、Error Analysis

有很多种方法，一种推荐的方法是查看 confusion matrix ，首先使用 cross_val_predict() 函数，然后调用 confusion_matrix() 函数。

#### 3.6、Multilabel classification

有时候我们希望我们的分类器能够为每个样本输出多个 labels。比如 人脸识别的分类器，如果在一个图片中有多个人就会出现这样的结果。

sklearn 中比较常用的是 KNN 算法。

#### 3.7、Multioutput Classification

全称是 multioutput-multiclass classification ，是多标签分类的概括，其中每个标签可以是多类的（即它可以有两个以上的可能值）。

例如，我们创建一个从图像中去除噪声的系统。它将输入一个具有多噪声的数字图像，然后输出一个干净的数字图像，表示为一个像素强度的数组，就像MNIST图像一样。 请注意，分类器的输出是多标签（每个像素一个标签），每个标签可以有多个值（像素强度范围从0到255）。 因此，这是一个多输出分类系统的例子。

#### 3.8、Exercise


## Part II, Neural Networks and Deep Learning




