#################################################################
#   One-nearest neighbor classifier
#################################################################

# 导入必要的库
from optparse import OptionParser
import sklearn
from sklearn.neighbors import NearestNeighbors
import scipy 
from scipy.stats import entropy
import numpy as np
import math

# 导入自定义工具模块
from . import model_utils

# 主函数定义：用于测试
def main():
    # 测试数据：概率分布示例
    a = [0.3, 0.2, 0.5]
    b = [0.4, 0.1, 0.5]

    # 测试特征向量
    feat_vecs = [
        [1,1,1,2,3],
        [10,23,1,24,32],
        [543,21,23,2,5]
    ]
    # 测试样本标识符
    items = [
        'a',
        'b',
        'c'
    ]
    # 测试样本标签映射
    item_to_labels = {
        'a':['hepatocyte', 'disease'],
        'b':['T-cell'],
        'c':['stem cell', 'cultured cell']
    }
    # 创建并测试模型
    model = OneNN('correlation')
    model.fit(feat_vecs, items, item_to_labels)
    print(model.predict([[10,10,10,20,30]]))

# Jensen-Shannon距离计算函数
def jensen_shannon(a, b):
    """
    计算两个概率分布之间的Jensen-Shannon距离
    """
    # 计算两个分布的中点
    m = [
        0.5 * (a[i] + b[i])
        for i in range(len(a))
    ]
    # 返回JS距离
    return math.sqrt(0.5 * (entropy(a, m) + entropy(b, m)))

# 一近邻分类器类定义
class OneNN:
    def __init__(self, params):
        """
        初始化一近邻分类器
        params: 包含距离度量方法的参数字典
        """
        # 选择距离度量方法
        metric = params['metric']
        if metric == 'correlation':
            self.metric_func = scipy.spatial.distance.correlation
        elif metric == 'euclidean':
            self.metric_func = scipy.spatial.distance.euclidean
        elif metric == 'jensen_shannon':
            self.metric_func = jensen_shannon
        self.metric = metric
        self.model = None
        self.item_to_labels = None

    # 模型训练方法
    def fit(
            self, 
            X, 
            train_items, 
            item_to_labels, 
            label_graph,
            item_to_group=None,
            verbose=False,
            features=None,
            model_dependency=None # 未使用的参数
        ):
        """
        训练一近邻分类器
        X: 训练数据特征矩阵
        train_items: 训练样本标识符
        item_to_labels: 样本到标签的映射
        label_graph: 标签图结构
        """
        # 保存训练数据信息
        self.items = train_items
        self.item_to_labels = item_to_labels
        self.label_graph = label_graph
        # 创建最近邻模型
        self.model = NearestNeighbors(
            metric=self.metric_func
        )
        self.model.fit(X)
        # 收集所有训练标签
        self.training_labels = set()
        for labels in self.item_to_labels.values():
            self.training_labels.update(labels)
        self.features = features

    # 预测方法
    def predict(self, X, test_items):
        """
        对新样本进行预测
        X: 测试数据特征矩阵
        test_items: 测试样本标识符
        """
        print('Finding nearest neighbors for {} samples...'.format(len(X)))
        # 找到最近邻
        neighb_dists, neighb_sets = self.model.kneighbors(
            X, 
            n_neighbors=1, 
            return_distance=True
        )
        print('done.')
        # 提取距离和邻居索引
        dists =[x[0] for x in neighb_dists]
        neighbs = [x[0] for x in neighb_sets]
        pred_labels = []

        # 计算每个样本的标签置信度
        label_to_conf_list = []
        for dist, neighb in zip(dists, neighbs):
            label_to_conf = {}
            item = self.items[neighb]
            neighb_labels = self.item_to_labels[item]
            # 为每个标签分配置信度
            for label in self.training_labels:
                if label in neighb_labels:
                    label_to_conf[label] = -1.0 * dist  # 距离越小，置信度越高
                else:
                    label_to_conf[label] = float('-inf')  # 不存在的标签置信度为负无穷
            label_to_conf_list.append(
                label_to_conf
            )
        # 转换为矩阵格式
        conf_matrix = model_utils.convert_to_matrix(
            label_to_conf_list,
            test_items
        )
        return conf_matrix, conf_matrix 

# 程序入口点            
if __name__ == "__main__":
    main()
