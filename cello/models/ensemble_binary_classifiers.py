#################################################################
#   Supervised hierarchical classification using a per-label
#   binary support vector machine. Variants of this algorithm
#   enforce label-graph consistency by propogating positive
#   predictions upward through the graph's 'is_a' relationship
#   edges, and propogates negative predictions downward.
#################################################################

# 导入必要的库
import sys
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import dill

# 导入二分类器模块
from . import binary_classifier as bc

# 定义常量
POS_CLASS = 1
VERBOSE = True

# 二分类器集成类定义
class EnsembleOfBinaryClassifiers(object):
    def __init__(
            self,
            params 
        ):
        # 初始化分类器参数
        self.binary_classif_algo = params['binary_classifier_algorithm']
        self.binary_classif_params = params['binary_classifier_params']
        self.assert_ambig_neg = params['assert_ambig_neg']
        # 检查是否使用组权重
        if 'group_weighted' in params:
            self.group_weighted = params['group_weighted']
        else:
            self.group_weighted = False

        # 每个标签的模型相关数据
        self.label_to_pos_items = None  # 标签到正样本的映射
        self.label_to_neg_items = None  # 标签到负样本的映射
        self.label_to_classifier = None # 标签到分类器的映射

        # 训练集中所有样本都有的标签
        self.trivial_labels = None

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
        参数说明:
            X: NxM训练集矩阵，N个样本，M个特征
            train_items: 项目标识符列表，对应X中的每一行
            item_to_labels: 字典，将每个项目标识符映射到label_graph中的标签集
            label_graph: 表示标签DAG的Graph对象
            item_to_group: 字典，将每个项目映射到其所属组
            verbose: 是否输出调试信息
        """
        # 保存训练数据
        self.train_items = train_items 
        self.label_graph = label_graph
        self.features = features

        # 创建项目到索引的映射
        item_to_index = {x:i for i,x in enumerate(self.train_items)}

        # 为每个标签创建项目集合
        label_to_items = defaultdict(lambda: set())
        for item in self.train_items:
            labels = item_to_labels[item]
            for label in labels:
                label_to_items[label].add(item)
        label_to_items = dict(label_to_items)

        # 为每个标签训练分类器
        self.label_to_pos_items = {}
        self.label_to_neg_items = {}
        self.label_to_classifier = {}

        # 处理每个标签
        self.trivial_labels = set()
        for label_i, label in enumerate(label_to_items.keys()):
            # 计算训练集
            pos_items, neg_items = self._compute_training_set(
                label,
                train_items,
                label_to_items,
                item_to_labels,
                label_graph
            )
            self.label_to_pos_items[label] = pos_items
            self.label_to_neg_items[label] = neg_items
            
            # 处理特殊情况和训练分类器
            if len(pos_items) > 0 and len(neg_items) == 0:
                print("Skipped training classifier for label {}. No negative examples.".format(label))
                self.trivial_labels.add(label)
            else:
                print('({}/{})'.format(label_i+1, len(label_to_items)))
                model = _train_classifier(
                    label,
                    self.binary_classif_algo, 
                    self.binary_classif_params,
                    pos_items, 
                    neg_items,
                    item_to_index,
                    X,
                    group_weighted=self.group_weighted,
                    item_to_group=item_to_group
                )
                self.label_to_classifier[label] = model

    # 计算训练集方法
    def _compute_training_set(
            self, 
            label,    
            train_items,
            label_to_items,
            item_to_labels,
            label_graph
        ):
        # 计算正样本
        pos_items = _compute_positive_examples(
            label,
            label_to_items,
            label_graph
        )
        # 计算负样本
        neg_items = _compute_negative_examples(
            label,
            train_items,
            pos_items,
            item_to_labels,
            label_graph,
            self.assert_ambig_neg
        )
        return pos_items, neg_items

    # 预测方法
    def predict(self, X, test_items, verbose=True):
        # 初始化预测结果存储
        label_to_scores = {}
        all_labels = sorted(self.label_to_classifier.keys())
        mat = []
        print('Making predictions for each classifier...')
        
        # 对每个标签进行预测
        for label in all_labels:
            classifier = self.label_to_classifier[label]
            pos_index = 0 
            for index, clss in enumerate(classifier.classes_):
                if clss == POS_CLASS:
                    pos_index = index
                    break
            scores = [
                x[pos_index]
                for x in classifier.predict_proba(X)
            ]
            mat.append(scores)
            
        # 处理特殊标签
        trivial_labels = sorted(self.trivial_labels)
        for label in trivial_labels:
            mat.append(list(np.full(len(test_items), 1.0)))

        # 整理预测结果为DataFrame格式
        all_labels += trivial_labels
        mat = np.array(mat).T
        df = pd.DataFrame(
            data=mat,
            index=test_items,
            columns=all_labels
        )
        return df, df

    # 获取分类器系数的属性
    @property
    def label_to_coefficients(self):
        return{
            label: classif.coef_
            for label, classif in self.label_to_classifier.items()
        }

# 训练单个分类器的辅助函数
def _train_classifier(
            label, 
            binary_classif_algo, 
            binary_classif_params, 
            pos_items, 
            neg_items, 
            item_to_index, 
            X, 
            group_weighted=False, 
            item_to_group=None
        ):
    # 准备训练数据
    pos_items = list(pos_items)
    neg_items = list(neg_items)
    if VERBOSE:
        print("Training classifier for label {}...".format(label))
        print("Number of positive items: {}".format(len(pos_items)))
        print("Number of negative items: {}".format(len(neg_items)))
    
    # 创建标签
    pos_classes = list(np.full(len(pos_items), 1))
    neg_classes = list(np.full(len(neg_items), -1))
    train_y = np.asarray(pos_classes + neg_classes)
    train_items = pos_items + neg_items

    # 准备特征矩阵
    pos_inds = [item_to_index[item] for item in pos_items]
    neg_inds = [item_to_index[item] for item in neg_items]
    pos_X = X[pos_inds,:]
    neg_X = X[neg_inds,:]
    train_X = np.concatenate([pos_X, neg_X])
    
    # 确保有正负样本
    assert len(pos_items) > 0 and len(neg_items) > 0
    
    # 构建分类器
    model = bc.build_binary_classifier(
        binary_classif_algo,
        binary_classif_params
    )
    
    # 使用组权重训练
    if group_weighted:
        assert item_to_group is not None
        # 限制item_to_group映射只包含训练集
        all_items = set(pos_items + neg_items)
        item_to_group = {
            item: group
            for item, group in item_to_group.items()
            if item in all_items
        }
        # 计算权重
        group_to_size = Counter(item_to_group.values())
        sample_weights = [
            1.0 / group_to_size[item_to_group[item]]
            for item in pos_items + neg_items
        ]
        print('Fitting with sample_weights')
        model.fit(train_X, train_y, sample_weights=sample_weights)
    else:
        model.fit(train_X, train_y)
    return model

# 计算正样本的辅助函数
def _compute_positive_examples(
        label,
        label_to_items,
        label_graph
    ):
    """
    计算给定标签的正样本。
    这个集合包含所有被标记为该标签后代的样本
    """
    positive_items = label_to_items[label].copy()
    desc_labels = label_graph.descendent_nodes(label)
    for desc_label in desc_labels:
        if desc_label in label_to_items:
            positive_items.update(label_to_items[desc_label])
    return list(positive_items)

# 计算负样本的辅助函数
def _compute_negative_examples(
        label, 
        all_items, 
        pos_items, 
        item_to_labels, 
        label_graph,
        assert_ambig_neg
    ):
    """
    计算给定标签的负样本集。
    这个集合包含所有未被标记为该标签后代，
    但同时也未被标记为该标签祖先的样本(可以包括兄弟节点)。
    """
    anc_labels = label_graph.ancestor_nodes(label)
    candidate_items = set(all_items) - set(pos_items)

    if assert_ambig_neg:
        neg_items = list(candidate_items)
    else:
        # 从负样本中移除那些最具体地标记为当前标签祖先的样本
        final_items = set()
        for item in candidate_items:
            ms_item_labels = label_graph.most_specific_nodes(
                item_to_labels[item]
            )
            ms_item_labels = set(ms_item_labels)
            if len(ms_item_labels & anc_labels) == 0:
                final_items.add(item)
        neg_items = list(final_items)
    return neg_items

