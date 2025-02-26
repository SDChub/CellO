#################################################################
#   TODO
#################################################################
import sys
from collections import defaultdict
import numpy as np
import math
import pandas as pd

from . import binary_classifier as bc

POS_CLASS = 1
NEG_CLASS = 0

class CascadedDiscriminativeClassifiers(object):
    def __init__(
            self, 
            params
        ):
        """
        Args:
            binary_classifier_type: 用作基础分类器的二分类器名称
            binary_classifier_params: 分类器参数字典，将参数名映射到参数值
        """
        # 初始化分类器参数
        self.binary_classif_algo = params['binary_classifier_algorithm']
        self.binary_classif_params = params['binary_classifier_params']
        self.assert_ambig_neg = params['assert_ambig_neg']

        # 每个标签的模型相关数据
        self.label_to_pos_items = None  # 标签到正样本的映射
        self.label_to_neg_items = None  # 标签到负样本的映射
        self.label_to_classifier = None # 标签到分类器的映射

        # 训练集中所有样本都有的标签
        self.trivial_labels = None

    def fit(
            self, 
            X, 
            train_items, 
            item_to_labels,  
            label_graph,
            item_to_group=None, 
            verbose=False,
            features=None,
            model_dependency=None # 未使用
        ):
        """
        Args:
            X: NxM特征矩阵，N是样本数，M是特征数
            train_items: 项目标识符列表，每个对应X中的特征向量
            item_to_labels: 字典，将每个项目标识符映射到label_graph中的标签集
            label_graph: 表示标签DAG的Graph对象
            item_to_group: 字典，将每个项目映射到其所属组
            verbose: 是否输出调试信息
        """
        # 保存训练数据
        self.train_items = train_items 
        self.item_to_labels = item_to_labels
        self.label_graph = label_graph
        self.features = features

        # 创建项目到索引的映射
        item_to_index = {
            x:i 
            for i,x in enumerate(self.train_items)
        }

        # 为每个标签创建项目集合
        self.label_to_items = defaultdict(lambda: set())
        for item in self.train_items:
            labels = self.item_to_labels[item]
            for label in labels:
                self.label_to_items[label].add(item)
        for label in label_graph.get_all_nodes():
            if label not in self.label_to_items:
                self.label_to_items[label] = set()
        self.label_to_items = dict(self.label_to_items)

        # 计算每个标签的训练集
        if self.assert_ambig_neg:
            # 使用断言模糊负样本的方法计算训练集
            label_to_pos_items, label_to_neg_items = _compute_training_sets_assert_ambiguous_negative(
                self.label_to_items,
                self.item_to_labels,
                self.label_graph
            )
        else:
            # 移除模糊样本的方法计算训练集
            label_to_pos_items, label_to_neg_items = _compute_training_sets_remove_ambiguous(
                self.label_to_items,
                self.item_to_labels,
                self.label_graph
            ) 
        self.label_to_pos_items = label_to_pos_items
        self.label_to_neg_items = label_to_neg_items
 
        # 在本体的每个节点上训练分类器
        self.trivial_labels = set()
        self.label_to_classifier = {}
        for label_i, curr_label in enumerate(self.label_to_items.keys()):
            # 获取正负样本
            pos_items = self.label_to_pos_items[curr_label]
            neg_items = self.label_to_neg_items[curr_label]
            pos_y = [POS_CLASS for x in pos_items]
            neg_y = [NEG_CLASS for x in neg_items]
            train_items = pos_items + neg_items
            train_y = np.asarray(pos_y + neg_y)
            
            # 提取特征
            pos_X = [
                X[
                    item_to_index[x]
                ]
                for x in pos_items
            ]
            neg_X = [
                X[
                    item_to_index[x]
                ]
                for x in neg_items
            ]
            train_X = np.asarray(pos_X + neg_X) 

            # 训练分类器
            if True:
                print("({}/{}) training classifier for label {}...".format(
                    label_i+1, 
                    len(self.label_to_items), 
                    curr_label
                ))
                print("Number of positive items: {}".format(len(pos_items)))
                print("Number of negative items: {}".format(len(neg_items)))
            
            # 处理特殊情况和训练分类器
            if len(pos_items) > 0 and len(neg_items) == 0:
                self.trivial_labels.add(curr_label)
            else:
                model = bc.build_binary_classifier(
                    self.binary_classif_algo,
                    self.binary_classif_params
                )
                model.fit(train_X, train_y)
                self.label_to_classifier[curr_label] = model

    def predict(self, X, test_items):
        # 运行所有分类器进行预测
        label_to_cond_log_probs = {}
        for label in self.label_graph.get_all_nodes():
            if label in self.label_to_classifier:
                classifier = self.label_to_classifier[label]
                pos_indices = [
                    i 
                    for i, classs in enumerate(classifier.classes_) 
                    if classs == POS_CLASS
                ]
                assert len(pos_indices) == 1
                pos_index = pos_indices[0]
                probs = [
                    x[pos_index] 
                    for x in classifier.predict_log_proba(X) 
                ]
                label_to_cond_log_probs[label] = probs
            else:
                if len(self.label_to_neg_items[label]) == 0.0:
                    label_to_cond_log_probs[label] = np.zeros(len(X))
                else:
                    raise Exception("No positive items for label %s" % label)

        # 计算每个标签的边际概率
        label_to_marginals = {}
        for label, log_probs in label_to_cond_log_probs.items():
            products = np.zeros(len(X))
            anc_labels = set(self.label_graph.ancestor_nodes(label)) - set([label])
            for anc_label in anc_labels:
                anc_probs = label_to_cond_log_probs[anc_label]
                products = np.add(products, anc_probs)
            products = np.add(products, log_probs)
            label_to_marginals[label] = np.exp(products)

        # 将结果组织成DataFrame格式
        confidence_df = pd.DataFrame(
            data=label_to_marginals,
            index=test_items
        )        
        scores_df = pd.DataFrame(
            data=label_to_cond_log_probs,
            index=test_items
        )
        sorted_labels = sorted(label_to_marginals.keys())
        confidence_df = confidence_df[sorted_labels]
        scores_df = scores_df[sorted_labels]
        return confidence_df, scores_df

def _compute_training_sets_assert_ambiguous_negative(
        label_to_items,
        item_to_labels,
        label_graph
    ):
    # 计算每个标签的正样本
    print("Computing positive labels...")
    label_to_pos_items = {}
    for curr_label in label_to_items:
        positive_items = label_to_items[curr_label].copy()
        desc_labels = label_graph.descendent_nodes(curr_label)
        for desc_label in desc_labels:
            if desc_label in label_to_items:
                positive_items.update(label_to_items[desc_label])
        label_to_pos_items[curr_label] = list(positive_items)

    # 计算每个标签的负样本
    print("Computing negative labels...")
    label_to_neg_items = {}
    for curr_label in label_to_items:
        negative_items = set()
        parent_labels = set(label_graph.target_to_sources[curr_label])
        for item, labels in item_to_labels.items():
             if frozenset(set(labels) & parent_labels) == frozenset(parent_labels):
                negative_items.add(item)
        negative_items -= set(label_to_pos_items[curr_label])
        label_to_neg_items[curr_label] = list(negative_items)
    return label_to_pos_items, label_to_neg_items
    
def _compute_training_sets_remove_ambiguous(
        label_to_items,
        item_to_labels,
        label_graph
    ):
    # 计算每个标签的正样本
    print("Computing positive labels...")
    label_to_pos_items = {}
    for curr_label in label_to_items:
        positive_items = label_to_items[curr_label].copy()
        desc_labels = label_graph.descendent_nodes(curr_label)
        for desc_label in desc_labels:
            if desc_label in label_to_items:
                positive_items.update(label_to_items[desc_label])
        label_to_pos_items[curr_label] = list(positive_items)

    # 计算每个标签的负样本
    print("Computing negative labels...")
    label_to_neg_items = {}
    for curr_label in label_to_items:
        negative_items = set()
        parent_labels = set(label_graph.target_to_sources[curr_label])
        for item, labels in item_to_labels.items():
             if frozenset(set(labels) & parent_labels) == frozenset(parent_labels):
                negative_items.add(item)
        negative_items -= set(label_to_pos_items[curr_label])

        # 计算当前标签的"模糊"样本
        ambig_items = set()
        for item in negative_items:
            item_ms_labels = label_graph.most_specific_nodes(item_to_labels[item]) 
            if parent_labels <= item_ms_labels:
                ambig_items.add(item)
        negative_items -= ambig_items

        label_to_neg_items[curr_label] = list(negative_items)
    return label_to_pos_items, label_to_neg_items

