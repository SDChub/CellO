#################################################################
#   Hierarchical classification via isotonic regression
#################################################################

# 通过保序回归实现层次分类

# 导入必要的库
import sys
import numpy as np
from quadprog import solve_qp  # 用于求解二次规划问题
import dill  # 用于模型序列化
import pandas as pd

# 导入自定义模块
from . import model
from .pca import PCA
from .ensemble_binary_classifiers import EnsembleOfBinaryClassifiers


# 保序回归分类器类定义
class IsotonicRegression:
    def __init__(self, params, trained_classifiers_f=None):
        # 初始化参数
        self.params = params

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
        model_dependency=None,
    ):
        """
        self.classifier.fit(
            train_X,            # 表达量矩阵
            train_items,        # experiment(cell) 索引 
            item_to_labels,     # exp_to_labels 实验样本ID列表, 就是数据集列表, 一条一条给模型
            label_graph,        # 细胞类型的层级关系图
            item_to_group=item_to_group,  # 实验样本到研究批次的分组信息
            verbose=verbose,    # 是否打印训练信息
            features=features,  # 调用者提供的训练的基因列表
            model_dependency=model_dependency,  # 模型依赖性信息
        )
        """

        # 保存调用者提供的训练的基因列表
        self.features = features

        # 处理预训练模型或从头训练
        if model_dependency is not None:
            # 加载预训练的模型
            with open(model_dependency, "rb") as f:
                self.ensemble = dill.load(f)
            # 验证预训练模型的兼容性
            assert _validate_pretrained_model(
                self.ensemble, train_items, label_graph, features
            )
            # 从预训练模型中获取特征和训练数据信息
            self.features = self.ensemble.classifier.features
            self.train_items = self.ensemble.classifier.train_items
            self.label_graph = self.ensemble.classifier.label_graph
        else:
            # 创建新的二分类器集成并训练
            # params如下: 
            #    "IR": {  # 保序回归算法参数
            #         "assert_ambig_neg": False,
            #         "binary_classifier_algorithm": "logistic_regression",
            #         "binary_classifier_params": {
            #             "penalty": "l2",
            #             "penalty_weight": 0.0006,
            #             "solver": "liblinear",
            #             "intercept_scaling": 1000.0,
            #             "downweight_by_class": True,
            #         }
            #     }
            self.ensemble = EnsembleOfBinaryClassifiers(self.params)
            self.ensemble.fit(
                X,  # cello 作者提供的表达量矩阵,其中过滤出了调用者提供的数据中的基因id
                train_items,        # experiment(cell) 索引 
                item_to_labels,     # exp_to_labels 实验样本ID列表, 就是数据集列表, 一条一条给模型
                label_graph,        # 细胞类型的层级关系图
                item_to_group=item_to_group,  # 实验样本到研究批次的分组信息
                verbose=verbose,    # 是否打印训练信息
                features=features,  # 调用者提供的训练的基因列表
            )
            # 保存训练数据信息
            self.features = features
            self.train_items = train_items
            self.label_graph = label_graph

    # 预测方法
    def predict(self, X, test_items):
        # 获取基础分类器的预测结果
        confidence_df, scores_df = self.ensemble.predict(X, test_items)
        labels_order = confidence_df.columns

        # 创建约束矩阵：确保预测结果满足层次关系
        constraints_matrix = []
        for row_label in labels_order:
            for constraint_label in self.label_graph.source_to_targets[row_label]:
                row = []
                for label in labels_order:
                    if label == row_label:
                        row.append(1.0)
                    elif label == constraint_label:
                        row.append(-1.0)
                    else:
                        row.append(0.0)
                constraints_matrix.append(row)
        b = np.zeros(len(constraints_matrix))
        constraints_matrix = np.array(constraints_matrix).T

        # 对每个样本进行保序回归优化
        pred_da = []
        for q_i in range(len(X)):
            # 设置二次规划问题的参数
            Q = np.eye(len(labels_order), len(labels_order))
            predictions = np.array(
                confidence_df[labels_order].iloc[q_i], dtype=np.double
            )
            print("Running solver on item {}/{}...".format(q_i + 1, len(X)))
            # 求解二次规划问题
            xf, f, xu, iters, lagr, iact = solve_qp(
                Q, predictions, constraints_matrix, b
            )
            pred_da.append(xf)

        # 将结果整理为DataFrame格式
        pred_df = pd.DataFrame(data=pred_da, columns=labels_order, index=test_items)
        return pred_df, confidence_df

    # 获取分类器系数的属性方法
    @property
    def label_to_coefficients(self):
        pca = None
        # 处理可能的PCA预处理
        if isinstance(self.ensemble, model.Model):
            # 查找PCA预处理器
            for prep in self.ensemble.preprocessors:
                if isinstance(prep, PCA):
                    pca = prep
            label_to_coefs = self.ensemble.classifier.label_to_coefficients
            # 如果没有PCA，直接返回系数
            if pca is None:
                return label_to_coefs
            else:
                # 如果有PCA，需要转换回原始特征空间
                components = pca.components_
                return {
                    label: np.matmul(components.T, coefs.T).squeeze()
                    for label, coefs in label_to_coefs.items()
                }
        else:
            raise Exception("This has not been implemented yet!")
        return {
            label: classif.coef_
            for label, classif in self.ensemble.classifier.label_to_classifier.items()
        }


# 验证预训练模型的辅助函数
def _validate_pretrained_model(ensemble, train_items, label_graph, features):
    # 检查标签图是否具有相同的标签集
    classif_labels = frozenset(ensemble.classifier.label_graph.get_all_nodes())
    curr_labels = frozenset(label_graph.get_all_nodes())
    if classif_labels != curr_labels:
        return False

    # 检查训练项是否相同
    classif_train_items = frozenset(ensemble.classifier.train_items)
    curr_train_items = frozenset(train_items)
    if classif_train_items != curr_train_items:
        return False

    # 检查特征是否相同
    if tuple(ensemble.classifier.features) != tuple(features):
        return False
    return True
