from optparse import OptionParser
import dill
import numpy as np
import json
import random
import pandas as pd

from .one_nn import OneNN
from .ensemble_binary_classifiers import EnsembleOfBinaryClassifiers
from .cascaded_discriminative_classifiers import CascadedDiscriminativeClassifiers
from .isotonic_regression import IsotonicRegression
from .scale import Scale
from .pca import PCA

CLASSIFIERS = {
    "onn": OneNN,
    "ind_one_vs_rest": EnsembleOfBinaryClassifiers,
    "cdc": CascadedDiscriminativeClassifiers,
    "isotonic_regression": IsotonicRegression,
}

PREPROCESSORS = {"scale": Scale, "pca": PCA}


class Model:
    def __init__(self, classifier, preprocessors=None):
        """
        初始化模型类
        Parameters:
            classifier: 执行监督分类的分类器对象
            preprocessors: 用于数据转换的预处理算法列表
        """
        if preprocessors is None:
            self.preprocessors = []
        else:
            self.preprocessors = preprocessors
        self.classifier = classifier

    def fit(
        self,
        train_X,
        train_items,
        item_to_labels,
        label_graph,
        item_to_group=None,
        verbose=False,
        features=None,
        model_dependency=None,
    ):
        """
        训练模型
        model.fit(
            train_X,            # cello 作者提供的表达量矩阵,其中过滤出了调用者提供的数据中的基因id
            train_items,        # experiment(cell) 索引 
            item_to_labels,     # exp_to_labels 实验样本ID列表, 就是数据集列表, 一条一条给模型
            label_graph,        # 细胞类型的层级关系图
            item_to_group=item_to_group,  # 实验样本到研究批次的分组信息
            verbose=verbose,    # 是否打印训练信息
            features=features,  # 调用者提供的训练的基因列表
            model_dependency=model_dependency,  # 模型依赖性信息
        )
        """
        for prep in self.preprocessors:
            prep.fit(train_X)
            train_X = prep.transform(train_X)
        self.classifier.fit(
            train_X,            # cello作者提供的表达量矩阵
            train_items,        # experiment(cell) 索引 
            item_to_labels,     # exp_to_labels 实验样本ID列表, 就是数据集列表, 一条一条给模型
            label_graph,        # 细胞类型的层级关系图
            item_to_group=item_to_group,  # 实验样本到研究批次的分组信息
            verbose=verbose,    # 是否打印训练信息
            features=features,  # 特征(基因)列表
            model_dependency=model_dependency,  # 模型依赖性信息
        )

    def _preprocess(self, X):
        """
        对输入数据应用预处理转换
        """
        if self.preprocessors is not None:
            for prep in self.preprocessors:
                X = prep.transform(X)
        return X

    def predict(self, X, test_items):
        """
        对新数据进行预测
        """
        X = self._preprocess(X)
        return self.classifier.predict(X, test_items)

    def decision_function(self, X, test_items):
        """
        计算决策函数值
        """
        if self.preprocessors is not None:
            for prep in self.preprocessors:
                X = prep.transform(X)
        return self.classifier.decision_function(X, test_items)

    def feature_weights(self):
        """
        获取模型的特征权重
        """
        label_to_weights = self.classifier.label_to_coefficients
        df = pd.DataFrame(data=label_to_weights, index=self.classifier.features)
        return df.transpose()


def train_model(
    classifier_name,
    params,
    train_X,
    train_items,
    item_to_labels,
    label_graph,
    preprocessor_names=None,
    preprocessor_params=None,
    verbose=False,
    item_to_group=None,
    tmp_dir=None,
    features=None,
    model_dependency=None,
):
    """
    创建并训练模型
        mod = model.train_model(
            ALGO_TO_INTERNAL[algo],     # 算法
            ALGO_TO_PARAMS[algo],       # 算法对应的配置
            X_train,                    # cello 作者提供的表达量矩阵,其中过滤出了调用者提供的数据中的基因id
            the_exps,               # experiment(cell) 索引
            exp_to_labels,              # experiment(cell) 到 细胞类型ID 的映射：一个实验样本, 对应多个细胞 label_id. 因为一个具体细胞类型可能属于很多细胞类型, 一个细胞类型可能包含多个具体细胞类型
            label_graph,            # 细胞 label_id 层级图
            item_to_group=exp_to_study,         # 实验样本到研究批次的分组信息
            features=train_genes,               # 调用者提供的训练的基因列表
            preprocessor_names=PREPROCESSORS,   # 预处理器名称列表
            preprocessor_params=PREPROCESSOR_PARAMS,  # 预处理器参数
        )
    """
    classifier = CLASSIFIERS[classifier_name](params)  # 默认IR
    preps = None
    if preprocessor_names:  # 初始化 PCA 降维函数,  后期可以直接使用fit调用该处理器
        assert preprocessor_params is not None
        assert len(preprocessor_params) == len(preprocessor_names)
        preps = [
            PREPROCESSORS[prep_name](prep_params)
            for prep_name, prep_params in zip(preprocessor_names, preprocessor_params)
        ]
    model = Model(classifier, preprocessors=preps)  # 初始化 Model 类
    model.fit(
        train_X,            # 表达量矩阵
        train_items,        # experiment(cell) 索引 
        item_to_labels,     # exp_to_labels 实验样本ID列表, 就是数据集列表, 一条一条给模型
        label_graph,        # 细胞类型的层级关系图
        item_to_group=item_to_group,  # 实验样本到研究批次的分组信息
        verbose=verbose,    # 是否打印训练信息
        features=features,  # 调用者提供的训练的基因列表
        model_dependency=model_dependency,  # 模型依赖性信息
    )
    return model


if __name__ == "__main__":
    main()
