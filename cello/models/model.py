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
    'onn': OneNN,
    'ind_one_vs_rest': EnsembleOfBinaryClassifiers,
    'cdc': CascadedDiscriminativeClassifiers,
    'isotonic_regression': IsotonicRegression
}

PREPROCESSORS = {
    'scale': Scale,
    'pca': PCA
}

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
            model_dependency=None
        ):
        """
        训练模型
        Parameters: 
            train_X: NxM的训练数据矩阵，N个样本，M个特征
            train_items: N个样本标识符列表
            item_to_labels: 样本到标签的映射字典
            label_graph: 标签有向无环图
            features: M个特征名称列表
        """
        for prep in self.preprocessors:
            prep.fit(train_X)
            train_X = prep.transform(
                train_X
            )
        self.classifier.fit(
            train_X,
            train_items,
            item_to_labels,
            label_graph,
            item_to_group=item_to_group,
            verbose=verbose,
            features=features,
            model_dependency=model_dependency
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
        df = pd.DataFrame(
            data=label_to_weights,
            index=self.classifier.features
        )
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
        model_dependency=None
    ):
    """
    创建并训练模型
    Parameters:
        classifier_name: 机器学习算法的字符串标识
        params: 算法参数字典
        train_X: 训练特征向量
        train_items: 特征向量对应的样本标识符列表
        item_to_labels: 样本到标签的映射字典
        label_graph: 标签DAG中标签到邻居的映射字典
        verbose: 是否输出调试信息
        tmp_dir: 算法中间文件的存储目录
    """
    classifier = CLASSIFIERS[classifier_name](params)
    preps = None
    if preprocessor_names:
        assert preprocessor_params is not None
        assert len(preprocessor_params) == len(preprocessor_names)
        preps = [
            PREPROCESSORS[prep_name](prep_params)
            for prep_name, prep_params in zip(preprocessor_names, preprocessor_params)
        ]
    model = Model(
        classifier,
        preprocessors=preps
    )
    model.fit(
        train_X,
        train_items,
        item_to_labels,
        label_graph,
        item_to_group=item_to_group,
        verbose=verbose,
        features=features,
        model_dependency=model_dependency
    )
    return model


if __name__ == "__main__":
    main()
