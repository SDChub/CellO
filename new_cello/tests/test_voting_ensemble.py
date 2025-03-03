"""
投票集成模型测试
"""

import os
import numpy as np
import pandas as pd
import pytest
from typing import Dict, List, Tuple, Any

from new_cello.models.ensemble.voting import VotingEnsembleModel
from new_cello.models.traditional.random_forest import RandomForestModel

class MockModel:
    """用于测试的模拟模型"""
    
    def __init__(self, return_value=None, return_proba=None):
        self.return_value = return_value
        self.return_proba = return_proba
        self.fit_called = False
        self.predict_called = False
        self.predict_proba_called = False
    
    def fit(self, X, y, **kwargs):
        self.fit_called = True
        return self
    
    def predict(self, X):
        self.predict_called = True
        if self.return_value is not None:
            return self.return_value
        return np.zeros(X.shape[0])
    
    def predict_proba(self, X):
        self.predict_proba_called = True
        if self.return_proba is not None:
            return self.return_proba
        return np.zeros((X.shape[0], 2))

class TestVotingEnsembleModel:
    """投票集成模型测试类"""
    
    @pytest.fixture
    def sample_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建样本数据用于测试
        
        Returns:
            Tuple: (特征矩阵, 标签向量)
        """
        # 创建随机特征矩阵 [n_samples, n_features]
        n_samples = 100
        n_features = 20
        X = np.random.rand(n_samples, n_features)
        
        # 创建随机标签 [n_samples]
        n_classes = 3
        y = np.random.randint(0, n_classes, size=n_samples)
        
        return X, y
    
    def test_init(self):
        """测试模型初始化"""
        # 创建空模型
        model = VotingEnsembleModel()
        assert model.models == []
        assert model.voting == 'hard'
        assert model.weights is None
        
        # 创建带模型的实例
        models = [MockModel(), MockModel()]
        weights = [0.7, 0.3]
        model = VotingEnsembleModel(models=models, voting='soft', weights=weights)
        assert model.models == models
        assert model.voting == 'soft'
        assert model.weights == weights
        
        # 测试权重验证
        with pytest.raises(ValueError):
            VotingEnsembleModel(models=models, weights=[0.7])  # 权重数量不匹配
        
        # 测试投票类型验证
        with pytest.raises(ValueError):
            VotingEnsembleModel(voting='invalid')  # 无效的投票类型
    
    def test_add_model(self):
        """测试添加模型"""
        model = VotingEnsembleModel()
        
        # 添加第一个模型
        mock1 = MockModel()
        model.add_model(mock1)
        assert len(model.models) == 1
        assert model.models[0] == mock1
        assert model.weights == [1.0]
        
        # 添加第二个模型，带权重
        mock2 = MockModel()
        model.add_model(mock2, weight=0.5)
        assert len(model.models) == 2
        assert model.models[1] == mock2
        assert model.weights == [1.0, 0.5]
    
    def test_fit(self, sample_data):
        """测试模型训练"""
        X, y = sample_data
        
        # 创建模拟模型
        mock1 = MockModel()
        mock2 = MockModel()
        
        # 创建集成模型
        model = VotingEnsembleModel(models=[mock1, mock2])
        
        # 训练模型
        model.fit(X, y)
        
        # 检查是否调用了所有模型的fit方法
        assert mock1.fit_called
        assert mock2.fit_called
        
        # 测试空模型列表
        empty_model = VotingEnsembleModel()
        with pytest.raises(ValueError):
            empty_model.fit(X, y)
    
    def test_hard_voting(self, sample_data):
        """测试硬投票"""
        X, _ = sample_data
        
        # 创建返回不同预测的模拟模型
        mock1 = MockModel(return_value=np.zeros(X.shape[0]))
        mock2 = MockModel(return_value=np.ones(X.shape[0]))
        mock3 = MockModel(return_value=np.ones(X.shape[0]))
        
        # 创建硬投票集成模型
        model = VotingEnsembleModel(models=[mock1, mock2, mock3], voting='hard')
        
        # 预测
        predictions = model.predict(X)
        
        # 检查预测结果（多数投票应该是1）
        assert np.all(predictions == 1)
        
        # 检查是否调用了所有模型的predict方法
        assert mock1.predict_called
        assert mock2.predict_called
        assert mock3.predict_called
    
    def test_soft_voting(self, sample_data):
        """测试软投票"""
        X, _ = sample_data
        n_samples = X.shape[0]
        
        # 创建返回不同概率的模拟模型
        proba1 = np.zeros((n_samples, 3))
        proba1[:, 0] = 0.7  # 类别0的概率高
        proba1[:, 1] = 0.2
        proba1[:, 2] = 0.1
        
        proba2 = np.zeros((n_samples, 3))
        proba2[:, 0] = 0.3
        proba2[:, 1] = 0.6  # 类别1的概率高
        proba2[:, 2] = 0.1
        
        mock1 = MockModel(return_proba=proba1)
        mock2 = MockModel(return_proba=proba2)
        
        # 创建软投票集成模型，权重相等
        model = VotingEnsembleModel(models=[mock1, mock2], voting='soft')
        
        # 预测
        predictions = model.predict(X)
        
        # 检查预测结果（平均概率应该是类别0）
        # (0.7 + 0.3)/2 = 0.5 for class 0, (0.2 + 0.6)/2 = 0.4 for class 1
        assert np.all(predictions == 0)
        
        # 创建软投票集成模型，权重不等
        model = VotingEnsembleModel(models=[mock1, mock2], voting='soft', weights=[0.3, 0.7])
        
        # 预测
        predictions = model.predict(X)
        
        # 检查预测结果（加权平均概率应该是类别1）
        # 0.7*0.3 + 0.3*0.7 = 0.42 for class 0, 0.2*0.3 + 0.6*0.7 = 0.48 for class 1
        assert np.all(predictions == 1)
    
    def test_predict_proba(self, sample_data):
        """测试概率预测"""
        X, _ = sample_data
        n_samples = X.shape[0]
        
        # 创建返回不同概率的模拟模型
        proba1 = np.zeros((n_samples, 3))
        proba1[:, 0] = 0.7
        proba1[:, 1] = 0.2
        proba1[:, 2] = 0.1
        
        proba2 = np.zeros((n_samples, 3))
        proba2[:, 0] = 0.3
        proba2[:, 1] = 0.6
        proba2[:, 2] = 0.1
        
        mock1 = MockModel(return_proba=proba1)
        mock2 = MockModel(return_proba=proba2)
        
        # 创建集成模型
        model = VotingEnsembleModel(models=[mock1, mock2])
        
        # 预测概率
        probas = model.predict_proba(X)
        
        # 检查概率形状
        assert probas.shape == (n_samples, 3)
        
        # 检查概率和为1
        assert np.allclose(np.sum(probas, axis=1), 1.0)
        
        # 检查平均概率
        expected_probas = (proba1 + proba2) / 2
        assert np.allclose(probas, expected_probas)
    
    def test_real_models(self, sample_data):
        """测试真实模型集成"""
        X, y = sample_data
        
        # 创建两个随机森林模型
        rf1 = RandomForestModel(n_estimators=10, random_state=42)
        rf2 = RandomForestModel(n_estimators=20, random_state=43)
        
        # 创建集成模型
        model = VotingEnsembleModel(models=[rf1, rf2])
        
        # 训练模型
        model.fit(X, y)
        
        # 预测
        predictions = model.predict(X)
        
        # 检查预测结果
        assert predictions.shape == (X.shape[0],)
        
        # 预测概率
        probas = model.predict_proba(X)
        
        # 检查概率
        assert probas.shape == (X.shape[0], len(np.unique(y)))
        assert np.allclose(np.sum(probas, axis=1), 1.0)
    
    def test_save_load(self, sample_data, tmp_path):
        """测试模型保存和加载"""
        X, y = sample_data
        
        # 创建两个随机森林模型
        rf1 = RandomForestModel(n_estimators=10, random_state=42)
        rf2 = RandomForestModel(n_estimators=20, random_state=43)
        
        # 训练模型
        rf1.fit(X, y)
        rf2.fit(X, y)
        
        # 创建集成模型
        model = VotingEnsembleModel(models=[rf1, rf2], weights=[0.7, 0.3])
        
        # 获取原始预测
        original_pred = model.predict(X)
        
        # 保存模型
        save_dir = os.path.join(tmp_path, "model")
        model.save(save_dir)
        
        # 检查保存的文件
        assert os.path.exists(os.path.join(save_dir, "config.pkl"))
        assert os.path.exists(os.path.join(save_dir, "models"))
        assert os.path.exists(os.path.join(save_dir, "models", "model_0"))
        assert os.path.exists(os.path.join(save_dir, "models", "model_1"))
        
        # 创建模型加载器
        def load_rf(model_dir):
            return RandomForestModel.load(model_dir)
        
        # 加载模型
        loaded_model = VotingEnsembleModel.load(save_dir, model_loaders=[load_rf, load_rf])
        
        # 检查加载的模型参数
        assert loaded_model.voting == model.voting
        assert loaded_model.weights == model.weights
        assert len(loaded_model.models) == len(model.models)
        
        # 检查加载的模型预测
        loaded_pred = loaded_model.predict(X)
        assert np.array_equal(original_pred, loaded_pred) 