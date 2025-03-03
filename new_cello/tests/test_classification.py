"""
核心分类模块测试
"""

import os
import numpy as np
import pandas as pd
import pytest
from typing import Dict, List, Tuple, Any

from new_cello.core.classification import classify, train_model, evaluate_model
from new_cello.models.traditional.random_forest import RandomForestModel

class MockModel:
    """用于测试的模拟模型"""
    
    def __init__(self, return_value=None, return_proba=None):
        self.return_value = return_value
        self.return_proba = return_proba
        self.fit_called = False
        self.predict_called = False
        self.predict_proba_called = False
        self.saved = False
    
    def fit(self, X, y, **kwargs):
        self.fit_called = True
        return self
    
    def predict(self, X, gene_names=None):
        self.predict_called = True
        if self.return_value is not None:
            return self.return_value
        return np.zeros(X.shape[0])
    
    def predict_proba(self, X, gene_names=None):
        self.predict_proba_called = True
        if self.return_proba is not None:
            return self.return_proba
        return np.zeros((X.shape[0], 2))
    
    def save(self, save_dir):
        self.saved = True
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "mock_model.txt"), 'w') as f:
            f.write("Mock model saved")

# 模拟get_model函数
def mock_get_model(model_type, **kwargs):
    """模拟get_model函数"""
    return MockModel()

# 替换classify和train_model中的get_model函数
import new_cello.core.classification
original_get_model = new_cello.core.classification.get_model

@pytest.fixture(autouse=True)
def setup_and_teardown():
    """设置和清理"""
    # 替换get_model函数
    new_cello.core.classification.get_model = mock_get_model
    
    yield
    
    # 恢复原始get_model函数
    new_cello.core.classification.get_model = original_get_model

class TestClassification:
    """分类模块测试类"""
    
    @pytest.fixture
    def sample_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        创建样本数据用于测试
        
        Returns:
            Tuple: (特征矩阵, 标签向量, 基因名称列表)
        """
        # 创建随机特征矩阵 [n_samples, n_features]
        n_samples = 50
        n_features = 100
        X = np.random.rand(n_samples, n_features)
        
        # 创建随机标签 [n_samples]
        n_classes = 3
        y = np.random.randint(0, n_classes, size=n_samples)
        
        # 创建基因名称列表
        gene_names = [f"gene_{i}" for i in range(n_features)]
        
        return X, y, gene_names
    
    def test_classify(self, sample_data, monkeypatch):
        """测试分类函数"""
        X, _, gene_names = sample_data
        
        # 设置模拟返回值
        expected_predictions = np.ones(X.shape[0])
        expected_probabilities = np.random.rand(X.shape[0], 3)
        expected_probabilities = expected_probabilities / expected_probabilities.sum(axis=1, keepdims=True)
        
        # 创建模拟模型
        mock_model = MockModel(return_value=expected_predictions, return_proba=expected_probabilities)
        
        # 替换_load_model函数
        def mock_load_model(*args, **kwargs):
            return mock_model
        
        monkeypatch.setattr(new_cello.core.classification, '_load_model', mock_load_model)
        
        # 测试从目录加载模型
        predictions, probabilities = classify(
            X,
            gene_names=gene_names,
            model_type="traditional",
            model_name="random_forest",
            model_dir="/mock/model/dir",
            use_gpu=False
        )
        
        # 检查结果
        assert np.array_equal(predictions, expected_predictions)
        assert np.array_equal(probabilities, expected_probabilities)
        assert mock_model.predict_called
        assert mock_model.predict_proba_called
        
        # 测试创建新模型
        mock_model = MockModel(return_value=expected_predictions, return_proba=expected_probabilities)
        monkeypatch.setattr(new_cello.core.classification, 'get_model', lambda *args, **kwargs: mock_model)
        
        predictions, probabilities = classify(
            X,
            gene_names=gene_names,
            model_type="traditional",
            model_name="random_forest",
            use_gpu=False
        )
        
        # 检查结果
        assert np.array_equal(predictions, expected_predictions)
        assert np.array_equal(probabilities, expected_probabilities)
        assert mock_model.predict_called
        assert mock_model.predict_proba_called
    
    def test_train_model(self, sample_data, tmp_path):
        """测试训练模型函数"""
        X, y, gene_names = sample_data
        
        # 训练模型
        model = train_model(
            X,
            y,
            gene_names=gene_names,
            model_type="traditional",
            model_name="random_forest",
            use_gpu=False,
            save_dir=str(tmp_path)
        )
        
        # 检查模型
        assert isinstance(model, MockModel)
        assert model.fit_called
        assert model.saved
        
        # 检查保存的文件
        assert os.path.exists(os.path.join(tmp_path, "mock_model.txt"))
    
    def test_evaluate_model(self, sample_data):
        """测试评估模型函数"""
        X, y, gene_names = sample_data
        
        # 创建模型
        model = RandomForestModel(n_estimators=10, random_state=42)
        
        # 训练模型
        model.fit(X, y)
        
        # 评估模型
        metrics = evaluate_model(model, X, y, gene_names=gene_names)
        
        # 检查指标
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'confusion_matrix' in metrics 