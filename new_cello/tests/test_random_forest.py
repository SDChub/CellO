"""
随机森林模型测试
"""

import os
import numpy as np
import pandas as pd
import pytest
from typing import Dict, List, Tuple

from new_cello.models.traditional.random_forest import RandomForestModel

class TestRandomForestModel:
    """随机森林模型测试类"""
    
    @pytest.fixture
    def sample_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        创建样本数据用于测试
        
        Returns:
            Tuple: (特征矩阵, 标签向量, 基因名称列表)
        """
        # 创建随机特征矩阵 [n_samples, n_features]
        n_samples = 100
        n_features = 200
        X = np.random.rand(n_samples, n_features)
        
        # 创建随机标签 [n_samples]
        n_classes = 3
        y = np.random.randint(0, n_classes, size=n_samples)
        
        # 创建基因名称列表
        gene_names = [f"gene_{i}" for i in range(n_features)]
        
        return X, y, gene_names
    
    def test_init(self):
        """测试模型初始化"""
        model = RandomForestModel(n_estimators=50, max_depth=10)
        assert model.n_estimators == 50
        assert model.max_depth == 10
        assert model.feature_selection == True
        assert model.scale_features == True
    
    def test_fit_predict(self, sample_data):
        """测试模型训练和预测"""
        X, y, gene_names = sample_data
        
        # 创建模型
        model = RandomForestModel(n_estimators=50, random_state=42)
        
        # 训练模型
        model.fit(X, y, gene_names=gene_names)
        
        # 预测
        y_pred = model.predict(X)
        
        # 检查预测结果的形状
        assert y_pred.shape == y.shape
        
        # 检查预测结果的类型
        assert np.issubdtype(y_pred.dtype, np.integer)
        
        # 检查预测概率
        y_prob = model.predict_proba(X)
        assert y_prob.shape == (X.shape[0], len(np.unique(y)))
        assert np.allclose(np.sum(y_prob, axis=1), 1.0)
    
    def test_feature_importance(self, sample_data):
        """测试特征重要性"""
        X, y, gene_names = sample_data
        
        # 创建模型
        model = RandomForestModel(n_estimators=50, random_state=42)
        
        # 训练模型
        model.fit(X, y, gene_names=gene_names)
        
        # 获取特征重要性
        importances = model.get_feature_importances()
        
        # 检查特征重要性的类型和大小
        assert isinstance(importances, dict)
        assert len(importances) > 0
        
        # 检查特征重要性的值
        for gene, importance in importances.items():
            assert importance >= 0.0
            assert importance <= 1.0
    
    def test_save_load(self, sample_data, tmp_path):
        """测试模型保存和加载"""
        X, y, gene_names = sample_data
        
        # 创建模型
        model = RandomForestModel(n_estimators=50, random_state=42)
        
        # 训练模型
        model.fit(X, y, gene_names=gene_names)
        
        # 获取原始预测
        original_pred = model.predict(X)
        
        # 保存模型
        save_dir = os.path.join(tmp_path, "model")
        model.save(save_dir)
        
        # 检查保存的文件
        assert os.path.exists(os.path.join(save_dir, "model.pkl"))
        assert os.path.exists(os.path.join(save_dir, "config.pkl"))
        
        # 加载模型
        loaded_model = RandomForestModel.load(save_dir)
        
        # 检查加载的模型参数
        assert loaded_model.n_estimators == model.n_estimators
        assert loaded_model.max_depth == model.max_depth
        assert loaded_model.feature_selection == model.feature_selection
        
        # 检查加载的模型预测
        loaded_pred = loaded_model.predict(X)
        assert np.array_equal(original_pred, loaded_pred)
    
    def test_feature_selection(self, sample_data):
        """测试特征选择"""
        X, y, gene_names = sample_data
        
        # 创建带特征选择的模型
        model_with_fs = RandomForestModel(
            n_estimators=50, 
            random_state=42,
            feature_selection=True,
            feature_selection_threshold="mean"
        )
        
        # 创建不带特征选择的模型
        model_without_fs = RandomForestModel(
            n_estimators=50, 
            random_state=42,
            feature_selection=False
        )
        
        # 训练模型
        model_with_fs.fit(X, y, gene_names=gene_names)
        model_without_fs.fit(X, y, gene_names=gene_names)
        
        # 检查特征选择器
        assert model_with_fs.feature_selector is not None
        assert model_without_fs.feature_selector is None
        
        # 检查选择的特征
        if hasattr(model_with_fs, 'selected_genes'):
            assert len(model_with_fs.selected_genes) < len(gene_names)
    
    def test_scale_features(self, sample_data):
        """测试特征缩放"""
        X, y, gene_names = sample_data
        
        # 创建带特征缩放的模型
        model_with_scaling = RandomForestModel(
            n_estimators=50, 
            random_state=42,
            scale_features=True
        )
        
        # 创建不带特征缩放的模型
        model_without_scaling = RandomForestModel(
            n_estimators=50, 
            random_state=42,
            scale_features=False
        )
        
        # 训练模型
        model_with_scaling.fit(X, y, gene_names=gene_names)
        model_without_scaling.fit(X, y, gene_names=gene_names)
        
        # 检查缩放器
        assert model_with_scaling.scaler is not None
        assert model_without_scaling.scaler is None 