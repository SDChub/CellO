"""
Transformer模型测试
"""

import os
import numpy as np
import pandas as pd
import pytest
from typing import Dict, List, Tuple
import torch

# 标记需要GPU的测试
gpu_required = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="需要GPU才能运行此测试"
)

# 尝试导入Transformer模型
try:
    from new_cello.models.deep_learning.transformer import TransformerModel, TRANSFORMERS_AVAILABLE
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# 如果Transformers库不可用，则跳过所有测试
pytestmark = pytest.mark.skipif(
    not TRANSFORMERS_AVAILABLE,
    reason="Transformers库未安装，无法测试Transformer模型"
)

class TestTransformerModel:
    """Transformer模型测试类"""
    
    @pytest.fixture
    def sample_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        创建样本数据用于测试
        
        Returns:
            Tuple: (特征矩阵, 标签向量, 基因名称列表)
        """
        # 创建随机特征矩阵 [n_samples, n_features]
        n_samples = 10  # 使用较小的样本数以加快测试速度
        n_features = 100
        X = np.random.rand(n_samples, n_features)
        
        # 创建随机标签 [n_samples]
        n_classes = 3
        y = np.random.randint(0, n_classes, size=n_samples)
        
        # 创建基因名称列表
        gene_names = [f"gene_{i}" for i in range(n_features)]
        
        return X, y, gene_names
    
    def test_init(self):
        """测试模型初始化"""
        model = TransformerModel(
            model_name="distilbert-base-uncased",
            num_classes=3,
            device="cpu"
        )
        assert model.model_name == "distilbert-base-uncased"
        assert model.num_classes == 3
        assert model.device == "cpu"
        assert model.base_model is not None
        assert model.tokenizer is not None
        assert model.classifier is not None
    
    def test_preprocess(self, sample_data):
        """测试数据预处理"""
        X, _, gene_names = sample_data
        
        # 创建模型
        model = TransformerModel(
            model_name="distilbert-base-uncased",
            num_classes=3,
            device="cpu"
        )
        
        # 预处理数据
        inputs = model.preprocess(X, gene_names)
        
        # 检查预处理结果
        assert isinstance(inputs, dict)
        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert isinstance(inputs["input_ids"], torch.Tensor)
        assert isinstance(inputs["attention_mask"], torch.Tensor)
        assert inputs["input_ids"].shape[0] == X.shape[0]  # 批次大小
    
    def test_forward(self, sample_data):
        """测试前向传播"""
        X, _, gene_names = sample_data
        
        # 创建模型
        model = TransformerModel(
            model_name="distilbert-base-uncased",
            num_classes=3,
            device="cpu"
        )
        
        # 预处理数据
        inputs = model.preprocess(X, gene_names)
        
        # 前向传播
        outputs = model.forward(inputs)
        
        # 检查输出
        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape == (X.shape[0], 3)  # [batch_size, num_classes]
    
    def test_predict(self, sample_data):
        """测试预测"""
        X, _, gene_names = sample_data
        
        # 创建模型
        model = TransformerModel(
            model_name="distilbert-base-uncased",
            num_classes=3,
            device="cpu"
        )
        
        # 预测
        predictions = model.predict(X, gene_names)
        
        # 检查预测结果
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (X.shape[0],)  # [batch_size]
        assert np.all((predictions >= 0) & (predictions < 3))  # 类别范围
    
    def test_get_embeddings(self, sample_data):
        """测试获取嵌入"""
        X, _, gene_names = sample_data
        
        # 创建没有分类头的模型
        model = TransformerModel(
            model_name="distilbert-base-uncased",
            num_classes=0,  # 不使用分类头
            device="cpu"
        )
        
        # 获取嵌入
        embeddings = model.get_embeddings(X, gene_names)
        
        # 检查嵌入
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (X.shape[0], model.base_model.config.hidden_size)  # [batch_size, hidden_size]
    
    def test_save_load(self, sample_data, tmp_path):
        """测试模型保存和加载"""
        X, _, gene_names = sample_data
        
        # 创建模型
        model = TransformerModel(
            model_name="distilbert-base-uncased",
            num_classes=3,
            device="cpu"
        )
        
        # 获取原始预测
        original_pred = model.predict(X, gene_names)
        
        # 保存模型
        save_dir = os.path.join(tmp_path, "model")
        model.save(save_dir)
        
        # 检查保存的文件
        assert os.path.exists(os.path.join(save_dir, "base_model"))
        assert os.path.exists(os.path.join(save_dir, "tokenizer"))
        assert os.path.exists(os.path.join(save_dir, "classifier.pt"))
        assert os.path.exists(os.path.join(save_dir, "config.pt"))
        
        # 加载模型
        loaded_model = TransformerModel.load(save_dir, device="cpu")
        
        # 检查加载的模型参数
        assert loaded_model.model_name == os.path.join(save_dir, "base_model")
        assert loaded_model.num_classes == model.num_classes
        
        # 检查加载的模型预测
        loaded_pred = loaded_model.predict(X, gene_names)
        assert np.array_equal(original_pred, loaded_pred)
    
    @gpu_required
    def test_gpu_support(self, sample_data):
        """测试GPU支持（需要GPU）"""
        X, _, gene_names = sample_data
        
        # 创建GPU模型
        model = TransformerModel(
            model_name="distilbert-base-uncased",
            num_classes=3,
            device="cuda"
        )
        
        # 检查设备
        assert model.device == "cuda"
        assert next(model.base_model.parameters()).device.type == "cuda"
        assert next(model.classifier.parameters()).device.type == "cuda"
        
        # 预测
        predictions = model.predict(X, gene_names)
        
        # 检查预测结果
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (X.shape[0],) 