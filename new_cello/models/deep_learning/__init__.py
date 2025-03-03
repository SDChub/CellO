"""
深度学习模型模块，使用Hugging Face的Transformers工具
"""

import logging
from enum import Enum

logger = logging.getLogger(__name__)

class DeepLearningModelType(Enum):
    """深度学习模型类型枚举"""
    TRANSFORMER = "transformer"
    GNN = "gnn"  # 图神经网络
    HYBRID = "hybrid"  # 混合模型

def get_deep_learning_model(model_name="transformer", **kwargs):
    """
    获取指定类型的深度学习模型
    
    Args:
        model_name (str): 模型名称，可以是 'transformer', 'gnn' 或 'hybrid'
        **kwargs: 传递给模型构造函数的参数
    
    Returns:
        object: 模型实例
    
    Raises:
        ValueError: 如果指定的模型类型不存在
    """
    try:
        model_type = DeepLearningModelType(model_name.lower())
    except ValueError:
        valid_types = [t.value for t in DeepLearningModelType]
        raise ValueError(f"未知的深度学习模型类型: {model_name}. 有效的类型是: {valid_types}")
    
    if model_type == DeepLearningModelType.TRANSFORMER:
        from new_cello.models.deep_learning.transformer import TransformerModel
        return TransformerModel(**kwargs)
    elif model_type == DeepLearningModelType.GNN:
        from new_cello.models.deep_learning.gnn import GraphNeuralNetworkModel
        return GraphNeuralNetworkModel(**kwargs)
    elif model_type == DeepLearningModelType.HYBRID:
        from new_cello.models.deep_learning.hybrid import HybridModel
        return HybridModel(**kwargs)
    else:
        valid_types = [t.value for t in DeepLearningModelType]
        raise ValueError(f"未知的深度学习模型类型: {model_type}. 有效的类型是: {valid_types}")

def list_models():
    """
    列出可用的深度学习模型
    
    Returns:
        list: 可用模型的列表
    """
    return [t.value for t in DeepLearningModelType]  
