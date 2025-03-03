"""
模型工厂模块，用于获取不同类型的模型
"""

import logging
from enum import Enum

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """模型类型枚举"""
    TRADITIONAL = "traditional"
    DEEP_LEARNING = "deep_learning"
    ENSEMBLE = "ensemble"

def get_model(model_type, **kwargs):
    """
    获取指定类型的模型
    
    Args:
        model_type (str or ModelType): 模型类型，可以是 'traditional', 'deep_learning' 或 'ensemble'
        **kwargs: 传递给模型构造函数的参数
    
    Returns:
        object: 模型实例
    
    Raises:
        ValueError: 如果指定的模型类型不存在
    """
    if isinstance(model_type, str):
        try:
            model_type = ModelType(model_type.lower())
        except ValueError:
            valid_types = [t.value for t in ModelType]
            raise ValueError(f"未知的模型类型: {model_type}. 有效的类型是: {valid_types}")
    
    if model_type == ModelType.TRADITIONAL:
        from new_cello.models.traditional import get_traditional_model
        return get_traditional_model(**kwargs)
    elif model_type == ModelType.DEEP_LEARNING:
        from new_cello.models.deep_learning import get_deep_learning_model
        return get_deep_learning_model(**kwargs)
    elif model_type == ModelType.ENSEMBLE:
        from new_cello.models.ensemble import get_ensemble_model
        return get_ensemble_model(**kwargs)
    else:
        valid_types = [t.value for t in ModelType]
        raise ValueError(f"未知的模型类型: {model_type}. 有效的类型是: {valid_types}")

def list_available_models(model_type=None):
    """
    列出可用的模型
    
    Args:
        model_type (str or ModelType, optional): 如果指定，只列出该类型的模型
    
    Returns:
        dict: 可用模型的字典，按类型分组
    """
    available_models = {}
    
    # 获取传统模型
    if model_type is None or (isinstance(model_type, str) and model_type.lower() == ModelType.TRADITIONAL.value) or model_type == ModelType.TRADITIONAL:
        from new_cello.models.traditional import list_models as list_traditional
        available_models[ModelType.TRADITIONAL.value] = list_traditional()
    
    # 获取深度学习模型
    if model_type is None or (isinstance(model_type, str) and model_type.lower() == ModelType.DEEP_LEARNING.value) or model_type == ModelType.DEEP_LEARNING:
        from new_cello.models.deep_learning import list_models as list_deep_learning
        available_models[ModelType.DEEP_LEARNING.value] = list_deep_learning()
    
    # 获取集成模型
    if model_type is None or (isinstance(model_type, str) and model_type.lower() == ModelType.ENSEMBLE.value) or model_type == ModelType.ENSEMBLE:
        from new_cello.models.ensemble import list_models as list_ensemble
        available_models[ModelType.ENSEMBLE.value] = list_ensemble()
    
    return available_models  
