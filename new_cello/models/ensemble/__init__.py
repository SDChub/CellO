"""
集成模型模块，结合传统机器学习和深度学习模型
"""

import logging
from enum import Enum

logger = logging.getLogger(__name__)

class EnsembleModelType(Enum):
    """集成模型类型枚举"""
    VOTING = "voting"
    STACKING = "stacking"
    WEIGHTED = "weighted"

def get_ensemble_model(model_name="voting", **kwargs):
    """
    获取指定类型的集成模型
    
    Args:
        model_name (str): 模型名称，可以是 'voting', 'stacking' 或 'weighted'
        **kwargs: 传递给模型构造函数的参数
    
    Returns:
        object: 模型实例
    
    Raises:
        ValueError: 如果指定的模型类型不存在
    """
    try:
        model_type = EnsembleModelType(model_name.lower())
    except ValueError:
        valid_types = [t.value for t in EnsembleModelType]
        raise ValueError(f"未知的集成模型类型: {model_name}. 有效的类型是: {valid_types}")
    
    if model_type == EnsembleModelType.VOTING:
        from new_cello.models.ensemble.voting import VotingEnsembleModel
        return VotingEnsembleModel(**kwargs)
    elif model_type == EnsembleModelType.STACKING:
        from new_cello.models.ensemble.stacking import StackingEnsembleModel
        return StackingEnsembleModel(**kwargs)
    elif model_type == EnsembleModelType.WEIGHTED:
        from new_cello.models.ensemble.weighted import WeightedEnsembleModel
        return WeightedEnsembleModel(**kwargs)
    else:
        valid_types = [t.value for t in EnsembleModelType]
        raise ValueError(f"未知的集成模型类型: {model_type}. 有效的类型是: {valid_types}")

def list_models():
    """
    列出可用的集成模型
    
    Returns:
        list: 可用模型的列表
    """
    return [t.value for t in EnsembleModelType]  
