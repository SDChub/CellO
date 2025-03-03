"""
传统机器学习模型模块，包括随机森林、SVM等
"""

import logging
from enum import Enum

logger = logging.getLogger(__name__)

class TraditionalModelType(Enum):
    """传统机器学习模型类型枚举"""
    RANDOM_FOREST = "random_forest"
    SVM = "svm"
    LOGISTIC_REGRESSION = "logistic_regression"
    GRADIENT_BOOSTING = "gradient_boosting"

def get_traditional_model(model_name="random_forest", **kwargs):
    """
    获取指定类型的传统机器学习模型
    
    Args:
        model_name (str): 模型名称，可以是 'random_forest', 'svm', 'logistic_regression' 或 'gradient_boosting'
        **kwargs: 传递给模型构造函数的参数
    
    Returns:
        object: 模型实例
    
    Raises:
        ValueError: 如果指定的模型类型不存在
    """
    try:
        model_type = TraditionalModelType(model_name.lower())
    except ValueError:
        valid_types = [t.value for t in TraditionalModelType]
        raise ValueError(f"未知的传统模型类型: {model_name}. 有效的类型是: {valid_types}")
    
    if model_type == TraditionalModelType.RANDOM_FOREST:
        from new_cello.models.traditional.random_forest import RandomForestModel
        return RandomForestModel(**kwargs)
    elif model_type == TraditionalModelType.SVM:
        from new_cello.models.traditional.svm import SVMModel
        return SVMModel(**kwargs)
    elif model_type == TraditionalModelType.LOGISTIC_REGRESSION:
        from new_cello.models.traditional.logistic_regression import LogisticRegressionModel
        return LogisticRegressionModel(**kwargs)
    elif model_type == TraditionalModelType.GRADIENT_BOOSTING:
        from new_cello.models.traditional.gradient_boosting import GradientBoostingModel
        return GradientBoostingModel(**kwargs)
    else:
        valid_types = [t.value for t in TraditionalModelType]
        raise ValueError(f"未知的传统模型类型: {model_type}. 有效的类型是: {valid_types}")

def list_models():
    """
    列出可用的传统机器学习模型
    
    Returns:
        list: 可用模型的列表
    """
    return [t.value for t in TraditionalModelType]  
