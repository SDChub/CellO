"""
核心分类功能模块
"""

import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any

from new_cello.models import get_model, ModelType
from new_cello.preprocess import preprocess_pipeline
from new_cello.utils.gpu import is_gpu_available, get_device

logger = logging.getLogger(__name__)

def classify(
    data: Union[np.ndarray, pd.DataFrame],
    gene_names: Optional[List[str]] = None,
    model_type: str = "ensemble",
    model_name: Optional[str] = None,
    model_dir: Optional[str] = None,
    use_gpu: bool = True,
    preprocess_params: Optional[Dict] = None,
    **kwargs
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    使用指定模型对数据进行分类
    
    Args:
        data: 基因表达矩阵，形状为 [n_samples, n_genes]
        gene_names: 基因名称列表
        model_type: 模型类型，可以是 'traditional', 'deep_learning' 或 'ensemble'
        model_name: 模型名称，如果为None则使用默认模型
        model_dir: 模型目录，如果提供则从该目录加载模型
        use_gpu: 是否使用GPU加速
        preprocess_params: 预处理参数
        **kwargs: 传递给模型的其他参数
        
    Returns:
        Tuple: (预测标签, 预测概率)
    """
    logger.info(f"使用 {model_type}/{model_name} 模型进行分类")
    
    # 检查GPU可用性
    if use_gpu and not is_gpu_available():
        logger.warning("GPU不可用，将使用CPU")
        use_gpu = False
    
    # 设置设备
    device = "cuda" if use_gpu else "cpu"
    logger.info(f"使用设备: {device}")
    
    # 预处理数据
    preprocess_params = preprocess_params or {}
    processed_data, processed_gene_names = preprocess_pipeline(
        data, 
        gene_names=gene_names,
        **preprocess_params
    )
    
    # 加载或创建模型
    if model_dir is not None:
        # 从目录加载模型
        logger.info(f"从 {model_dir} 加载模型")
        model = _load_model(model_type, model_name, model_dir, device)
    else:
        # 创建新模型
        logger.info("创建新模型")
        model = get_model(model_type, model_name=model_name, device=device, **kwargs)
    
    # 进行预测
    predictions = model.predict(processed_data, processed_gene_names)
    
    # 如果模型支持概率预测，获取概率
    probabilities = None
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(processed_data, processed_gene_names)
    
    return predictions, probabilities

def train_model(
    data: Union[np.ndarray, pd.DataFrame],
    labels: Union[np.ndarray, pd.Series],
    gene_names: Optional[List[str]] = None,
    model_type: str = "ensemble",
    model_name: Optional[str] = None,
    use_gpu: bool = True,
    preprocess_params: Optional[Dict] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Any:
    """
    训练模型
    
    Args:
        data: 基因表达矩阵，形状为 [n_samples, n_genes]
        labels: 标签向量
        gene_names: 基因名称列表
        model_type: 模型类型，可以是 'traditional', 'deep_learning' 或 'ensemble'
        model_name: 模型名称，如果为None则使用默认模型
        use_gpu: 是否使用GPU加速
        preprocess_params: 预处理参数
        save_dir: 保存目录，如果提供则将模型保存到该目录
        **kwargs: 传递给模型的其他参数
        
    Returns:
        训练好的模型
    """
    logger.info(f"训练 {model_type}/{model_name} 模型")
    
    # 检查GPU可用性
    if use_gpu and not is_gpu_available():
        logger.warning("GPU不可用，将使用CPU")
        use_gpu = False
    
    # 设置设备
    device = "cuda" if use_gpu else "cpu"
    logger.info(f"使用设备: {device}")
    
    # 预处理数据
    preprocess_params = preprocess_params or {}
    processed_data, processed_gene_names = preprocess_pipeline(
        data, 
        gene_names=gene_names,
        **preprocess_params
    )
    
    # 创建模型
    model = get_model(model_type, model_name=model_name, device=device, **kwargs)
    
    # 训练模型
    if hasattr(model, 'fit'):
        # 检查模型的fit方法是否接受gene_names参数
        if 'gene_names' in model.fit.__code__.co_varnames:
            model.fit(processed_data, labels, gene_names=processed_gene_names)
        else:
            model.fit(processed_data, labels)
    else:
        raise ValueError(f"模型 {model_type}/{model_name} 没有fit方法")
    
    # 保存模型
    if save_dir is not None:
        logger.info(f"保存模型到 {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        
        if hasattr(model, 'save'):
            model.save(save_dir)
        else:
            logger.warning(f"模型 {model_type}/{model_name} 没有save方法，无法保存")
    
    return model

def evaluate_model(
    model: Any,
    data: Union[np.ndarray, pd.DataFrame],
    labels: Union[np.ndarray, pd.Series],
    gene_names: Optional[List[str]] = None,
    preprocess_params: Optional[Dict] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    评估模型性能
    
    Args:
        model: 模型实例
        data: 基因表达矩阵，形状为 [n_samples, n_genes]
        labels: 标签向量
        gene_names: 基因名称列表
        preprocess_params: 预处理参数
        **kwargs: 其他参数
        
    Returns:
        Dict: 包含评估指标的字典
    """
    logger.info("评估模型性能")
    
    # 预处理数据
    preprocess_params = preprocess_params or {}
    processed_data, processed_gene_names = preprocess_pipeline(
        data, 
        gene_names=gene_names,
        **preprocess_params
    )
    
    # 进行预测
    predictions = model.predict(processed_data, processed_gene_names)
    
    # 如果模型支持概率预测，获取概率
    probabilities = None
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(processed_data, processed_gene_names)
    
    # 计算评估指标
    from new_cello.evaluation import calculate_metrics
    metrics = calculate_metrics(labels, predictions, probabilities)
    
    return metrics

def _load_model(model_type: str, model_name: Optional[str], model_dir: str, device: str) -> Any:
    """
    从目录加载模型
    
    Args:
        model_type: 模型类型
        model_name: 模型名称
        model_dir: 模型目录
        device: 设备
        
    Returns:
        加载的模型
    """
    # 根据模型类型选择加载方法
    if model_type == "traditional" or model_type == ModelType.TRADITIONAL.value:
        if model_name is None:
            model_name = "random_forest"
            
        if model_name == "random_forest":
            from new_cello.models.traditional.random_forest import RandomForestModel
            return RandomForestModel.load(model_dir)
        elif model_name == "svm":
            from new_cello.models.traditional.svm import SVMModel
            return SVMModel.load(model_dir)
        else:
            raise ValueError(f"不支持的传统模型: {model_name}")
            
    elif model_type == "deep_learning" or model_type == ModelType.DEEP_LEARNING.value:
        if model_name is None:
            model_name = "transformer"
            
        if model_name == "transformer":
            from new_cello.models.deep_learning.transformer import TransformerModel
            return TransformerModel.load(model_dir, device=device)
        else:
            raise ValueError(f"不支持的深度学习模型: {model_name}")
            
    elif model_type == "ensemble" or model_type == ModelType.ENSEMBLE.value:
        if model_name is None:
            model_name = "voting"
            
        if model_name == "voting":
            from new_cello.models.ensemble.voting import VotingEnsembleModel
            
            # 创建模型加载器
            def load_model(model_dir):
                # 尝试加载配置文件，确定子模型类型
                config_path = os.path.join(model_dir, "config.json")
                if os.path.exists(config_path):
                    import json
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    sub_model_type = config.get("model_type", "traditional")
                    sub_model_name = config.get("model_name", "random_forest")
                    return _load_model(sub_model_type, sub_model_name, model_dir, device)
                else:
                    # 默认尝试加载随机森林模型
                    from new_cello.models.traditional.random_forest import RandomForestModel
                    return RandomForestModel.load(model_dir)
            
            # 获取子模型目录
            models_dir = os.path.join(model_dir, "models")
            if os.path.exists(models_dir):
                model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
                model_loaders = [load_model] * len(model_dirs)
                return VotingEnsembleModel.load(model_dir, model_loaders=model_loaders)
            else:
                raise ValueError(f"模型目录 {models_dir} 不存在")
        else:
            raise ValueError(f"不支持的集成模型: {model_name}")
    else:
        raise ValueError(f"不支持的模型类型: {model_type}") 