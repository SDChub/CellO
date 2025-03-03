"""
批次效应校正方法实现
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# 检查是否可用的批次校正库
COMBAT_AVAILABLE = False
HARMONY_AVAILABLE = False
SCANORAMA_AVAILABLE = False
MNN_AVAILABLE = False

# 尝试导入ComBat
try:
    import scanpy as sc
    COMBAT_AVAILABLE = True
except ImportError:
    logger.warning("scanpy未安装，ComBat批次校正将不可用。请使用 'pip install scanpy' 安装。")

# 尝试导入Harmony
try:
    import harmonypy
    HARMONY_AVAILABLE = True
except ImportError:
    logger.warning("harmonypy未安装，Harmony批次校正将不可用。请使用 'pip install harmonypy' 安装。")

# 尝试导入Scanorama
try:
    import scanorama
    SCANORAMA_AVAILABLE = True
except ImportError:
    logger.warning("scanorama未安装，Scanorama批次校正将不可用。请使用 'pip install scanorama' 安装。")

# 尝试导入MNN Correct
try:
    import mnnpy
    MNN_AVAILABLE = True
except ImportError:
    logger.warning("mnnpy未安装，MNN批次校正将不可用。请使用 'pip install mnnpy' 安装。")

def correct_batch_effect(
    data: Union[np.ndarray, pd.DataFrame],
    batch_labels: Union[List[int], np.ndarray, pd.Series],
    method: str = 'combat',
    **kwargs
) -> Union[np.ndarray, pd.DataFrame]:
    """
    批次效应校正的统一接口
    
    Args:
        data: 基因表达矩阵，形状为 [n_samples, n_genes]
        batch_labels: 批次标签，长度为n_samples
        method: 校正方法，可选 'combat', 'harmony', 'scanorama', 'mnn'
        **kwargs: 传递给具体校正方法的其他参数
        
    Returns:
        校正后的数据，与输入类型相同
    """
    logger.info(f"使用 {method} 方法进行批次效应校正")
    
    # 确保批次标签是数组
    if isinstance(batch_labels, (list, pd.Series)):
        batch_labels = np.array(batch_labels)
    
    # 根据方法选择校正函数
    if method == 'combat':
        return combat_correct(data, batch_labels, **kwargs)
    elif method == 'harmony':
        return harmony_correct(data, batch_labels, **kwargs)
    elif method == 'scanorama':
        return scanorama_correct(data, batch_labels, **kwargs)
    elif method == 'mnn':
        return mnncorrect(data, batch_labels, **kwargs)
    elif method == 'none':
        logger.info("跳过批次校正")
        return data
    else:
        raise ValueError(f"未知的批次校正方法: {method}")

def combat_correct(
    data: Union[np.ndarray, pd.DataFrame],
    batch_labels: np.ndarray,
    **kwargs
) -> Union[np.ndarray, pd.DataFrame]:
    """
    使用ComBat方法进行批次效应校正
    
    Args:
        data: 基因表达矩阵，形状为 [n_samples, n_genes]
        batch_labels: 批次标签，长度为n_samples
        **kwargs: 传递给scanpy.pp.combat的其他参数
        
    Returns:
        校正后的数据，与输入类型相同
    """
    if not COMBAT_AVAILABLE:
        logger.warning("scanpy未安装，无法使用ComBat批次校正。返回原始数据。")
        return data
    
    logger.info("使用ComBat进行批次效应校正")
    
    # 确保数据是DataFrame
    is_dataframe = isinstance(data, pd.DataFrame)
    if is_dataframe:
        index = data.index
        columns = data.columns
        data_values = data.values
    else:
        data_values = data
    
    # 创建AnnData对象
    adata = sc.AnnData(X=data_values)
    adata.obs['batch'] = batch_labels
    
    # 应用ComBat校正
    sc.pp.combat(adata, key='batch', **kwargs)
    
    # 提取校正后的数据
    corrected_data = adata.X
    
    # 返回与输入相同类型的数据
    if is_dataframe:
        return pd.DataFrame(corrected_data, index=index, columns=columns)
    else:
        return corrected_data

def harmony_correct(
    data: Union[np.ndarray, pd.DataFrame],
    batch_labels: np.ndarray,
    n_components: int = 50,
    **kwargs
) -> Union[np.ndarray, pd.DataFrame]:
    """
    使用Harmony方法进行批次效应校正
    
    Args:
        data: 基因表达矩阵，形状为 [n_samples, n_genes]
        batch_labels: 批次标签，长度为n_samples
        n_components: PCA组件数量
        **kwargs: 传递给harmonypy.run_harmony的其他参数
        
    Returns:
        校正后的数据，与输入类型相同
    """
    if not HARMONY_AVAILABLE:
        logger.warning("harmonypy未安装，无法使用Harmony批次校正。返回原始数据。")
        return data
    
    logger.info("使用Harmony进行批次效应校正")
    
    # 确保数据是DataFrame
    is_dataframe = isinstance(data, pd.DataFrame)
    if is_dataframe:
        index = data.index
        columns = data.columns
        data_values = data.values
    else:
        data_values = data
    
    # 执行PCA降维
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(n_components, data_values.shape[1], data_values.shape[0]))
        pca_result = pca.fit_transform(data_values)
    except ImportError:
        logger.warning("scikit-learn未安装，无法执行PCA。使用原始数据。")
        pca_result = data_values
        pca = None
    
    # 应用Harmony校正
    harmony_result = harmonypy.run_harmony(pca_result, batch_labels, **kwargs)
    corrected_pca = harmony_result.Z_corr
    
    # 如果执行了PCA，需要转换回原始空间
    if pca is not None:
        # 使用PCA的逆变换（近似）
        corrected_data = np.dot(corrected_pca, pca.components_)
        
        # 添加均值
        if hasattr(pca, 'mean_'):
            corrected_data += pca.mean_
    else:
        corrected_data = corrected_pca
    
    # 返回与输入相同类型的数据
    if is_dataframe:
        return pd.DataFrame(corrected_data, index=index, columns=columns)
    else:
        return corrected_data

def scanorama_correct(
    data: Union[np.ndarray, pd.DataFrame],
    batch_labels: np.ndarray,
    **kwargs
) -> Union[np.ndarray, pd.DataFrame]:
    """
    使用Scanorama方法进行批次效应校正
    
    Args:
        data: 基因表达矩阵，形状为 [n_samples, n_genes]
        batch_labels: 批次标签，长度为n_samples
        **kwargs: 传递给scanorama.correct的其他参数
        
    Returns:
        校正后的数据，与输入类型相同
    """
    if not SCANORAMA_AVAILABLE:
        logger.warning("scanorama未安装，无法使用Scanorama批次校正。返回原始数据。")
        return data
    
    logger.info("使用Scanorama进行批次效应校正")
    
    # 确保数据是DataFrame
    is_dataframe = isinstance(data, pd.DataFrame)
    if is_dataframe:
        index = data.index
        columns = data.columns
        data_values = data.values
    else:
        data_values = data
    
    # 按批次分割数据
    unique_batches = np.unique(batch_labels)
    datasets = []
    batch_indices = []
    
    for batch in unique_batches:
        batch_mask = batch_labels == batch
        datasets.append(data_values[batch_mask].copy())
        batch_indices.append(np.where(batch_mask)[0])
    
    # 应用Scanorama校正
    corrected_datasets, _ = scanorama.correct(datasets, **kwargs)
    
    # 重建校正后的数据矩阵
    corrected_data = np.zeros_like(data_values)
    for i, indices in enumerate(batch_indices):
        corrected_data[indices] = corrected_datasets[i]
    
    # 返回与输入相同类型的数据
    if is_dataframe:
        return pd.DataFrame(corrected_data, index=index, columns=columns)
    else:
        return corrected_data

def mnncorrect(
    data: Union[np.ndarray, pd.DataFrame],
    batch_labels: np.ndarray,
    **kwargs
) -> Union[np.ndarray, pd.DataFrame]:
    """
    使用MNN (Mutual Nearest Neighbors) 方法进行批次效应校正
    
    Args:
        data: 基因表达矩阵，形状为 [n_samples, n_genes]
        batch_labels: 批次标签，长度为n_samples
        **kwargs: 传递给mnnpy.mnn_correct的其他参数
        
    Returns:
        校正后的数据，与输入类型相同
    """
    if not MNN_AVAILABLE:
        logger.warning("mnnpy未安装，无法使用MNN批次校正。返回原始数据。")
        return data
    
    logger.info("使用MNN进行批次效应校正")
    
    # 确保数据是DataFrame
    is_dataframe = isinstance(data, pd.DataFrame)
    if is_dataframe:
        index = data.index
        columns = data.columns
        data_values = data.values
    else:
        data_values = data
    
    # 按批次分割数据
    unique_batches = np.unique(batch_labels)
    batch_data = {}
    
    for i, batch in enumerate(unique_batches):
        batch_mask = batch_labels == batch
        # MNN需要转置数据 [genes, cells]
        batch_data[f"batch{i}"] = data_values[batch_mask].T
    
    # 应用MNN校正
    corrected_data, _, _ = mnnpy.mnn_correct(
        **batch_data,
        var_subset=None,  # 使用所有基因
        **kwargs
    )
    
    # 合并校正后的数据
    combined_data = np.hstack([batch.T for batch in corrected_data])  # 转置回 [cells, genes]
    
    # 重新排序以匹配原始数据顺序
    corrected_ordered = np.zeros_like(data_values)
    start_idx = 0
    
    for i, batch in enumerate(unique_batches):
        batch_mask = batch_labels == batch
        batch_size = np.sum(batch_mask)
        corrected_ordered[batch_mask] = combined_data[start_idx:start_idx + batch_size]
        start_idx += batch_size
    
    # 返回与输入相同类型的数据
    if is_dataframe:
        return pd.DataFrame(corrected_ordered, index=index, columns=columns)
    else:
        return corrected_ordered 