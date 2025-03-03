"""
ComBat批次效应校正实现
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# 尝试导入ComBat相关库
try:
    import scanpy as sc
    from scanpy.preprocessing._combat import combat
    COMBAT_AVAILABLE = True
except ImportError:
    try:
        from sklearn.preprocessing import StandardScaler
        import patsy
        COMBAT_AVAILABLE = True
    except ImportError:
        COMBAT_AVAILABLE = False
        logger.warning("scanpy或patsy库未安装，ComBat批次校正将不可用。请使用 'pip install scanpy' 或 'pip install patsy' 安装。")

def combat_correct(
    data: Union[np.ndarray, pd.DataFrame],
    batch_labels: Union[np.ndarray, pd.Series, List[int]],
    covariates: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    parametric: bool = True,
    mean_only: bool = False,
    **kwargs
) -> Union[np.ndarray, pd.DataFrame]:
    """
    使用ComBat方法进行批次效应校正
    
    Args:
        data: 基因表达矩阵，形状为 [n_samples, n_genes]
        batch_labels: 批次标签，长度为n_samples
        covariates: 协变量矩阵，形状为 [n_samples, n_covariates]，用于保留生物学变异
        parametric: 是否使用参数化方法估计批次效应
        mean_only: 是否只校正均值（不校正方差）
        **kwargs: 其他参数
        
    Returns:
        校正后的数据，与输入类型相同
    """
    if not COMBAT_AVAILABLE:
        raise ImportError("scanpy或patsy库未安装，无法使用ComBat批次校正")
    
    logger.info("使用ComBat方法进行批次效应校正")
    
    # 确保数据是数值型
    if isinstance(data, pd.DataFrame):
        data_values = data.values
        is_dataframe = True
        index = data.index
        columns = data.columns
    else:
        data_values = data
        is_dataframe = False
        index = None
        columns = None
    
    # 确保批次标签是数组
    if isinstance(batch_labels, (pd.Series, list)):
        batch_labels = np.array(batch_labels)
    
    # 检查数据和批次标签的形状
    if len(batch_labels) != data_values.shape[0]:
        raise ValueError(f"批次标签长度 ({len(batch_labels)}) 与样本数 ({data_values.shape[0]}) 不匹配")
    
    # 检查批次数量
    unique_batches = np.unique(batch_labels)
    if len(unique_batches) < 2:
        logger.warning("只有一个批次，无需校正")
        return data
    
    logger.info(f"检测到 {len(unique_batches)} 个批次")
    
    try:
        # 使用scanpy的combat函数
        if 'combat' in globals():
            # 转置数据以符合scanpy的要求 [n_genes, n_samples]
            data_t = data_values.T
            
            # 准备协变量设计矩阵
            if covariates is not None:
                if isinstance(covariates, pd.DataFrame):
                    covariates = covariates.values
                
                # 创建设计矩阵
                design = patsy.dmatrix("~ 0 + C(batch)", {"batch": batch_labels})
                
                # 添加协变量
                for i in range(covariates.shape[1]):
                    cov_col = covariates[:, i]
                    design = np.hstack((design, cov_col.reshape(-1, 1)))
            else:
                design = None
            
            # 应用ComBat校正
            corrected_data_t = combat(
                data_t,
                batch_labels,
                design=design,
                parametric=parametric,
                mean_only=mean_only
            )
            
            # 转置回原始形状 [n_samples, n_genes]
            corrected_data = corrected_data_t.T
        else:
            # 使用scanpy的pp.combat函数
            # 创建AnnData对象
            adata = sc.AnnData(X=data_values)
            adata.obs['batch'] = batch_labels
            
            # 添加协变量
            if covariates is not None:
                if isinstance(covariates, pd.DataFrame):
                    for col in covariates.columns:
                        adata.obs[col] = covariates[col].values
                    covariates_names = covariates.columns.tolist()
                else:
                    for i in range(covariates.shape[1]):
                        adata.obs[f'cov_{i}'] = covariates[:, i]
                    covariates_names = [f'cov_{i}' for i in range(covariates.shape[1])]
                
                # 应用ComBat校正
                sc.pp.combat(adata, key='batch', covariates=covariates_names)
            else:
                # 应用ComBat校正
                sc.pp.combat(adata, key='batch')
            
            corrected_data = adata.X
    except Exception as e:
        logger.error(f"ComBat校正失败: {str(e)}")
        logger.info("尝试使用自定义ComBat实现")
        
        # 使用自定义ComBat实现
        corrected_data = _custom_combat(
            data_values,
            batch_labels,
            covariates=covariates,
            parametric=parametric,
            mean_only=mean_only
        )
    
    # 返回与输入相同类型的数据
    if is_dataframe:
        return pd.DataFrame(corrected_data, index=index, columns=columns)
    else:
        return corrected_data

def _custom_combat(
    data: np.ndarray,
    batch_labels: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    parametric: bool = True,
    mean_only: bool = False
) -> np.ndarray:
    """
    自定义ComBat实现，用于当scanpy不可用时
    
    Args:
        data: 基因表达矩阵，形状为 [n_samples, n_genes]
        batch_labels: 批次标签，长度为n_samples
        covariates: 协变量矩阵，形状为 [n_samples, n_covariates]
        parametric: 是否使用参数化方法估计批次效应
        mean_only: 是否只校正均值（不校正方差）
        
    Returns:
        校正后的数据
    """
    # 获取数据维度
    n_samples, n_genes = data.shape
    
    # 标准化数据
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data)
    
    # 获取唯一批次
    unique_batches = np.unique(batch_labels)
    n_batches = len(unique_batches)
    
    # 计算每个批次的均值和方差
    batch_means = np.zeros((n_batches, n_genes))
    batch_vars = np.zeros((n_batches, n_genes))
    
    for i, batch in enumerate(unique_batches):
        batch_mask = batch_labels == batch
        batch_data = data_std[batch_mask]
        batch_means[i] = np.mean(batch_data, axis=0)
        batch_vars[i] = np.var(batch_data, axis=0)
    
    # 计算全局均值和方差
    global_mean = np.mean(data_std, axis=0)
    global_var = np.var(data_std, axis=0)
    
    # 校正数据
    corrected_data = np.zeros_like(data_std)
    
    for i, batch in enumerate(unique_batches):
        batch_mask = batch_labels == batch
        batch_data = data_std[batch_mask]
        
        # 均值校正
        mean_diff = batch_means[i] - global_mean
        corrected_batch = batch_data - mean_diff
        
        # 方差校正（如果需要）
        if not mean_only:
            var_ratio = global_var / np.maximum(batch_vars[i], 1e-8)
            corrected_batch = corrected_batch * np.sqrt(var_ratio)
        
        corrected_data[batch_mask] = corrected_batch
    
    # 反标准化
    corrected_data = scaler.inverse_transform(corrected_data)
    
    return corrected_data 