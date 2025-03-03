"""
数据预处理模块，用于处理和准备单细胞RNA-seq数据
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple

logger = logging.getLogger(__name__)

# 导入批次校正模块
try:
    from new_cello.preprocess.batch_correction import correct_batch_effect
    BATCH_CORRECTION_AVAILABLE = True
except ImportError:
    BATCH_CORRECTION_AVAILABLE = False
    logger.warning("批次校正模块未加载，批次校正功能将不可用。")

def normalize_data(
    data: Union[np.ndarray, pd.DataFrame], 
    method: str = 'log1p',
    scale_factor: float = 1e6,
    **kwargs
) -> Union[np.ndarray, pd.DataFrame]:
    """
    标准化基因表达数据
    
    Args:
        data: 基因表达矩阵，形状为 [n_samples, n_genes]
        method: 标准化方法，可选 'log1p', 'cpm', 'tpm', 'none'
        scale_factor: 用于CPM/TPM标准化的缩放因子
        **kwargs: 其他参数
        
    Returns:
        标准化后的数据，与输入类型相同
    """
    logger.info(f"使用 {method} 方法标准化数据")
    
    # 确保数据是数值型
    if isinstance(data, pd.DataFrame):
        data_values = data.values
        is_dataframe = True
    else:
        data_values = data
        is_dataframe = False
    
    # 应用标准化方法
    if method == 'log1p':
        # log(1+x) 转换
        normalized = np.log1p(data_values)
    elif method == 'cpm':
        # 每百万读数计数 (CPM)
        row_sums = data_values.sum(axis=1, keepdims=True)
        normalized = data_values * scale_factor / row_sums
    elif method == 'tpm':
        # 每百万转录本 (TPM)
        # 注意：这需要基因长度信息，这里简化处理
        row_sums = data_values.sum(axis=1, keepdims=True)
        normalized = data_values * scale_factor / row_sums
    elif method == 'none':
        # 不进行标准化
        normalized = data_values
    else:
        raise ValueError(f"未知的标准化方法: {method}")
    
    # 返回与输入相同类型的数据
    if is_dataframe:
        return pd.DataFrame(normalized, index=data.index, columns=data.columns)
    else:
        return normalized

def filter_genes(
    data: Union[np.ndarray, pd.DataFrame],
    gene_names: Optional[List[str]] = None,
    min_cells: int = 3,
    min_counts: int = 1,
    **kwargs
) -> Tuple[Union[np.ndarray, pd.DataFrame], Optional[List[str]]]:
    """
    过滤低表达基因
    
    Args:
        data: 基因表达矩阵，形状为 [n_samples, n_genes]
        gene_names: 基因名称列表
        min_cells: 基因必须在至少这么多细胞中表达
        min_counts: 基因在细胞中的最小表达量
        **kwargs: 其他参数
        
    Returns:
        过滤后的数据和基因名称列表
    """
    logger.info(f"过滤基因，要求至少在 {min_cells} 个细胞中表达，每个细胞至少 {min_counts} 个计数")
    
    # 确保数据是数值型
    if isinstance(data, pd.DataFrame):
        data_values = data.values
        is_dataframe = True
        if gene_names is None:
            gene_names = data.columns.tolist()
    else:
        data_values = data
        is_dataframe = False
    
    # 计算每个基因在多少细胞中表达
    cells_per_gene = np.sum(data_values >= min_counts, axis=0)
    
    # 过滤基因
    gene_mask = cells_per_gene >= min_cells
    filtered_data = data_values[:, gene_mask]
    
    logger.info(f"过滤前基因数: {data_values.shape[1]}, 过滤后基因数: {filtered_data.shape[1]}")
    
    # 更新基因名称列表
    if gene_names is not None:
        filtered_gene_names = [gene_names[i] for i in range(len(gene_names)) if gene_mask[i]]
    else:
        filtered_gene_names = None
    
    # 返回与输入相同类型的数据
    if is_dataframe:
        return pd.DataFrame(filtered_data, index=data.index, columns=filtered_gene_names), filtered_gene_names
    else:
        return filtered_data, filtered_gene_names

def filter_cells(
    data: Union[np.ndarray, pd.DataFrame],
    min_genes: int = 200,
    max_genes: Optional[int] = None,
    min_counts: int = 500,
    max_counts: Optional[int] = None,
    **kwargs
) -> Union[np.ndarray, pd.DataFrame]:
    """
    过滤低质量细胞
    
    Args:
        data: 基因表达矩阵，形状为 [n_samples, n_genes]
        min_genes: 细胞必须表达至少这么多基因
        max_genes: 细胞最多表达这么多基因（可用于过滤双细胞）
        min_counts: 细胞必须有至少这么多总计数
        max_counts: 细胞最多有这么多总计数
        **kwargs: 其他参数
        
    Returns:
        过滤后的数据
    """
    logger.info(f"过滤细胞，要求表达至少 {min_genes} 个基因，总计数至少 {min_counts}")
    
    # 确保数据是数值型
    if isinstance(data, pd.DataFrame):
        data_values = data.values
        is_dataframe = True
    else:
        data_values = data
        is_dataframe = False
    
    # 计算每个细胞表达的基因数和总计数
    genes_per_cell = np.sum(data_values > 0, axis=1)
    counts_per_cell = np.sum(data_values, axis=1)
    
    # 创建过滤掩码
    cell_mask = genes_per_cell >= min_genes
    if max_genes is not None:
        cell_mask &= genes_per_cell <= max_genes
        
    cell_mask &= counts_per_cell >= min_counts
    if max_counts is not None:
        cell_mask &= counts_per_cell <= max_counts
    
    # 过滤细胞
    filtered_data = data_values[cell_mask, :]
    
    logger.info(f"过滤前细胞数: {data_values.shape[0]}, 过滤后细胞数: {filtered_data.shape[0]}")
    
    # 返回与输入相同类型的数据
    if is_dataframe:
        return pd.DataFrame(filtered_data, index=data.index[cell_mask], columns=data.columns)
    else:
        return filtered_data

def batch_correct(
    data: Union[np.ndarray, pd.DataFrame],
    batch_labels: List[int],
    method: str = 'combat',
    **kwargs
) -> Union[np.ndarray, pd.DataFrame]:
    """
    批次效应校正
    
    Args:
        data: 基因表达矩阵，形状为 [n_samples, n_genes]
        batch_labels: 每个样本的批次标签
        method: 校正方法，可选 'combat', 'harmony', 'scanorama', 'mnn', 'none'
        **kwargs: 其他参数
        
    Returns:
        校正后的数据，与输入类型相同
    """
    logger.info(f"使用 {method} 方法进行批次效应校正")
    
    # 检查批次校正模块是否可用
    if not BATCH_CORRECTION_AVAILABLE:
        logger.warning("批次校正模块未加载，无法进行批次校正。返回原始数据。")
        return data
    
    # 调用批次校正函数
    return correct_batch_effect(data, batch_labels, method=method, **kwargs)

def select_highly_variable_genes(
    data: Union[np.ndarray, pd.DataFrame],
    gene_names: Optional[List[str]] = None,
    n_top_genes: int = 2000,
    method: str = 'seurat',
    **kwargs
) -> Tuple[Union[np.ndarray, pd.DataFrame], List[str]]:
    """
    选择高变异基因
    
    Args:
        data: 基因表达矩阵，形状为 [n_samples, n_genes]
        gene_names: 基因名称列表
        n_top_genes: 选择的高变异基因数量
        method: 选择方法，可选 'seurat', 'cell_ranger', 'dispersion'
        **kwargs: 其他参数
        
    Returns:
        包含高变异基因的数据和基因名称列表
    """
    logger.info(f"使用 {method} 方法选择 {n_top_genes} 个高变异基因")
    
    # 确保数据是数值型
    if isinstance(data, pd.DataFrame):
        data_values = data.values
        is_dataframe = True
        if gene_names is None:
            gene_names = data.columns.tolist()
    else:
        data_values = data
        is_dataframe = False
        if gene_names is None:
            gene_names = [f"gene_{i}" for i in range(data_values.shape[1])]
    
    # 计算每个基因的均值和方差
    means = np.mean(data_values, axis=0)
    vars = np.var(data_values, axis=0)
    
    # 应用选择方法
    if method == 'seurat':
        # Seurat方法：标准化方差
        dispersion = vars / (means + 1e-5)
        dispersion[np.isnan(dispersion)] = 0
    elif method == 'cell_ranger':
        # Cell Ranger方法：均值归一化方差
        dispersion = vars / (means + 1e-5)
        dispersion[np.isnan(dispersion)] = 0
    elif method == 'dispersion':
        # 简单方差
        dispersion = vars
    else:
        raise ValueError(f"未知的高变异基因选择方法: {method}")
    
    # 选择前n_top_genes个高变异基因
    top_genes_idx = np.argsort(dispersion)[::-1][:n_top_genes]
    top_genes_idx = np.sort(top_genes_idx)  # 按原始顺序排序
    
    # 提取高变异基因数据
    hvg_data = data_values[:, top_genes_idx]
    hvg_gene_names = [gene_names[i] for i in top_genes_idx]
    
    logger.info(f"选择了 {len(hvg_gene_names)} 个高变异基因")
    
    # 返回与输入相同类型的数据
    if is_dataframe:
        return pd.DataFrame(hvg_data, index=data.index, columns=hvg_gene_names), hvg_gene_names
    else:
        return hvg_data, hvg_gene_names

def preprocess_pipeline(
    data: Union[np.ndarray, pd.DataFrame],
    gene_names: Optional[List[str]] = None,
    normalize_method: str = 'log1p',
    min_cells: int = 3,
    min_genes: int = 200,
    n_top_genes: int = 2000,
    batch_labels: Optional[List[int]] = None,
    batch_correction_method: str = 'none',
    **kwargs
) -> Tuple[Union[np.ndarray, pd.DataFrame], List[str]]:
    """
    完整的预处理流程
    
    Args:
        data: 基因表达矩阵，形状为 [n_samples, n_genes]
        gene_names: 基因名称列表
        normalize_method: 标准化方法
        min_cells: 基因必须在至少这么多细胞中表达
        min_genes: 细胞必须表达至少这么多基因
        n_top_genes: 选择的高变异基因数量
        batch_labels: 每个样本的批次标签
        batch_correction_method: 批次校正方法
        **kwargs: 其他参数
        
    Returns:
        预处理后的数据和基因名称列表
    """
    logger.info("开始数据预处理流程")
    
    # 1. 过滤细胞
    filtered_data = filter_cells(data, min_genes=min_genes, **kwargs)
    
    # 2. 过滤基因
    filtered_data, filtered_gene_names = filter_genes(
        filtered_data, gene_names=gene_names, min_cells=min_cells, **kwargs
    )
    
    # 3. 标准化
    normalized_data = normalize_data(filtered_data, method=normalize_method, **kwargs)
    
    # 4. 批次校正（如果提供了批次标签）
    if batch_labels is not None and batch_correction_method != 'none':
        corrected_data = batch_correct(
            normalized_data, batch_labels=batch_labels, method=batch_correction_method, **kwargs
        )
    else:
        corrected_data = normalized_data
    
    # 5. 选择高变异基因
    hvg_data, hvg_gene_names = select_highly_variable_genes(
        corrected_data, gene_names=filtered_gene_names, n_top_genes=n_top_genes, **kwargs
    )
    
    logger.info("数据预处理流程完成")
    
    return hvg_data, hvg_gene_names 