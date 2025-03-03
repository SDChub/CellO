"""
Scanorama批次效应校正实现
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# 尝试导入Scanorama相关库
try:
    import scanorama
    SCANORAMA_AVAILABLE = True
except ImportError:
    try:
        import scanpy as sc
        SCANORAMA_AVAILABLE = True
    except ImportError:
        SCANORAMA_AVAILABLE = False
        logger.warning("scanorama或scanpy库未安装，Scanorama批次校正将不可用。请使用 'pip install scanorama' 或 'pip install scanpy' 安装。")

def scanorama_correct(
    data: Union[np.ndarray, pd.DataFrame],
    batch_labels: Union[np.ndarray, pd.Series, List[int]],
    n_pcs: int = 100,
    knn: int = 20,
    sigma: float = 15,
    alpha: float = 0.1,
    batch_size: int = 5000,
    random_state: int = 42,
    **kwargs
) -> Union[np.ndarray, pd.DataFrame]:
    """
    使用Scanorama方法进行批次效应校正
    
    Args:
        data: 基因表达矩阵，形状为 [n_samples, n_genes]
        batch_labels: 批次标签，长度为n_samples
        n_pcs: 使用的主成分数量
        knn: k近邻数量
        sigma: 高斯核宽度
        alpha: 对齐强度
        batch_size: 批处理大小
        random_state: 随机种子
        **kwargs: 其他参数
        
    Returns:
        校正后的数据，与输入类型相同
    """
    if not SCANORAMA_AVAILABLE:
        raise ImportError("scanorama或scanpy库未安装，无法使用Scanorama批次校正")
    
    logger.info("使用Scanorama方法进行批次效应校正")
    
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
        # 尝试使用scanorama
        if 'scanorama' in globals():
            # 将数据按批次分割
            datasets = []
            batch_indices = []
            
            for batch in unique_batches:
                batch_mask = batch_labels == batch
                batch_data = data_values[batch_mask]
                datasets.append(batch_data)
                batch_indices.append(np.where(batch_mask)[0])
            
            # 运行Scanorama
            corrected_datasets, corrected_genes = scanorama.correct_scanpy(
                datasets,
                [columns] * len(datasets) if columns is not None else None,
                return_dimred=False,
                return_dense=True,
                dimred=n_pcs,
                knn=knn,
                sigma=sigma,
                alpha=alpha,
                batch_size=batch_size,
                seed=random_state
            )
            
            # 重建完整的校正数据
            corrected_data = np.zeros_like(data_values)
            for i, indices in enumerate(batch_indices):
                corrected_data[indices] = corrected_datasets[i]
        else:
            # 使用scanpy的scanorama包装器
            adata = sc.AnnData(X=data_values)
            adata.obs['batch'] = batch_labels
            
            # 运行Scanorama
            sc.external.pp.scanorama_integrate(
                adata,
                'batch',
                basis='X',
                adjusted_basis='X_scanorama',
                knn=knn,
                sigma=sigma,
                alpha=alpha,
                batch_size=batch_size,
                random_state=random_state
            )
            
            # 获取校正后的数据
            if 'X_scanorama' in adata.obsm:
                corrected_data = adata.obsm['X_scanorama']
            else:
                corrected_data = adata.X
    except Exception as e:
        logger.error(f"Scanorama校正失败: {str(e)}")
        logger.info("尝试使用自定义Scanorama实现")
        
        # 使用自定义Scanorama实现
        corrected_data = _custom_scanorama(
            data_values,
            batch_labels,
            n_pcs=n_pcs,
            knn=knn,
            random_state=random_state
        )
    
    # 返回与输入相同类型的数据
    if is_dataframe:
        return pd.DataFrame(corrected_data, index=index, columns=columns)
    else:
        return corrected_data

def _custom_scanorama(
    data: np.ndarray,
    batch_labels: np.ndarray,
    n_pcs: int = 100,
    knn: int = 20,
    random_state: int = 42
) -> np.ndarray:
    """
    自定义Scanorama实现，用于当scanorama不可用时
    
    Args:
        data: 基因表达矩阵，形状为 [n_samples, n_genes]
        batch_labels: 批次标签，长度为n_samples
        n_pcs: 使用的主成分数量
        knn: k近邻数量
        random_state: 随机种子
        
    Returns:
        校正后的数据
    """
    # 导入必要的库
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    
    # 获取数据维度
    n_samples, n_genes = data.shape
    
    # 获取唯一批次
    unique_batches = np.unique(batch_labels)
    n_batches = len(unique_batches)
    
    # 如果数据维度很高，先进行PCA降维
    if n_genes > 100:
        n_pcs = min(n_pcs, n_samples // 2)
        pca = PCA(n_components=n_pcs, random_state=random_state)
        data_pca = pca.fit_transform(data)
    else:
        data_pca = data.copy()
        n_pcs = n_genes
    
    # 将数据按批次分割
    batch_data_pca = {}
    for batch in unique_batches:
        batch_mask = batch_labels == batch
        batch_data_pca[batch] = data_pca[batch_mask]
    
    # 对每个批次找到锚点
    anchors = {}
    for batch in unique_batches:
        # 为每个批次找到与其他批次最相似的样本
        batch_anchors = []
        
        # 当前批次的数据
        current_data = batch_data_pca[batch]
        
        for other_batch in unique_batches:
            if batch == other_batch:
                continue
                
            # 其他批次的数据
            other_data = batch_data_pca[other_batch]
            
            # 找到最近的样本对
            nn_current = NearestNeighbors(n_neighbors=min(knn, other_data.shape[0]), algorithm='ball_tree')
            nn_current.fit(current_data)
            
            nn_other = NearestNeighbors(n_neighbors=min(knn, current_data.shape[0]), algorithm='ball_tree')
            nn_other.fit(other_data)
            
            # 找到互为最近邻的样本对
            distances_current, indices_current = nn_current.kneighbors(other_data)
            distances_other, indices_other = nn_other.kneighbors(current_data)
            
            # 找到互为最近邻的样本对
            for i in range(other_data.shape[0]):
                for j, idx in enumerate(indices_current[i]):
                    if i in indices_other[idx]:
                        batch_anchors.append((idx, i, other_batch))
                        break
        
        anchors[batch] = batch_anchors
    
    # 使用锚点计算批次间的转换
    corrected_data_pca = data_pca.copy()
    
    for batch in unique_batches:
        batch_mask = batch_labels == batch
        batch_indices = np.where(batch_mask)[0]
        
        # 如果没有锚点，跳过
        if not anchors[batch]:
            continue
        
        # 计算批次间的平均偏移
        offsets = []
        for idx, other_idx, other_batch in anchors[batch]:
            # 当前批次中锚点的索引
            global_idx = batch_indices[idx]
            
            # 其他批次中锚点的索引
            other_batch_mask = batch_labels == other_batch
            other_batch_indices = np.where(other_batch_mask)[0]
            global_other_idx = other_batch_indices[other_idx]
            
            # 计算偏移
            offset = data_pca[global_other_idx] - data_pca[global_idx]
            offsets.append(offset)
        
        # 计算平均偏移
        if offsets:
            mean_offset = np.mean(offsets, axis=0)
            
            # 应用偏移
            corrected_data_pca[batch_indices] += mean_offset
    
    # 如果进行了PCA，转换回原始空间
    if n_genes > 100:
        corrected_data = np.dot(corrected_data_pca, pca.components_) + pca.mean_
    else:
        corrected_data = corrected_data_pca
    
    return corrected_data 