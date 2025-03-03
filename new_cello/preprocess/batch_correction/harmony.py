"""
Harmony批次效应校正实现
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# 尝试导入Harmony相关库
try:
    import harmonypy
    HARMONY_AVAILABLE = True
except ImportError:
    try:
        import scanpy as sc
        HARMONY_AVAILABLE = True
    except ImportError:
        HARMONY_AVAILABLE = False
        logger.warning("harmonypy或scanpy库未安装，Harmony批次校正将不可用。请使用 'pip install harmonypy' 或 'pip install scanpy' 安装。")

def harmony_correct(
    data: Union[np.ndarray, pd.DataFrame],
    batch_labels: Union[np.ndarray, pd.Series, List[int]],
    n_clusters: int = 50,
    max_iter_harmony: int = 20,
    theta: float = 2.0,
    lambda_val: float = 1.0,
    sigma: float = 0.1,
    n_pcs: Optional[int] = None,
    random_state: int = 42,
    **kwargs
) -> Union[np.ndarray, pd.DataFrame]:
    """
    使用Harmony方法进行批次效应校正
    
    Args:
        data: 基因表达矩阵，形状为 [n_samples, n_genes]，或降维后的数据 [n_samples, n_pcs]
        batch_labels: 批次标签，长度为n_samples
        n_clusters: 聚类数量
        max_iter_harmony: 最大迭代次数
        theta: 多样性惩罚参数
        lambda_val: 岭回归参数
        sigma: 高斯核宽度
        n_pcs: 使用的主成分数量，如果为None则使用所有特征
        random_state: 随机种子
        **kwargs: 其他参数
        
    Returns:
        校正后的数据，与输入类型相同
    """
    if not HARMONY_AVAILABLE:
        raise ImportError("harmonypy或scanpy库未安装，无法使用Harmony批次校正")
    
    logger.info("使用Harmony方法进行批次效应校正")
    
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
        # 尝试使用harmonypy
        if 'harmonypy' in sys.modules:
            # 如果数据维度很高，先进行PCA降维
            if data_values.shape[1] > 100 and n_pcs is not None:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=n_pcs, random_state=random_state)
                data_pca = pca.fit_transform(data_values)
                
                # 运行Harmony
                harmony_out = harmonypy.run_harmony(
                    data_pca.T,  # Harmony需要 [n_pcs, n_samples] 格式
                    batch_labels,
                    n_clusters=n_clusters,
                    max_iter_harmony=max_iter_harmony,
                    theta=theta,
                    lambda_val=lambda_val,
                    sigma=sigma,
                    random_state=random_state
                )
                
                # 获取校正后的PCA坐标
                corrected_pca = harmony_out.Z_corr.T  # 转回 [n_samples, n_pcs] 格式
                
                # 转换回原始空间
                corrected_data = np.dot(corrected_pca, pca.components_) + pca.mean_
            else:
                # 直接在原始空间运行Harmony
                harmony_out = harmonypy.run_harmony(
                    data_values.T,  # Harmony需要 [n_genes, n_samples] 格式
                    batch_labels,
                    n_clusters=n_clusters,
                    max_iter_harmony=max_iter_harmony,
                    theta=theta,
                    lambda_val=lambda_val,
                    sigma=sigma,
                    random_state=random_state
                )
                
                # 获取校正后的数据
                corrected_data = harmony_out.Z_corr.T  # 转回 [n_samples, n_genes] 格式
        else:
            # 使用scanpy的harmony包装器
            adata = sc.AnnData(X=data_values)
            adata.obs['batch'] = batch_labels
            
            # 如果数据维度很高，先进行PCA降维
            if data_values.shape[1] > 100 and n_pcs is not None:
                sc.pp.pca(adata, n_comps=n_pcs, random_state=random_state)
            
            # 运行Harmony
            sc.external.pp.harmony_integrate(
                adata,
                key='batch',
                n_clusters=n_clusters,
                max_iter_harmony=max_iter_harmony,
                theta=theta,
                lambda_val=lambda_val,
                sigma=sigma,
                random_state=random_state
            )
            
            # 获取校正后的数据
            if 'X_pca_harmony' in adata.obsm:
                # 如果进行了PCA，需要转换回原始空间
                from sklearn.decomposition import PCA
                pca = PCA(n_components=n_pcs, random_state=random_state)
                pca.fit(data_values)
                
                corrected_pca = adata.obsm['X_pca_harmony']
                corrected_data = np.dot(corrected_pca, pca.components_) + pca.mean_
            else:
                # 直接使用校正后的数据
                corrected_data = adata.X
    except Exception as e:
        logger.error(f"Harmony校正失败: {str(e)}")
        logger.info("尝试使用自定义Harmony实现")
        
        # 使用自定义Harmony实现
        corrected_data = _custom_harmony(
            data_values,
            batch_labels,
            n_clusters=n_clusters,
            max_iter=max_iter_harmony,
            random_state=random_state
        )
    
    # 返回与输入相同类型的数据
    if is_dataframe:
        return pd.DataFrame(corrected_data, index=index, columns=columns)
    else:
        return corrected_data

def _custom_harmony(
    data: np.ndarray,
    batch_labels: np.ndarray,
    n_clusters: int = 50,
    max_iter: int = 20,
    random_state: int = 42
) -> np.ndarray:
    """
    自定义Harmony实现，用于当harmonypy不可用时
    
    Args:
        data: 基因表达矩阵，形状为 [n_samples, n_genes]
        batch_labels: 批次标签，长度为n_samples
        n_clusters: 聚类数量
        max_iter: 最大迭代次数
        random_state: 随机种子
        
    Returns:
        校正后的数据
    """
    # 导入必要的库
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.linear_model import Ridge
    
    # 获取数据维度
    n_samples, n_genes = data.shape
    
    # 如果数据维度很高，先进行PCA降维
    if n_genes > 100:
        n_pcs = min(50, n_samples // 2)
        pca = PCA(n_components=n_pcs, random_state=random_state)
        data_pca = pca.fit_transform(data)
    else:
        data_pca = data.copy()
        n_pcs = n_genes
    
    # 获取唯一批次
    unique_batches = np.unique(batch_labels)
    n_batches = len(unique_batches)
    
    # 初始化校正后的数据
    corrected_pca = data_pca.copy()
    
    # Harmony迭代
    for iter_idx in range(max_iter):
        # 聚类
        kmeans = KMeans(n_clusters=min(n_clusters, n_samples // 2), random_state=random_state)
        cluster_labels = kmeans.fit_predict(corrected_pca)
        
        # 计算每个聚类中每个批次的比例
        cluster_batch_counts = np.zeros((kmeans.n_clusters, n_batches))
        for i, batch in enumerate(unique_batches):
            for j in range(kmeans.n_clusters):
                cluster_batch_counts[j, i] = np.sum((cluster_labels == j) & (batch_labels == batch))
        
        # 计算每个聚类的总样本数
        cluster_totals = np.sum(cluster_batch_counts, axis=1, keepdims=True)
        
        # 计算每个聚类中每个批次的比例
        cluster_batch_props = cluster_batch_counts / (cluster_totals + 1e-8)
        
        # 计算每个批次的全局比例
        batch_props = np.sum(cluster_batch_counts, axis=0) / np.sum(cluster_batch_counts)
        
        # 计算每个聚类中每个批次的偏差
        batch_bias = cluster_batch_props - batch_props
        
        # 对每个批次进行校正
        for i, batch in enumerate(unique_batches):
            batch_mask = batch_labels == batch
            
            # 对每个聚类计算校正因子
            for j in range(kmeans.n_clusters):
                cluster_mask = cluster_labels == j
                
                # 计算校正因子
                correction = batch_bias[j, i]
                
                # 应用校正
                if np.sum(batch_mask & cluster_mask) > 0:
                    corrected_pca[batch_mask & cluster_mask] -= correction
    
    # 如果进行了PCA，转换回原始空间
    if n_genes > 100:
        corrected_data = np.dot(corrected_pca, pca.components_) + pca.mean_
    else:
        corrected_data = corrected_pca
    
    return corrected_data 