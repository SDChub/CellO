"""
数据加载模块，用于加载和处理单细胞RNA-seq数据
"""

import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any

logger = logging.getLogger(__name__)

def load_csv(
    file_path: str,
    gene_column: Optional[str] = None,
    transpose: bool = False,
    **kwargs
) -> Tuple[pd.DataFrame, Optional[List[str]]]:
    """
    从CSV文件加载基因表达数据
    
    Args:
        file_path: CSV文件路径
        gene_column: 包含基因名称的列名，如果为None则假设第一列是基因名称
        transpose: 是否转置数据（如果基因是行而不是列）
        **kwargs: 传递给pd.read_csv的其他参数
        
    Returns:
        基因表达数据框和基因名称列表
    """
    logger.info(f"从CSV文件加载数据: {file_path}")
    
    try:
        # 读取CSV文件
        if gene_column is not None:
            # 如果指定了基因列名，将其设为索引
            data = pd.read_csv(file_path, **kwargs)
            gene_names = data[gene_column].tolist()
            data = data.drop(columns=[gene_column])
        else:
            # 否则假设第一列是基因名称
            data = pd.read_csv(file_path, index_col=0, **kwargs)
            gene_names = data.index.tolist()
        
        # 如果需要转置数据
        if transpose:
            data = data.T
            gene_names = data.columns.tolist()
            
        logger.info(f"成功加载数据，形状: {data.shape}")
        return data, gene_names
    except Exception as e:
        logger.error(f"加载CSV文件时出错: {str(e)}")
        raise

def load_h5ad(
    file_path: str,
    **kwargs
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    从H5AD文件加载基因表达数据（AnnData格式）
    
    Args:
        file_path: H5AD文件路径
        **kwargs: 其他参数
        
    Returns:
        基因表达数据框、基因名称列表和元数据字典
    """
    logger.info(f"从H5AD文件加载数据: {file_path}")
    
    try:
        # 尝试导入scanpy
        import scanpy as sc
    except ImportError:
        logger.error("无法导入scanpy库，请使用 'pip install scanpy' 安装")
        raise
    
    try:
        # 读取H5AD文件
        adata = sc.read_h5ad(file_path)
        
        # 提取表达矩阵
        data = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X)
        data.index = adata.obs_names
        data.columns = adata.var_names
        
        # 提取基因名称
        gene_names = adata.var_names.tolist()
        
        # 提取元数据
        metadata = {
            'obs': adata.obs,
            'var': adata.var,
            'uns': adata.uns
        }
        
        logger.info(f"成功加载数据，形状: {data.shape}")
        return data, gene_names, metadata
    except Exception as e:
        logger.error(f"加载H5AD文件时出错: {str(e)}")
        raise

def load_10x(
    directory: str,
    genome: Optional[str] = None,
    **kwargs
) -> Tuple[pd.DataFrame, List[str]]:
    """
    从10X Genomics格式加载基因表达数据
    
    Args:
        directory: 包含10X数据的目录
        genome: 基因组名称（如果有多个基因组）
        **kwargs: 其他参数
        
    Returns:
        基因表达数据框和基因名称列表
    """
    logger.info(f"从10X格式加载数据: {directory}")
    
    try:
        # 尝试导入scanpy
        import scanpy as sc
    except ImportError:
        logger.error("无法导入scanpy库，请使用 'pip install scanpy' 安装")
        raise
    
    try:
        # 读取10X数据
        adata = sc.read_10x_mtx(directory, genome=genome)
        
        # 提取表达矩阵
        data = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X)
        data.index = adata.obs_names
        data.columns = adata.var_names
        
        # 提取基因名称
        gene_names = adata.var_names.tolist()
        
        logger.info(f"成功加载数据，形状: {data.shape}")
        return data, gene_names
    except Exception as e:
        logger.error(f"加载10X数据时出错: {str(e)}")
        raise

def load_labels(
    file_path: str,
    label_column: str,
    id_column: Optional[str] = None,
    **kwargs
) -> pd.Series:
    """
    加载样本标签
    
    Args:
        file_path: 标签文件路径
        label_column: 包含标签的列名
        id_column: 包含样本ID的列名，如果为None则使用文件的索引
        **kwargs: 传递给pd.read_csv的其他参数
        
    Returns:
        样本标签序列
    """
    logger.info(f"从文件加载标签: {file_path}")
    
    try:
        # 读取标签文件
        labels_df = pd.read_csv(file_path, **kwargs)
        
        # 提取标签
        if id_column is not None:
            labels = labels_df.set_index(id_column)[label_column]
        else:
            labels = labels_df[label_column]
            
        logger.info(f"成功加载标签，样本数: {len(labels)}")
        return labels
    except Exception as e:
        logger.error(f"加载标签文件时出错: {str(e)}")
        raise

def save_data(
    data: pd.DataFrame,
    gene_names: List[str],
    file_path: str,
    format: str = 'csv',
    **kwargs
) -> None:
    """
    保存基因表达数据
    
    Args:
        data: 基因表达数据框
        gene_names: 基因名称列表
        file_path: 保存文件路径
        format: 保存格式，可选 'csv', 'h5ad'
        **kwargs: 其他参数
    """
    logger.info(f"保存数据到文件: {file_path}, 格式: {format}")
    
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        if format == 'csv':
            # 保存为CSV
            data.to_csv(file_path, **kwargs)
        elif format == 'h5ad':
            # 保存为H5AD
            try:
                import scanpy as sc
                import anndata
            except ImportError:
                logger.error("无法导入scanpy或anndata库，请使用 'pip install scanpy anndata' 安装")
                raise
                
            # 创建AnnData对象
            adata = anndata.AnnData(X=data.values)
            adata.obs_names = data.index
            adata.var_names = gene_names
            
            # 保存
            adata.write(file_path)
        else:
            raise ValueError(f"不支持的保存格式: {format}")
            
        logger.info(f"数据保存成功")
    except Exception as e:
        logger.error(f"保存数据时出错: {str(e)}")
        raise

def split_data(
    data: pd.DataFrame,
    labels: pd.Series,
    test_size: float = 0.2,
    stratify: bool = True,
    random_state: int = 42,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    将数据分割为训练集和测试集
    
    Args:
        data: 基因表达数据框
        labels: 样本标签序列
        test_size: 测试集比例
        stratify: 是否进行分层抽样
        random_state: 随机种子
        **kwargs: 其他参数
        
    Returns:
        训练数据、测试数据、训练标签、测试标签
    """
    logger.info(f"将数据分割为训练集和测试集，测试集比例: {test_size}")
    
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        logger.error("无法导入sklearn库，请使用 'pip install scikit-learn' 安装")
        raise
    
    try:
        # 确保索引一致
        common_indices = data.index.intersection(labels.index)
        if len(common_indices) < len(data):
            logger.warning(f"数据和标签的索引不完全匹配，只使用共同的 {len(common_indices)} 个样本")
            data = data.loc[common_indices]
            labels = labels.loc[common_indices]
        
        # 分割数据
        stratify_param = labels if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=test_size, 
            stratify=stratify_param, random_state=random_state
        )
        
        logger.info(f"数据分割完成，训练集: {X_train.shape[0]} 样本，测试集: {X_test.shape[0]} 样本")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"分割数据时出错: {str(e)}")
        raise 