"""
可视化绘图函数
"""

import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union, Optional, Tuple, Any

logger = logging.getLogger(__name__)

def plot_expression_heatmap(
    data: Union[np.ndarray, pd.DataFrame],
    gene_names: Optional[List[str]] = None,
    sample_names: Optional[List[str]] = None,
    n_top_genes: int = 50,
    cluster_genes: bool = True,
    cluster_samples: bool = True,
    cmap: str = 'viridis',
    figsize: Tuple[int, int] = (12, 10),
    title: str = 'Gene Expression Heatmap',
    output_path: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    绘制基因表达热图
    
    Args:
        data: 基因表达矩阵，形状为 [n_samples, n_genes]
        gene_names: 基因名称列表
        sample_names: 样本名称列表
        n_top_genes: 显示的顶部基因数量
        cluster_genes: 是否对基因进行聚类
        cluster_samples: 是否对样本进行聚类
        cmap: 颜色映射
        figsize: 图形大小
        title: 图形标题
        output_path: 输出文件路径
        **kwargs: 传递给seaborn.clustermap的其他参数
        
    Returns:
        matplotlib图形对象
    """
    logger.info(f"绘制基因表达热图，显示前 {n_top_genes} 个基因")
    
    # 确保数据是DataFrame
    if isinstance(data, np.ndarray):
        if gene_names is None:
            gene_names = [f"gene_{i}" for i in range(data.shape[1])]
        if sample_names is None:
            sample_names = [f"sample_{i}" for i in range(data.shape[0])]
        
        data_df = pd.DataFrame(data, index=sample_names, columns=gene_names)
    else:
        data_df = data
    
    # 选择变异最大的基因
    if data_df.shape[1] > n_top_genes:
        gene_var = data_df.var(axis=0)
        top_genes = gene_var.nlargest(n_top_genes).index
        data_df = data_df[top_genes]
    
    # 绘制热图
    if cluster_genes or cluster_samples:
        # 使用clustermap进行聚类
        row_cluster = cluster_samples
        col_cluster = cluster_genes
        
        g = sns.clustermap(
            data_df,
            cmap=cmap,
            figsize=figsize,
            row_cluster=row_cluster,
            col_cluster=col_cluster,
            **kwargs
        )
        
        # 添加标题
        plt.suptitle(title, y=1.02)
        fig = g.fig
    else:
        # 使用普通热图
        plt.figure(figsize=figsize)
        sns.heatmap(data_df, cmap=cmap, **kwargs)
        plt.title(title)
        plt.tight_layout()
        fig = plt.gcf()
    
    # 保存图形
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        logger.info(f"热图已保存到 {output_path}")
    
    return fig

def plot_dimensionality_reduction(
    embedding: np.ndarray,
    labels: Optional[Union[np.ndarray, pd.Series]] = None,
    method: str = 'UMAP',
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    cmap: str = 'tab10',
    alpha: float = 0.7,
    s: int = 30,
    legend: bool = True,
    output_path: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    绘制降维结果
    
    Args:
        embedding: 降维后的坐标，形状为 [n_samples, 2]
        labels: 样本标签
        method: 降维方法名称
        figsize: 图形大小
        title: 图形标题
        cmap: 颜色映射
        alpha: 点的透明度
        s: 点的大小
        legend: 是否显示图例
        output_path: 输出文件路径
        **kwargs: 传递给plt.scatter的其他参数
        
    Returns:
        matplotlib图形对象
    """
    logger.info(f"绘制{method}降维结果")
    
    # 创建图形
    plt.figure(figsize=figsize)
    
    # 绘制散点图
    if labels is not None:
        # 确保标签是数组
        if isinstance(labels, pd.Series):
            labels = labels.values
        
        # 获取唯一标签
        unique_labels = np.unique(labels)
        
        # 为每个标签绘制散点
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[plt.cm.get_cmap(cmap)(i / len(unique_labels))],
                label=label,
                alpha=alpha,
                s=s,
                **kwargs
            )
        
        if legend:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # 没有标签，使用单一颜色
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            alpha=alpha,
            s=s,
            **kwargs
        )
    
    # 设置标题和轴标签
    if title:
        plt.title(title)
    else:
        plt.title(f"{method} Embedding")
    
    plt.xlabel(f"{method}1")
    plt.ylabel(f"{method}2")
    
    plt.tight_layout()
    
    # 保存图形
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        logger.info(f"{method}图已保存到 {output_path}")
    
    return plt.gcf()

def plot_umap(
    embedding: np.ndarray,
    labels: Optional[Union[np.ndarray, pd.Series]] = None,
    **kwargs
) -> plt.Figure:
    """
    绘制UMAP降维结果
    
    Args:
        embedding: UMAP坐标，形状为 [n_samples, 2]
        labels: 样本标签
        **kwargs: 传递给plot_dimensionality_reduction的其他参数
        
    Returns:
        matplotlib图形对象
    """
    return plot_dimensionality_reduction(embedding, labels, method='UMAP', **kwargs)

def plot_tsne(
    embedding: np.ndarray,
    labels: Optional[Union[np.ndarray, pd.Series]] = None,
    **kwargs
) -> plt.Figure:
    """
    绘制t-SNE降维结果
    
    Args:
        embedding: t-SNE坐标，形状为 [n_samples, 2]
        labels: 样本标签
        **kwargs: 传递给plot_dimensionality_reduction的其他参数
        
    Returns:
        matplotlib图形对象
    """
    return plot_dimensionality_reduction(embedding, labels, method='t-SNE', **kwargs)

def plot_pca(
    embedding: np.ndarray,
    labels: Optional[Union[np.ndarray, pd.Series]] = None,
    **kwargs
) -> plt.Figure:
    """
    绘制PCA降维结果
    
    Args:
        embedding: PCA坐标，形状为 [n_samples, 2]
        labels: 样本标签
        **kwargs: 传递给plot_dimensionality_reduction的其他参数
        
    Returns:
        matplotlib图形对象
    """
    return plot_dimensionality_reduction(embedding, labels, method='PCA', **kwargs)

def plot_cell_type_distribution(
    labels: Union[np.ndarray, pd.Series, List],
    figsize: Tuple[int, int] = (10, 6),
    title: str = 'Cell Type Distribution',
    sort_by_count: bool = True,
    horizontal: bool = False,
    cmap: str = 'viridis',
    output_path: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    绘制细胞类型分布
    
    Args:
        labels: 细胞类型标签
        figsize: 图形大小
        title: 图形标题
        sort_by_count: 是否按计数排序
        horizontal: 是否使用水平条形图
        cmap: 颜色映射
        output_path: 输出文件路径
        **kwargs: 传递给plt.bar或plt.barh的其他参数
        
    Returns:
        matplotlib图形对象
    """
    logger.info("绘制细胞类型分布")
    
    # 计算每种细胞类型的数量
    if isinstance(labels, (np.ndarray, list)):
        labels = pd.Series(labels)
    
    counts = labels.value_counts()
    
    # 排序
    if sort_by_count:
        counts = counts.sort_values(ascending=False)
    
    # 创建图形
    plt.figure(figsize=figsize)
    
    # 获取颜色
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(counts)))
    
    # 绘制条形图
    if horizontal:
        plt.barh(counts.index, counts.values, color=colors, **kwargs)
        plt.xlabel('Count')
        plt.ylabel('Cell Type')
    else:
        plt.bar(counts.index, counts.values, color=colors, **kwargs)
        plt.xlabel('Cell Type')
        plt.ylabel('Count')
        plt.xticks(rotation=90)
    
    plt.title(title)
    plt.tight_layout()
    
    # 保存图形
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        logger.info(f"细胞类型分布图已保存到 {output_path}")
    
    return plt.gcf()

def plot_feature_importance(
    feature_importance: Dict[str, float],
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    title: str = 'Feature Importance',
    color: str = 'skyblue',
    horizontal: bool = True,
    output_path: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    绘制特征重要性
    
    Args:
        feature_importance: 特征名称到重要性的映射
        top_n: 显示前N个重要特征
        figsize: 图形大小
        title: 图形标题
        color: 条形颜色
        horizontal: 是否使用水平条形图
        output_path: 输出文件路径
        **kwargs: 传递给plt.bar或plt.barh的其他参数
        
    Returns:
        matplotlib图形对象
    """
    logger.info(f"绘制前 {top_n} 个重要特征")
    
    # 转换为DataFrame并排序
    importance_df = pd.DataFrame({
        'feature': list(feature_importance.keys()),
        'importance': list(feature_importance.values())
    })
    importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
    
    # 创建图形
    plt.figure(figsize=figsize)
    
    # 绘制条形图
    if horizontal:
        plt.barh(importance_df['feature'], importance_df['importance'], color=color, **kwargs)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
    else:
        plt.bar(importance_df['feature'], importance_df['importance'], color=color, **kwargs)
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.xticks(rotation=90)
    
    plt.title(title)
    plt.tight_layout()
    
    # 保存图形
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        logger.info(f"特征重要性图已保存到 {output_path}")
    
    return plt.gcf() 