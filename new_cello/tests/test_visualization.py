"""
可视化模块测试
"""

import os
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from new_cello.core.visualization.plots import (
    plot_expression_heatmap,
    plot_umap,
    plot_tsne,
    plot_pca,
    plot_cell_type_distribution,
    plot_feature_importance
)

class TestVisualization:
    """可视化测试类"""
    
    @pytest.fixture
    def sample_expression_data(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        创建样本表达数据用于测试
        
        Returns:
            Tuple: (表达矩阵, 基因名称列表, 样本名称列表)
        """
        # 创建随机表达矩阵 [n_samples, n_genes]
        n_samples = 20
        n_genes = 50
        
        # 使用稀疏矩阵模拟单细胞数据（大部分值为0）
        data = np.random.exponential(scale=1.0, size=(n_samples, n_genes))
        data = data * (np.random.rand(n_samples, n_genes) > 0.8)  # 80%的值为0
        
        # 创建基因名称和样本名称
        gene_names = [f"gene_{i}" for i in range(n_genes)]
        sample_names = [f"sample_{i}" for i in range(n_samples)]
        
        return data, gene_names, sample_names
    
    @pytest.fixture
    def sample_embedding(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建样本降维结果用于测试
        
        Returns:
            Tuple: (降维坐标, 标签)
        """
        # 创建随机降维坐标 [n_samples, 2]
        n_samples = 100
        embedding = np.random.randn(n_samples, 2)
        
        # 创建随机标签
        n_classes = 3
        labels = np.random.randint(0, n_classes, size=n_samples)
        
        return embedding, labels
    
    @pytest.fixture
    def sample_feature_importance(self) -> Dict[str, float]:
        """
        创建样本特征重要性用于测试
        
        Returns:
            Dict: 特征名称到重要性的映射
        """
        # 创建随机特征重要性
        n_features = 30
        features = [f"feature_{i}" for i in range(n_features)]
        importances = np.random.rand(n_features)
        
        # 归一化重要性
        importances = importances / np.sum(importances)
        
        return dict(zip(features, importances))
    
    def test_plot_expression_heatmap(self, sample_expression_data, tmp_path):
        """测试绘制表达热图"""
        data, gene_names, sample_names = sample_expression_data
        
        # 绘制热图
        fig = plot_expression_heatmap(
            data,
            gene_names=gene_names,
            sample_names=sample_names,
            n_top_genes=20,
            cluster_genes=True,
            cluster_samples=True,
            title='Test Heatmap'
        )
        
        # 检查图形
        assert isinstance(fig, plt.Figure)
        
        # 保存图形
        output_path = os.path.join(tmp_path, "heatmap.png")
        plt.savefig(output_path)
        plt.close(fig)
        
        # 检查文件是否存在
        assert os.path.exists(output_path)
    
    def test_plot_umap(self, sample_embedding, tmp_path):
        """测试绘制UMAP图"""
        embedding, labels = sample_embedding
        
        # 绘制UMAP图
        fig = plot_umap(
            embedding,
            labels=labels,
            title='Test UMAP'
        )
        
        # 检查图形
        assert isinstance(fig, plt.Figure)
        
        # 保存图形
        output_path = os.path.join(tmp_path, "umap.png")
        plt.savefig(output_path)
        plt.close(fig)
        
        # 检查文件是否存在
        assert os.path.exists(output_path)
    
    def test_plot_tsne(self, sample_embedding, tmp_path):
        """测试绘制t-SNE图"""
        embedding, labels = sample_embedding
        
        # 绘制t-SNE图
        fig = plot_tsne(
            embedding,
            labels=labels,
            title='Test t-SNE'
        )
        
        # 检查图形
        assert isinstance(fig, plt.Figure)
        
        # 保存图形
        output_path = os.path.join(tmp_path, "tsne.png")
        plt.savefig(output_path)
        plt.close(fig)
        
        # 检查文件是否存在
        assert os.path.exists(output_path)
    
    def test_plot_pca(self, sample_embedding, tmp_path):
        """测试绘制PCA图"""
        embedding, labels = sample_embedding
        
        # 绘制PCA图
        fig = plot_pca(
            embedding,
            labels=labels,
            title='Test PCA'
        )
        
        # 检查图形
        assert isinstance(fig, plt.Figure)
        
        # 保存图形
        output_path = os.path.join(tmp_path, "pca.png")
        plt.savefig(output_path)
        plt.close(fig)
        
        # 检查文件是否存在
        assert os.path.exists(output_path)
    
    def test_plot_cell_type_distribution(self, sample_embedding, tmp_path):
        """测试绘制细胞类型分布图"""
        _, labels = sample_embedding
        
        # 绘制细胞类型分布图
        fig = plot_cell_type_distribution(
            labels,
            title='Test Cell Type Distribution',
            sort_by_count=True
        )
        
        # 检查图形
        assert isinstance(fig, plt.Figure)
        
        # 保存图形
        output_path = os.path.join(tmp_path, "cell_type_distribution.png")
        plt.savefig(output_path)
        plt.close(fig)
        
        # 检查文件是否存在
        assert os.path.exists(output_path)
    
    def test_plot_feature_importance(self, sample_feature_importance, tmp_path):
        """测试绘制特征重要性图"""
        # 绘制特征重要性图
        fig = plot_feature_importance(
            sample_feature_importance,
            top_n=10,
            title='Test Feature Importance',
            horizontal=True
        )
        
        # 检查图形
        assert isinstance(fig, plt.Figure)
        
        # 保存图形
        output_path = os.path.join(tmp_path, "feature_importance.png")
        plt.savefig(output_path)
        plt.close(fig)
        
        # 检查文件是否存在
        assert os.path.exists(output_path) 