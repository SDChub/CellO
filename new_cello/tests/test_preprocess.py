"""
数据预处理模块测试
"""

import numpy as np
import pandas as pd
import pytest
from typing import Dict, List, Tuple

from new_cello.preprocess import (
    normalize_data,
    filter_genes,
    filter_cells,
    select_highly_variable_genes,
    preprocess_pipeline
)

class TestPreprocessing:
    """数据预处理测试类"""
    
    @pytest.fixture
    def sample_data(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        创建样本数据用于测试
        
        Returns:
            Tuple: (表达数据框, 基因名称列表)
        """
        # 创建随机表达矩阵 [n_samples, n_genes]
        n_samples = 50
        n_genes = 100
        
        # 使用稀疏矩阵模拟单细胞数据（大部分值为0）
        data = np.random.exponential(scale=1.0, size=(n_samples, n_genes))
        data = data * (np.random.rand(n_samples, n_genes) > 0.8)  # 80%的值为0
        
        # 创建基因名称和样本名称
        gene_names = [f"gene_{i}" for i in range(n_genes)]
        sample_names = [f"sample_{i}" for i in range(n_samples)]
        
        # 创建DataFrame
        df = pd.DataFrame(data, index=sample_names, columns=gene_names)
        
        return df, gene_names
    
    def test_normalize_data_log1p(self, sample_data):
        """测试log1p标准化"""
        data, _ = sample_data
        
        # 应用log1p标准化
        normalized = normalize_data(data, method='log1p')
        
        # 检查结果类型
        assert isinstance(normalized, pd.DataFrame)
        assert normalized.shape == data.shape
        
        # 检查log1p转换是否正确应用
        expected = np.log1p(data.values)
        assert np.allclose(normalized.values, expected)
    
    def test_normalize_data_cpm(self, sample_data):
        """测试CPM标准化"""
        data, _ = sample_data
        
        # 应用CPM标准化
        normalized = normalize_data(data, method='cpm', scale_factor=1e6)
        
        # 检查结果类型
        assert isinstance(normalized, pd.DataFrame)
        assert normalized.shape == data.shape
        
        # 检查CPM转换是否正确应用
        row_sums = data.sum(axis=1).values.reshape(-1, 1)
        expected = data.values * 1e6 / row_sums
        assert np.allclose(normalized.values, expected, rtol=1e-5)
    
    def test_normalize_data_numpy(self, sample_data):
        """测试NumPy数组输入"""
        data, _ = sample_data
        
        # 转换为NumPy数组
        data_np = data.values
        
        # 应用标准化
        normalized = normalize_data(data_np, method='log1p')
        
        # 检查结果类型
        assert isinstance(normalized, np.ndarray)
        assert normalized.shape == data_np.shape
        
        # 检查log1p转换是否正确应用
        expected = np.log1p(data_np)
        assert np.allclose(normalized, expected)
    
    def test_filter_genes(self, sample_data):
        """测试基因过滤"""
        data, gene_names = sample_data
        
        # 应用基因过滤
        filtered_data, filtered_genes = filter_genes(
            data, 
            gene_names=gene_names,
            min_cells=5,  # 基因必须在至少5个细胞中表达
            min_counts=1   # 表达量至少为1
        )
        
        # 检查结果类型
        assert isinstance(filtered_data, pd.DataFrame)
        assert isinstance(filtered_genes, list)
        
        # 检查过滤是否正确应用
        assert filtered_data.shape[1] <= data.shape[1]  # 列数应该减少或保持不变
        assert len(filtered_genes) == filtered_data.shape[1]
        
        # 检查每个保留的基因是否满足条件
        for gene_idx, gene in enumerate(filtered_genes):
            gene_expr = filtered_data.iloc[:, gene_idx]
            assert np.sum(gene_expr >= 1) >= 5  # 至少5个细胞表达该基因
    
    def test_filter_cells(self, sample_data):
        """测试细胞过滤"""
        data, _ = sample_data
        
        # 应用细胞过滤
        filtered_data = filter_cells(
            data,
            min_genes=10,  # 细胞必须表达至少10个基因
            min_counts=20  # 细胞总计数至少为20
        )
        
        # 检查结果类型
        assert isinstance(filtered_data, pd.DataFrame)
        
        # 检查过滤是否正确应用
        assert filtered_data.shape[0] <= data.shape[0]  # 行数应该减少或保持不变
        
        # 检查每个保留的细胞是否满足条件
        for idx, row in filtered_data.iterrows():
            assert np.sum(row > 0) >= 10  # 至少表达10个基因
            assert np.sum(row) >= 20  # 总计数至少为20
    
    def test_select_highly_variable_genes(self, sample_data):
        """测试高变异基因选择"""
        data, gene_names = sample_data
        
        # 应用高变异基因选择
        n_top_genes = 20
        hvg_data, hvg_genes = select_highly_variable_genes(
            data,
            gene_names=gene_names,
            n_top_genes=n_top_genes,
            method='seurat'
        )
        
        # 检查结果类型
        assert isinstance(hvg_data, pd.DataFrame)
        assert isinstance(hvg_genes, list)
        
        # 检查选择的基因数量
        assert hvg_data.shape[1] == n_top_genes
        assert len(hvg_genes) == n_top_genes
        
        # 检查选择的基因是否是原始基因的子集
        assert all(gene in gene_names for gene in hvg_genes)
    
    def test_preprocess_pipeline(self, sample_data):
        """测试完整预处理流程"""
        data, gene_names = sample_data
        
        # 应用完整预处理流程
        processed_data, processed_genes = preprocess_pipeline(
            data,
            gene_names=gene_names,
            normalize_method='log1p',
            min_cells=3,
            min_genes=5,
            n_top_genes=30
        )
        
        # 检查结果类型
        assert isinstance(processed_data, pd.DataFrame)
        assert isinstance(processed_genes, list)
        
        # 检查处理后的数据形状
        assert processed_data.shape[1] == 30  # n_top_genes
        assert len(processed_genes) == 30
        
        # 检查处理后的数据是否包含NaN或无穷大
        assert not np.any(np.isnan(processed_data.values))
        assert not np.any(np.isinf(processed_data.values))
    
    def test_preprocess_pipeline_with_batch_correction(self, sample_data):
        """测试带批次校正的预处理流程"""
        data, gene_names = sample_data
        
        # 创建批次标签（一半样本为批次0，一半为批次1）
        n_samples = data.shape[0]
        batch_labels = np.zeros(n_samples)
        batch_labels[n_samples//2:] = 1
        
        # 应用带批次校正的预处理流程
        processed_data, processed_genes = preprocess_pipeline(
            data,
            gene_names=gene_names,
            normalize_method='log1p',
            min_cells=3,
            min_genes=5,
            n_top_genes=30,
            batch_labels=batch_labels,
            batch_correction_method='none'  # 使用'none'避免依赖外部库
        )
        
        # 检查结果类型
        assert isinstance(processed_data, pd.DataFrame)
        assert isinstance(processed_genes, list)
        
        # 检查处理后的数据形状
        assert processed_data.shape[1] == 30  # n_top_genes
        assert len(processed_genes) == 30