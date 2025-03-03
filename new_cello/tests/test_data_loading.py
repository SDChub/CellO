"""
数据加载模块测试
"""

import os
import numpy as np
import pandas as pd
import pytest
from typing import Dict, List, Tuple
import tempfile

from new_cello.data import (
    load_csv,
    load_labels,
    save_data,
    split_data
)

class TestDataLoading:
    """数据加载测试类"""
    
    @pytest.fixture
    def sample_expression_data(self) -> pd.DataFrame:
        """
        创建样本表达数据用于测试
        
        Returns:
            pd.DataFrame: 表达数据框
        """
        # 创建随机表达矩阵 [n_samples, n_genes]
        n_samples = 30
        n_genes = 50
        
        # 使用稀疏矩阵模拟单细胞数据（大部分值为0）
        data = np.random.exponential(scale=1.0, size=(n_samples, n_genes))
        data = data * (np.random.rand(n_samples, n_genes) > 0.8)  # 80%的值为0
        
        # 创建基因名称和样本名称
        gene_names = [f"gene_{i}" for i in range(n_genes)]
        sample_names = [f"sample_{i}" for i in range(n_samples)]
        
        # 创建DataFrame
        df = pd.DataFrame(data, index=sample_names, columns=gene_names)
        
        return df
    
    @pytest.fixture
    def sample_labels(self) -> pd.DataFrame:
        """
        创建样本标签数据用于测试
        
        Returns:
            pd.DataFrame: 标签数据框
        """
        # 创建样本名称
        n_samples = 30
        sample_names = [f"sample_{i}" for i in range(n_samples)]
        
        # 创建随机标签
        labels = np.random.randint(0, 3, size=n_samples)
        
        # 创建DataFrame
        df = pd.DataFrame({
            'sample_id': sample_names,
            'label': labels
        })
        
        return df
    
    def test_load_csv(self, sample_expression_data, tmp_path):
        """测试从CSV加载数据"""
        # 保存数据到临时CSV文件
        csv_path = os.path.join(tmp_path, "expression.csv")
        sample_expression_data.to_csv(csv_path)
        
        # 加载数据
        loaded_data, gene_names = load_csv(csv_path)
        
        # 检查加载的数据
        assert isinstance(loaded_data, pd.DataFrame)
        assert isinstance(gene_names, list)
        assert loaded_data.shape == sample_expression_data.shape
        assert np.allclose(loaded_data.values, sample_expression_data.values)
        assert gene_names == sample_expression_data.columns.tolist()
    
    def test_load_csv_with_transpose(self, sample_expression_data, tmp_path):
        """测试从CSV加载并转置数据"""
        # 转置数据
        transposed_data = sample_expression_data.T
        
        # 保存转置数据到临时CSV文件
        csv_path = os.path.join(tmp_path, "expression_transposed.csv")
        transposed_data.to_csv(csv_path)
        
        # 加载数据并转置回来
        loaded_data, gene_names = load_csv(csv_path, transpose=True)
        
        # 检查加载的数据
        assert isinstance(loaded_data, pd.DataFrame)
        assert isinstance(gene_names, list)
        assert loaded_data.shape == sample_expression_data.shape
        assert np.allclose(loaded_data.values, sample_expression_data.values, rtol=1e-5)
    
    def test_load_labels(self, sample_labels, tmp_path):
        """测试加载标签"""
        # 保存标签到临时CSV文件
        csv_path = os.path.join(tmp_path, "labels.csv")
        sample_labels.to_csv(csv_path, index=False)
        
        # 加载标签
        loaded_labels = load_labels(csv_path, label_column='label', id_column='sample_id')
        
        # 检查加载的标签
        assert isinstance(loaded_labels, pd.Series)
        assert len(loaded_labels) == len(sample_labels)
        assert np.array_equal(loaded_labels.values, sample_labels['label'].values)
        assert list(loaded_labels.index) == sample_labels['sample_id'].tolist()
    
    def test_save_data(self, sample_expression_data, tmp_path):
        """测试保存数据"""
        # 准备数据
        data = sample_expression_data
        gene_names = data.columns.tolist()
        
        # 保存数据到临时CSV文件
        csv_path = os.path.join(tmp_path, "saved_expression.csv")
        save_data(data, gene_names, csv_path, format='csv')
        
        # 检查文件是否存在
        assert os.path.exists(csv_path)
        
        # 加载保存的数据
        loaded_data = pd.read_csv(csv_path, index_col=0)
        
        # 检查加载的数据
        assert loaded_data.shape == data.shape
        assert np.allclose(loaded_data.values, data.values)
        assert loaded_data.columns.tolist() == gene_names
    
    def test_split_data(self, sample_expression_data, sample_labels):
        """测试数据分割"""
        # 准备数据
        data = sample_expression_data
        labels = pd.Series(sample_labels['label'].values, index=sample_labels['sample_id'])
        
        # 确保索引一致
        common_indices = data.index.intersection(labels.index)
        data = data.loc[common_indices]
        labels = labels.loc[common_indices]
        
        # 分割数据
        test_size = 0.3
        X_train, X_test, y_train, y_test = split_data(
            data, labels, test_size=test_size, stratify=True, random_state=42
        )
        
        # 检查分割结果
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        
        # 检查分割比例
        assert len(X_train) + len(X_test) == len(data)
        assert len(y_train) + len(y_test) == len(labels)
        assert abs(len(X_test) / len(data) - test_size) < 0.1  # 允许一些误差
        
        # 检查训练集和测试集没有重叠
        assert len(set(X_train.index) & set(X_test.index)) == 0
        assert len(set(y_train.index) & set(y_test.index)) == 0
    
    def test_split_data_with_mismatched_indices(self, sample_expression_data, sample_labels):
        """测试索引不匹配时的数据分割"""
        # 准备数据
        data = sample_expression_data
        
        # 修改标签索引，使其与数据索引部分不匹配
        modified_indices = [f"sample_{i}" for i in range(10, 40)]  # 只有部分重叠
        labels = pd.Series(
            sample_labels['label'].values[:30], 
            index=modified_indices
        )
        
        # 分割数据
        X_train, X_test, y_train, y_test = split_data(
            data, labels, test_size=0.3, stratify=False, random_state=42
        )
        
        # 检查分割结果
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        
        # 检查只使用了共同的索引
        common_indices = data.index.intersection(labels.index)
        assert len(X_train) + len(X_test) == len(common_indices)
        assert len(y_train) + len(y_test) == len(common_indices)
        
        # 检查训练集和测试集的索引一致性
        assert set(X_train.index) == set(y_train.index)
        assert set(X_test.index) == set(y_test.index) 