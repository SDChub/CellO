"""
批次校正模块测试
"""

import numpy as np
import pandas as pd
import pytest
from typing import Dict, List, Tuple

# 尝试导入批次校正模块
try:
    from new_cello.preprocess.batch_correction import (
        correct_batch_effect,
        combat_correct,
        harmony_correct,
        scanorama_correct,
        mnncorrect
    )
    from new_cello.preprocess.batch_correction.methods import (
        COMBAT_AVAILABLE,
        HARMONY_AVAILABLE,
        SCANORAMA_AVAILABLE,
        MNN_AVAILABLE
    )
    BATCH_CORRECTION_AVAILABLE = True
except ImportError:
    BATCH_CORRECTION_AVAILABLE = False

# 如果批次校正模块不可用，则跳过所有测试
pytestmark = pytest.mark.skipif(
    not BATCH_CORRECTION_AVAILABLE,
    reason="批次校正模块未安装，无法测试批次校正功能"
)

class TestBatchCorrection:
    """批次校正测试类"""
    
    @pytest.fixture
    def sample_batch_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建样本批次数据用于测试
        
        Returns:
            Tuple: (表达矩阵, 批次标签)
        """
        # 创建两个批次的数据，每个批次有不同的均值
        n_samples_batch1 = 50
        n_samples_batch2 = 40
        n_genes = 100
        
        # 批次1：均值为0的数据
        batch1 = np.random.normal(0, 1, size=(n_samples_batch1, n_genes))
        
        # 批次2：均值为2的数据
        batch2 = np.random.normal(2, 1, size=(n_samples_batch2, n_genes))
        
        # 合并数据
        data = np.vstack([batch1, batch2])
        
        # 创建批次标签
        batch_labels = np.array([0] * n_samples_batch1 + [1] * n_samples_batch2)
        
        return data, batch_labels
    
    @pytest.fixture
    def sample_batch_dataframe(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        创建样本批次DataFrame用于测试
        
        Returns:
            Tuple: (表达数据框, 批次标签)
        """
        # 获取NumPy数组数据
        data, batch_labels = self.sample_batch_data()
        
        # 创建样本名称和基因名称
        sample_names = [f"sample_{i}" for i in range(data.shape[0])]
        gene_names = [f"gene_{i}" for i in range(data.shape[1])]
        
        # 创建DataFrame
        df = pd.DataFrame(data, index=sample_names, columns=gene_names)
        
        return df, batch_labels
    
    def test_correct_batch_effect_interface(self, sample_batch_data):
        """测试批次校正统一接口"""
        data, batch_labels = sample_batch_data
        
        # 测试统一接口
        corrected = correct_batch_effect(data, batch_labels, method='none')
        
        # 检查结果
        assert isinstance(corrected, np.ndarray)
        assert corrected.shape == data.shape
        
        # 测试无效方法
        with pytest.raises(ValueError):
            correct_batch_effect(data, batch_labels, method='invalid_method')
    
    @pytest.mark.skipif(not COMBAT_AVAILABLE, reason="ComBat不可用")
    def test_combat_correct(self, sample_batch_data):
        """测试ComBat批次校正"""
        data, batch_labels = sample_batch_data
        
        # 应用ComBat校正
        corrected = combat_correct(data, batch_labels)
        
        # 检查结果
        assert isinstance(corrected, np.ndarray)
        assert corrected.shape == data.shape
        
        # 检查批次效应是否减少
        batch1_mean = np.mean(corrected[batch_labels == 0], axis=0)
        batch2_mean = np.mean(corrected[batch_labels == 1], axis=0)
        
        # 校正后两个批次的均值差异应该减小
        original_mean_diff = np.abs(np.mean(data[batch_labels == 0]) - np.mean(data[batch_labels == 1]))
        corrected_mean_diff = np.abs(np.mean(batch1_mean) - np.mean(batch2_mean))
        
        assert corrected_mean_diff < original_mean_diff
    
    @pytest.mark.skipif(not HARMONY_AVAILABLE, reason="Harmony不可用")
    def test_harmony_correct(self, sample_batch_data):
        """测试Harmony批次校正"""
        data, batch_labels = sample_batch_data
        
        # 应用Harmony校正
        corrected = harmony_correct(data, batch_labels, n_components=10)
        
        # 检查结果
        assert isinstance(corrected, np.ndarray)
        assert corrected.shape == data.shape
    
    @pytest.mark.skipif(not SCANORAMA_AVAILABLE, reason="Scanorama不可用")
    def test_scanorama_correct(self, sample_batch_data):
        """测试Scanorama批次校正"""
        data, batch_labels = sample_batch_data
        
        # 应用Scanorama校正
        corrected = scanorama_correct(data, batch_labels)
        
        # 检查结果
        assert isinstance(corrected, np.ndarray)
        assert corrected.shape == data.shape
    
    @pytest.mark.skipif(not MNN_AVAILABLE, reason="MNN不可用")
    def test_mnn_correct(self, sample_batch_data):
        """测试MNN批次校正"""
        data, batch_labels = sample_batch_data
        
        # 应用MNN校正
        corrected = mnncorrect(data, batch_labels)
        
        # 检查结果
        assert isinstance(corrected, np.ndarray)
        assert corrected.shape == data.shape
    
    def test_dataframe_input(self, sample_batch_dataframe):
        """测试DataFrame输入"""
        df, batch_labels = sample_batch_dataframe
        
        # 应用批次校正
        corrected = correct_batch_effect(df, batch_labels, method='none')
        
        # 检查结果
        assert isinstance(corrected, pd.DataFrame)
        assert corrected.shape == df.shape
        assert list(corrected.index) == list(df.index)
        assert list(corrected.columns) == list(df.columns)
    
    def test_preprocess_integration(self, sample_batch_dataframe):
        """测试与预处理流程的集成"""
        from new_cello.preprocess import preprocess_pipeline
        
        df, batch_labels = sample_batch_dataframe
        
        # 应用预处理流程，包括批次校正
        processed_data, processed_gene_names = preprocess_pipeline(
            df,
            gene_names=df.columns.tolist(),
            normalize_method='log1p',
            min_cells=1,
            min_genes=1,
            n_top_genes=50,
            batch_labels=batch_labels,
            batch_correction_method='none'  # 使用'none'避免依赖外部库
        )
        
        # 检查结果
        assert isinstance(processed_data, pd.DataFrame)
        assert processed_data.shape[1] == 50  # n_top_genes
        assert len(processed_gene_names) == 50 