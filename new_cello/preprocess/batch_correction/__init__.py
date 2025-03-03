"""
批次效应校正模块
"""

from new_cello.preprocess.batch_correction.methods import (
    correct_batch_effect,
    combat_correct,
    harmony_correct,
    scanorama_correct,
    mnncorrect
)

def correct_batch_effect(
    data, 
    batch_labels, 
    method='combat', 
    **kwargs
):
    """
    批次效应校正的统一接口
    
    Args:
        data: 基因表达矩阵，形状为 [n_samples, n_genes]
        batch_labels: 批次标签，长度为n_samples
        method: 校正方法，可选 'combat', 'harmony', 'scanorama'
        **kwargs: 传递给具体校正方法的其他参数
        
    Returns:
        校正后的数据，与输入类型相同
    """
    if method == 'combat':
        return combat_correct(data, batch_labels, **kwargs)
    elif method == 'harmony':
        return harmony_correct(data, batch_labels, **kwargs)
    elif method == 'scanorama':
        return scanorama_correct(data, batch_labels, **kwargs)
    else:
        raise ValueError(f"未知的批次校正方法: {method}") 