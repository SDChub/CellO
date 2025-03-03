"""
预测命令行接口
"""

import argparse
import logging
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from new_cello import models
from new_cello.data import load_csv, load_h5ad, load_10x
from new_cello.preprocess import preprocess_pipeline
from new_cello.evaluation import calculate_metrics, plot_confusion_matrix, plot_roc_curve

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = False):
    """设置日志级别"""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用CellO模型进行细胞类型预测')
    
    # 输入数据参数
    parser.add_argument('--input', '-i', required=True, help='输入数据文件路径')
    parser.add_argument('--format', choices=['csv', 'h5ad', '10x'], default='csv', help='输入数据格式')
    parser.add_argument('--transpose', action='store_true', help='是否转置输入数据')
    
    # 模型参数
    parser.add_argument('--model-dir', required=True, help='模型目录路径')
    parser.add_argument('--model-type', choices=['traditional', 'deep_learning', 'ensemble'], 
                        default='ensemble', help='模型类型')
    parser.add_argument('--model-name', help='模型名称')
    
    # 预处理参数
    parser.add_argument('--normalize', choices=['log1p', 'cpm', 'tpm', 'none'], 
                        default='log1p', help='标准化方法')
    parser.add_argument('--n-top-genes', type=int, default=2000, help='使用的高变异基因数量')
    
    # 输出参数
    parser.add_argument('--output', '-o', required=True, help='输出预测结果的文件路径')
    parser.add_argument('--output-proba', help='输出预测概率的文件路径')
    parser.add_argument('--plot-dir', help='保存可视化结果的目录')
    
    # 评估参数
    parser.add_argument('--true-labels', help='真实标签文件路径')
    parser.add_argument('--label-column', default='label', help='标签列名')
    
    # 其他参数
    parser.add_argument('--verbose', '-v', action='store_true', help='启用详细日志')
    parser.add_argument('--gpu', action='store_true', help='使用GPU加速')
    
    return parser.parse_args()

def load_data(args):
    """加载输入数据"""
    logger.info(f"加载输入数据: {args.input}")
    
    if args.format == 'csv':
        data, gene_names = load_csv(args.input, transpose=args.transpose)
    elif args.format == 'h5ad':
        data, gene_names, _ = load_h5ad(args.input)
    elif args.format == '10x':
        data, gene_names = load_10x(args.input)
    else:
        raise ValueError(f"不支持的数据格式: {args.format}")
    
    logger.info(f"数据加载完成，形状: {data.shape}")
    return data, gene_names

def load_model(args):
    """加载模型"""
    logger.info(f"加载模型: {args.model_dir}")
    
    # 确定模型类型和名称
    model_type = args.model_type
    model_name = args.model_name
    
    # 加载模型
    if model_type == 'traditional':
        from new_cello.models.traditional import get_traditional_model
        if model_name is None:
            model_name = "random_forest"
        model = get_traditional_model(model_name)
    elif model_type == 'deep_learning':
        from new_cello.models.deep_learning import get_deep_learning_model
        if model_name is None:
            model_name = "transformer"
        model = get_deep_learning_model(model_name)
    elif model_type == 'ensemble':
        from new_cello.models.ensemble import get_ensemble_model
        if model_name is None:
            model_name = "voting"
        model = get_ensemble_model(model_name)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 加载模型权重
    if hasattr(model, 'load'):
        model = model.load(args.model_dir)
    else:
        logger.warning(f"模型没有load方法，使用未训练的模型")
    
    logger.info(f"模型加载完成: {model_type}/{model_name}")
    return model

def preprocess_data(data, gene_names, args):
    """预处理数据"""
    logger.info("预处理数据")
    
    # 应用预处理流程
    processed_data, processed_gene_names = preprocess_pipeline(
        data,
        gene_names=gene_names,
        normalize_method=args.normalize,
        n_top_genes=args.n_top_genes
    )
    
    logger.info(f"数据预处理完成，形状: {processed_data.shape}")
    return processed_data, processed_gene_names

def predict(model, data, gene_names, args):
    """使用模型进行预测"""
    logger.info("开始预测")
    
    # 进行预测
    predictions = model.predict(data)
    
    # 如果模型有predict_proba方法，获取概率
    probabilities = None
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(data)
    
    logger.info(f"预测完成，样本数: {len(predictions)}")
    return predictions, probabilities

def save_results(data, predictions, probabilities, args):
    """保存预测结果"""
    logger.info(f"保存预测结果: {args.output}")
    
    # 创建结果DataFrame
    results = pd.DataFrame({
        'sample_id': data.index,
        'predicted_label': predictions
    })
    
    # 保存预测结果
    results.to_csv(args.output, index=False)
    
    # 如果有概率和输出路径，保存概率
    if probabilities is not None and args.output_proba:
        logger.info(f"保存预测概率: {args.output_proba}")
        
        # 创建概率DataFrame
        proba_df = pd.DataFrame(
            probabilities,
            index=data.index,
            columns=[f'class_{i}' for i in range(probabilities.shape[1])]
        )
        
        # 保存概率
        proba_df.to_csv(args.output_proba)
    
    logger.info("结果保存完成")

def evaluate(predictions, probabilities, args):
    """评估预测结果（如果提供了真实标签）"""
    if not args.true_labels:
        logger.info("未提供真实标签，跳过评估")
        return
    
    logger.info(f"加载真实标签: {args.true_labels}")
    
    # 加载真实标签
    true_labels_df = pd.read_csv(args.true_labels)
    true_labels = true_labels_df[args.label_column].values
    
    # 计算评估指标
    metrics = calculate_metrics(true_labels, predictions, probabilities)
    
    # 打印评估结果
    logger.info(f"评估结果:")
    logger.info(f"  准确率: {metrics['accuracy']:.4f}")
    logger.info(f"  精确率: {metrics['precision']:.4f}")
    logger.info(f"  召回率: {metrics['recall']:.4f}")
    logger.info(f"  F1分数: {metrics['f1']:.4f}")
    
    # 如果提供了绘图目录，保存可视化结果
    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        
        # 绘制混淆矩阵
        cm_path = os.path.join(args.plot_dir, 'confusion_matrix.png')
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            save_path=cm_path
        )
        
        # 如果有概率，绘制ROC曲线
        if probabilities is not None:
            roc_path = os.path.join(args.plot_dir, 'roc_curve.png')
            plot_roc_curve(
                true_labels,
                probabilities,
                save_path=roc_path
            )
    
    return metrics

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志
    setup_logging(args.verbose)
    
    try:
        # 加载数据
        data, gene_names = load_data(args)
        
        # 预处理数据
        processed_data, processed_gene_names = preprocess_data(data, gene_names, args)
        
        # 加载模型
        model = load_model(args)
        
        # 进行预测
        predictions, probabilities = predict(model, processed_data, processed_gene_names, args)
        
        # 保存结果
        save_results(data, predictions, probabilities, args)
        
        # 评估结果
        evaluate(predictions, probabilities, args)
        
        logger.info("预测流程完成")
        
    except Exception as e:
        logger.error(f"预测过程中出错: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()  
