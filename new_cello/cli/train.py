"""
训练命令行接口
"""

import argparse
import logging
import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from new_cello import models
from new_cello.data import load_csv, load_h5ad, load_10x, load_labels, split_data
from new_cello.preprocess import preprocess_pipeline
from new_cello.evaluation import calculate_metrics, plot_confusion_matrix, plot_roc_curve, cross_validate

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
    parser = argparse.ArgumentParser(description='训练CellO模型')
    
    # 输入数据参数
    parser.add_argument('--input', '-i', required=True, help='输入数据文件路径')
    parser.add_argument('--format', choices=['csv', 'h5ad', '10x'], default='csv', help='输入数据格式')
    parser.add_argument('--transpose', action='store_true', help='是否转置输入数据')
    parser.add_argument('--labels', required=True, help='标签文件路径')
    parser.add_argument('--label-column', default='label', help='标签列名')
    parser.add_argument('--id-column', help='样本ID列名')
    
    # 模型参数
    parser.add_argument('--model-type', choices=['traditional', 'deep_learning', 'ensemble'], 
                        default='ensemble', help='模型类型')
    parser.add_argument('--model-name', help='模型名称')
    parser.add_argument('--model-params', help='模型参数JSON字符串或文件路径')
    
    # 训练参数
    parser.add_argument('--test-size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--cross-validation', type=int, default=0, help='交叉验证折数，0表示不进行交叉验证')
    parser.add_argument('--random-state', type=int, default=42, help='随机种子')
    
    # 预处理参数
    parser.add_argument('--normalize', choices=['log1p', 'cpm', 'tpm', 'none'], 
                        default='log1p', help='标准化方法')
    parser.add_argument('--n-top-genes', type=int, default=2000, help='使用的高变异基因数量')
    parser.add_argument('--min-cells', type=int, default=3, help='基因必须在至少这么多细胞中表达')
    parser.add_argument('--min-genes', type=int, default=200, help='细胞必须表达至少这么多基因')
    
    # 输出参数
    parser.add_argument('--output-dir', '-o', required=True, help='输出目录路径')
    parser.add_argument('--save-preprocessed', action='store_true', help='是否保存预处理后的数据')
    
    # 其他参数
    parser.add_argument('--verbose', '-v', action='store_true', help='启用详细日志')
    parser.add_argument('--gpu', action='store_true', help='使用GPU加速')
    
    return parser.parse_args()

def load_data(args):
    """加载输入数据和标签"""
    logger.info(f"加载输入数据: {args.input}")
    
    # 加载表达数据
    if args.format == 'csv':
        data, gene_names = load_csv(args.input, transpose=args.transpose)
    elif args.format == 'h5ad':
        data, gene_names, _ = load_h5ad(args.input)
    elif args.format == '10x':
        data, gene_names = load_10x(args.input)
    else:
        raise ValueError(f"不支持的数据格式: {args.format}")
    
    logger.info(f"数据加载完成，形状: {data.shape}")
    
    # 加载标签
    logger.info(f"加载标签: {args.labels}")
    labels = load_labels(args.labels, args.label_column, args.id_column)
    
    # 确保数据和标签的索引一致
    common_indices = data.index.intersection(labels.index)
    if len(common_indices) < len(data):
        logger.warning(f"数据和标签的索引不完全匹配，只使用共同的 {len(common_indices)} 个样本")
        data = data.loc[common_indices]
        labels = labels.loc[common_indices]
    
    logger.info(f"标签加载完成，样本数: {len(labels)}")
    return data, gene_names, labels

def preprocess_data(data, gene_names, args):
    """预处理数据"""
    logger.info("预处理数据")
    
    # 应用预处理流程
    processed_data, processed_gene_names = preprocess_pipeline(
        data,
        gene_names=gene_names,
        normalize_method=args.normalize,
        min_cells=args.min_cells,
        min_genes=args.min_genes,
        n_top_genes=args.n_top_genes
    )
    
    logger.info(f"数据预处理完成，形状: {processed_data.shape}")
    
    # 如果需要保存预处理后的数据
    if args.save_preprocessed:
        preprocessed_dir = os.path.join(args.output_dir, 'preprocessed')
        os.makedirs(preprocessed_dir, exist_ok=True)
        
        # 保存预处理后的数据
        preprocessed_path = os.path.join(preprocessed_dir, 'preprocessed_data.csv')
        processed_data.to_csv(preprocessed_path)
        
        # 保存基因名称
        gene_names_path = os.path.join(preprocessed_dir, 'gene_names.txt')
        with open(gene_names_path, 'w') as f:
            for gene in processed_gene_names:
                f.write(f"{gene}\n")
                
        logger.info(f"预处理后的数据已保存到 {preprocessed_dir}")
    
    return processed_data, processed_gene_names

def create_model(args, gene_names=None):
    """创建模型"""
    logger.info(f"创建模型: {args.model_type}")
    
    # 确定模型类型和名称
    model_type = args.model_type
    model_name = args.model_name
    
    # 解析模型参数
    model_params = {}
    if args.model_params:
        if os.path.isfile(args.model_params):
            # 从文件加载参数
            with open(args.model_params, 'r') as f:
                model_params = json.load(f)
        else:
            # 解析JSON字符串
            model_params = json.loads(args.model_params)
    
    # 如果使用GPU，添加设备参数
    if args.gpu:
        model_params['device'] = 'cuda'
    
    # 创建模型
    if model_type == 'traditional':
        from new_cello.models.traditional import get_traditional_model
        if model_name is None:
            model_name = "random_forest"
        model = get_traditional_model(model_name, **model_params)
    elif model_type == 'deep_learning':
        from new_cello.models.deep_learning import get_deep_learning_model
        if model_name is None:
            model_name = "transformer"
        model = get_deep_learning_model(model_name, **model_params)
    elif model_type == 'ensemble':
        from new_cello.models.ensemble import get_ensemble_model
        if model_name is None:
            model_name = "voting"
        model = get_ensemble_model(model_name, **model_params)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    logger.info(f"模型创建完成: {model_type}/{model_name}")
    return model

def train_model(model, X_train, y_train, gene_names=None):
    """训练模型"""
    logger.info("开始训练模型")
    
    # 训练模型
    if hasattr(model, 'fit'):
        # 如果模型接受gene_names参数
        if 'gene_names' in model.fit.__code__.co_varnames:
            model.fit(X_train, y_train, gene_names=gene_names)
        else:
            model.fit(X_train, y_train)
    else:
        raise ValueError("模型没有fit方法")
    
    logger.info("模型训练完成")
    return model

def evaluate_model(model, X_test, y_test):
    """评估模型"""
    logger.info("评估模型")
    
    # 进行预测
    y_pred = model.predict(X_test)
    
    # 如果模型有predict_proba方法，获取概率
    y_prob = None
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)
    
    # 计算评估指标
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    
    # 打印评估结果
    logger.info(f"评估结果:")
    logger.info(f"  准确率: {metrics['accuracy']:.4f}")
    logger.info(f"  精确率: {metrics['precision']:.4f}")
    logger.info(f"  召回率: {metrics['recall']:.4f}")
    logger.info(f"  F1分数: {metrics['f1']:.4f}")
    
    return metrics, y_pred, y_prob

def save_model(model, args, metrics=None):
    """保存模型和评估结果"""
    logger.info(f"保存模型到: {args.output_dir}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存模型
    model_dir = os.path.join(args.output_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    if hasattr(model, 'save'):
        model.save(model_dir)
    else:
        logger.warning("模型没有save方法，无法保存")
    
    # 保存模型配置
    config = {
        'model_type': args.model_type,
        'model_name': args.model_name,
        'preprocessing': {
            'normalize': args.normalize,
            'n_top_genes': args.n_top_genes,
            'min_cells': args.min_cells,
            'min_genes': args.min_genes
        }
    }
    
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # 如果有评估指标，保存评估结果
    if metrics is not None:
        # 创建评估目录
        eval_dir = os.path.join(args.output_dir, 'evaluation')
        os.makedirs(eval_dir, exist_ok=True)
        
        # 保存基本指标
        basic_metrics = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        }
        
        if 'roc_auc' in metrics:
            basic_metrics['roc_auc'] = metrics['roc_auc']
        
        metrics_path = os.path.join(eval_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(basic_metrics, f, indent=2)
        
        # 保存混淆矩阵可视化
        cm_path = os.path.join(eval_dir, 'confusion_matrix.png')
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            save_path=cm_path
        )
    
    logger.info(f"模型和配置已保存到 {args.output_dir}")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志
    setup_logging(args.verbose)
    
    try:
        # 加载数据和标签
        data, gene_names, labels = load_data(args)
        
        # 预处理数据
        processed_data, processed_gene_names = preprocess_data(data, gene_names, args)
        
        # 如果需要进行交叉验证
        if args.cross_validation > 1:
            logger.info(f"执行 {args.cross_validation} 折交叉验证")
            
            # 创建模型工厂函数
            def model_factory(**kwargs):
                return create_model(args)
            
            # 执行交叉验证
            cv_results = cross_validate(
                model_factory,
                processed_data,
                labels,
                n_splits=args.cross_validation,
                random_state=args.random_state
            )
            
            # 保存交叉验证结果
            cv_dir = os.path.join(args.output_dir, 'cross_validation')
            os.makedirs(cv_dir, exist_ok=True)
            
            cv_metrics_path = os.path.join(cv_dir, 'cv_metrics.json')
            with open(cv_metrics_path, 'w') as f:
                json.dump(cv_results['mean_metrics'], f, indent=2)
                
            logger.info(f"交叉验证结果已保存到 {cv_dir}")
        
        # 分割数据为训练集和测试集
        X_train, X_test, y_train, y_test = split_data(
            processed_data,
            labels,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        # 创建模型
        model = create_model(args, processed_gene_names)
        
        # 训练模型
        model = train_model(model, X_train, y_train, processed_gene_names)
        
        # 评估模型
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)
        
        # 保存模型和评估结果
        save_model(model, args, metrics)
        
        logger.info("训练流程完成")
        
    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()  
