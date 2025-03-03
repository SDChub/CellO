"""
可视化命令行接口
"""

import argparse
import logging
import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union, Optional, Tuple, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from new_cello.data import load_csv, load_h5ad, load_10x, load_labels
from new_cello.preprocess import preprocess_pipeline
from new_cello.evaluation import (
    plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve,
    plot_feature_importance
)

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
    parser = argparse.ArgumentParser(description='可视化CellO预测结果')
    
    # 输入数据参数
    parser.add_argument('--predictions', '-p', required=True, help='预测结果文件路径')
    parser.add_argument('--probabilities', help='预测概率文件路径')
    parser.add_argument('--true-labels', help='真实标签文件路径')
    parser.add_argument('--label-column', default='label', help='标签列名')
    parser.add_argument('--feature-importance', help='特征重要性文件路径')
    
    # 可视化参数
    parser.add_argument('--plot-type', choices=['confusion', 'roc', 'pr', 'feature', 'all'], 
                        default='all', help='可视化类型')
    parser.add_argument('--class-names', help='类别名称文件路径，每行一个名称')
    parser.add_argument('--top-n-features', type=int, default=20, help='显示前N个重要特征')
    
    # 输出参数
    parser.add_argument('--output-dir', '-o', required=True, help='输出目录路径')
    parser.add_argument('--dpi', type=int, default=300, help='图像DPI')
    parser.add_argument('--figsize', help='图像大小，格式为"宽,高"')
    
    # 其他参数
    parser.add_argument('--verbose', '-v', action='store_true', help='启用详细日志')
    
    return parser.parse_args()

def load_data(args):
    """加载预测结果、概率和真实标签"""
    data = {}
    
    # 加载预测结果
    logger.info(f"加载预测结果: {args.predictions}")
    predictions_df = pd.read_csv(args.predictions)
    data['predictions'] = predictions_df
    
    # 加载预测概率（如果提供）
    if args.probabilities:
        logger.info(f"加载预测概率: {args.probabilities}")
        probabilities_df = pd.read_csv(args.probabilities)
        data['probabilities'] = probabilities_df
    
    # 加载真实标签（如果提供）
    if args.true_labels:
        logger.info(f"加载真实标签: {args.true_labels}")
        true_labels_df = pd.read_csv(args.true_labels)
        data['true_labels'] = true_labels_df
    
    # 加载特征重要性（如果提供）
    if args.feature_importance:
        logger.info(f"加载特征重要性: {args.feature_importance}")
        try:
            # 尝试加载JSON格式
            with open(args.feature_importance, 'r') as f:
                feature_importance = json.load(f)
            data['feature_importance'] = feature_importance
        except json.JSONDecodeError:
            # 尝试加载CSV格式
            feature_importance_df = pd.read_csv(args.feature_importance)
            feature_importance = dict(zip(
                feature_importance_df.iloc[:, 0],
                feature_importance_df.iloc[:, 1]
            ))
            data['feature_importance'] = feature_importance
    
    # 加载类别名称（如果提供）
    if args.class_names:
        logger.info(f"加载类别名称: {args.class_names}")
        with open(args.class_names, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        data['class_names'] = class_names
    
    return data

def prepare_confusion_matrix(data):
    """准备混淆矩阵数据"""
    if 'true_labels' not in data or 'predictions' not in data:
        logger.warning("缺少真实标签或预测结果，无法绘制混淆矩阵")
        return None
    
    # 获取真实标签和预测标签
    true_labels = data['true_labels'][args.label_column].values
    pred_labels = data['predictions']['predicted_label'].values
    
    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    return {
        'cm': cm,
        'class_names': data.get('class_names')
    }

def prepare_roc_curve(data):
    """准备ROC曲线数据"""
    if 'true_labels' not in data or 'probabilities' not in data:
        logger.warning("缺少真实标签或预测概率，无法绘制ROC曲线")
        return None
    
    # 获取真实标签和预测概率
    true_labels = data['true_labels'][args.label_column].values
    probabilities = data['probabilities'].iloc[:, 1:].values  # 跳过第一列（通常是样本ID）
    
    return {
        'y_true': true_labels,
        'y_prob': probabilities,
        'class_names': data.get('class_names')
    }

def prepare_pr_curve(data):
    """准备精确率-召回率曲线数据"""
    if 'true_labels' not in data or 'probabilities' not in data:
        logger.warning("缺少真实标签或预测概率，无法绘制精确率-召回率曲线")
        return None
    
    # 获取真实标签和预测概率
    true_labels = data['true_labels'][args.label_column].values
    probabilities = data['probabilities'].iloc[:, 1:].values  # 跳过第一列（通常是样本ID）
    
    return {
        'y_true': true_labels,
        'y_prob': probabilities,
        'class_names': data.get('class_names')
    }

def prepare_feature_importance(data, args):
    """准备特征重要性数据"""
    if 'feature_importance' not in data:
        logger.warning("缺少特征重要性数据，无法绘制特征重要性图")
        return None
    
    return {
        'feature_importance': data['feature_importance'],
        'top_n': args.top_n_features
    }

def create_visualizations(data, args):
    """创建可视化图表"""
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 解析图像大小
    figsize = (12, 8)  # 默认大小
    if args.figsize:
        try:
            width, height = map(float, args.figsize.split(','))
            figsize = (width, height)
        except:
            logger.warning(f"无法解析图像大小: {args.figsize}，使用默认大小")
    
    # 根据可视化类型创建图表
    if args.plot_type in ['confusion', 'all']:
        cm_data = prepare_confusion_matrix(data)
        if cm_data:
            logger.info("绘制混淆矩阵")
            cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
            plot_confusion_matrix(
                cm_data['cm'],
                class_names=cm_data['class_names'],
                figsize=figsize,
                save_path=cm_path
            )
    
    if args.plot_type in ['roc', 'all']:
        roc_data = prepare_roc_curve(data)
        if roc_data:
            logger.info("绘制ROC曲线")
            roc_path = os.path.join(args.output_dir, 'roc_curve.png')
            plot_roc_curve(
                roc_data['y_true'],
                roc_data['y_prob'],
                class_names=roc_data['class_names'],
                figsize=figsize,
                save_path=roc_path
            )
    
    if args.plot_type in ['pr', 'all']:
        pr_data = prepare_pr_curve(data)
        if pr_data:
            logger.info("绘制精确率-召回率曲线")
            pr_path = os.path.join(args.output_dir, 'pr_curve.png')
            plot_precision_recall_curve(
                pr_data['y_true'],
                pr_data['y_prob'],
                class_names=pr_data['class_names'],
                figsize=figsize,
                save_path=pr_path
            )
    
    if args.plot_type in ['feature', 'all']:
        feature_data = prepare_feature_importance(data, args)
        if feature_data:
            logger.info(f"绘制特征重要性（前 {feature_data['top_n']} 个）")
            feature_path = os.path.join(args.output_dir, 'feature_importance.png')
            plot_feature_importance(
                feature_data['feature_importance'],
                top_n=feature_data['top_n'],
                figsize=figsize,
                save_path=feature_path
            )
    
    logger.info(f"可视化结果已保存到 {args.output_dir}")

def main():
    """主函数"""
    # 解析命令行参数
    global args
    args = parse_args()
    
    # 设置日志
    setup_logging(args.verbose)
    
    try:
        # 加载数据
        data = load_data(args)
        
        # 创建可视化
        create_visualizations(data, args)
        
        logger.info("可视化流程完成")
        
    except Exception as e:
        logger.error(f"可视化过程中出错: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main() 
