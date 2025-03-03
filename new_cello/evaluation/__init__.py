"""
评估模块，用于评估模型性能
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union, Optional, Tuple, Any, Callable

logger = logging.getLogger(__name__)

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = 'weighted',
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    计算分类评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率（可选）
        average: 多类别指标的平均方式
        class_names: 类别名称列表
        
    Returns:
        包含各种评估指标的字典
    """
    logger.info("计算评估指标")
    
    try:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix, classification_report
        )
    except ImportError:
        logger.error("无法导入sklearn库，请使用 'pip install scikit-learn' 安装")
        raise
    
    # 初始化结果字典
    metrics = {}
    
    # 计算基本指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # 如果提供了概率，计算AUC
    if y_prob is not None:
        try:
            if y_prob.shape[1] > 2:  # 多类别
                metrics['roc_auc'] = roc_auc_score(
                    pd.get_dummies(y_true), y_prob, 
                    average=average, multi_class='ovr'
                )
            else:  # 二分类
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
        except Exception as e:
            logger.warning(f"计算ROC AUC时出错: {str(e)}")
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    # 计算每个类别的指标
    class_metrics = {}
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # 使用类别名称（如果提供）
    if class_names is not None:
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        for i, cls in enumerate(unique_classes):
            if i < len(class_names):
                class_name = class_names[i]
                if str(i) in report:
                    class_metrics[class_name] = report[str(i)]
                elif str(cls) in report:
                    class_metrics[class_name] = report[str(cls)]
    else:
        # 否则使用数字标签
        for cls in np.unique(np.concatenate([y_true, y_pred])):
            if str(cls) in report:
                class_metrics[f'class_{cls}'] = report[str(cls)]
    
    metrics['class_metrics'] = class_metrics
    
    logger.info(f"评估指标计算完成，准确率: {metrics['accuracy']:.4f}, F1分数: {metrics['f1']:.4f}")
    return metrics

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'Blues',
    normalize: bool = False,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制混淆矩阵
    
    Args:
        cm: 混淆矩阵
        class_names: 类别名称列表
        figsize: 图形大小
        cmap: 颜色映射
        normalize: 是否归一化
        save_path: 保存路径（可选）
        
    Returns:
        matplotlib图形对象
    """
    logger.info("绘制混淆矩阵")
    
    # 如果需要归一化
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # 如果没有提供类别名称，使用数字
    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]
    
    # 创建图形
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap=cmap,
        xticklabels=class_names, yticklabels=class_names
    )
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.title('混淆矩阵')
    
    # 保存图形（如果指定了路径）
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"混淆矩阵已保存到 {save_path}")
    
    return plt.gcf()

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制ROC曲线
    
    Args:
        y_true: 真实标签
        y_prob: 预测概率
        class_names: 类别名称列表
        figsize: 图形大小
        save_path: 保存路径（可选）
        
    Returns:
        matplotlib图形对象
    """
    logger.info("绘制ROC曲线")
    
    try:
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
    except ImportError:
        logger.error("无法导入sklearn库，请使用 'pip install scikit-learn' 安装")
        raise
    
    # 获取类别数
    classes = np.unique(y_true)
    n_classes = len(classes)
    
    # 如果没有提供类别名称，使用数字
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    elif len(class_names) < n_classes:
        # 如果提供的类别名称不足，补充数字
        class_names = class_names + [f'Class {i}' for i in range(len(class_names), n_classes)]
    
    # 二值化标签
    y_bin = label_binarize(y_true, classes=classes)
    
    # 创建图形
    plt.figure(figsize=figsize)
    
    # 如果是二分类
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[1]} (AUC = {roc_auc:.2f})')
    else:
        # 多分类，为每个类别绘制ROC曲线
        for i, cls in enumerate(classes):
            if i < len(y_prob[0]):  # 确保有足够的概率列
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    
    # 添加对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # 设置图形属性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('接收者操作特征曲线')
    plt.legend(loc="lower right")
    
    # 保存图形（如果指定了路径）
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"ROC曲线已保存到 {save_path}")
    
    return plt.gcf()

def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制精确率-召回率曲线
    
    Args:
        y_true: 真实标签
        y_prob: 预测概率
        class_names: 类别名称列表
        figsize: 图形大小
        save_path: 保存路径（可选）
        
    Returns:
        matplotlib图形对象
    """
    logger.info("绘制精确率-召回率曲线")
    
    try:
        from sklearn.metrics import precision_recall_curve, average_precision_score
        from sklearn.preprocessing import label_binarize
    except ImportError:
        logger.error("无法导入sklearn库，请使用 'pip install scikit-learn' 安装")
        raise
    
    # 获取类别数
    classes = np.unique(y_true)
    n_classes = len(classes)
    
    # 如果没有提供类别名称，使用数字
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    elif len(class_names) < n_classes:
        # 如果提供的类别名称不足，补充数字
        class_names = class_names + [f'Class {i}' for i in range(len(class_names), n_classes)]
    
    # 二值化标签
    y_bin = label_binarize(y_true, classes=classes)
    
    # 创建图形
    plt.figure(figsize=figsize)
    
    # 如果是二分类
    if n_classes == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
        ap = average_precision_score(y_true, y_prob[:, 1])
        plt.plot(recall, precision, lw=2, label=f'{class_names[1]} (AP = {ap:.2f})')
    else:
        # 多分类，为每个类别绘制精确率-召回率曲线
        for i, cls in enumerate(classes):
            if i < len(y_prob[0]):  # 确保有足够的概率列
                precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
                ap = average_precision_score(y_bin[:, i], y_prob[:, i])
                plt.plot(recall, precision, lw=2, label=f'{class_names[i]} (AP = {ap:.2f})')
    
    # 设置图形属性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('精确率-召回率曲线')
    plt.legend(loc="lower left")
    
    # 保存图形（如果指定了路径）
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"精确率-召回率曲线已保存到 {save_path}")
    
    return plt.gcf()

def plot_feature_importance(
    feature_importance: Dict[str, float],
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制特征重要性
    
    Args:
        feature_importance: 特征名称到重要性的映射
        top_n: 显示前N个重要特征
        figsize: 图形大小
        save_path: 保存路径（可选）
        
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
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title(f'前 {top_n} 个重要特征')
    plt.xlabel('重要性')
    plt.ylabel('特征')
    
    # 保存图形（如果指定了路径）
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"特征重要性图已保存到 {save_path}")
    
    return plt.gcf()

def cross_validate(
    model_factory: Callable,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    stratify: bool = True,
    random_state: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """
    执行交叉验证
    
    Args:
        model_factory: 创建模型的工厂函数
        X: 特征数据
        y: 标签数据
        n_splits: 折数
        stratify: 是否进行分层抽样
        random_state: 随机种子
        **kwargs: 传递给模型工厂的其他参数
        
    Returns:
        包含交叉验证结果的字典
    """
    logger.info(f"执行 {n_splits} 折交叉验证")
    
    try:
        from sklearn.model_selection import StratifiedKFold, KFold
    except ImportError:
        logger.error("无法导入sklearn库，请使用 'pip install scikit-learn' 安装")
        raise
    
    # 选择交叉验证方法
    if stratify:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # 初始化结果
    results = {
        'fold_metrics': [],
        'fold_predictions': [],
        'fold_probabilities': [],
        'mean_metrics': {}
    }
    
    # 执行交叉验证
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        logger.info(f"训练折 {i+1}/{n_splits}")
        
        # 分割数据
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 创建并训练模型
        model = model_factory(**kwargs)
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 如果模型有predict_proba方法，获取概率
        y_prob = None
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
        
        # 计算指标
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        
        # 存储结果
        results['fold_metrics'].append(metrics)
        results['fold_predictions'].append((y_test, y_pred))
        if y_prob is not None:
            results['fold_probabilities'].append((y_test, y_prob))
    
    # 计算平均指标
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        values = [fold[metric] for fold in results['fold_metrics']]
        results['mean_metrics'][metric] = np.mean(values)
        results['mean_metrics'][f'{metric}_std'] = np.std(values)
    
    logger.info(f"交叉验证完成，平均准确率: {results['mean_metrics']['accuracy']:.4f} ± {results['mean_metrics']['accuracy_std']:.4f}")
    return results 