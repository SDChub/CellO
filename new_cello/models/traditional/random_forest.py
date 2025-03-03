"""
随机森林模型实现，基于scikit-learn
"""

import logging
import os
import numpy as np
import pickle
from typing import Dict, List, Union, Optional, Tuple

# 尝试导入scikit-learn库，如果不可用则记录警告
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectFromModel
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn库未安装，随机森林模型将不可用。请使用 'pip install scikit-learn' 安装。")

class RandomForestModel:
    """
    基于随机森林的细胞分类模型
    
    使用随机森林算法对单细胞RNA-seq数据进行分类
    """
    
    def __init__(
        self, 
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, int, float] = "sqrt",
        n_jobs: int = -1,
        random_state: int = 42,
        feature_selection: bool = True,
        feature_selection_threshold: Union[str, float] = "mean",
        scale_features: bool = True,
        **kwargs
    ):
        """
        初始化随机森林模型
        
        Args:
            n_estimators: 森林中树的数量
            max_depth: 树的最大深度
            min_samples_split: 分裂内部节点所需的最小样本数
            min_samples_leaf: 叶节点所需的最小样本数
            max_features: 寻找最佳分裂时考虑的特征数量
            n_jobs: 并行作业数
            random_state: 随机种子
            feature_selection: 是否进行特征选择
            feature_selection_threshold: 特征选择阈值
            scale_features: 是否对特征进行标准化
            **kwargs: 其他参数
        """
        self.logger = logging.getLogger(__name__)
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn库未安装，无法使用随机森林模型")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.feature_selection = feature_selection
        self.feature_selection_threshold = feature_selection_threshold
        self.scale_features = scale_features
        
        # 初始化模型
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs
        )
        
        # 初始化特征选择器和标准化器
        self.feature_selector = None
        self.scaler = StandardScaler() if scale_features else None
        
        self.logger.info(f"随机森林模型初始化完成，参数: n_estimators={n_estimators}, max_depth={max_depth}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, gene_names: Optional[List[str]] = None) -> 'RandomForestModel':
        """
        训练模型
        
        Args:
            X: 特征矩阵 [n_samples, n_features]
            y: 标签向量 [n_samples]
            gene_names: 基因名称列表，用于特征选择后的报告
            
        Returns:
            self: 训练后的模型
        """
        self.logger.info(f"开始训练随机森林模型，数据形状: {X.shape}")
        
        # 保存基因名称
        self.gene_names = gene_names
        
        # 标准化特征
        if self.scale_features:
            X = self.scaler.fit_transform(X)
            self.logger.info("特征标准化完成")
        
        # 特征选择
        if self.feature_selection:
            self.logger.info(f"执行特征选择，阈值: {self.feature_selection_threshold}")
            self.feature_selector = SelectFromModel(
                RandomForestClassifier(
                    n_estimators=100, 
                    random_state=self.random_state
                ),
                threshold=self.feature_selection_threshold
            )
            X_selected = self.feature_selector.fit_transform(X, y)
            
            # 记录选择的特征
            if gene_names is not None:
                selected_indices = self.feature_selector.get_support(indices=True)
                self.selected_genes = [gene_names[i] for i in selected_indices]
                self.logger.info(f"特征选择完成，从 {X.shape[1]} 个特征中选择了 {len(self.selected_genes)} 个特征")
            else:
                self.logger.info(f"特征选择完成，从 {X.shape[1]} 个特征中选择了 {X_selected.shape[1]} 个特征")
                
            # 使用选择的特征训练模型
            self.model.fit(X_selected, y)
        else:
            # 直接使用所有特征训练模型
            self.model.fit(X, y)
            
        self.logger.info("随机森林模型训练完成")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测样本的类别
        
        Args:
            X: 特征矩阵 [n_samples, n_features]
            
        Returns:
            np.ndarray: 预测的类别 [n_samples]
        """
        # 标准化特征
        if self.scale_features:
            X = self.scaler.transform(X)
        
        # 特征选择
        if self.feature_selection and self.feature_selector is not None:
            X = self.feature_selector.transform(X)
        
        # 预测
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测样本属于各个类别的概率
        
        Args:
            X: 特征矩阵 [n_samples, n_features]
            
        Returns:
            np.ndarray: 预测的概率 [n_samples, n_classes]
        """
        # 标准化特征
        if self.scale_features:
            X = self.scaler.transform(X)
        
        # 特征选择
        if self.feature_selection and self.feature_selector is not None:
            X = self.feature_selector.transform(X)
        
        # 预测概率
        return self.model.predict_proba(X)
    
    def get_feature_importances(self) -> Dict[str, float]:
        """
        获取特征重要性
        
        Returns:
            Dict[str, float]: 特征名称到重要性的映射
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("模型尚未训练，无法获取特征重要性")
        
        importances = self.model.feature_importances_
        
        if self.feature_selection and self.feature_selector is not None:
            # 如果进行了特征选择，只返回选择的特征的重要性
            if hasattr(self, 'selected_genes'):
                return {gene: imp for gene, imp in zip(self.selected_genes, importances)}
            else:
                return {f"feature_{i}": imp for i, imp in enumerate(importances)}
        else:
            # 如果没有进行特征选择，返回所有特征的重要性
            if hasattr(self, 'gene_names') and self.gene_names is not None:
                return {gene: imp for gene, imp in zip(self.gene_names, importances)}
            else:
                return {f"feature_{i}": imp for i, imp in enumerate(importances)}
    
    def save(self, save_dir: str):
        """
        保存模型到指定目录
        
        Args:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型
        with open(os.path.join(save_dir, "model.pkl"), 'wb') as f:
            pickle.dump(self.model, f)
        
        # 保存特征选择器
        if self.feature_selection and self.feature_selector is not None:
            with open(os.path.join(save_dir, "feature_selector.pkl"), 'wb') as f:
                pickle.dump(self.feature_selector, f)
        
        # 保存标准化器
        if self.scale_features and self.scaler is not None:
            with open(os.path.join(save_dir, "scaler.pkl"), 'wb') as f:
                pickle.dump(self.scaler, f)
        
        # 保存配置
        config = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "feature_selection": self.feature_selection,
            "feature_selection_threshold": self.feature_selection_threshold,
            "scale_features": self.scale_features
        }
        
        # 保存基因名称和选择的基因
        if hasattr(self, 'gene_names') and self.gene_names is not None:
            config["gene_names"] = self.gene_names
        
        if hasattr(self, 'selected_genes') and self.selected_genes is not None:
            config["selected_genes"] = self.selected_genes
        
        with open(os.path.join(save_dir, "config.pkl"), 'wb') as f:
            pickle.dump(config, f)
            
        self.logger.info(f"模型已保存到 {save_dir}")
    
    @classmethod
    def load(cls, load_dir: str) -> 'RandomForestModel':
        """
        从指定目录加载模型
        
        Args:
            load_dir: 加载目录
            
        Returns:
            RandomForestModel: 加载的模型
        """
        # 加载配置
        config_path = os.path.join(load_dir, "config.pkl")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        # 提取基因名称和选择的基因
        gene_names = config.pop("gene_names", None)
        selected_genes = config.pop("selected_genes", None)
        
        # 创建模型实例
        model = cls(**config)
        
        # 加载模型
        model_path = os.path.join(load_dir, "model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
        with open(model_path, 'rb') as f:
            model.model = pickle.load(f)
        
        # 加载特征选择器
        feature_selector_path = os.path.join(load_dir, "feature_selector.pkl")
        if os.path.exists(feature_selector_path) and model.feature_selection:
            with open(feature_selector_path, 'rb') as f:
                model.feature_selector = pickle.load(f)
        
        # 加载标准化器
        scaler_path = os.path.join(load_dir, "scaler.pkl")
        if os.path.exists(scaler_path) and model.scale_features:
            with open(scaler_path, 'rb') as f:
                model.scaler = pickle.load(f)
        
        # 设置基因名称和选择的基因
        if gene_names is not None:
            model.gene_names = gene_names
            
        if selected_genes is not None:
            model.selected_genes = selected_genes
            
        return model 