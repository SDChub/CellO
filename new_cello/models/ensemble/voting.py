"""
投票集成模型实现，结合多个基础模型的预测结果
"""

import logging
import os
import numpy as np
import pickle
from typing import Dict, List, Union, Optional, Tuple, Any

class VotingEnsembleModel:
    """
    投票集成模型
    
    结合多个基础模型的预测结果，通过投票方式确定最终预测
    """
    
    def __init__(
        self, 
        models: Optional[List[Any]] = None,
        voting: str = 'hard',
        weights: Optional[List[float]] = None,
        **kwargs
    ):
        """
        初始化投票集成模型
        
        Args:
            models: 基础模型列表
            voting: 投票类型，'hard'（多数投票）或'soft'（加权概率）
            weights: 各模型的权重，如果为None则权重相等
            **kwargs: 其他参数
        """
        self.logger = logging.getLogger(__name__)
        
        self.models = models or []
        self.voting = voting
        self.weights = weights
        
        if weights is not None and len(weights) != len(models):
            raise ValueError(f"权重数量 ({len(weights)}) 必须与模型数量 ({len(models)}) 相同")
            
        if voting not in ['hard', 'soft']:
            raise ValueError(f"投票类型必须是 'hard' 或 'soft'，而不是 '{voting}'")
            
        self.logger.info(f"初始化投票集成模型，投票类型: {voting}，模型数量: {len(self.models) if self.models else 0}")
    
    def add_model(self, model: Any, weight: float = 1.0):
        """
        添加基础模型
        
        Args:
            model: 要添加的模型
            weight: 模型权重
        """
        self.models.append(model)
        
        if self.weights is None:
            self.weights = [1.0] * len(self.models)
        else:
            self.weights.append(weight)
            
        self.logger.info(f"添加模型，当前模型数量: {len(self.models)}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'VotingEnsembleModel':
        """
        训练所有基础模型
        
        Args:
            X: 特征矩阵 [n_samples, n_features]
            y: 标签向量 [n_samples]
            **kwargs: 传递给基础模型的参数
            
        Returns:
            self: 训练后的模型
        """
        if not self.models:
            raise ValueError("没有基础模型可训练")
            
        self.logger.info(f"开始训练 {len(self.models)} 个基础模型")
        
        for i, model in enumerate(self.models):
            self.logger.info(f"训练模型 {i+1}/{len(self.models)}")
            if hasattr(model, 'fit'):
                model.fit(X, y, **kwargs)
            else:
                self.logger.warning(f"模型 {i+1} 没有 'fit' 方法，跳过训练")
                
        self.logger.info("所有基础模型训练完成")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测样本的类别
        
        Args:
            X: 特征矩阵 [n_samples, n_features]
            
        Returns:
            np.ndarray: 预测的类别 [n_samples]
        """
        if not self.models:
            raise ValueError("没有基础模型可用于预测")
            
        self.logger.info(f"使用 {len(self.models)} 个基础模型进行预测")
        
        if self.voting == 'hard':
            # 硬投票：每个模型预测一个类别，取多数
            predictions = []
            for i, model in enumerate(self.models):
                self.logger.info(f"获取模型 {i+1}/{len(self.models)} 的预测")
                if hasattr(model, 'predict'):
                    pred = model.predict(X)
                    predictions.append(pred)
                else:
                    self.logger.warning(f"模型 {i+1} 没有 'predict' 方法，跳过")
            
            if not predictions:
                raise ValueError("没有有效的预测结果")
                
            # 转换为数组并进行投票
            predictions = np.array(predictions)
            if self.weights is None:
                # 无权重，直接取众数
                final_pred = np.apply_along_axis(
                    lambda x: np.bincount(x).argmax(), 
                    axis=0, 
                    arr=predictions
                )
            else:
                # 有权重，加权投票
                weights = np.array(self.weights[:len(predictions)])[:, np.newaxis]
                weighted_votes = np.zeros((X.shape[0], len(np.unique(np.concatenate(predictions)))))
                
                for pred, weight in zip(predictions, weights):
                    for i, cls in enumerate(np.unique(pred)):
                        weighted_votes[pred == cls, i] += weight
                        
                final_pred = np.argmax(weighted_votes, axis=1)
                
            return final_pred
        else:
            # 软投票：每个模型预测概率，取加权平均
            probas = []
            for i, model in enumerate(self.models):
                self.logger.info(f"获取模型 {i+1}/{len(self.models)} 的概率预测")
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X)
                    probas.append(prob)
                else:
                    self.logger.warning(f"模型 {i+1} 没有 'predict_proba' 方法，跳过")
            
            if not probas:
                raise ValueError("没有有效的概率预测结果")
                
            # 确保所有概率矩阵具有相同的形状
            n_classes = max(p.shape[1] for p in probas)
            for i in range(len(probas)):
                if probas[i].shape[1] < n_classes:
                    # 扩展概率矩阵
                    extended = np.zeros((probas[i].shape[0], n_classes))
                    extended[:, :probas[i].shape[1]] = probas[i]
                    probas[i] = extended
            
            # 转换为数组并计算加权平均
            probas = np.array(probas)
            if self.weights is None:
                # 无权重，直接平均
                avg_proba = np.mean(probas, axis=0)
            else:
                # 有权重，加权平均
                weights = np.array(self.weights[:len(probas)])[:, np.newaxis, np.newaxis]
                avg_proba = np.sum(probas * weights, axis=0) / np.sum(weights)
                
            # 返回概率最高的类别
            return np.argmax(avg_proba, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测样本属于各个类别的概率
        
        Args:
            X: 特征矩阵 [n_samples, n_features]
            
        Returns:
            np.ndarray: 预测的概率 [n_samples, n_classes]
        """
        if not self.models:
            raise ValueError("没有基础模型可用于预测概率")
            
        self.logger.info(f"使用 {len(self.models)} 个基础模型进行概率预测")
        
        # 收集所有模型的概率预测
        probas = []
        for i, model in enumerate(self.models):
            self.logger.info(f"获取模型 {i+1}/{len(self.models)} 的概率预测")
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X)
                probas.append(prob)
            else:
                self.logger.warning(f"模型 {i+1} 没有 'predict_proba' 方法，跳过")
        
        if not probas:
            raise ValueError("没有有效的概率预测结果")
            
        # 确保所有概率矩阵具有相同的形状
        n_classes = max(p.shape[1] for p in probas)
        for i in range(len(probas)):
            if probas[i].shape[1] < n_classes:
                # 扩展概率矩阵
                extended = np.zeros((probas[i].shape[0], n_classes))
                extended[:, :probas[i].shape[1]] = probas[i]
                probas[i] = extended
        
        # 转换为数组并计算加权平均
        probas = np.array(probas)
        if self.weights is None:
            # 无权重，直接平均
            avg_proba = np.mean(probas, axis=0)
        else:
            # 有权重，加权平均
            weights = np.array(self.weights[:len(probas)])[:, np.newaxis, np.newaxis]
            avg_proba = np.sum(probas * weights, axis=0) / np.sum(weights)
            
        return avg_proba
    
    def save(self, save_dir: str):
        """
        保存模型到指定目录
        
        Args:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存配置
        config = {
            "voting": self.voting,
            "weights": self.weights
        }
        
        with open(os.path.join(save_dir, "config.pkl"), 'wb') as f:
            pickle.dump(config, f)
            
        # 保存每个基础模型
        os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
        for i, model in enumerate(self.models):
            model_dir = os.path.join(save_dir, "models", f"model_{i}")
            os.makedirs(model_dir, exist_ok=True)
            
            if hasattr(model, 'save'):
                model.save(model_dir)
            else:
                # 如果模型没有save方法，尝试使用pickle保存
                try:
                    with open(os.path.join(model_dir, "model.pkl"), 'wb') as f:
                        pickle.dump(model, f)
                except Exception as e:
                    self.logger.error(f"无法保存模型 {i}: {str(e)}")
                    
        self.logger.info(f"集成模型已保存到 {save_dir}")
    
    @classmethod
    def load(cls, load_dir: str, model_loaders: Optional[List[callable]] = None) -> 'VotingEnsembleModel':
        """
        从指定目录加载模型
        
        Args:
            load_dir: 加载目录
            model_loaders: 用于加载各个基础模型的函数列表
            
        Returns:
            VotingEnsembleModel: 加载的模型
        """
        # 加载配置
        config_path = os.path.join(load_dir, "config.pkl")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        # 创建模型实例
        model = cls(models=[], **config)
        
        # 加载基础模型
        models_dir = os.path.join(load_dir, "models")
        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"模型目录不存在: {models_dir}")
            
        # 获取所有模型目录
        model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        model_dirs.sort()  # 确保顺序一致
        
        for i, model_dir in enumerate(model_dirs):
            full_model_dir = os.path.join(models_dir, model_dir)
            
            # 如果提供了加载器，使用加载器加载模型
            if model_loaders and i < len(model_loaders):
                try:
                    base_model = model_loaders[i](full_model_dir)
                    model.add_model(base_model)
                    continue
                except Exception as e:
                    model.logger.error(f"使用加载器加载模型 {i} 失败: {str(e)}")
            
            # 尝试使用pickle加载模型
            model_path = os.path.join(full_model_dir, "model.pkl")
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        base_model = pickle.load(f)
                    model.add_model(base_model)
                except Exception as e:
                    model.logger.error(f"加载模型 {i} 失败: {str(e)}")
            else:
                model.logger.warning(f"模型文件不存在: {model_path}")
                
        if not model.models:
            model.logger.warning("没有成功加载任何基础模型")
            
        return model 