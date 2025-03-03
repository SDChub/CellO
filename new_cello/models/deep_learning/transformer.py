"""
Transformer模型实现，基于Hugging Face的Transformers库
"""

import logging
import os
import torch
import numpy as np
from typing import Dict, List, Union, Optional, Tuple

# 尝试导入transformers库，如果不可用则记录警告
try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers库未安装，Transformer模型将不可用。请使用 'pip install transformers' 安装。")

class TransformerModel:
    """
    基于Transformer的细胞分类模型
    
    使用预训练的Transformer模型对单细胞RNA-seq数据进行特征提取和分类
    """
    
    def __init__(
        self, 
        model_name: str = "distilbert-base-uncased",
        num_classes: int = 0,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        """
        初始化Transformer模型
        
        Args:
            model_name: 预训练模型名称或路径
            num_classes: 分类类别数量
            device: 运行设备 ('cuda' 或 'cpu')
            cache_dir: 模型缓存目录
            **kwargs: 其他参数
        """
        self.logger = logging.getLogger(__name__)
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers库未安装，无法使用Transformer模型")
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.cache_dir = cache_dir
        
        # 设置设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.logger.info(f"使用设备: {self.device}")
        self.logger.info(f"加载预训练模型: {model_name}")
        
        # 加载预训练模型和分词器
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            self.base_model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
            self.base_model.to(self.device)
            
            # 添加分类头
            if num_classes > 0:
                self.classifier = torch.nn.Linear(self.base_model.config.hidden_size, num_classes)
                self.classifier.to(self.device)
            else:
                self.classifier = None
                
            self.logger.info("模型加载成功")
        except Exception as e:
            self.logger.error(f"加载模型时出错: {str(e)}")
            raise
    
    def preprocess(self, gene_expression_data: np.ndarray, gene_names: List[str]) -> Dict:
        """
        预处理基因表达数据为模型输入格式
        
        Args:
            gene_expression_data: 基因表达矩阵 [n_samples, n_genes]
            gene_names: 基因名称列表
            
        Returns:
            Dict: 模型输入字典
        """
        # 将基因表达数据转换为模型可接受的格式
        # 这里我们将基因表达数据转换为"文本"格式，每个基因作为一个"词"
        batch_size = gene_expression_data.shape[0]
        texts = []
        
        for i in range(batch_size):
            # 为每个样本创建一个"文本"，格式为 "gene1:value1 gene2:value2 ..."
            # 只包含表达值大于0的基因
            sample = gene_expression_data[i]
            expressed_genes = [f"{gene_names[j]}:{sample[j]:.4f}" for j in range(len(gene_names)) if sample[j] > 0]
            text = " ".join(expressed_genes)
            texts.append(text)
        
        # 使用tokenizer处理文本
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # 将输入移动到正确的设备
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
            
        return inputs
    
    def forward(self, inputs: Dict) -> torch.Tensor:
        """
        前向传播
        
        Args:
            inputs: 模型输入字典
            
        Returns:
            torch.Tensor: 模型输出
        """
        # 获取基础模型的输出
        outputs = self.base_model(**inputs)
        
        # 使用[CLS]标记的输出作为整个序列的表示
        sequence_output = outputs.last_hidden_state[:, 0, :]
        
        # 如果有分类头，则应用分类头
        if self.classifier is not None:
            logits = self.classifier(sequence_output)
            return logits
        
        # 否则返回序列输出
        return sequence_output
    
    def predict(self, gene_expression_data: np.ndarray, gene_names: List[str]) -> np.ndarray:
        """
        预测样本的类别
        
        Args:
            gene_expression_data: 基因表达矩阵 [n_samples, n_genes]
            gene_names: 基因名称列表
            
        Returns:
            np.ndarray: 预测结果
        """
        if self.classifier is None:
            raise ValueError("模型未配置分类头，无法进行预测")
        
        # 预处理数据
        inputs = self.preprocess(gene_expression_data, gene_names)
        
        # 设置为评估模式
        self.base_model.eval()
        self.classifier.eval()
        
        # 进行预测
        with torch.no_grad():
            logits = self.forward(inputs)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
        
        return predictions
    
    def get_embeddings(self, gene_expression_data: np.ndarray, gene_names: List[str]) -> np.ndarray:
        """
        获取样本的嵌入表示
        
        Args:
            gene_expression_data: 基因表达矩阵 [n_samples, n_genes]
            gene_names: 基因名称列表
            
        Returns:
            np.ndarray: 样本嵌入 [n_samples, embedding_dim]
        """
        # 预处理数据
        inputs = self.preprocess(gene_expression_data, gene_names)
        
        # 设置为评估模式
        self.base_model.eval()
        
        # 获取嵌入
        with torch.no_grad():
            embeddings = self.forward(inputs)
            
        return embeddings.cpu().numpy()
    
    def save(self, save_dir: str):
        """
        保存模型到指定目录
        
        Args:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存基础模型和分词器
        self.base_model.save_pretrained(os.path.join(save_dir, "base_model"))
        self.tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))
        
        # 如果有分类头，保存分类头
        if self.classifier is not None:
            torch.save(self.classifier.state_dict(), os.path.join(save_dir, "classifier.pt"))
            
        # 保存配置
        config = {
            "model_name": self.model_name,
            "num_classes": self.num_classes
        }
        
        torch.save(config, os.path.join(save_dir, "config.pt"))
        self.logger.info(f"模型已保存到 {save_dir}")
    
    @classmethod
    def load(cls, load_dir: str, device: Optional[str] = None):
        """
        从指定目录加载模型
        
        Args:
            load_dir: 加载目录
            device: 运行设备
            
        Returns:
            TransformerModel: 加载的模型
        """
        # 加载配置
        config_path = os.path.join(load_dir, "config.pt")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        config = torch.load(config_path)
        
        # 创建模型实例
        model = cls(
            model_name=os.path.join(load_dir, "base_model"),
            num_classes=config["num_classes"],
            device=device
        )
        
        # 如果有分类头，加载分类头
        classifier_path = os.path.join(load_dir, "classifier.pt")
        if os.path.exists(classifier_path) and model.classifier is not None:
            model.classifier.load_state_dict(torch.load(classifier_path))
            
        return model 