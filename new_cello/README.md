# New CellO

Enhanced CellO: Cell Ontology-based classification with deep learning support and GPU acceleration.

## 项目概述

New CellO是对原始CellO项目的增强版本，引入了深度学习模型、GPU加速和模块化架构。该项目用于对人类RNA-seq数据进行细胞类型分类，基于细胞本体论（Cell Ontology）进行层次化预测。

## 主要改进

1. **深度学习集成**：使用Hugging Face的Transformers工具，将深度学习模型集成到项目中，提高分类准确性
2. **数据处理优化**：优化算法实现，支持GPU加速，提高大规模数据集的处理效率
3. **模块化重构**：降低组件耦合度，每个模块分开，提高代码可维护性和可扩展性
4. **本体论处理增强**：改进细胞本体论处理，支持更复杂的层次分类和关系查询
5. **可视化功能**：提供丰富的可视化工具，帮助理解和解释分类结果

## 项目结构

```
new_cello/
├── core/                  # 核心功能
│   ├── classification/    # 分类算法
│   ├── ontology/          # 本体论处理
│   └── visualization/     # 可视化工具
├── models/                # 模型实现
│   ├── traditional/       # 传统机器学习模型
│   ├── deep_learning/     # 深度学习模型
│   └── ensemble/          # 集成模型
├── data/                  # 数据处理
│   ├── loaders/           # 数据加载器
│   └── preprocessors/     # 数据预处理
├── preprocess/            # 预处理功能
├── evaluation/            # 评估指标
├── utils/                 # 工具函数
│   ├── gpu/               # GPU加速工具
│   └── io/                # 输入输出工具
├── cli/                   # 命令行接口
│   ├── predict.py         # 预测接口
│   ├── train.py           # 训练接口
│   └── visualize.py       # 可视化接口
├── tests/                 # 测试代码
└── docs/                  # 文档
```

## 安装

```bash
# 克隆仓库
git clone https://github.com/example/new_cello.git
cd new_cello

# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .
```

## 使用方法

### 命令行使用

```bash
# 使用深度学习模型进行预测
new_cello_predict -m deep_learning -i <输入文件> -o <输出前缀>

# 使用传统模型进行预测
new_cello_predict -m traditional -i <输入文件> -o <输出前缀>

# 训练自定义模型
new_cello_train -m <模型类型> -i <输入文件> --labels <标签文件> -o <模型输出路径>

# 可视化结果
new_cello_visualize -p <预测结果> -o <输出目录>
```

### Python API使用

```python
import scanpy as sc
import new_cello
from new_cello.core import classify, train_model
from new_cello.core.visualization import plot_umap, plot_cell_type_distribution

# 加载数据
adata = sc.read_csv('expression_matrix.csv')

# 预处理数据
adata = new_cello.preprocess(adata)

# 使用深度学习模型进行分类
predictions, probabilities = classify(
    adata.X, 
    gene_names=adata.var_names.tolist(),
    model_type='deep_learning', 
    use_gpu=True
)

# 将预测结果添加到AnnData对象
adata.obs['predicted_cell_type'] = predictions

# 可视化结果
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color='predicted_cell_type')

# 绘制细胞类型分布
plot_cell_type_distribution(predictions, title='Cell Type Distribution')
```

## 模型类型

New CellO支持三种类型的模型：

1. **传统机器学习模型**
   - 随机森林
   - SVM
   - 逻辑回归
   - 梯度提升

2. **深度学习模型**
   - Transformer
   - 图神经网络
   - 混合模型

3. **集成模型**
   - 投票集成
   - 堆叠集成
   - 加权集成

## 本体论支持

New CellO使用细胞本体论进行层次化分类，支持以下功能：

- 加载和解析OBO格式的本体论文件
- 查询术语之间的关系（祖先、后代、父级、子级）
- 查找多个术语的最近公共祖先
- 可视化本体论结构

## 可视化功能

New CellO提供多种可视化工具：

- 基因表达热图
- UMAP/t-SNE/PCA降维可视化
- 细胞类型分布图
- 特征重要性图
- 本体论结构可视化

## 贡献

欢迎贡献代码、报告问题或提出改进建议。请参阅[贡献指南](CONTRIBUTING.md)了解更多信息。

## 许可证

MIT License 
