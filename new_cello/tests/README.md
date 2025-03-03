# New CellO 测试

本目录包含 New CellO 项目的测试用例。

## 测试结构

- `test_random_forest.py`: 随机森林模型测试
- `test_transformer.py`: Transformer模型测试
- `test_voting_ensemble.py`: 投票集成模型测试
- `test_preprocess.py`: 数据预处理模块测试
- `test_data_loading.py`: 数据加载模块测试
- `test_classification.py`: 核心分类模块测试
- `test_ontology.py`: 本体论处理模块测试
- `test_visualization.py`: 可视化模块测试
- `test_batch_correction.py`: 批次效应校正模块测试
- `conftest.py`: pytest配置文件

## 运行测试

### 运行所有测试

```bash
cd new_cello
pytest tests/
```

### 运行特定测试文件

```bash
pytest tests/test_random_forest.py
```

### 运行特定测试函数

```bash
pytest tests/test_random_forest.py::TestRandomForestModel::test_fit_predict
```

### 运行不需要GPU的测试

```bash
pytest tests/ -k "not gpu"
```

### 跳过需要特定依赖的测试

```bash
# 跳过需要本体论处理库的测试
pytest tests/ -k "not test_ontology"

# 跳过需要Transformers库的测试
pytest tests/ -k "not test_transformer"

# 跳过需要批次校正库的测试
pytest tests/ -k "not test_batch_correction"
```

## 测试覆盖率

要生成测试覆盖率报告，请运行：

```bash
pytest tests/ --cov=new_cello --cov-report=html
```

这将在 `htmlcov/` 目录中生成HTML格式的覆盖率报告。

## 测试依赖

测试需要以下依赖：

- pytest
- pytest-cov
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- torch (可选，用于Transformer模型测试)
- transformers (可选，用于Transformer模型测试)
- networkx (可选，用于本体论测试)
- pronto (可选，用于本体论测试)
- scanpy (可选，用于批次校正测试)
- harmonypy (可选，用于批次校正测试)
- scanorama (可选，用于批次校正测试)
- mnnpy (可选，用于批次校正测试) 