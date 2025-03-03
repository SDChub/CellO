"""
pytest配置文件
"""

import pytest
import logging
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 配置日志
@pytest.fixture(autouse=True)
def setup_logging():
    """设置测试日志级别"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # 在测试期间降低日志级别，避免过多输出
    for handler in logging.root.handlers:
        handler.setLevel(logging.WARNING)
    
    # 测试完成后恢复日志级别
    yield
    for handler in logging.root.handlers:
        handler.setLevel(logging.INFO)

# 跳过需要GPU的测试（如果没有GPU）
def pytest_configure(config):
    """配置pytest"""
    try:
        import torch
        if not torch.cuda.is_available():
            config.addinivalue_line(
                "markers", "gpu: 标记需要GPU的测试"
            )
    except ImportError:
        config.addinivalue_line(
            "markers", "gpu: 标记需要GPU的测试"
        )

def pytest_collection_modifyitems(config, items):
    """修改测试项目"""
    # 检查是否有GPU
    has_gpu = False
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        pass
    
    # 如果没有GPU，跳过标记为需要GPU的测试
    if not has_gpu:
        skip_gpu = pytest.mark.skip(reason="需要GPU才能运行此测试")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu) 