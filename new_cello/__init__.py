"""
New CellO: Enhanced Cell Ontology-based classification with deep learning support
"""

__version__ = '0.1.0'

# 导入主要功能
from new_cello.core.classification import classify
from new_cello.data_processing.preprocessors import preprocess
from new_cello.models import get_model
from new_cello.utils.gpu import is_gpu_available

# 设置默认日志级别
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 检查GPU可用性
gpu_available = is_gpu_available()
if gpu_available:
    logging.info("GPU is available and will be used for acceleration")
else:
    logging.info("GPU is not available, falling back to CPU")  
