"""
GPU检测和加速工具
"""

import logging

logger = logging.getLogger(__name__)

def is_gpu_available():
    """
    检查是否有可用的GPU
    
    Returns:
        bool: 如果GPU可用则返回True，否则返回False
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        logger.warning("PyTorch未安装，无法检测GPU")
        return False
    except Exception as e:
        logger.warning(f"检测GPU时出错: {str(e)}")
        return False

def get_device():
    """
    获取可用的设备（GPU或CPU）
    
    Returns:
        torch.device: 可用的设备
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    except ImportError:
        logger.warning("PyTorch未安装，默认使用CPU")
        return "cpu"
    except Exception as e:
        logger.warning(f"获取设备时出错: {str(e)}")
        return "cpu"

def get_gpu_memory_info():
    """
    获取GPU内存信息
    
    Returns:
        dict: GPU内存信息，包括总内存和已用内存
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        
        # 获取GPU数量
        gpu_count = torch.cuda.device_count()
        
        # 获取每个GPU的内存信息
        memory_info = {}
        for i in range(gpu_count):
            memory_info[i] = {
                "total": torch.cuda.get_device_properties(i).total_memory / 1024**3,  # GB
                "reserved": torch.cuda.memory_reserved(i) / 1024**3,  # GB
                "allocated": torch.cuda.memory_allocated(i) / 1024**3,  # GB
                "free": (torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i)) / 1024**3  # GB
            }
        
        return {
            "available": True,
            "gpu_count": gpu_count,
            "memory_info": memory_info
        }
    except ImportError:
        logger.warning("PyTorch未安装，无法获取GPU内存信息")
        return {"available": False}
    except Exception as e:
        logger.warning(f"获取GPU内存信息时出错: {str(e)}")
        return {"available": False, "error": str(e)}  
