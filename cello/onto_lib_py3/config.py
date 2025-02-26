# 导入必要的库
# pkg_resources用于访问包资源
import pkg_resources as pr
# os用于操作系统相关功能
import os
# 从os.path导入join函数用于路径拼接
from os.path import join
# 导入json处理库
import json

# 定义当前包名为资源包名
resource_package = __name__

# 定义OBO文件目录路径
# 使用pkg_resources获取obo目录的实际文件系统路径
OBO_DIR = pr.resource_filename(resource_package, "obo")

# 定义本体前缀到文件名映射的配置文件路径
# 这个文件存储了不同本体前缀对应的文件名
PREFIX_TO_FNAME = pr.resource_filename(
    resource_package, 
    "ont_prefix_to_filename.json"
)

# 定义函数：获取本体名称到文件位置的映射
def ontology_name_to_location():
    """
    读取配置文件并返回本体前缀到完整文件路径的映射字典
    返回: dict, key为本体前缀，value为对应的完整文件路径
    """
    # 初始化空字典存储前缀到位置的映射
    prefix_to_location = {}
    # 打开并读取配置文件
    with open(PREFIX_TO_FNAME, "r") as f:
        # 加载JSON文件，并遍历每个前缀和文件名
        for prefix, fname in json.load(f).items():
            # 将前缀映射到完整的文件路径（OBO_DIR + 文件名）
            prefix_to_location[prefix] = join(OBO_DIR, fname)
    # 返回映射字典
    return prefix_to_location
    
if __name__ == "__main__":
    print(ontology_name_to_location())

