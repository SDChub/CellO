# 导入包资源管理模块
import pkg_resources as pr
# 导入JSON处理模块
import json

# 导入自定义模块
from . import config
from . import ontology_graph

# 定义加载本体的主要函数
def load(ontology_index):
    """
    根据本体索引加载和配置本体
    参数:
        ontology_index: 本体配置的索引标识符
    返回:
        og: 构建的本体图对象
        include_ontologies: 包含的本体项目列表
        restrict_to_roots: 限制的子图根节点列表
    """
    # 获取当前包名
    resource_package = __name__
    # 获取本体配置文件的路径
    config_f = pr.resource_filename(resource_package, "ontology_configurations.json")
    
    # 读取并解析配置文件
    with open(config_f, "r") as f:
        j = json.load(f)
    # 获取特定本体的配置
    ont_config = j[ontology_index]

    # 从配置中提取各项参数
    # 需要包含的本体项目列表
    include_ontologies = ont_config["included_ontology_projects"]
    # 限制的ID空间
    restrict_to_idspaces = ont_config["id_spaces"]
    # 是否限制到特定子图
    is_restrict_roots = ont_config["restrict_to_specific_subgraph"]
    # 如果需要限制子图，获取根节点列表
    restrict_to_roots = ont_config["subgraph_roots"] if is_restrict_roots else None
    # 需要排除的术语列表
    exclude_terms = ont_config["exclude_terms"]

    # 创建本体名称到位置的映射字典
    # 只包含配置中指定的本体
    ont_to_loc = {x:y for x,y
        in config.ontology_name_to_location().items()
        if x in include_ontologies}
    
    # 构建本体图对象
    og = ontology_graph.build_ontology(ont_to_loc,
        restrict_to_idspaces=restrict_to_idspaces,  # 限制ID空间
        include_obsolete=False,                     # 不包含过时的术语
        restrict_to_roots=restrict_to_roots,        # 限制子图的根节点
        exclude_terms= exclude_terms)               # 排除特定术语

    # 返回构建的本体图和相关配置信息
    return og, include_ontologies, restrict_to_roots

# 主函数定义
def main():
    # 加载索引为"4"的本体
    # 并打印特定细胞系(CVCL:C792)的信息
    og, i, r = load("4")
    print(og.id_to_term["CVCL:C792"])

# 程序入口点
if __name__ == "__main__":
    main()
