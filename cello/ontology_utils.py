# 导入必要的模块
from . import the_ontology

# 定义全局变量_cell_ontology，用于缓存细胞本体图实例
_cell_ontology = None

# 获取细胞本体图的函数，使用单例模式
def cell_ontology():
    """
    获取细胞本体图实例
    使用单例模式确保只创建一个实例
    返回：细胞本体图对象
    """
    global _cell_ontology
    # 如果实例不存在，则创建新实例
    if _cell_ontology is None:
        _cell_ontology = the_ontology.the_ontology()
    return _cell_ontology

# 根据术语ID获取术语名称的函数
def get_term_name(term_id):
    """
    根据术语ID获取其名称
    参数：
        term_id: 术语的唯一标识符
    返回：
        术语名称，如果不存在则返回None
    """
    try:
        # 从细胞本体图中查找术语名称
        return cell_ontology().id_to_term[term_id].name
    except KeyError:
        # 如果术语ID不存在，返回None
        return None

