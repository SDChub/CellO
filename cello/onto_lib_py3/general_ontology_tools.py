# 导入命令行参数解析库
from optparse import OptionParser

# 导入自定义本体相关模块
from . import ontology_graph
from . import load_ontology

# 定义本体名称到ID的映射字典
# EFO_CL_DOID_UBERON_CVCL是组合本体，包含多个生物医学本体
ONT_NAME_TO_ONT_ID = {"EFO_CL_DOID_UBERON_CVCL":"17"}

# 创建本体ID到本体图对象的映射字典
# 使用load_ontology.load()加载每个本体
ONT_ID_TO_OG = {
    x:load_ontology.load(x)[0] 
    for x in list(ONT_NAME_TO_ONT_ID.values())
}

# 主函数定义
def main():
    """
    示例代码（已注释）:
    results = get_ancestors_within_radius("CL:0000034", 4)
    for res in results:
        print ONT_ID_TO_OG["17"].id_to_term[res].name
    """
    # 打印特定术语的名称和同义词
    print(get_term_name_and_synonyms("CL:0000134"))   

#########################################################
#   示例函数部分
#########################################################

# 演示is_descendant函数使用的示例函数
def example_is_descendant():
    # 检查间充质干细胞是否是干细胞的后代（应返回True）
    print(is_descendant(
        "CL:0000134",   # mesenchymal stem cell 
        "CL:0000034"    # stem cell
    ))

    # 检查间充质干细胞是否是神经元的后代（应返回False）
    print(is_descendant(
        "CL:0000134",   # mesenchymal stem cell 
        "CL:0000540"    # neuron
    ))

#########################################################
#   API函数部分
#########################################################

# 获取本体对象的函数
def get_ontology_object():
    return ONT_ID_TO_OG["17"]

# 根据术语ID获取术语名称的函数
def get_term_name(term_id):
    og = ONT_ID_TO_OG["17"]
    return og.id_to_term[term_id].name

# 获取术语的名称和所有同义词的函数
def get_term_name_and_synonyms(term_id):
    og = ONT_ID_TO_OG["17"]
    t_strs = set()
    term = og.id_to_term[term_id]
    t_strs.add(term.name)
    for syn in term.synonyms:
        t_strs.add(syn.syn_str)
    return list(t_strs) 

# 检查一个术语是否是另一个术语的后代
def is_descendant(descendent, ancestor):
    og = ONT_ID_TO_OG["17"]
    # 获取所有上级术语（包括is_a和part_of关系）
    sup_terms = og.recursive_relationship(
        descendent, 
        recurs_relationships=['is_a', 'part_of']
    )
    return ancestor in set(sup_terms)

# 在指定半径范围内获取所有后代术语
def get_descendents_within_radius(term_id, radius):
    return _get_terms_within_radius(
        term_id, 
        radius, 
        relationships=['inv_is_a']
    )

# 在指定半径范围内获取所有祖先术语
def get_ancestors_within_radius(term_id, radius):
    return _get_terms_within_radius(
        term_id,
        radius,
        relationships=['is_a']
    )

#########################################################
#   辅助函数部分
#########################################################

# 在指定关系和半径范围内获取相关术语的通用函数
def _get_terms_within_radius(
    term_id, 
    radius, 
    relationships
    ):
    # 获取本体图对象
    og = ONT_ID_TO_OG["17"]

    # 初始化结果集和下一批要处理的术语
    result_terms = set()
    next_batch = set([term_id])
    
    # 在指定半径范围内迭代处理
    for i in range(radius):
        new_next_batch = set()
        # 处理当前批次中的每个术语
        for curr_t_id in next_batch:
            curr_term = term = og.id_to_term[curr_t_id]
            # 对每种关系类型
            for rel in relationships:
                # 如果存在该类型的关系
                if rel in curr_term.relationships:
                    # 添加相关术语到下一批
                    new_next_batch.update(
                        curr_term.relationships[rel]
                    )
        # 更新结果集和下一批要处理的术语
        result_terms.update(new_next_batch) 
        next_batch = new_next_batch
                         
    return result_terms

# 程序入口点
if __name__ == "__main__":
    main()
