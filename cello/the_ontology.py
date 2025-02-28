from .onto_lib_py3 import load_ontology

# 定义基因本体(GO)配置ID
GO_ONT_CONFIG_ID = '18'
# 定义单位本体配置ID
UNIT_OG_ID = '7'

# 定义本体补丁，用于添加额外的边（关系）
ONTO_PATCH = {
    # 添加边的列表，每个边包含源术语、目标术语和边的类型
    "add_edges": [
        {
            "source_term": "CL:2000001",    # 外周血单核细胞
            "target_term": "CL:0000081",    # 血细胞
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0000670",    # 原始生殖细胞
            "target_term": "CL:0002321",    # 胚胎细胞
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0008001",    # 造血前体细胞
            "target_term": "CL:0011115",    # 前体细胞
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0002246",    # 外周血干细胞
            "target_term": "CL:0000081",    # 血细胞
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0000542",    # 淋巴细胞
            "target_term": "CL:0000842",    # 单核细胞
            "edge_type": "is_a" 
        },
        {
            "source_term": "CL:0000066",    # 上皮细胞
            "target_term": "CL:0002371",    # 体细胞
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0001035",    # 骨细胞
            "target_term": "CL:0002371",    # 体细胞
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0000018",    # 精子细胞
            "target_term": "CL:0011115",    # 前体细胞  
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0000017",    # 精母细胞
            "target_term": "CL:0011115",    # 前体细胞
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0000235",    # 巨噬细胞
            "target_term": "CL:0000842",    # 单核细胞
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0000235",    # 巨噬细胞
            "target_term": "CL:0000145",    # 专业抗原呈递细胞
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0000451",    # 树突状细胞
            "target_term": "CL:0000145",    # 专业抗原呈递细胞
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0000236",    # B细胞
            "target_term": "CL:0000145",    # 专业抗原呈递细胞
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0002371",    # 体细胞
            "target_term": "CL:0000255",    # 真核细胞
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0000163",    # 内分泌细胞
            "target_term": "CL:0002371",    # 体细胞
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0008024",    # 胰腺内分泌细胞
            "target_term": "CL:0000164",    # 肠内分泌细胞
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0000583",    # 肺泡巨噬细胞
            "target_term": "CL:1001603",    # 肺巨噬细胞
            "edge_type": "is_a"
        },
        {
            'source_term': 'CL:0000091',        # 库普弗细胞
            "target_term": "UBERON:0002107",    # 肝脏
            "edge_type": "part_of"
        }
    ]
}

# 定义补丁函数，用于向本体图添加额外的边
def patch_the_ontology(og):
    """
    为本体图添加额外的边（关系）
    参数:
        og: 本体图对象
    返回:
        修改后的本体图对象
    """
    # 遍历补丁中定义的所有边
    for edge_info in ONTO_PATCH['add_edges']:
        # 获取源术语和目标术语的ID
        source_id = edge_info['source_term']
        target_id = edge_info['target_term']
        source_term = None
        target_term = None
        
        # 检查源术语和目标术语是否存在于本体图中
        if source_id in og.id_to_term:
            source_term = og.id_to_term[source_id]
        if target_id in og.id_to_term:
            target_term = og.id_to_term[target_id]
            
        # 如果源术语或目标术语不存在，跳过当前边
        if source_term is None or target_term is None:
            continue
            
        # 获取边的类型和反向边的类型
        edge_type = edge_info['edge_type']
        inv_edge_type = "inv_%s" % edge_type
        
        # 在源术语中添加关系
        if edge_type in source_term.relationships:
            source_term.relationships[edge_type].append(target_id)
        else:
            source_term.relationships[edge_type] = [target_id]
            
        # 在目标术语中添加反向关系
        if inv_edge_type in target_term.relationships:
            target_term.relationships[inv_edge_type].append(source_id)
        else:
            target_term.relationships[inv_edge_type] = [source_id]
    
    return og

#ONT_NAME_TO_ONT_ID = {"EFO_CL_DOID_UBERON_CVCL":"17"}

#ont_id_to_og = None
#def _ont_id_to_og():
#    global ont_id_to_og
#    if ont_id_to_og is None:
#        ont_id_to_og = {x: patch_the_ontology(load_ontology.load(x)[0]) for x in ONT_NAME_TO_ONT_ID.values()}
#    return ont_id_to_og

# 定义获取本体的函数
def the_ontology():
    """
    加载并返回补丁后的本体图
    返回:
        修改后的本体图对象
    """
    return patch_the_ontology(load_ontology.load('17')[0])
    #return _ont_id_to_og()['17']

# # 定义获取单位本体的函数
# def unit_ontology():
#     """
#     加载并返回单位本体
#     返回:
#         单位本体图对象
#     """
#     return load_ontology.load(UNIT_OG_ID)[0]

# 定义获取基因本体的函数
def go_ontology():
    """
    加载并返回基因本体(GO)
    返回:
        基因本体图对象
    """
    return load_ontology.load(GO_ONT_CONFIG_ID)[0] 

# 主函数
def main():
    og = the_ontology()
    og = patch_the_ontology(og)
    print( og.id_to_term['CL:0000081'])
    print( og.id_to_term['CL:2000001'])
    print( og.id_to_term['CL:0000542'])
    # 加载基因本体并打印特定术语的信息
    # og = go_ontology()
    # print(og.id_to_term['GO:0002312'])

# 程序入口点
if __name__ == "__main__":
    main()
