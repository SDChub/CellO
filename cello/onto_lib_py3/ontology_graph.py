#!/usr/bin/python

# 导入必要的库
import io
import re
from optparse import OptionParser
from queue import Queue

# 导入自定义配置模块
from . import config  #定义文件路径

# 导入资源管理相关模块
import pkg_resources as pr # 通过这个包找到文件路径
import os
from os.path import join
import json

# 定义当前包名
resource_package = __name__ # 用于后期通过pkg_resources找到文件路径

# 定义实体类型常量
ENTITY_TERM = "TERM"
ENTITY_TYPE_DEF = "TYPE_DEF"
ENTITY_EXCLUDED_TERM = "EXCLUDED_TERM"

# 调试输出控制标志
VERBOSE = False

# 同义词类定义. 理解为struct结构体, 存储了同义词字符串和同义词类型
class Synonym:
    """
    表示术语的同义词。存储同义词字符串和同义词类型。
    """
    def __init__(self, syn_str, syn_type):
        self.syn_str = syn_str    # 同义词字符串
        self.syn_type = syn_type  # 同义词类型

    # 定义对象的字符串表示
    def __repr__(self):
        return str((self.syn_str, self.syn_type))

# 术语类定义
class Term:
    def __init__(self, id, name, definition=None,
        synonyms=[], comment=None, xrefs=None,
        relationships={}, property_values=[], subsets=[]):
        """
        参数:
            id: 术语标识符 (例如 'CL:0000555')
            name: 术语名称
            definition: 术语定义
            synonyms: 表示术语同义词的Synonym对象列表
            comment: 术语相关注释
            xrefs: 该术语外部定义的URI列表
            relationships: 字典，将关系类型映射到通过该类型与此术语相关的术语ID
            property_values: 术语的属性值列表
            subsets: 术语所属的子集列表
        """
        self.id = id
        self.name = name
        self.definition = definition
        self.synonyms = synonyms
        self.comment = comment
        self.xrefs = xrefs
        self.relationships = relationships
        self.property_values = property_values
        self.subsets = subsets

    # 定义对象的字符串表示
    def __repr__(self):
        rep = {
            "id":self.id,
            "name":self.name,
            "definition":self.definition,
            "synonyms": self.synonyms,
            "relationships": self.relationships,
            "subsets": self.subsets}
        return str(rep)

    # 获取is_a关系的术语
    def is_a(self):
        return self.get_related_terms("is_a")

    # 获取反向is_a关系的术语
    def inv_is_a(self):
        return self.get_related_terms("inv_is_a")

    # 获取指定关系类型的相关术语
    def get_related_terms(self, relation):
        if relation in self.relationships:
            return self.relationships[relation]
        else:
            return []

# 本体图类定义
class OntologyGraph:
    def __init__(self, id_to_term,
        enriched_synonyms_file=None):
        self.id_to_term = id_to_term
        # 初始化 id_to_term 字典，用于存储 ID 到class Term对象 的映射

    # 打印子类型名称
    def subtype_names(self, supertype_name):
        id = self.name_to_ids[supertype_name]
        for t in self.id_to_term[id].inv_is_a():
            print(self.id_to_term[t].name)

    #def graphviz(self, root_id=None):
    #    g = pgv.AGraph(directed='True')
    #
    #    # Breadth-first traversal from root
    #    visited_ids = set()
    #    curr_id = root_id
    #    q = Queue()
    #    q.put(curr_id)
    #    while not q.empty():
    #        curr_id = q.get()
    #        visited_ids.add(curr_id)
    #        for sub_id in self.id_to_term[curr_id].inv_is_a():
    #            if not sub_id in visited_ids:
    #                g.add_edge(self.id_to_term[curr_id].name,
    #                    self.id_to_term[sub_id].name)
    #                q.put(sub_id)
    #    print(str(g))

    # 获取直接子术语
    def direct_subterms(self, id):
        return set([
            self.id_to_term[x]
            for x in self.id_to_term[id].relationships["inv_is_a"]
        ])

    # 递归获取所有子术语
    def recursive_subterms(self, id):
        return self.recursive_relationship(id, ["inv_is_a"])

    # 递归获取所有父术语
    def recursive_superterms(self, id):
        return self.recursive_relationship(id, ["is_a"])

    # 递归获取指定关系类型的相关术语
    def recursive_relationship(self, t_id, recurs_relationships):
        """
        谨慎使用此方法
        """
        if t_id not in self.id_to_term:
            return set()
        gathered_ids = set()
        curr_id = t_id
        q = Queue()
        q.put(curr_id)
        visited_ids = set()
        while not q.empty():
            curr_id = q.get()
            visited_ids.add(curr_id)
            gathered_ids.add(curr_id)
            for rel in recurs_relationships:
                if curr_id not in self.id_to_term:
                    continue
                if rel in self.id_to_term[curr_id].relationships:
                    for rel_id in self.id_to_term[curr_id].relationships[rel]:
                        if not rel_id in visited_ids:
                            q.put(rel_id)
        return gathered_ids

# 可映射本体图类定义（继承自OntologyGraph）
class MappableOntologyGraph(OntologyGraph):
    # 返回空列表的辅助方法
    def empty_list(self):
        return []

    def __init__(self, id_to_term, nonmappable_terms):
        """
        初始化可映射本体图
        参数:
            id_to_term: 术语ID到术语对象的字典映射
            nonmappable_terms: 不可映射术语ID的集合(如临时术语、草稿术语等)
        """
        # 调用父类OntologyGraph的初始化方法
        OntologyGraph.__init__(self, id_to_term)
        
        # 初始化不可映射术语集合
        if not nonmappable_terms:
            self.nonmappable_terms = set()  # 如果没有提供,创建空集合
        else:
            self.nonmappable_terms = set(nonmappable_terms)  # 转换为集合类型
            
        # 计算可映射术语ID集合
        # 使用集合差集操作:所有术语ID减去不可映射术语ID
        self.mappable_term_ids = set(list(self.id_to_term.keys())).difference(self.nonmappable_terms)

    # 获取所有可映射的术语ID
    def get_mappable_term_ids(self):
        """
        返回所有可映射的术语ID集合
        """
        return self.mappable_term_ids

    # 获取所有可映射的术语对象
    def get_mappable_terms(self):
        """
        返回所有可映射的术语对象列表
        通过列表推导式过滤掉不可映射的术语
        """
        return [y
            for x,y in self.id_to_term.items()
            if x not in self.nonmappable_terms
        ]

# 构建本体的函数
def build_ontology(ont_to_loc, restrict_to_idspaces=None,
    include_obsolete=False, restrict_to_roots=None,
    exclude_terms=None):
    """
    构建本体图对象
    参数:
        ont_to_loc: 本体名称到文件位置的映射
        restrict_to_idspaces: 限制的ID空间
        include_obsolete: 是否包含过时的术语
        restrict_to_roots: 限制的根节点
        exclude_terms: 要排除的术语
    """
    # 解析OBO文件构建基础本体图
    og = parse_obos(ont_to_loc,
        restrict_to_idspaces=restrict_to_idspaces,
        include_obsolete=include_obsolete)

    # 添加扩展的同义词
    cvcl_syns_f = pr.resource_filename(
        resource_package,
        join("metadata", "term_to_extra_synonyms.json")
    )
    term_to_syns = None
    with open(cvcl_syns_f, "r") as f:
        term_to_syns = json.load(f)
    for term in list(og.id_to_term.values()):
        if term.id in term_to_syns:
            for syn in term_to_syns[term.id]:
                term.synonyms.add(Synonym(syn, "ENRICHED"))

    # 移除指定的同义词
    term_to_remove_syns_f = pr.resource_filename(
        resource_package,
        join("metadata", "term_to_remove_synonyms.json")
    )
    term_remove_syns = None
    with open(term_to_remove_syns_f, "r") as f:
        term_remove_syns = json.load(f)
    for t_id, rem_syn_data in term_remove_syns.items():
        if t_id in og.id_to_term:
            exclude_syns = set(rem_syn_data["exclude_synonyms"])
            term = og.id_to_term[t_id]
            term.synonyms = [
                x
                for x in term.synonyms
                if x.syn_str not in exclude_syns
            ]

    # 如果指定了根节点限制，构建子图
    if restrict_to_roots:
        keep_ids = set() # The IDs that we will keep

        # Get the subterms of terms that we want to keep
        for root_id in restrict_to_roots:
            keep_ids.update(og.recursive_subterms(root_id))

        # Build the ontology-graph object
        id_to_term = {}
        for t_id in keep_ids:
            t_name = og.id_to_term[t_id].name
            id_to_term[t_id] =  og.id_to_term[t_id]

            # Update the relationships between terms to remove dangling edges
            for rel, rel_ids in og.id_to_term[t_id].relationships.items():
                og.id_to_term[t_id].relationships[rel] = [
                    x
                    for x in rel_ids
                    if x in keep_ids
                ]

        return MappableOntologyGraph(id_to_term, exclude_terms)
    else:
        return MappableOntologyGraph(og.id_to_term, exclude_terms)


def most_specific_terms(term_ids, og, sup_relations=["is_a"]):
    """
    从给定的术语集合S中找出所有在S中没有子术语的术语
    (即找出最具体/最专门的术语), 就是最底层的叶子节点,  不能再向下划分了,  上面都是他爹.
    
    参数:
        term_ids: 术语ID集合
        og: 本体图对象
        sup_relations: 用于定义父子关系的关系类型列表,默认为["is_a"]
    返回:
        最具体术语ID的列表
    """
    # 过滤掉不在本体图中的术语ID
    term_ids = set([x for x in term_ids if x in og.id_to_term])

    # 如果没有有效术语,直接返回空集
    if len(term_ids) < 1:
        return term_ids

    # 获取所有术语对象
    terms = [og.id_to_term[x] for x in term_ids]
    most_specific_terms = []
    
    # 构建术语ID到其所有父术语ID的映射
    term_id_to_superterm_ids = {}
    for term in terms:
        # 使用recursive_relationship获取每个术语的所有父术语
        term_id_to_superterm_ids[term.id] = og.recursive_relationship(term.id, sup_relations)

    # 创建"比...更一般"的关系树
    have_relations = set()  # 存储有关系的术语
    more_general_than = {} # 存储术语间的层次关系
    # 遍历所有术语对,建立层次关系
    for term_a in list(term_id_to_superterm_ids.keys()):
        for term_b, b_superterms in term_id_to_superterm_ids.items():
            if term_a == term_b:
                continue
            # 如果term_a是term_b的父术语
            if term_a in b_superterms:
                if not term_a in list(more_general_than.keys()):
                    more_general_than[term_a] = []
                # 记录term_a比term_b更一般
                more_general_than[term_a].append(term_b)
                # 记录这两个术语有关系
                have_relations.update([term_a, term_b])

    # 收集关系树的叶子节点(即最具体的术语)
    for subs in list(more_general_than.values()):
        for s in subs:
            # 如果一个术语不是其他任何术语的父术语,那么它就是最具体的
            if not s in list(more_general_than.keys()):
                most_specific_terms.append(s)
                
    # 返回最具体术语列表
    # 包括:1)关系树的叶子节点 2)没有参与任何关系的孤立术语
    return list(set(most_specific_terms + list(set(term_ids) - have_relations)))


def parse_obos(ont_to_loc, restrict_to_idspaces=None, include_obsolete=False):

    def add_inverse_relationship_to_parents(term, relation, inverse_relation):
        for sup_term_id in [x for x in term.get_related_terms(relation)]:
            if sup_term_id in id_to_term:
                sup_term = id_to_term[sup_term_id]
                if inverse_relation not in sup_term.relationships:
                    sup_term.relationships[inverse_relation] = []
                sup_term.relationships[inverse_relation].append(term.id)
            else:
                if VERBOSE:
                    print("Warning! Attempted to create inverse edge in term %s. \
                        Not found in not in the ontology" % sup_term_id)
                # Remove superterm from term's relationship list because it
                # is not in the current ontology
                while sup_term_id in term.relationships[relation]:
                    term.relationships[relation].remove(sup_term_id)
                if not term.relationships[relation]:
                    del term.relationships[relation]


    id_to_term = {}
    name_to_ids = {}

    # Iterate through OBO files and build up the ontology
    print("Loading ontology...")
    for ont, loc in ont_to_loc.items():
        i_to_t, n_to_is = parse_obo(loc,
            restrict_to_idspaces=restrict_to_idspaces,
            include_obsolete=include_obsolete)
        id_to_term.update(i_to_t)
        for name, ids in n_to_is.items():
            if name not in name_to_ids:
                name_to_ids[name] = ids
            else:
                name_to_ids[name].update(ids)

    for term in list(id_to_term.values()):
        add_inverse_relationship_to_parents(term, "is_a", "inv_is_a")
        add_inverse_relationship_to_parents(term, "part_of", "inv_part_of")
        add_inverse_relationship_to_parents(term, "located_in", "inv_located_in")

    #return OntologyGraph(id_to_term, name_to_ids)
    return OntologyGraph(id_to_term)

# 解析OBO格式的本体文件
# 参数:
#   obo_file: OBO文件路径
#   restrict_to_idspaces: ID前缀列表,用于限制只包含特定ID空间的术语
#   include_obsolete: 是否包含已过时的术语
def parse_obo(obo_file, restrict_to_idspaces=None, include_obsolete=False):

    # 处理一组术语定义行
    # 参数:
    #   curr_lines: 当前要处理的行组
    #   restrict_to_idspaces: ID空间限制
    #   name_to_ids: 术语名称到ID的映射字典
    #   id_to_term: 术语ID到术语对象的映射字典
    def process_chunk_of_lines(curr_lines, restrict_to_idspaces,
        name_to_ids, id_to_term):
        # 解析实体(术语或类型定义)
        entity = parse_entity(curr_lines, restrict_to_idspaces)
        if not entity:
            if VERBOSE:
                print("ERROR!")
        # 如果是术语实体
        elif entity[0] == ENTITY_TERM:
            term = entity[1]
            is_obsolete = entity[2]
            # 如果术语未过时或允许包含过时术语
            if not is_obsolete or include_obsolete:
                # 添加到ID-术语映射
                id_to_term[term.id] = term
                # 添加到名称-ID映射
                if term.name not in name_to_ids:
                    name_to_ids[term.name]= set()
                name_to_ids[term.name].add(term.id)

    # 为术语添加反向关系到父术语
    # 参数:
    #   term: 要处理的术语
    #   relation: 关系类型(如"is_a")
    #   inverse_relation: 反向关系类型(如"inv_is_a")
    def add_inverse_relationship_to_parents(term, relation, inverse_relation):
        # 遍历所有具有指定关系的父术语
        for sup_term_id in [x for x in term.get_related_terms(relation)]:
            # 如果父术语存在于本体中
            if sup_term_id in id_to_term:
                sup_term = id_to_term[sup_term_id]
                # 添加反向关系
                if inverse_relation not in sup_term.relationships:
                    sup_term.relationships[inverse_relation] = []
                sup_term.relationships[inverse_relation].append(term.id)
            else:
                # 如果父术语不在本体中,输出警告并移除关系
                if VERBOSE:
                    print("Warning! Attempted to create inverse edge in term %s. \
                        Not found in not in the ontology" % sup_term_id)
                while sup_term_id in term.relationships[relation]:
                    term.relationships[relation].remove(sup_term_id)
                if not term.relationships[relation]:
                    del term.relationships[relation]

    # 初始化存储结构
    header_info = {}  # 存储文件头信息
    name_to_ids = {}  # 存储名称到ID的映射
    id_to_term = {}   # 存储ID到术语的映射

    # 打开并读取OBO文件
    with io.open(obo_file, "r", encoding="utf-8") as f:
        # 处理文件头部信息
        for line in f:
            if not line.strip():
                break  # 遇到空行表示头部结束
            # 解析头部信息并存储
            header_info[line.split(":")[0].strip()] = ":".join(line.split(":")[1:]).strip()

        # 处理术语定义部分
        curr_lines = []  # 存储当前处理的行组
        for line in f:
            # 遇到空行时处理已收集的行组
            if not line.strip():
                if not curr_lines:  # 如果没有收集到行,继续
                    continue
                # 处理当前行组
                process_chunk_of_lines(curr_lines, restrict_to_idspaces,
                    name_to_ids, id_to_term)
                curr_lines = []  # 重置行组
            else:
                # 收集非空行
                curr_lines.append(line)
                
        # 处理文件末尾的最后一组行
        if curr_lines:
            process_chunk_of_lines(curr_lines, restrict_to_idspaces,
                name_to_ids, id_to_term)

    # 返回构建的映射字典
    return id_to_term, name_to_ids



def parse_entity(lines, restrict_to_idspaces):
    # 解析术语的属性,将每行解析为键值对
    # 返回一个字典,键为属性名,值为该属性的所有值列表
    def parse_term_attrs(lines):
        attrs = {}
        for line in lines:
            tokens = line.split(":")
            if not tokens[0].strip() in list(attrs.keys()):
                attrs[tokens[0].strip()] = []
            attrs[tokens[0].strip()].append(":".join(tokens[1:]).strip())
        return attrs

    # 解析术语之间的关系
    # 包括is_a关系和其他类型的关系(如part_of)
    # 返回关系类型到相关术语ID列表的映射字典
    def parse_relationships(attrs):
        relationships = {}
        # 处理is_a关系
        is_a = [x.split("!")[0].split()[0].strip() for x in attrs["is_a"]] if "is_a" in attrs else set()
        if restrict_to_idspaces:
            is_a = [x for x in is_a if x.split(":")[0] in restrict_to_idspaces]
        if len(is_a) > 0:
            relationships["is_a"] = []
            for is_a_t in is_a:
                relationships["is_a"].append(is_a_t)

        # 处理其他类型的关系
        if "relationship" in attrs:
            for rel_raw in attrs["relationship"]:
                rel = rel_raw.split()[0]
                rel_term_id = rel_raw.split()[1]
                if rel not in relationships:
                    relationships[rel] = []
                relationships[rel].append(rel_term_id)

        return relationships

    # 从原始同义词文本中提取同义词信息
    # 返回Synonym对象的集合,每个对象包含同义词字符串和类型
    def extract_synonyms(raw_syns):
        """
        Args:
            raw_syns: all of the lines of the OBO file corresponding to synonyms
                of a given term.
        Returns:
            A set of tuples where the first element is the synonym string and
            the second element is the synonym type (e.g. 'EXACT' or 'NARROW')
        """
        synonyms = set()
        for syn in raw_syns:
            m = re.search('\".+\"', syn)
            if m:
                syn_type = syn.split('"')[2].strip().split()[0]
                parsed_syn = m.group(0)[1:-1].strip()
                synonyms.add(Synonym(parsed_syn, syn_type))
        return synonyms
        
    #     # 输入的raw_syns列表包含以下内容：
    # raw_syns = [
    #     '"T-lymphocyte" EXACT []',
    #     '"T lymphocyte" EXACT []',
    #     '"thymus-derived lymphocyte" BROAD []',
    #     '"mature alpha-beta T cell" NARROW []'
    # ]

    # # 函数处理每一行：
    # for syn in raw_syns:
    #     # 使用正则表达式查找引号中的文本
    #     m = re.search('\".+\"', syn)  # 匹配双引号之间的所有内容
    #     if m:
    #         # 提取同义词类型（EXACT、BROAD或NARROW）
    #         syn_type = syn.split('"')[2].strip().split()[0]
    #         # 提取同义词文本（去除引号和多余空格）
    #         parsed_syn = m.group(0)[1:-1].strip()
    #         # 创建Synonym对象并添加到集合中
    #         synonyms.add(Synonym(parsed_syn, syn_type))

    # 从原始外部引用文本中提取引用ID
    # 返回外部引用ID的列表
    def extract_xrefs(raw_xrefs):
        xrefs =set()
        for xref in raw_xrefs:
            xrefs.add(xref.split("!")[0].strip())
        return list(xrefs)

    # 检查术语是否应该被包含在本体中
    # 基于ID空间限制进行过滤
    def is_include_term(attrs):
        if restrict_to_idspaces:
            term_prefix = attrs["id"][0].split(":")[0]
            if term_prefix in restrict_to_idspaces:
                return True
            else:
                return False
        else:
            return True

    # 检查术语是否已过时
    # 通过查找is_obsolete标记实现
    def parse_is_obsolete(attrs):
        """
        Check if "is_obsolete: true" is included in the term.
        If so, this term is obsolete.
        """
        is_obsolete = False
        if "is_obsolete" in attrs:
            is_obsolete = True if attrs["is_obsolete"][0] == "true" else False
        return is_obsolete

    # 解析术语的同义词
    # 调用extract_synonyms处理原始同义词数据
    def parse_synonyms(attrs):
        return extract_synonyms(attrs["synonym"]) if "synonym" in list(attrs.keys()) else set()

    # 获取术语的定义文本
    def parse_definition(attrs):
        return attrs["def"][0] if "def" in list(attrs.keys()) else None

    # 解析术语的外部引用
    # 调用extract_xrefs处理原始引用数据
    def parse_xrefs(attrs):
        xrefs = []
        if "xref" in attrs:
            xrefs = extract_xrefs(attrs["xref"])
        return xrefs

    # 从原始属性值文本中提取属性-值对
    # 返回(属性,值)元组的集合
    def extract_property_values(raw_prop_vals):
        prop_vals = set()
        for prop_val in raw_prop_vals:
            if "\"" in prop_val:
                m = re.search('\".+\"', prop_val)
                if m:
                    prop = prop_val.split('"')[0].strip()
                    val = m.group(0)[1:-1].strip()
            else:
                prop = prop_val.split()[0].strip()
                val = prop_val.split()[1].strip()
            prop_vals.add((prop, val))
        return prop_vals

    # 获取术语所属的子集
    def parse_subsets(attrs):
        if "subset" in attrs:
            return set(attrs["subset"])
        return set()

    # 解析术语的属性值
    # 调用extract_property_values处理原始属性值数据
    def parse_property_values(attrs):
        return extract_property_values(attrs["property_value"]) if "property_value" in list(attrs.keys()) else set()

    # 获取术语的注释文本
    def parse_comment(attrs):
        if "comment" in attrs:
            return attrs["comment"][0]
        return None

    # 验证术语是否有效
    # 目前仅检查是否有name属性
    def is_valid_term(attrs):
        if "name" not in attrs:
            return False
        return True

    # 主要解析逻辑开始
    # 如果是术语定义
    if lines[0].strip() == "[Term]":
        # 解析所有属性
        attrs = parse_term_attrs(lines)

        # 检查是否应该包含该术语
        if not is_include_term(attrs):
            return (ENTITY_EXCLUDED_TERM, None)

        # 验证术语有效性
        if not is_valid_term(attrs):
            return ("ERROR PARSING ENTITY", None)

        # 解析术语的各个组成部分
        definition = parse_definition(attrs)
        synonyms = parse_synonyms(attrs)
        is_obsolete = parse_is_obsolete(attrs)
        xrefs = parse_xrefs(attrs)
        comment = parse_comment(attrs)
        relationships = parse_relationships(attrs)
        property_values = parse_property_values(attrs)
        subsets = parse_subsets(attrs)

        # 创建并返回Term对象
        term = Term(attrs["id"][0], attrs["name"][0].strip(),
            definition=definition, synonyms=set(synonyms), xrefs=xrefs,
            relationships=relationships, property_values=property_values,
            comment=comment, subsets=subsets)

        return (ENTITY_TERM, term, is_obsolete)

    # 如果是类型定义(暂未实现完整支持)
    elif lines[0].strip() == "[Typedef]":
        return (ENTITY_TYPE_DEF, None, None)

    # 无法识别的实体类型
    else:
        if VERBOSE:
            print("Unable to parse chunk: %s" % lines)
