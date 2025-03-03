# 导入所需的Python库
import json
from os.path import join
import h5py
import sys
from collections import defaultdict

from . import the_ontology
from .graph_lib.graph import DirectedAcyclicGraph


# 加载cello作者处理好的数据,包含以下内容:
# labels.json 细胞类型数据  （细胞的层级数据, 即一个细胞类型包含多个具体细胞类型,就像T细胞可以再分为更多细胞类型）
# expr_matrix.h5: 已有的表达数据矩阵
# experiment_to_study.json: 实验样本到研究批次的分组信息
# experiment_to_tags.json: 实验样本ID 到 技术标签的映射


def load(features, rsrc_loc):
    # 定义各种数据文件的路径
    labels_f = join(rsrc_loc, "resources", "training_set", "labels.json")
    studys_f = join(rsrc_loc, "resources", "training_set", "experiment_to_study.json")
    tags_f = join(rsrc_loc, "resources", "training_set", "experiment_to_tags.json")
    expr_matrix_f = join(rsrc_loc, "resources", "training_set", "{}.h5".format(features))

    # 加载本体数据, 最全的一个本体图
    og = the_ontology.the_ontology()

    # 加载标签数据和构建标签图
    with open(labels_f, "r") as f:
        labels_data = json.load(f)
        source_to_targets = labels_data["label_graph"]
        exp_to_labels = labels_data["labels"]
    label_graph = DirectedAcyclicGraph(source_to_targets)

    # 创建细胞类型ID到名称的映射
    label_to_name = {
        label: og.id_to_term[label].name for label in source_to_targets.keys()
    }

    # 每一条experiment(cell) 对应的最具体的细胞类型
    exp_to_ms_labels = {
        exp: label_graph.most_specific_nodes(labels)
        for exp, labels in exp_to_labels.items()
    }

    # 加载研究元数据
    with open(studys_f, "r") as f:
        exp_to_study = json.load(f)
    # 创建研究到实验的反向映射
    study_to_exps = defaultdict(lambda: set())
    for exp, study in exp_to_study.items():
        study_to_exps[study].add(exp)
    study_to_exps = dict(study_to_exps)

    # 加载技术标签, 该tag标签说明每条的item的处理方法
    with open(tags_f, "r") as f:
        exp_to_tags = json.load(f)

    # 加载表达矩阵数据
    print("Loading expression data from {}...".format(expr_matrix_f))
    with h5py.File(expr_matrix_f, "r") as f:
        # 处理cell ID, 每一条experiment可以看做是一个cell
        the_exps = [str(x)[2:-1] for x in f["experiment"][:]]
        # 处理基因ID
        gene_ids = [str(x)[2:-1] for x in f["gene_id"][:]]
        # 加载表达数据矩阵
        data_matrix = f["expression"][:]
    print("Loaded matrix of shape {}".format(data_matrix.shape))
    print("done.")

    # 创建实验到索引的映射, 索引 每一条experimen(cell)
    exp_to_index = {exp: ind for ind, exp in enumerate(the_exps)}

    # 返回所有加载的数据
    return (
        og,  # 本体数据
        label_graph,  # 细胞 label_id 层级图
        label_to_name,  # 细胞 label_id 到 细胞类型名称的映射
        the_exps,  # experiment(cell)
        exp_to_index,  # 索引 每一条experiment(cell)
        exp_to_labels,  # experiment(cell) 到 细胞类型ID 的映射：一个实验样本, 对应多个细胞 label_id. 因为一个具体细胞类型可能属于很多细胞类型, 一个细胞类型可能包含多个具体细胞类型
        exp_to_tags,  # 技术标签：说明该 实验样本ID 使用的技术("poly_a_rna", "uncultured")
        exp_to_study,  # 实验样本ID 到 研究ID的映射
        study_to_exps,  # 研究ID 到 实验样本ID的映射
        exp_to_ms_labels,  # 每一条experiment(cell) 对应的最具体的细胞类型, 用于训练的应该就是这个
        data_matrix,  # 表达数据矩阵
        gene_ids,  # 基因ID
    )


# 加载稀疏数据集的函数
def load_sparse_dataset():
    # 定义数据文件路径
    labels_f = join(data_dir, "labels.json")
    studys_f = join(data_dir, "experiment_to_study.json")
    tags_f = join(data_dir, "experiment_to_tags.json")
    expr_matrix_f = join(data_dir, "{}.h5".format(features))

    # 加载本体数据
    og = the_ontology.the_ontology()

    # 加载标签数据和构建标签图
    with open(labels_f, "r") as f:
        labels_data = json.load(f)
        source_to_targets = labels_data["label_graph"]
        exp_to_labels = labels_data["labels"]
    label_graph = DirectedAcyclicGraph(source_to_targets)

    # 创建标签ID到可读名称的映射
    label_to_name = {
        label: og.id_to_term[label].name for label in source_to_targets.keys()
    }

    # 为每个实验创建最具体标签的映射
    exp_to_ms_labels = {
        exp: label_graph.most_specific_nodes(labels)
        for exp, labels in exp_to_labels.items()
    }

    # 加载研究元数据
    with open(studys_f, "r") as f:
        exp_to_study = json.load(f)
    study_to_exps = defaultdict(lambda: set())
    for exp, study in exp_to_study.items():
        study_to_exps[study].add(exp)
    study_to_exps = dict(study_to_exps)

    # 加载技术标签
    with open(tags_f, "r") as f:
        exp_to_tags = json.load(f)

    # 加载表达矩阵数据
    print("Loading expression data from {}...".format(expr_matrix_f))
    with h5py.File(expr_matrix_f, "r") as f:
        # 处理实验ID（与普通加载不同，这里不去除字符串两端的字符）
        the_exps = [str(x) for x in f["experiment"][:]]
        # 处理基因ID
        gene_ids = [str(x) for x in f["gene_id"][:]]
        # 加载表达数据矩阵
        data_matrix = f["expression"][:]
    print("Loaded matrix of shape {}.".format(data_matrix.shape))
    print("done.")

    # 创建实验到索引的映射
    exp_to_index = {exp: ind for ind, exp in enumerate(the_exps)}

    # 返回所有加载的数据
    return (
        og,
        label_graph,
        label_to_name,
        the_exps,
        exp_to_index,
        exp_to_labels,
        exp_to_tags,
        exp_to_study,
        study_to_exps,
        exp_to_ms_labels,
        data_matrix,
        gene_ids,
    )


# 主程序入口
if __name__ == "__main__":
    main()
