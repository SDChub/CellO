"""
The CellO API

Authors: Matthew Bernstein <mbernstein@morgridge.org>
"""

from optparse import OptionParser
from os.path import join
import pandas as pd
from anndata import AnnData
import h5py
import dill
from scipy.io import mmread
import numpy as np
import json
from collections import defaultdict
import sys
import os

from . import the_ontology
from . import load_expression_matrix
from . import load_training_data
from . import download_resources
from . import ontology_utils as ou
from . import models
from .models import model
from .graph_lib.graph import DirectedAcyclicGraph

# Units keywords
COUNTS_UNITS = 'COUNTS'  # 原始计数
CPM_UNITS = 'CPM'  # 每百万读数计数
LOG1_CPM_UNITS = 'LOG1_CPM'  # log(CPM+1)转换
TPM_UNITS = 'TPM'  # 每百万转录本计数
LOG1_TPM_UNITS = 'LOG1_TPM'  # log(TPM+1)转换

# Assay keywords
FULL_LENGTH_ASSAY = 'FULL_LENGTH'  # 全长测序
THREE_PRIMED_ASSAY = '3_PRIME'  # 3'端测序

UNITS = 'log_tpm'

ALGO_TO_INTERNAL = {
    'IR': 'isotonic_regression',  # 保序回归
    'CLR': 'cdc'  # 级联逻辑回归
}
ALGO_TO_PARAMS = {
    'IR': {  # 保序回归算法参数
        "assert_ambig_neg": False,
        "binary_classifier_algorithm": "logistic_regression",
        "binary_classifier_params": {
            "penalty": "l2",
            "penalty_weight": 0.0006,
            "solver": "liblinear",
            "intercept_scaling": 1000.0,
            "downweight_by_class": True
        }
    },
    'CLR': {  # 级联逻辑回归算法参数
        "assert_ambig_neg": False,
        "binary_classifier_algorithm": "logistic_regression",
        "binary_classifier_params": {
            "penalty": "l2",
            "penalty_weight": 0.001,
            "solver": "liblinear",
            "intercept_scaling": 1000.0,
            "downweight_by_class": True
        }
    }
}
PREPROCESSORS = ['pca']
PREPROCESSOR_PARAMS = [{
    "n_components": 3000  # PCA保留3000个主成分
}]

QUALIFIER_TERMS = set([
    'CL:2000001',   # 外周血单核细胞
    'CL:0000081',   # 血细胞
    'CL:0000080',   # 循环细胞
    'CL:0002321'    # 胚胎细胞
])

def train_model(ad, rsrc_loc, algo='IR', log_dir=None):
    """
    基于输入数据集的基因训练CellO模型

    参数:
    ----------
    ad : AnnData对象
        n个细胞 x m个基因的表达矩阵

    algo : 字符串
        用于训练模型的算法名称。'IR'使用保序回归,'CLR'使用级联逻辑回归

    rsrc_loc: 字符串
        已下载的"resources"目录的位置

    log_dir : 字符串
        用于写入日志信息的目录路径

    返回:
    -------
    训练好的CellO模型
    """
    _download_resources(rsrc_loc)

    genes = ad.var.index


    # 矩阵数据就是 实验样本ID X 基因ID 的矩阵
    r = load_training_data.load(UNITS, rsrc_loc)
    og = r[0]               # 本体数据
    label_graph = r[1]      # 细胞label_id 层级图
    label_to_name = r[2]    # 细胞label_id 到 细胞类型名称的映射
    the_exps = r[3]         # 实验样本ID
    exp_to_index = r[4]     # 实验样本ID 到 索引的映射
    exp_to_labels = r[5]    # 实验样本ID 到 细胞类型ID的映射：一个实验样本, 对应多个细胞 label_id. 因为一个具体细胞类型可能属于很多细胞类型, 一个细胞类型可能包含多个具体细胞类型
    exp_to_tags = r[6]      # 技术标签：说明该 实验样本ID 使用的技术("poly_a_rna", "uncultured")
    exp_to_study = r[7]     # 实验样本ID 到 研究ID的映射
    study_to_exps = r[8]    # 研究ID 到 实验样本ID的映射
    exp_to_ms_labels = r[9] # 实验样本ID 到 最具体细胞类型的映射. 应该就是最终使用细胞label
    X = r[10]               # 表达数据矩阵
    all_genes = r[11]       # 基因ID

    # 匹配测试数据中的基因到训练数据中的基因
    # 拿到训练基因和基因在all_genes基因表中对应的索引
    train_genes, gene_to_indices = _match_genes(
        genes, 
        all_genes, 
        rsrc_loc,  #用于处理HGNC基因符号
        log_dir=log_dir
    )

    # 提取训练基因对应的表达数据列
    # 注意:如果测试集中的一个基因映射到多个训练基因,则对这些训练基因的表达值求和
    # 为啥会映射到多个训练基因? 因为基因名称不唯一, 比如CD14, CD14+, CD14++, CD14+++, 这些基因名称都表示同一个基因
    X_train = []
    for gene in train_genes:
        # 获取当前基因对应的所有训练基因索引
        indices = gene_to_indices[gene]
        # 对所有映射的训练基因表达值求和
        X_train.append(np.sum(X[:,indices], axis=1))
    # 转换为numpy数组并转置,使其符合样本×特征的格式
    X_train = np.array(X_train).T
    # 验证特征数量与训练基因数量一致
    assert X_train.shape[1] == len(train_genes)

    # 使用处理好的训练数据训练模型
    print('正在训练模型...')
    mod = model.train_model(
        ALGO_TO_INTERNAL[algo],     # 将算法名称转换为内部使用的名称
        ALGO_TO_PARAMS[algo],   # 获取对应算法的参数配置
        X_train,               # 训练数据矩阵
        the_exps,          # 实验样本ID列表 理解为细胞ID
        exp_to_labels,  # 实验样本到细胞类型标签的映射, 一个实验样本, 对应多个细胞 label_id.
        label_graph,     # 细胞类型标签的层级关系图
        item_to_group=exp_to_study,  # 实验样本到研究批次的分组信息
        features=train_genes,        # 特征(基因)列表
        preprocessor_names=PREPROCESSORS,          # 预处理器名称列表
        preprocessor_params=PREPROCESSOR_PARAMS    # 预处理器参数
    )
    print('完成。')
    return mod


def load_training_set():
    """
    加载默认单位(log_tpm)的训练数据集
    
    返回:
    -------
    r : tuple
        包含训练数据的元组,包括:
        - 本体数据
        - 标签层级图
        - 标签到名称的映射
        - 实验样本列表
        - 各种索引和映射关系
        - 表达矩阵数据
        - 基因列表
    """
    r = load_training_data.load(UNITS)
    return r


def predict(
        ad,
        mod,
        algo='IR',
        clust_key='leiden',
        log_dir=None,
        remove_anatomical_subterms=None,
        rsrc_loc=None
    ):
    """
    对给定的表达矩阵进行细胞类型分类预测

    参数:
    ----------
    ad : AnnData对象
        n个细胞 x m个基因的表达矩阵

    mod : Model对象, 默认: None
        训练好的分类模型。如果为None，将使用预训练的默认模型；
        如果'train'为True，则训练新模型。

    algo : 字符串, 可选, 默认: 'IR'
        使用的层次分类算法。必须是{'IR', 'CLR'}中的一个。
        这些关键字代表:
            IR: 保序回归校正
            CLR: 级联逻辑回归

    clust_key : 字符串, 默认: 'leiden'
        在观察注释'.obs'中表示细胞聚类身份的键名

    log_dir : 字符串, 默认: None
        日志文件的输出目录

    remove_anatomical_subterms : 列表, 默认: None
        需要从预测结果中移除的解剖学子术语列表

    rsrc_loc : 字符串, 默认: 当前目录
        CellO资源文件的位置。如果在此路径下找不到资源文件，
        将自动下载。

    返回值:
    -------
    (probabilities, binary_classifications, most_specific_results) : 三个Pandas DataFrame
        - probabilities: 存储每个细胞类型在所有输入细胞中的原始概率
        - binary_classifications: 存储每个细胞类型的二值化(1或0)分类结果
        - most_specific_results: 存储每个细胞的最具体细胞类型预测结果
    """

    # 设置资源文件位置为当前工作目录
    if rsrc_loc is None:
        rsrc_loc = os.getcwd()

    # 如果资源文件不存在则下载
    _download_resources(rsrc_loc)

    # 检查模型是否与数据兼容
    # 只要包含了模型所需要的基因就可以, 多余的基因, 模型用不到,会自动忽略.
    is_compatible = check_compatibility(ad, mod)
    if not is_compatible:
        print("错误：数据矩阵中的基因与分类器期望的基因不匹配。")
        print("请使用cello_train_model.py程序在此输入基因集上训练分类器，")
        print("或使用'-t'标志运行cello_classify。")
        sys.exit()

    # 计算原始分类器概率, 通过聚类, 计算每个细胞属于每个聚类的概率
    results_df, cell_to_clust = _raw_probabilities(
        ad,
        mod,
        algo=algo,
        clust_key=clust_key,
        log_dir=log_dir
    )

    # 按解剖学实体过滤
    if remove_anatomical_subterms is not None:
        print("过滤以下类型的细胞预测结果:\n{}".format(
            "\n".join([
                "{} ({})".format(
                    ou.cell_ontology().id_to_term[term].name,
                    term
                )
                for term in remove_anatomical_subterms
            ])
        ))
        results_df = _filter_by_anatomical_entity(
            results_df, 
            remove_subterms_of=remove_anatomical_subterms
        )

    # ===================== 概率二值化处理部分 =====================
    # 步骤1: 获取经验阈值
    # 经验阈值是通过分析训练数据集得出的最优决策阈值
    # 每个细胞类型都有其特定的阈值，用于将预测概率转换为二值化结果(0/1)
    threshold_df = _retrieve_empirical_thresholds(ad, algo, rsrc_loc)
    
    # 步骤2: 获取细胞类型的层级关系图
    # 层级关系图是一个有向无环图(DAG)，表示细胞类型之间的父子关系
    # 例如：T细胞是CD4+T细胞和CD8+T细胞的父类型
    # 这种关系对于保证预测结果的生物学一致性很重要
    label_graph = _retrieve_label_graph(rsrc_loc)
    
    # 步骤3: 执行概率二值化转换
    # 参数说明：
    # - results_df: 原始预测概率矩阵，每行是一个细胞，每列是一个细胞类型
    # - threshold_df: 包含每个细胞类型的决策阈值
    # - label_graph: 细胞类型的层级关系图
    # 处理逻辑：
    # 1. 将每个预测概率与其对应的阈值比较
    # 2. 根据层级关系调整预测结果（如果父类型为负，所有子类型也设为负）
    # 3. 生成最终的二值化结果矩阵
    binary_results_df = _binarize_probabilities(
        results_df,
        threshold_df,
        label_graph
    )
    # ==========================================================

    # 如果有多个最具体的细胞类型，选择其中一个
    finalized_binary_results_df, ms_results_df = _select_one_most_specific(
        binary_results_df,
        results_df,
        threshold_df,
        label_graph,
        precision_thresh=0.0
    )

    # 将聚类预测映射回细胞
    if cell_to_clust is not None:
        results_da = [
            results_df.loc[cell_to_clust[cell]]
            for cell in ad.obs.index
        ]
        results_df = pd.DataFrame(
            data=results_da,
            index=ad.obs.index,
            columns=results_df.columns
        )

        finalized_binary_results_da = [
            finalized_binary_results_df.loc[cell_to_clust[cell]]
            for cell in ad.obs.index
        ]
        finalized_binary_results_df = pd.DataFrame(
            data=finalized_binary_results_da,
            index=ad.obs.index,
            columns=finalized_binary_results_df.columns
        )

        ms_results_da = [
            ms_results_df.loc[cell_to_clust[cell]]
            for cell in ad.obs.index
        ]
        ms_results_df = pd.DataFrame(
            data=ms_results_da,
            index=ad.obs.index,
            columns=ms_results_df.columns
        )
    return results_df, finalized_binary_results_df, ms_results_df


def _retrieve_pretrained_model(ad, algo, rsrc_loc):
    """
    检查是否有预训练模型与输入数据集兼容

    参数:
    ----------
    ad : AnnData对象
        输入的表达矩阵数据

    algo : 字符串
        使用的算法类型 ('IR' 或 'CLR')

    rsrc_loc : 字符串
        资源文件的位置

    返回值:
    -------
    mod : Model对象或None
        如果找到兼容的预训练模型则返回该模型，否则返回None
    """
    
    # 确保资源文件存在
    _download_resources(rsrc_loc)

    print('正在检查是否有预训练模型与此输入数据集兼容...')
    
    # 预训练模型文件名列表
    pretrained_ir = [
        'ir.dill',
        'ir.10x.dill',
    ]
    pretrained_clr = [
        'clr.dill',
        'clr.10x.dill'
    ]
    
    mod = None
    assert algo in ALGO_TO_INTERNAL.keys()
    
    # 检查IR算法的预训练模型
    if algo == 'IR':
        for model_fname in pretrained_ir:
            model_f = join(
                rsrc_loc,
                "resources",
                "trained_models", 
                model_fname
            )
            with open(model_f, 'rb') as f:
                mod = dill.load(f)  
            feats = mod.classifier.features
            # 检查特征是否完全匹配
            if frozenset(feats) == frozenset(ad.var.index):
                return mod
                
    # 检查CLR算法的预训练模型
    elif algo == 'CLR':
        for model_fname in pretrained_clr:
            model_f = join(
                rsrc_loc,
                "resources",
                "trained_models", 
                model_fname
            )
            with open(model_f, 'rb') as f:
                mod = dill.load(f)
            feats = mod.classifier.features
            # 检查特征是否完全匹配
            if frozenset(feats) == frozenset(ad.var.index):
                return mod
                
    print('未找到兼容的预训练模型。')
    return None


def _download_resources(rsrc_loc):
    if not os.path.isdir(join(rsrc_loc, "resources")):
        msg = """
        Could not find the CellO resources directory called
        'resources' in '{}'. Will download resources to current 
        directory.
        """.format(rsrc_loc)
        print(msg)
        download_resources.download(rsrc_loc)
    else:
        print("Found CellO resources at '{}'.".format(join(rsrc_loc, 'resources')))


def retreive_pretrained_model_from_local(ad, model_dir):
    """
    从本地目录搜索预训练模型,返回第一个与输入数据集基因匹配的模型
    
    这是一个辅助函数,用于管理可能用于不同数据集的大型预训练模型集合。

    参数:
    ----------
    ad : AnnData对象
        n个细胞 x m个基因的表达矩阵

    model_dir: 字符串
        存储预训练模型(dill文件格式)的目录路径

    返回值:
    -------
    Model对象或None
        返回目录中第一个与输入数据集兼容的模型对象,如果未找到则返回None
    """
    # 遍历目录中的所有模型文件
    for model_fname in os.listdir(model_dir):
        model_f = join(model_dir, model_fname)
        # 加载模型
        with open(model_f, 'rb') as f:
            mod = dill.load(f)
        # 获取模型的特征(基因)列表
        feats = mod.classifier.features
        # 检查模型特征是否是输入数据特征的子集
        if frozenset(feats) < frozenset(ad.var.index):
            print("在文件中找到兼容模型: ", model_f)
            return mod
    return None

     
def check_compatibility(ad, mod):
    """
    检查模型与数据的兼容性
    
    参数:
    ad: AnnData对象 - 输入数据
    mod: Model对象 - 待检查的模型
    
    返回:
    bool - 如果模型的特征是输入数据特征的子集则返回True
    只要包含了模型所需要的基因就可以, 多余的基因, 模型用不到,会自动忽略.
    属于是 找到必要条件
    """
    return frozenset(mod.classifier.features) <= frozenset(ad.var.index)


def _raw_probabilities(
        ad, 
        mod,
        algo='IR', 
        clust_key='leiden',
        log_dir=None
    ):
    """
    计算原始分类器概率
    
    参数:
    ----------
    ad : AnnData对象
        输入的表达矩阵
    mod : Model对象
        训练好的分类器模型
    algo : str, 默认='IR'
        使用的算法类型
    clust_key : str, 默认='leiden'
        聚类结果的键名
    log_dir : str, 可选
        日志输出目录
        
    返回:
    -------
    tuple(DataFrame, dict)
        - 分类概率矩阵
        - 细胞到聚类的映射字典
    """
    # 验证模型与数据的兼容性
    assert check_compatibility(ad, mod)

    # 重排列特征列以匹配模型
    features = mod.classifier.features
    ad = ad[:,features]

    # 处理聚类信息
    if clust_key:
        # 检查聚类键是否存在
        if clust_key not in ad.obs.columns:
            sys.exit(
                """
                错误: 在AnnData对象的'.obs'变量中未找到聚类键名 {}。
                """.format(clust_key)
            )

        # 按聚类合并细胞
        ad_clust = _combine_by_cluster(ad, clust_key=clust_key)
        # 如果只有一个聚类,扩展表达矩阵维度
        # AnnData会压缩维度,所以需要保持为Numpy数组
        if len(ad_clust.X.shape) == 1:
            expr = np.expand_dims(ad_clust.X, 0)
        else:
            expr = ad_clust.X
        # 预测聚类的分类概率
        conf_df, score_df = mod.predict(expr, ad_clust.obs.index)
        # 构建细胞到聚类的映射
        cell_to_clust = {
            cell: str(clust)
            for cell, clust in zip(ad.obs.index, ad.obs[clust_key])
        }
    else:
        # 不使用聚类,直接预测每个细胞
        cell_to_clust = None
        conf_df, score_df = mod.predict(ad.X, ad.obs.index)
    
    # conf_df 包含每个聚类对应每种细胞类型的预测概率
    # cell_to_clust 包含每个细胞对应的聚类
    return conf_df, cell_to_clust


def _aggregate_expression(X):
    """
    聚合表达矩阵中的计数以形成伪整体表达谱
    
    参数:
    X : 矩阵
        log(TPM+1)格式的表达矩阵,行对应细胞,列对应基因
        
    返回:
    ndarray
        聚合后的表达谱
    """
    # 将log(TPM+1)转回TPM,并归一化
    X = np.expm1(X) / 1e6
    # 对所有细胞求和
    x_clust = np.squeeze(np.array(np.sum(X, axis=0)))
    # 再次归一化
    sum_x_clust = float(sum(x_clust))
    x_clust = np.array([x/sum_x_clust for x in x_clust])
    x_clust *= 1e6
    # 转回log(TPM+1)格式
    x_clust = np.log(x_clust+1)
    return x_clust


def _combine_by_cluster(ad, clust_key='leiden'):
    """
    将AnnData对象中的细胞按聚类合并
    
    参数:
    ----------
    ad : AnnData对象
        原始的单细胞数据
    clust_key : str, 默认='leiden'
        聚类结果的键名
        
    返回:
    -------
    AnnData
        每个元素代表一个聚类而不是单个细胞的新AnnData对象
    """
    clusters = []
    X_mean_clust = []
    # 遍历每个聚类
    for clust in sorted(set(ad.obs[clust_key])):
        # 获取该聚类的所有细胞
        cells = ad.obs.loc[ad.obs[clust_key] == clust].index
        X_clust = ad[cells,:].X
        # 计算聚类的平均表达谱
        x_clust = _aggregate_expression(X_clust)
        X_mean_clust.append(x_clust)
        clusters.append(str(clust))
    # 构建新的AnnData对象
    X_mean_clust = np.array(X_mean_clust)
    ad_mean_clust = AnnData(
        X=X_mean_clust,
        var=ad.var,
        obs=pd.DataFrame(
            data=clusters,
            index=clusters
        )
    )
    return ad_mean_clust


def _retrieve_empirical_thresholds(ad, algo, rsrc_loc):
    """
    检索经验阈值
    
    参数:
    ----------
    ad : AnnData对象
        输入的表达矩阵
    algo : str
        使用的算法类型('IR'或'CLR')
    rsrc_loc : str
        资源文件位置
        
    返回:
    -------
    DataFrame
        包含每个标签的经验阈值的数据框
    """
    print('检查是否有预训练模型与此输入数据集兼容...')
    # 预训练模型和对应的阈值文件
    pretrained_ir = [
        ('ir.dill', 'ir.all_genes_thresholds.tsv'), 
        ('ir.10x.dill', 'ir.10x_genes_thresholds.tsv') 
    ]
    pretrained_clr = [
        ('clr.dill', 'clr.all_genes_thresholds.tsv'),
        ('clr.10x.dill', 'clr.10x_genes_thresholds.tsv')
    ]
    mod = None
    max_genes_common = 0
    best_thresh_f = None
    
    # 根据算法类型检查对应的预训练模型
    if algo == 'IR':
        for model_fname, thresh_fname in pretrained_ir:
            model_f = join(rsrc_loc, "resources", "trained_models", model_fname)
            with open(model_f, 'rb') as f:
                mod = dill.load(f)  
            feats = mod.classifier.features
            # 计算模型特征与输入数据集特征的重叠比例
            matched_genes, _ = _match_genes(ad.var.index, feats, rsrc_loc, verbose=False)
            common = len(frozenset(feats) & frozenset(matched_genes)) / len(feats)
            if common >= max_genes_common:
                max_genes_common = common
                best_thresh_f = join(rsrc_loc, "resources", "trained_models", thresh_fname)
    elif algo == 'CLR':
        for model_fname, thresh_fname in pretrained_clr:
            model_f = join(rsrc_loc, "resources", "trained_models", model_fname)
            with open(model_f, 'rb') as f:
                mod = dill.load(f)
            feats = mod.classifier.features
            matched_genes, _ = _match_genes(ad.var.index, feats, rsrc_loc, verbose=False)
            common = len(frozenset(feats) & frozenset(matched_genes)) / len(feats)
            if common >= max_genes_common:
                max_genes_common = common
                best_thresh_f = join(rsrc_loc, "resources", "trained_models", thresh_fname)
                
    print('使用存储在{}中的阈值'.format(best_thresh_f))
    thresh_df = pd.read_csv(best_thresh_f, sep='\t', index_col=0)
    return thresh_df


def _retrieve_label_graph(rsrc_loc):
    """
    从资源文件中加载标签图结构
    
    参数:
    ----------
    rsrc_loc : str
        资源文件位置
        
    返回:
    -------
    DirectedAcyclicGraph
        表示标签层级关系的有向无环图
    """
    labels_f = join(rsrc_loc, "resources", "training_set", "labels.json")
    with open(labels_f, 'r') as f:
        labels_data = json.load(f)
        source_to_targets = labels_data['label_graph']  # 标签之间的父子关系
        exp_to_labels = labels_data['labels']  # 实验样本到标签的映射
    label_graph = DirectedAcyclicGraph(source_to_targets)
    return label_graph


def _filter_by_anatomical_entity(results_df, remove_subterms_of):
    """
    根据解剖学实体过滤预测结果
    
    参数:
    ----------
    results_df : DataFrame
        预测结果数据框
    remove_subterms_of : list
        需要移除其子术语的解剖学实体列表
        
    返回:
    -------
    DataFrame
        过滤后的预测结果
    """
    labels = set(results_df.columns)
    all_subterms = set()
    # 移除指定解剖学实体的所有子术语
    for term in remove_subterms_of:
        subterms = ou.cell_ontology().recursive_relationship(
            term, 
            ['inv_is_a', 'inv_part_of', 'inv_located_in']
        )
        labels -= subterms
    labels = sorted(labels)
    results_df = results_df[labels]
    return results_df


def _binarize_probabilities(results_df, decision_df, label_graph):
    """
    将概率值二值化为0/1分类结果
    
    参数:
    ----------
    results_df : DataFrame
        原始预测概率
    decision_df : DataFrame
        包含每个标签阈值的决策表, 通过这个判断选择的模型
    label_graph : DirectedAcyclicGraph
        标签层级关系图
        
    返回:
    -------
    DataFrame
        二值化的分类结果
    """
    # 将每个标签映射到其经验阈值
    # 拿到每个标签的阈值,如果超过这个阈值,就认为这个细胞属于这个标签
    label_to_thresh = {
        label: decision_df.loc[label]['threshold']
        for label in decision_df.index
    }

    print('正在二值化分类结果...')
    # 获取每个细胞类型标签的后代节点
    label_to_descendents = {
        label: label_graph.descendent_nodes(label)
        for label in label_graph.get_all_nodes()
    }

    da = []
    the_labels = sorted(set(results_df.columns) & set(label_to_thresh.keys()))
    # 遍历每个样本
    for exp_i, exp in enumerate(results_df.index):
        if (exp_i+1) % 100 == 0:
            print('已处理{}个样本。'.format(exp_i+1))
        # 将每个标签映射到其分类得分
        label_to_conf = {
            label: results_df.loc[exp][label]
            for label in results_df.columns
        }
        # 计算每个标签是否超过其阈值
        label_to_is_above = {
            label: int(conf > label_to_thresh[label])
            for label, conf in label_to_conf.items()
            if label in the_labels
        }
        # 将每个标签映射到其二值化结果
        label_to_bin = {
            label: is_above
            for label, is_above in label_to_is_above.items()
        }
        # 将负预测传播到所有后代节点
        for label, over_thresh in label_to_is_above.items():
            if not bool(over_thresh):
                desc_labels = label_to_descendents[label] # 获取某个细胞类型标签的后代节点
                for desc_label in set(desc_labels) & set(label_to_bin.keys()):
                    label_to_bin[desc_label] = int(False)
        da.append([
            label_to_bin[label]
            for label in the_labels
        ])
    df = pd.DataFrame(
        data=da,
        index=results_df.index,
        columns=the_labels
    )
    return df


def _select_one_most_specific(binary_results_df, results_df, decision_df, label_graph, precision_thresh=0.0):
    """
    从二值化结果中选择最具体的细胞类型
    
    参数:
    ----------
    binary_results_df : DataFrame
        二值化的分类结果
    results_df : DataFrame
        原始预测概率
    decision_df : DataFrame
        决策阈值表
    label_graph : DirectedAcyclicGraph
        标签层级关系图
    precision_thresh : float, 默认=0.0
        精确度阈值
        
    返回:
    -------
    tuple(DataFrame, DataFrame)
        - 最终的二值化结果
        - 最具体的细胞类型预测结果
    """
    # 解析决策阈值表
    label_to_f1 = {
        label: decision_df.loc[label]['F1-score']
        for label in decision_df.index
    }
    label_to_prec = {
        label: decision_df.loc[label]['precision']
        for label in decision_df.index
    }
    label_to_thresh = {
        label: decision_df.loc[label]['empirical_threshold']
        for label in decision_df.index
    }

    # 将每个标签映射到其祖先节点
    label_to_ancestors = {
        label: label_graph.ancestor_nodes(label)
        for label in label_graph.get_all_nodes()
    }

    # 根据经验精确度过滤标签
    hard_labels = set([
        label
        for label, prec in label_to_prec.items()
        if prec < precision_thresh
    ])
    
    # 将每个实验映射到其预测标签
    print('正在将每个样本映射到其预测标签...')
    consider_labels = set(binary_results_df.columns) - hard_labels
    exp_to_pred_labels = {
        exp: [
            label
            for label in consider_labels
            if binary_results_df.loc[exp][label] == 1
        ]
        for exp in binary_results_df.index
    }

    print('正在计算最具体的预测标签...')
    # 获取每个实验的最具体预测标签
    exp_to_ms_pred_labels = {
        exp: label_graph.most_specific_nodes(set(pred_labels) - QUALIFIER_TERMS)
        for exp, pred_labels in exp_to_pred_labels.items()
    }

    # 选择概率最高的细胞
    exp_to_select_pred_label = {
        exp: max(
            [
                (label, results_df.loc[exp][label])
                for label in ms_pred_labels
            ],
            key=lambda x: x[1]
        )[0]
        for exp, ms_pred_labels in exp_to_ms_pred_labels.items()
        if len(ms_pred_labels) > 0
    } 
   
    # 将每个实验映射到其最终标签
    exp_to_update_pred = {}
    for exp, select_label in exp_to_select_pred_label.items():
        print('样本 {} 预测为 "{} ({})"'.format(
            exp, 
            ou.cell_ontology().id_to_term[select_label].name, 
            select_label
        ))
        all_labels = label_to_ancestors[select_label] 
        exp_to_update_pred[exp] = all_labels

    # 添加限定词细胞类型
    for exp in exp_to_update_pred:
        for qual_label in QUALIFIER_TERMS:
            if qual_label in exp_to_pred_labels[exp]:
                all_labels = label_to_ancestors[qual_label]
                exp_to_update_pred[exp].update(all_labels)

    # 创建过滤后的结果数据框
    da = []
    for exp in binary_results_df.index:
        row = []
        for label in binary_results_df.columns:
            if exp in exp_to_update_pred and label in exp_to_update_pred[exp]:
                row.append(1)
            else:
                row.append(0)
        da.append(row)

    df = pd.DataFrame(
        data=da,
        columns=binary_results_df.columns,
        index=binary_results_df.index
    )

    # 最具体的细胞类型标签
    da = []
    for exp in binary_results_df.index:
        if exp in exp_to_select_pred_label:
            da.append(exp_to_select_pred_label[exp])
        else:
            # 该实验没有预测结果
            da.append('')
    df_ms = pd.DataFrame(
        data=da,
        index=binary_results_df.index,
        columns=['most_specific_cell_type']
    )
    return df, df_ms


# 匹配基因名称
def _match_genes(test_genes, all_genes, rsrc_loc, verbose=True, log_dir=None, ret_ids=False):
    # 将全部基因映射到其索引
    gene_to_index = {
        gene: index
        for index, gene in enumerate(all_genes)
    }
    
    # 处理Ensembl基因ID(无版本号)
    if 'ENSG' in test_genes[0] and '.' not in test_genes[0]:
        print("推断输入文件使用Ensembl基因ID。")
        if '.' in test_genes[0]:
            print("推断基因ID包含版本号")
            test_genes = [
                gene_id.split('.')[0]
                for gene_id in test_genes
            ]
        train_genes = sorted(set(test_genes) & set(all_genes))
        not_found = set(all_genes) - set(test_genes)
        train_ids = train_genes
        gene_to_indices = {
            gene: [gene_to_index[gene]]
            for gene in train_genes
        }
    
    # 处理Ensembl基因ID(带版本号)
    elif 'ENSG' in test_genes[0] and '.' in test_genes[0]:
        print("推断输入文件使用带版本号的Ensembl基因ID")
        all_genes = set(all_genes)
        train_ids = []
        train_genes = []
        not_found = []
        gene_to_indices = {}
        for gene in test_genes:
            gene_no_version = gene.split('.')[0]
            if gene_no_version in all_genes:
                train_ids.append(gene_no_version)
                train_genes.append(gene)
                gene_to_indices[gene] = [gene_to_index[gene_no_version]]
            else:
                not_found.append(gene)
    
    # 处理HGNC基因符号
    elif len(set(['CD14', 'SOX2', 'NANOG', 'PECAM1']) & set(test_genes)) > 0:
        if verbose:
            print("推断输入文件使用HGNC基因符号。")
        genes_f = join(
            rsrc_loc,
            'resources',
            'gene_metadata', 
            'biomart_id_to_symbol.tsv'
        )
        with open(genes_f, 'r') as f:
            sym_to_ids = defaultdict(lambda: [])
            for l in f:
                gene_id, gene_sym = l.split('\t')
                gene_id = gene_id.strip()
                gene_sym = gene_sym.strip()
                sym_to_ids[gene_sym].append(gene_id)
        # 收集训练基因
        train_ids = []
        train_genes = []
        all_genes_s = set(all_genes)
        not_found = []
        gene_to_indices = defaultdict(lambda: []) 
        for sym in test_genes:
            if sym in sym_to_ids:
                ids = sym_to_ids[sym]
                for idd in ids:
                    if idd in all_genes_s:
                        train_genes.append(sym)
                        train_ids.append(idd)
                        gene_to_indices[sym].append(gene_to_index[idd])
            else:
                not_found.append(sym)
    else:
        raise ValueError("无法确定基因集合。请确保输入数据集指定了HUGO基因符号或Entrez基因ID。")
    
    gene_to_indices = dict(gene_to_indices)
    print('在输入文件的{}个基因中,{}个在{}个训练集基因中找到。'.format(
        len(test_genes),
        len(train_ids),
        len(all_genes)
    ))
    
    # 写入日志文件
    if log_dir:
        with open(join(log_dir, 'genes_absent_from_training_set.tsv'), 'w') as f:
            f.write('\n'.join(sorted(not_found)))
        with open(join(log_dir, 'genes_found_in_training_set.tsv'), 'w') as f:
            f.write('\n'.join(sorted(train_genes)))
    return train_genes, gene_to_indices


if __name__ == "__main__":
    main()


