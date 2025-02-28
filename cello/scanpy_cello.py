"""
使用CellO对人类细胞进行细胞本体分类。

这里我们按照Scanpy外部API的约定实现CellO的运行函数
(https://scanpy.readthedocs.io/en/stable/external/)。

作者: Matthew Bernstein
邮箱: mbernstein@morgridge.org
"""

# 导入必要的库
from anndata import AnnData  # 用于存储单细胞数据
import dill  # 用于序列化Python对象
from collections import defaultdict  # 用于处理默认字典
import pandas as pd  # 用于数据处理
import io  # 用于处理字节流
from matplotlib import pyplot as plt  # 用于绘图
import matplotlib.image as mpimg  # 用于图像处理

# 导入CellO自定义模块
from .plot_annotations import probabilities_on_graph
from . import ontology_utils as ou
from . import cello as ce

def cello(
        adata: AnnData,  # AnnData对象，包含基因表达数据
        clust_key: str = 'leiden',  # 聚类结果的键名
        rsrc_loc: str = '.',  # CellO资源文件位置
        algo: str = 'IR',  # 分类算法选择
        out_prefix: str = None,  # 输出文件前缀
        model_file: str = None,  # 预训练模型文件路径
        log_dir: str = None,  # 日志目录
        term_ids: bool = False,  # 是否使用术语ID
        remove_anatomical_subterms: list = None  # 需要过滤的解剖学术语
    ):
    """
    CellO [Bernstein21]_
    
    使用细胞本体对人类细胞进行层次分类。
    
    更多信息、教程和错误报告，请访问CellO的GitHub页面：
    https://github.com/deweylab/CellO
    
    参数说明：
    ----------
    adata: 注释数据矩阵，要求表达数据已经用log(TPM+1)标准化
    clust_key: 聚类注释在adata.obs中的键名
    rsrc_loc: CellO资源文件路径
    algo: 层次分类算法，'IR'表示保序回归，'CLR'表示级联逻辑回归
    out_prefix: 训练模型的输出前缀
    model_file: 预训练模型文件路径
    log_dir: 日志目录
    term_ids: 是否使用本体术语ID
    remove_anatomical_subterms: 需要过滤的解剖学术语列表
    """
    
    # 尝试导入cello包
    try:
        import cello as ce
    except ImportError:
        raise ImportError(
            '需要安装cello包：请在终端运行 `pip install --user cello`'
        )
    
    # 加载模型
    if model_file:
        print('从{}加载模型...'.format(model_file))
        with open(model_file, 'rb') as f:
            mod=dill.load(f)
    else:
        # 加载或训练模型
        mod = ce._retrieve_pretrained_model(adata, algo, rsrc_loc)
        if mod is None:
            # 如果没有预训练模型，则训练新模型
            mod = ce.train_model(
                adata, 
                rsrc_loc, 
                algo=algo, 
                log_dir=log_dir
            )
            # 保存训练好的模型
            if out_prefix:
                out_model_f = '{}.model.dill'.format(out_prefix)
                print('将训练好的模型写入{}'.format(out_model_f))
                with open(out_model_f, 'wb') as f:
                    dill.dump(mod, f)
            else:
                print("未提供'out_prefix'参数。训练好的模型不会被保存。")
    
    # 运行分类
    results_df, finalized_binary_results_df, ms_results_df = ce.predict(
        adata,
        mod,
        algo=algo,
        clust_key=clust_key,
        rsrc_loc=rsrc_loc,
        log_dir=log_dir,
        remove_anatomical_subterms=remove_anatomical_subterms
    )
        
    # 将结果合并到AnnData对象中
    if term_ids:
        # 使用术语ID作为列名
        column_to_term_id = {
            '{} (probability)'.format(c): c
            for c in results_df.columns
        }
        results_df.columns = [
            '{} (probability)'.format(c)
            for c in results_df.columns
        ]
        finalized_binary_results_df.columns = [
            '{} (binary)'.format(c)
            for c in finalized_binary_results_df.columns
        ]
    else:
        # 使用术语名称作为列名
        column_to_term_id = {
            '{} (probability)'.format(ou.cell_ontology().id_to_term[c].name): c
            for c in results_df.columns
        }
        results_df.columns = [
            '{} (probability)'.format(
                ou.cell_ontology().id_to_term[c].name
            )
            for c in results_df.columns
        ]
        finalized_binary_results_df.columns = [
            '{} (binary)'.format(
                ou.cell_ontology().id_to_term[c].name
            )
            for c in finalized_binary_results_df.columns
        ]
        ms_results_df['most_specific_cell_type'] = [
            ou.cell_ontology().id_to_term[c].name
            for c in ms_results_df['most_specific_cell_type']
        ]

    # 删除已存在的相关列
    drop_cols = [
        col
        for col in adata.obs.columns
        if '(probability)' in str(col)
        or '(binary)' in str(col)
        or col == 'Most specific cell type'
    ]
    adata.obs = adata.obs.drop(drop_cols, axis=1)

    # 将二值结果转换为分类变量
    finalized_binary_results_df = finalized_binary_results_df.astype(bool).astype(str).astype('category')

    # 将结果添加到adata对象中
    adata.obs = adata.obs.join(results_df).join(finalized_binary_results_df)
    adata.uns['CellO_column_mappings'] = column_to_term_id
    if term_ids:
        adata.obs['Most specific cell type'] = [
            ou.cell_ontology().id_to_term[c].name
            for c in ms_results_df['most_specific_cell_type']
        ]
    else:
        adata.obs['Most specific cell type'] = ms_results_df['most_specific_cell_type']
   

def normalize_and_cluster(
        adata: AnnData, 
        n_pca_components: int = 50, 
        n_neighbors: int = 15,
        n_top_genes: int = 10000,
        cluster_res: float = 2.0
    ):
    """
    对原始UMI计数矩阵进行标准化和聚类
    
    参数说明：
    adata: 包含原始UMI计数的AnnData对象
    n_pca_components: PCA降维的组分数
    n_neighbors: 计算最近邻图的邻居数
    n_top_genes: 用于聚类的高变基因数量
    cluster_res: Leiden聚类的分辨率参数
    """
    
    # 检查是否安装了scanpy
    try:
        import scanpy as sc
    except ImportError:
        sys.exit("'normalize_and_cluster'函数需要安装scanpy包。请运行'pip install scanpy'进行安装")
    
    # 数据预处理和聚类
    sc.pp.normalize_total(adata, target_sum=1e6)  # CPM标准化
    sc.pp.log1p(adata)  # 对数转换
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)  # 选择高变基因
    sc.pp.pca(adata, n_comps=n_pca_components, use_highly_variable=True)  # PCA降维
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)  # 构建KNN图
    sc.tl.leiden(adata, resolution=cluster_res)  # Leiden聚类


def cello_probs(adata, cell_or_clust, rsrc_loc, p_thresh, width=10, height=10, clust_key=None, dpi=300, return_graph=False):
    """
    可视化CellO的预测概率
    
    参数说明：
    adata: AnnData对象
    cell_or_clust: 要可视化的细胞或聚类ID
    rsrc_loc: 资源文件位置
    p_thresh: 概率阈值
    width, height: 图形尺寸
    clust_key: 聚类键名
    dpi: 图像分辨率
    return_graph: 是否返回图形对象
    """
    
    # 提取预测结果
    results_df = adata.obs[[col for col in adata.uns['CellO_column_mappings']]]
    results_df.columns = [
        adata.uns['CellO_column_mappings'][c] for c in results_df.columns
    ]

    # 基于聚类ID进行绘图
    if clust_key:
        try:
            assert cell_or_clust in set(adata.obs[clust_key])
        except AssertionError:
            raise KeyError(f"绘图错误。在`adata.obs`列中未找到聚类{clust_key}。")

        # 为每个聚类提取结果
        clust_to_indices = defaultdict(lambda: [])
        for index, clust in zip(adata.obs.index, adata.obs[clust_key]):
            clust_to_indices[clust].append(index)

        # 构建聚类结果DataFrame
        clusts = sorted(clust_to_indices.keys())
        results_df = pd.DataFrame(
            [
                results_df.loc[clust_to_indices[clust][0]]
                for clust in clusts
            ],
            index=clusts,
            columns=results_df.columns
        )

    # 生成概率图
    g = probabilities_on_graph(
        cell_or_clust,
        results_df,
        rsrc_loc,
        p_thresh=p_thresh
    )

    # 将图形转换为图像
    f = io.BytesIO(g.draw(format='png', prog='dot', args=f'-Gdpi={dpi}'))

    # 创建matplotlib图形
    fig, ax = plt.subplots(figsize=(width, height))
    im = mpimg.imread(f)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(im)
    plt.show()

    # 返回结果
    if return_graph:
        return fig, ax, g
    else:
        return fig, ax


def write_to_tsv(adata, filename):
    """
    将CellO的输出写入TSV文件
    
    参数：
    adata: AnnData对象
    filename: 输出文件名
    """
    # 选择要保存的列
    keep_cols = [
        col 
        for col in adata.obs.columns
        if '(probability)' in col
        or '(binary)' in col
        or 'Most specific cell type' in col
    ]
    # 提取数据并保存
    df = adata.obs[keep_cols]
    df.to_csv(filename, sep='\t')
