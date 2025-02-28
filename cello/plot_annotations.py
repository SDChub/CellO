"""
创建用于查看CellO输出的可视化图表。

作者: Matthew Bernstein mbernstein@morgridge.org 
"""

# 导入图形可视化相关的库
import pygraphviz
from pygraphviz import AGraph
import matplotlib as mpl

# 导入CellO自定义模块
from . import cello
from .graph_lib import graph
from . import ontology_utils as ou


def probabilities_on_graph(
        cell_or_clust,
        results_df, 
        rsrc_loc,
        clust=True,
        root_label=None, 
        p_thresh=0.0
    ):
    """
    在细胞本体图上可视化预测概率
    
    参数:
    cell_or_clust: 要绘制概率的细胞或聚类的名称
    results_df: 存储CellO输出概率的DataFrame，行对应细胞，列对应细胞类型
    rsrc_loc: CellO资源目录的位置
    clust: 默认True，表示cell_or_clust是聚类ID；False表示是细胞ID
    root_label: 细胞类型名称或ID。只绘制以该细胞类型为根的子图
    p_thresh: 概率阈值。只绘制概率超过该阈值的细胞类型子图
    """

    # 获取标签图(细胞类型层次结构)
    label_graph = cello._retrieve_label_graph(rsrc_loc) 

    # 判断DataFrame的列是使用本体术语ID还是术语名称
    is_term_ids = 'CL:' in results_df.columns[0]

    # 将cell_or_clust赋值给cell变量
    cell = cell_or_clust

    # 创建跨越所有概率超过阈值的术语的子图(一个细胞是T, 还是CD14,还是更具体的)
    # 将所有概率超过阈值的细胞类型添加到span_labels集合中
    span_labels = set([
        label
        for label, prob in zip(results_df.columns, results_df.loc[cell])
        if prob > p_thresh
    ])
    
    # 如果指定了根标签，则只保留该根下的节点
    if root_label:
        if not is_term_ids:
            root_id = ou.get_term_id(root_label)
        else:
            root_id = root_label
        span_labels &= label_graph._downstream_nodes(
            root_label, 
            label_graph.source_to_targets
        )
    
    # 构建跨越选定节点的子图
    label_graph = graph.subgraph_spanning_nodes(
        label_graph,
        span_labels
    )

    # 创建标签到概率的映射字典
    label_to_prob = {
        label: prob
        for label, prob in zip(results_df.columns, results_df.loc[cell])
        if label in label_graph.source_to_targets
    }
    
    # 根据是否使用术语ID创建不同格式的标签名称
    if is_term_ids:
        label_to_name = {
            label: '{}\n{:.2f}'.format(
                ou.get_term_name(label), 
                prob
            )
            for label, prob in label_to_prob.items()
        }
    else:
        label_to_name = {
            label: '{}\n{:.2f}'.format(
                label,
                prob
            )
            for label, prob in label_to_prob.items()
        }

    # 渲染并返回图形
    g = _render_graph(
        label_graph.source_to_targets,
        label_to_name,
        "Probabilities for {}".format(cell),
        label_to_prob
    )
    return g


def _render_graph(
        source_to_targets,
        node_to_label,
        metric_name,
        node_to_value
    ):
    """
    渲染图形的内部函数
    
    参数:
    source_to_targets: 源节点到目标节点的映射
    node_to_label: 节点到标签的映射
    metric_name: 度量名称
    node_to_value: 节点到值的映射
    """
    
    # 创建有向图实例
    g = AGraph(directed=True)

    # 收集图中的所有节点
    all_nodes = set(source_to_targets.keys())
    for targets in source_to_targets.values():
        all_nodes.update(targets)

    # 构建graphviz图
    for node in all_nodes:
        # 跳过没有对应值的节点
        if node not in node_to_value:
            continue

        # 使用viridis颜色映射创建节点颜色
        cmap = mpl.cm.get_cmap('viridis')
        rgba = cmap(node_to_value[node])
        color_value = mpl.colors.rgb2hex(rgba)

        # 根据节点值选择字体颜色
        if node_to_value[node] > 0.5:
            font_color = 'black'
        else:
            font_color = 'white'

        # 添加节点及其样式
        g.add_node(
            node_to_label[node],
            label=node_to_label[node],
            fontname='arial',
            style='filled', 
            fontcolor=font_color,
            fillcolor=color_value
        )

    # 添加边
    for source, targets in source_to_targets.items():
        for target in targets:
            if source in node_to_value and target in node_to_value:
                g.add_edge(node_to_label[source], node_to_label[target])
    return g


