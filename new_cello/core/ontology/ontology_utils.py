"""
细胞本体论处理工具
"""

import logging
import os
import networkx as nx
from typing import Dict, List, Set, Optional, Any, Union

logger = logging.getLogger(__name__)

# 尝试导入本体论处理库
try:
    import pronto
    PRONTO_AVAILABLE = True
except ImportError:
    PRONTO_AVAILABLE = False
    logger.warning("pronto库未安装，本体论处理功能将受限。请使用 'pip install pronto' 安装。")

class CellOntology:
    """细胞本体论类"""
    
    def __init__(self, ontology_path: Optional[str] = None):
        """
        初始化细胞本体论
        
        Args:
            ontology_path: 本体论文件路径，如果为None则使用默认路径
        """
        self.ontology = None
        self.graph = None
        
        if ontology_path is not None:
            self.load(ontology_path)
    
    def load(self, ontology_path: str):
        """
        加载本体论文件
        
        Args:
            ontology_path: 本体论文件路径
        """
        if not PRONTO_AVAILABLE:
            raise ImportError("pronto库未安装，无法加载本体论")
        
        logger.info(f"加载本体论文件: {ontology_path}")
        
        # 加载本体论
        self.ontology = pronto.Ontology(ontology_path)
        
        # 构建有向图
        self.graph = nx.DiGraph()
        
        # 添加节点
        for term in self.ontology:
            self.graph.add_node(term.id, name=term.name)
        
        # 添加边
        for term in self.ontology:
            for parent in term.parents():
                self.graph.add_edge(parent.id, term.id)
        
        logger.info(f"本体论加载完成，包含 {len(self.graph.nodes)} 个节点和 {len(self.graph.edges)} 条边")
    
    def get_term(self, term_id: str) -> Optional[Any]:
        """
        获取指定ID的术语
        
        Args:
            term_id: 术语ID
            
        Returns:
            术语对象，如果不存在则返回None
        """
        if self.ontology is None:
            raise ValueError("本体论未加载")
        
        return self.ontology.get(term_id)
    
    def get_term_name(self, term_id: str) -> Optional[str]:
        """
        获取术语名称
        
        Args:
            term_id: 术语ID
            
        Returns:
            术语名称，如果不存在则返回None
        """
        if self.graph is None:
            raise ValueError("本体论未加载")
        
        if term_id in self.graph.nodes:
            return self.graph.nodes[term_id].get('name')
        return None
    
    def get_ancestors(self, term_id: str) -> Set[str]:
        """
        获取术语的所有祖先
        
        Args:
            term_id: 术语ID
            
        Returns:
            祖先术语ID集合
        """
        if self.graph is None:
            raise ValueError("本体论未加载")
        
        if term_id not in self.graph.nodes:
            return set()
        
        return set(nx.ancestors(self.graph, term_id))
    
    def get_descendants(self, term_id: str) -> Set[str]:
        """
        获取术语的所有后代
        
        Args:
            term_id: 术语ID
            
        Returns:
            后代术语ID集合
        """
        if self.graph is None:
            raise ValueError("本体论未加载")
        
        if term_id not in self.graph.nodes:
            return set()
        
        return set(nx.descendants(self.graph, term_id))
    
    def get_parents(self, term_id: str) -> Set[str]:
        """
        获取术语的直接父级
        
        Args:
            term_id: 术语ID
            
        Returns:
            父级术语ID集合
        """
        if self.graph is None:
            raise ValueError("本体论未加载")
        
        if term_id not in self.graph.nodes:
            return set()
        
        return set(self.graph.predecessors(term_id))
    
    def get_children(self, term_id: str) -> Set[str]:
        """
        获取术语的直接子级
        
        Args:
            term_id: 术语ID
            
        Returns:
            子级术语ID集合
        """
        if self.graph is None:
            raise ValueError("本体论未加载")
        
        if term_id not in self.graph.nodes:
            return set()
        
        return set(self.graph.successors(term_id))
    
    def get_path_to_root(self, term_id: str) -> List[str]:
        """
        获取从术语到根的路径
        
        Args:
            term_id: 术语ID
            
        Returns:
            术语ID列表，从根到指定术语
        """
        if self.graph is None:
            raise ValueError("本体论未加载")
        
        if term_id not in self.graph.nodes:
            return []
        
        # 找到所有根节点（没有入边的节点）
        roots = [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]
        
        # 对于每个根节点，找到到指定术语的最短路径
        paths = []
        for root in roots:
            try:
                path = nx.shortest_path(self.graph, root, term_id)
                paths.append(path)
            except nx.NetworkXNoPath:
                continue
        
        # 返回最短的路径
        if paths:
            return min(paths, key=len)
        return []
    
    def get_common_ancestor(self, term_ids: List[str]) -> Optional[str]:
        """
        获取多个术语的最近公共祖先
        
        Args:
            term_ids: 术语ID列表
            
        Returns:
            最近公共祖先的ID，如果不存在则返回None
        """
        if self.graph is None:
            raise ValueError("本体论未加载")
        
        if not term_ids:
            return None
        
        # 检查所有术语是否存在
        for term_id in term_ids:
            if term_id not in self.graph.nodes:
                return None
        
        # 获取第一个术语的所有祖先
        ancestors = self.get_ancestors(term_ids[0])
        ancestors.add(term_ids[0])  # 包括术语本身
        
        # 与其他术语的祖先取交集
        for term_id in term_ids[1:]:
            term_ancestors = self.get_ancestors(term_id)
            term_ancestors.add(term_id)  # 包括术语本身
            ancestors = ancestors.intersection(term_ancestors)
        
        if not ancestors:
            return None
        
        # 找到深度最大的公共祖先
        max_depth = -1
        deepest_ancestor = None
        
        for ancestor in ancestors:
            # 计算祖先的深度（到根的距离）
            path = self.get_path_to_root(ancestor)
            depth = len(path) - 1  # 减去根节点
            
            if depth > max_depth:
                max_depth = depth
                deepest_ancestor = ancestor
        
        return deepest_ancestor
    
    def visualize(self, term_ids: Optional[List[str]] = None, output_path: Optional[str] = None):
        """
        可视化本体论或指定术语的子图
        
        Args:
            term_ids: 术语ID列表，如果为None则可视化整个本体论
            output_path: 输出文件路径，如果为None则显示图形
        """
        if self.graph is None:
            raise ValueError("本体论未加载")
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib库未安装，无法可视化本体论")
            return
        
        # 创建子图
        if term_ids:
            # 包含指定术语及其祖先和后代
            nodes = set(term_ids)
            for term_id in term_ids:
                nodes.update(self.get_ancestors(term_id))
                nodes.update(self.get_descendants(term_id))
            
            subgraph = self.graph.subgraph(nodes)
        else:
            subgraph = self.graph
        
        # 绘制图形
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(subgraph, seed=42)
        
        # 绘制节点
        nx.draw_networkx_nodes(subgraph, pos, node_size=500, alpha=0.8)
        
        # 绘制边
        nx.draw_networkx_edges(subgraph, pos, arrows=True)
        
        # 绘制标签
        labels = {node: self.get_term_name(node) for node in subgraph.nodes}
        nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=8)
        
        plt.axis('off')
        plt.tight_layout()
        
        # 保存或显示图形
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"图形已保存到 {output_path}")
        else:
            plt.show()

# 全局本体论实例
_ontology = None

def load_ontology(ontology_path: str) -> CellOntology:
    """
    加载本体论
    
    Args:
        ontology_path: 本体论文件路径
        
    Returns:
        CellOntology实例
    """
    global _ontology
    _ontology = CellOntology(ontology_path)
    return _ontology

def get_ontology() -> Optional[CellOntology]:
    """
    获取全局本体论实例
    
    Returns:
        CellOntology实例，如果未加载则返回None
    """
    return _ontology

def get_term_ancestors(term_id: str) -> Set[str]:
    """
    获取术语的所有祖先
    
    Args:
        term_id: 术语ID
        
    Returns:
        祖先术语ID集合
    """
    if _ontology is None:
        raise ValueError("本体论未加载，请先调用load_ontology")
    
    return _ontology.get_ancestors(term_id)

def get_term_descendants(term_id: str) -> Set[str]:
    """
    获取术语的所有后代
    
    Args:
        term_id: 术语ID
        
    Returns:
        后代术语ID集合
    """
    if _ontology is None:
        raise ValueError("本体论未加载，请先调用load_ontology")
    
    return _ontology.get_descendants(term_id)

def get_term_parents(term_id: str) -> Set[str]:
    """
    获取术语的直接父级
    
    Args:
        term_id: 术语ID
        
    Returns:
        父级术语ID集合
    """
    if _ontology is None:
        raise ValueError("本体论未加载，请先调用load_ontology")
    
    return _ontology.get_parents(term_id)

def get_term_children(term_id: str) -> Set[str]:
    """
    获取术语的直接子级
    
    Args:
        term_id: 术语ID
        
    Returns:
        子级术语ID集合
    """
    if _ontology is None:
        raise ValueError("本体论未加载，请先调用load_ontology")
    
    return _ontology.get_children(term_id) 