# 导入必要的系统模块
import sys
# 导入集合相关的数据结构
import collections
from collections import defaultdict, deque

# 调试模式开关
DEBUG = False

# 主函数：用于测试图的基本功能
def main():
    # 创建一个示例有向无环图
    source_to_targets = {
        'A': set(['B', 'C', 'D']),
        'B': set(['C', 'D', 'E']),
        'C': set(['E']),
        'D': set(['E']),
        'E': set()
    }
    # 构建图对象
    graph = DirectedAcyclicGraph(source_to_targets)
    # 执行传递闭包约简
    reduced_graph = transitive_reduction_on_dag(graph)
    # 执行拓扑排序并打印结果
    print(topological_sort(reduced_graph))


# 有向无环图（DAG）类定义
class DirectedAcyclicGraph:
    """
    实现一个有向图数据结构
    """
    def __init__(self, source_to_targets, target_to_sources=None):
        """
        初始化DAG
        参数：
            source_to_targets: 源节点到目标节点的映射字典
            target_to_sources: 目标节点到源节点的映射字典（可选）
        """
        # 存储源节点到目标节点的映射
        self.source_to_targets = source_to_targets
        # 如果没有提供反向映射，则构建它
        if not target_to_sources:
            self.target_to_sources = defaultdict(lambda: set())
            for source, targets in source_to_targets.items():
                for target in targets:
                    self.target_to_sources[target].add(source)
        else:
            self.target_to_sources = target_to_sources
            
        # 确保所有值都是集合类型
        self.target_to_sources = {
            target: set(sources)
            for target, sources in self.target_to_sources.items()
        }
        self.source_to_targets = {
            source: set(targets)
            for source, targets in self.source_to_targets.items()
        }
        
        # 确保所有节点都在两个映射中都存在
        for node in source_to_targets:
            if node not in self.target_to_sources:
                self.target_to_sources[node] = set()
        for node in self.target_to_sources:
            if node not in self.source_to_targets:
                self.source_to_targets[node] = set()

    # 添加边的方法
    def add_edge(self, source, target):
        """
        向图中添加一条边
        参数：
            source: 源节点
            target: 目标节点
        """
        if source not in self.source_to_targets:
            self.source_to_targets[source] = set()
        if target not in self.target_to_sources:
            self.target_to_sources[target] = set()
        self.source_to_targets[source].add(target)
        self.target_to_sources[target].add(source)

    # 获取后代节点的方法
    def descendent_nodes(self, node):
        """
        获取指定节点的所有后代节点
        """
        return self._downstream_nodes(
            node,
            self.source_to_targets
        )

    # 获取祖先节点的方法
    def ancestor_nodes(self, node):
        """
        获取指定节点的所有祖先节点
        """
        return self._downstream_nodes(
            node,
            self.target_to_sources
        )

    # 获取最具体节点的方法
    def most_specific_nodes(self, nodes):
        """
        从给定节点集合中找出最具体的节点（没有更具体后代的节点）
        """
        most_specific_nodes = set()
        # 将节点映射到其超节点
        node_to_supernodes = {}
        for node in nodes:
            node_to_supernodes[node] = self.ancestor_nodes(node)  #找到当前节点所有的祖先节点

        # 创建"比...更一般"的树形结构, 即当前节点和它的所有祖先节点之间的关系
        have_relations = set() 
        more_general_than = defaultdict(lambda: set())
        for node_a in node_to_supernodes.keys():
            for node_b, b_supernodes in node_to_supernodes.items():
                if node_a == node_b:
                    continue
                if node_a in b_supernodes:
                    more_general_than[node_a].add(node_b)
                    have_relations.update([node_a, node_b])
        more_general_than = dict(more_general_than)

        # 收集树的叶子节点, 排除当前节点的所有祖先节点, 剩下的就是当前节点的具体节点
        for subs in more_general_than.values():
            for s in subs:
                if not s in more_general_than.keys():
                    most_specific_nodes.add(s)

        # 添加没有关系的独立节点
        loner_nodes = set(nodes) - have_relations
        return most_specific_nodes | loner_nodes

    # 获取最一般节点的方法
    def most_general_nodes(self, nodes):
        """
        从给定节点集合中找出最一般的节点（没有更一般祖先的节点）
        """
        most_general_nodes = set()
        # 将节点映射到其子节点
        node_to_subnodes = {}
        for node in nodes:
            node_to_subnodes[node] = self.descendent_nodes(node)

        # 创建"比...更具体"的树形结构
        have_relations = set()
        more_specific_than = defaultdict(lambda: set())
        for node_a in node_to_subnodes.keys():
            for node_b, b_subnodes in node_to_subnodes.items():
                if node_a == node_b:
                    continue
                if node_a in b_subnodes:
                    more_specific_than[node_a].add(node_b)
                    have_relations.update([node_a, node_b])
        more_specific_than = dict(more_specific_than)

        # 收集树的叶子节点
        for sups in more_specific_than.values():
            for s in sups:
                if not s in more_specific_than.keys():
                    most_general_nodes.add(s)

        # 添加没有关系的独立节点
        loner_nodes = set(nodes) - have_relations
        return most_general_nodes | loner_nodes

    # 获取下游节点的内部方法
    def _downstream_nodes(self, node, orig_to_dests):
        """
        使用广度优先搜索获取下游节点
        """
        visited = set([node])   # 访问过的节点集合, 先把当前节点加入集合
        q = deque([node])      # 创建一个队列, 先把当前节点加入队列
        while len(q) > 0:       # 当队列不为空时, 继续遍历
            orig = q.popleft()  # 从队列中取出一个节点
            if orig not in orig_to_dests:  # 如果当前节点不在映射中, 则跳过
                continue
            for dest in orig_to_dests[orig]:
                if dest not in visited:
                    visited.add(dest)
                    q.append(dest)
        return visited

    # 获取所有节点的方法
    def get_all_nodes(self):
        """
        获取图中的所有节点
        """
        all_nodes = set(self.source_to_targets.keys())
        for target, sources in self.target_to_sources.items():
            all_nodes.update(sources)   # 添加一个可迭代对象用update
            all_nodes.add(target)   # 添加一个节点用add
        return all_nodes

    # 复制图的方法
    def copy(self):
        """
        创建图的副本
        """
        return DirectedAcyclicGraph(self.source_to_targets.copy())

    # 比较两个图是否相等的方法
    def __eq__(self, other):
        """
        判断两个图是否相等
        """
        if not isinstance(other, DirectedAcyclicGraph):
            return False
        self_all_nodes = frozenset(self.get_all_nodes())
        other_all_nodes = frozenset(other.get_all_nodes())
        if self_all_nodes != other_all_nodes:
            return False
        # TODO: 完成剩余的比较逻辑
    
# 无向图类定义
class UndirectedGraph:
    """
    实现一个通用的无向图，不包含边权重
    使用哈希表实现，适合稀疏图
    """
    def __init__(self,edges):
        # 初始化邻接表
        self.node_to_neighbors = defaultdict(lambda: set())
        # 添加所有边
        for edge in edges:
            node_a = edge[0]
            node_b = edge[1]
            self.node_to_neighbors[node_a].add(node_b)
            self.node_to_neighbors[node_b].add(node_a)

    # 获取所有节点的方法
    def get_all_nodes(self):
        """
        返回图中的所有节点
        """
        return set(self.node_to_neighbors.keys())

# 对DAG进行传递闭包约简的函数
def transitive_reduction_on_dag(dag):
    """
    计算DAG的传递闭包约简
    注意：此函数不适用于有环图
    """    
    # 需要移除的边集合
    remove_edges = set()

    # For each node u, for each child v of u, for each descendant
    # v' of v, if v' is a child of u, then remove (u, v')
    # 对每个节点u，对于u的每个子节点v，对于v的每个后代v'
    # 如果v'也是u的子节点，则移除边(u, v')
    for parent, children in dag.source_to_targets.items():
        for child in children:
            descendants = set(dag.descendent_nodes(child)) - set([child])
            for remove_target in descendants & children:
                remove_edges.add((parent, remove_target))
    
    # 创建约简后的图
    reduced_source_to_targets = dag.source_to_targets.copy()
    for edge in remove_edges:
        source = edge[0]
        target = edge[1]
        reduced_source_to_targets[source].remove(target)
    reduced_graph = DirectedAcyclicGraph(reduced_source_to_targets)
    return reduced_graph

# 拓扑排序函数
def topological_sort(dag):
    """
    对DAG进行拓扑排序
    """
    # 初始化：找出没有入边的节点,入度为0的节点
    removed_nodes = set([
        node 
        for node in dag.get_all_nodes()
        if len(dag.target_to_sources[node]) == 0
    ])
    sorted_nodes = sorted(removed_nodes)
    remaining_nodes = dag.get_all_nodes() - removed_nodes
    
    # 迭代直到处理完所有节点
    while removed_nodes < dag.get_all_nodes():
        next_removed = set()
        for node in remaining_nodes:
            incoming_nodes = set(dag.target_to_sources[node]) - removed_nodes
            if len(incoming_nodes) == 0:
                next_removed.add(node)
        removed_nodes.update(next_removed)
        sorted_nodes += sorted(next_removed)
        remaining_nodes = dag.get_all_nodes() - removed_nodes
    return sorted_nodes

# 图的道德化函数
def moralize(graph):
    """
    将有向图转换为无向图，并连接所有具有共同子节点的节点对
    """
    # 添加具有共同子节点的节点之间的边
    for node_a in graph.get_all_nodes():
        children_a = set(graph.source_to_targets[node_a])
        for node_b in graph.get_all_nodes():
            children_b = set(graph.source_to_targets[node_b])
            if len(children_a & children_b) > 0:
                add_edges.append((node_a, node_b))
    undir_edges = set(add_edges)
    # 转换为无向图
    for source, targets in graph.source_to_targets:
        for target in targets:
            undir_edges.add((source, target))
    return UndirectedGraph(edges)        

# 构建跨越指定节点的子图
def subgraph_spanning_nodes(graph, span_nodes):
    # 1. 找出最一般的节点（顶层节点）
    most_general_nodes = graph.most_general_nodes(span_nodes)

    # 2. 使用广度优先搜索构建子图
    q = deque(most_general_nodes)  # 创建队列存储待处理节点
    subgraph_source_to_targets = defaultdict(lambda: set())  # 存储子图的边

    # 3. 广度优先遍历
    while len(q) > 0:
        source = q.popleft()  # 获取当前要处理的节点
        subgraph_source_to_targets[source] = set()
        
        # 4. 检查当前节点的所有后代节点
        for target in graph.source_to_targets[source]:
            # 获取目标节点的所有后代节点
            target_descendants = graph._downstream_nodes(
                target,
                graph.source_to_targets
            )
            # 5. 如果目标节点的后代包含指定节点，则添加到子图
            if len(target_descendants.intersection(span_nodes)) > 0:
                subgraph_source_to_targets[source].add(target)
                q.append(target)
                
    # 6. 返回新的有向无环图
    return DirectedAcyclicGraph(subgraph_source_to_targets)

# 程序入口点
if __name__ == "__main__":
    main()
