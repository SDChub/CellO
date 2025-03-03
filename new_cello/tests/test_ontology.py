"""
本体论处理模块测试
"""

import os
import pytest
import tempfile
from typing import Dict, List, Set, Optional

# 尝试导入本体论处理模块
try:
    import networkx as nx
    import pronto
    from new_cello.core.ontology import (
        load_ontology,
        get_term_ancestors,
        get_term_descendants,
        get_term_parents,
        get_term_children
    )
    from new_cello.core.ontology.ontology_utils import CellOntology, PRONTO_AVAILABLE
    
    ONTOLOGY_AVAILABLE = PRONTO_AVAILABLE
except ImportError:
    ONTOLOGY_AVAILABLE = False

# 如果本体论处理库不可用，则跳过所有测试
pytestmark = pytest.mark.skipif(
    not ONTOLOGY_AVAILABLE,
    reason="本体论处理库未安装，无法测试本体论模块"
)

class TestOntology:
    """本体论处理测试类"""
    
    @pytest.fixture
    def sample_ontology(self) -> str:
        """
        创建样本本体论文件用于测试
        
        Returns:
            str: 本体论文件路径
        """
        # 创建临时OBO文件
        with tempfile.NamedTemporaryFile(suffix='.obo', delete=False) as f:
            # 写入简单的OBO格式本体论
            f.write(b"""
format-version: 1.2
ontology: test-cell-ontology

[Term]
id: CL:0000000
name: cell

[Term]
id: CL:0000001
name: primary cell
is_a: CL:0000000 ! cell

[Term]
id: CL:0000002
name: animal cell
is_a: CL:0000000 ! cell

[Term]
id: CL:0000003
name: native cell
is_a: CL:0000001 ! primary cell

[Term]
id: CL:0000004
name: neuron
is_a: CL:0000002 ! animal cell

[Term]
id: CL:0000005
name: T cell
is_a: CL:0000002 ! animal cell

[Term]
id: CL:0000006
name: motor neuron
is_a: CL:0000004 ! neuron

[Term]
id: CL:0000007
name: sensory neuron
is_a: CL:0000004 ! neuron

[Term]
id: CL:0000008
name: CD4-positive T cell
is_a: CL:0000005 ! T cell

[Term]
id: CL:0000009
name: CD8-positive T cell
is_a: CL:0000005 ! T cell
            """)
            ontology_path = f.name
        
        yield ontology_path
        
        # 清理临时文件
        os.unlink(ontology_path)
    
    def test_load_ontology(self, sample_ontology):
        """测试加载本体论"""
        # 加载本体论
        ontology = load_ontology(sample_ontology)
        
        # 检查是否成功加载
        assert ontology is not None
        assert ontology.graph is not None
        assert len(ontology.graph.nodes) == 10  # 10个术语
        assert len(ontology.graph.edges) == 9   # 9条边
    
    def test_get_term_ancestors(self, sample_ontology):
        """测试获取术语祖先"""
        # 加载本体论
        load_ontology(sample_ontology)
        
        # 获取术语祖先
        ancestors = get_term_ancestors("CL:0000006")  # 运动神经元
        
        # 检查祖先
        assert "CL:0000000" in ancestors  # 细胞
        assert "CL:0000002" in ancestors  # 动物细胞
        assert "CL:0000004" in ancestors  # 神经元
        assert len(ancestors) == 3
    
    def test_get_term_descendants(self, sample_ontology):
        """测试获取术语后代"""
        # 加载本体论
        load_ontology(sample_ontology)
        
        # 获取术语后代
        descendants = get_term_descendants("CL:0000004")  # 神经元
        
        # 检查后代
        assert "CL:0000006" in descendants  # 运动神经元
        assert "CL:0000007" in descendants  # 感觉神经元
        assert len(descendants) == 2
    
    def test_get_term_parents(self, sample_ontology):
        """测试获取术语父级"""
        # 加载本体论
        load_ontology(sample_ontology)
        
        # 获取术语父级
        parents = get_term_parents("CL:0000004")  # 神经元
        
        # 检查父级
        assert "CL:0000002" in parents  # 动物细胞
        assert len(parents) == 1
    
    def test_get_term_children(self, sample_ontology):
        """测试获取术语子级"""
        # 加载本体论
        load_ontology(sample_ontology)
        
        # 获取术语子级
        children = get_term_children("CL:0000005")  # T细胞
        
        # 检查子级
        assert "CL:0000008" in children  # CD4阳性T细胞
        assert "CL:0000009" in children  # CD8阳性T细胞
        assert len(children) == 2
    
    def test_cell_ontology_class(self, sample_ontology):
        """测试CellOntology类"""
        # 创建本体论实例
        ontology = CellOntology(sample_ontology)
        
        # 测试获取术语
        term = ontology.get_term("CL:0000000")
        assert term is not None
        assert term.name == "cell"
        
        # 测试获取术语名称
        name = ontology.get_term_name("CL:0000001")
        assert name == "primary cell"
        
        # 测试获取到根的路径
        path = ontology.get_path_to_root("CL:0000006")  # 运动神经元
        assert path == ["CL:0000000", "CL:0000002", "CL:0000004", "CL:0000006"]
        
        # 测试获取公共祖先
        common_ancestor = ontology.get_common_ancestor(["CL:0000006", "CL:0000007"])
        assert common_ancestor == "CL:0000004"  # 神经元
        
        # 测试获取不同分支的公共祖先
        common_ancestor = ontology.get_common_ancestor(["CL:0000006", "CL:0000008"])
        assert common_ancestor == "CL:0000002"  # 动物细胞 