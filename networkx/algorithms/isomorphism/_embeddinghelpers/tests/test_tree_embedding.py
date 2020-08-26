from networkx.algorithms.isomorphism._embeddinghelpers.tree_embedding import (
    maximum_common_ordered_tree_embedding, forest_str)

from networkx.algorithms.isomorphism._embeddinghelpers.demodata import (
    random_ordered_tree
)
import networkx as nx
import pytest
from networkx.utils import create_py_random_state


def test_null_common_embedding():
    """
    The empty graph is not a tree and should raise an error
    """
    empty = nx.OrderedDiGraph()
    non_empty = random_ordered_tree(n=1)

    with pytest.raises(nx.NetworkXPointlessConcept):
        maximum_common_ordered_tree_embedding(empty, empty)

    with pytest.raises(nx.NetworkXPointlessConcept):
        maximum_common_ordered_tree_embedding(empty, non_empty)

    with pytest.raises(nx.NetworkXPointlessConcept):
        maximum_common_ordered_tree_embedding(non_empty, empty)


def test_self_common_embedding():
    """
    The common embedding of a tree with itself should always be itself
    """
    rng = create_py_random_state(seed=85652972257)
    for n in range(1, 10):
        tree = random_ordered_tree(n=n, seed=rng)
        embedding1, embedding2 = maximum_common_ordered_tree_embedding(tree, tree)
        assert tree.edges == embedding1.edges


def test_common_tree_embedding_small():
    tree1 = nx.OrderedDiGraph([(0, 1)])
    tree2 = nx.OrderedDiGraph([(0, 1), (1, 2)])
    print(forest_str(tree1))
    print(forest_str(tree2))

    embedding1, embedding2 = maximum_common_ordered_tree_embedding(tree1, tree2)
    print(forest_str(embedding1))
    print(forest_str(embedding2))


def test_common_tree_embedding_small2():
    tree1 = nx.OrderedDiGraph([(0, 1), (2, 3), (4, 5), (5, 6)])
    tree2 = nx.OrderedDiGraph([(0, 1), (1, 2), (0, 3)])
    print(forest_str(tree1))
    print(forest_str(tree2))

    embedding1, embedding2 = maximum_common_ordered_tree_embedding(tree1, tree2, node_affinity=None)
    print(forest_str(embedding1))
    print(forest_str(embedding2))
