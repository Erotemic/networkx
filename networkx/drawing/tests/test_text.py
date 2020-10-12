import networkx as nx
from textwrap import dedent


def test_directed_tree_str():
    # Create a directed forest with labels
    graph = nx.balanced_tree(r=2, h=2, create_using=nx.DiGraph)
    for node in graph.nodes:
        graph.nodes[node]["label"] = "node_" + chr(ord("a") + node)

    node_target = dedent(
        """
        ╙── 0
            ├─➤ 2
            │   ├─➤ 6
            │   └─➤ 5
            └─➤ 1
                ├─➤ 4
                └─➤ 3
        """
    ).strip()

    label_target = dedent(
        """
        ╙── node_a
            ├─➤ node_c
            │   ├─➤ node_g
            │   └─➤ node_f
            └─➤ node_b
                ├─➤ node_e
                └─➤ node_d
        """
    ).strip()

    # Basic node case
    ret = nx.forest_str(graph, use_labels=False)
    print(ret)
    assert ret == node_target

    # Basic label case
    ret = nx.forest_str(graph, use_labels=True)
    print(ret)
    assert ret == label_target

    # Custom write function case
    lines = []
    ret = nx.forest_str(graph, write=lines.append, use_labels=False)
    assert ret is None
    assert lines == node_target.split("\n")

    # Smoke test to ensure passing the print function works. To properly test
    # this case we would need to capture stdout. (for potential reference
    # implementation see :class:`ubelt.util_stream.CaptureStdout`)
    ret = nx.forest_str(graph, write=print)
    assert ret is None


def test_empty_graph():
    assert nx.forest_str(nx.DiGraph()) == "<empty graph>"
    assert nx.forest_str(nx.Graph()) == "<empty graph>"


def test_directed_multi_tree_forest():
    tree1 = nx.balanced_tree(r=2, h=2, create_using=nx.DiGraph)
    tree2 = nx.balanced_tree(r=2, h=2, create_using=nx.DiGraph)
    tree2 = nx.relabel_nodes(tree2, {n: n + len(tree1) for n in tree2.nodes})
    forest = nx.union(tree1, tree2)
    ret = nx.forest_str(forest, sources=[0, 7])
    print(ret)

    target = dedent(
        """
        ╟── 7
        ╎   ├─➤ 9
        ╎   │   ├─➤ 13
        ╎   │   └─➤ 12
        ╎   └─➤ 8
        ╎       ├─➤ 11
        ╎       └─➤ 10
        ╙── 0
            ├─➤ 2
            │   ├─➤ 6
            │   └─➤ 5
            └─➤ 1
                ├─➤ 4
                └─➤ 3
        """
    ).strip()
    assert ret == target

    tree3 = nx.balanced_tree(r=2, h=2, create_using=nx.DiGraph)
    tree3 = nx.relabel_nodes(tree3, {n: n + len(forest) for n in tree3.nodes})
    forest = nx.union(forest, tree3)
    ret = nx.forest_str(forest, sources=[0, 7, 14])
    print(ret)

    target = dedent(
        """
        ╟── 14
        ╎   ├─➤ 16
        ╎   │   ├─➤ 20
        ╎   │   └─➤ 19
        ╎   └─➤ 15
        ╎       ├─➤ 18
        ╎       └─➤ 17
        ╟── 7
        ╎   ├─➤ 9
        ╎   │   ├─➤ 13
        ╎   │   └─➤ 12
        ╎   └─➤ 8
        ╎       ├─➤ 11
        ╎       └─➤ 10
        ╙── 0
            ├─➤ 2
            │   ├─➤ 6
            │   └─➤ 5
            └─➤ 1
                ├─➤ 4
                └─➤ 3
        """
    ).strip()
    assert ret == target


def test_undirected_multi_tree_forest():
    tree1 = nx.balanced_tree(r=2, h=2, create_using=nx.Graph)
    tree2 = nx.balanced_tree(r=2, h=2, create_using=nx.Graph)
    tree2 = nx.relabel_nodes(tree2, {n: n + len(tree1) for n in tree2.nodes})
    forest = nx.union(tree1, tree2)
    ret = nx.forest_str(forest, sources=[0, 7])
    print(ret)

    target = dedent(
        """
        ╟── 7
        ╎   ├── 9
        ╎   │   ├── 13
        ╎   │   └── 12
        ╎   └── 8
        ╎       ├── 11
        ╎       └── 10
        ╙── 0
            ├── 2
            │   ├── 6
            │   └── 5
            └── 1
                ├── 4
                └── 3
        """
    ).strip()
    assert ret == target


def test_undirected_tree_str():
    # Create a directed forest with labels
    graph = nx.balanced_tree(r=2, h=2, create_using=nx.Graph)

    # arbitrary starting point
    nx.forest_str(graph)

    node_target0 = dedent(
        """
        ╙── 0
            ├── 2
            │   ├── 6
            │   └── 5
            └── 1
                ├── 4
                └── 3
        """
    ).strip()

    # defined starting point
    ret = nx.forest_str(graph, sources=[0])
    print(ret)
    assert ret == node_target0

    # defined starting point
    node_target2 = dedent(
        """
        ╙── 2
            ├── 6
            ├── 5
            └── 0
                └── 1
                    ├── 4
                    └── 3
        """
    ).strip()
    ret = nx.forest_str(graph, sources=[2])
    print(ret)
    assert ret == node_target2


def test_forest_str_errors():
    import pytest

    ugraph = nx.generators.complete_graph(3, create_using=nx.Graph)

    with pytest.raises(nx.NetworkXNotImplemented):
        nx.forest_str(ugraph)

    dgraph = nx.generators.complete_graph(3, create_using=nx.DiGraph)

    with pytest.raises(nx.NetworkXNotImplemented):
        nx.forest_str(dgraph)
