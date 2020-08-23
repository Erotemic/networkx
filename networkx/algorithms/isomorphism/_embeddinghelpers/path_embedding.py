import networkx as nx
from .tree_embedding import maximum_common_ordered_tree_embedding


def maximum_common_path_embedding(paths1, paths2, sep='/', impl='iter-prehash2', mode='number'):
    """
    Finds the maximum path embedding common between two sets of paths

    Parameters
    ----------
    paths1, paths2: List[str]
        a list of paths

    sep: str
        path separator character

    impl: str
        backend runtime to use

    mode: str
        backend representation to use

    Examples
    --------
    >>> n = 80
    >>> paths1 = ['{}/{}'.format(i, j) for i in range(0, (n // 10) + 1) for j in range(0, n)]
    >>> paths2 = ['q/r/sp/' + p for p in paths1]
    >>> len(paths1)

    >>> rng = None
    >>> import kwarray
    >>> rng = kwarray.ensure_rng(rng)
    >>> def random_paths(rng, max_depth=10):
    >>>     depth = rng.randint(1, max_depth)
    >>>     parts = list(map(chr, rng.randint(ord('a'), ord('z'), size=depth)))
    >>>     path = '/'.join(parts)
    >>>     return path
    >>> n = 200
    >>> paths1 = sorted({random_paths(rng) for _ in range(n)})
    >>> paths2 = sorted({random_paths(rng) for _ in range(n)})
    >>> paths1 = paths1 + ['a/' + k for k in paths2[0:n // 3]]
    """
    # the longest common balanced sequence problem
    def _affinity(tok1, tok2):
        score = 0
        for t1, t2 in zip(tok1[::-1], tok2[::-1]):
            if t1 == t2:
                score += 1
            else:
                break
        return score
    node_affinity = _affinity

    tree1 = paths_to_otree(paths1, sep=sep)
    tree2 = paths_to_otree(paths2, sep=sep)

    subtree1, subtree2 = maximum_common_ordered_tree_embedding(
            tree1, tree2, node_affinity=node_affinity, impl=impl, mode=mode)

    subpaths1 = [sep.join(node) for node in subtree1.nodes if subtree1.out_degree[node] == 0]
    subpaths2 = [sep.join(node) for node in subtree2.nodes if subtree2.out_degree[node] == 0]
    return subpaths1, subpaths2


def paths_to_otree(paths, sep='/'):
    """

    Parameters
    ----------
    paths: List[str]
        a list of paths

    Returns
    -------
    nx.OrderedDiGraph

    """
    tree = nx.OrderedDiGraph()
    for path in sorted(paths):
        parts = tuple(path.split(sep))
        node_path = []
        for i in range(1, len(parts) + 1):
            node = parts[0:i]
            tree.add_node(node)
            tree.nodes[node]['label'] = node[-1]
            node_path.append(node)
        for u, v in zip(node_path[:-1], node_path[1:]):
            tree.add_edge(u, v)
    return tree
