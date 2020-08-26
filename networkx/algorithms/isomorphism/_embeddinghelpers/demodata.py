def random_paths(
        size=10, max_depth=10, common=0, prefix_depth1=0, prefix_depth2=0,
        sep='/', labels=26, seed=None):
    """
    Returns two randomly created paths (as in directory structures) for use in
    testing and benchmarking :func:`maximum_common_path_embedding`.

    Parameters
    ----------
    size : int
        The number of independant random paths

    max_depth : int
        Maximum depth for the independant random paths

    common : int
        The number of shared common paths

    prefix_depth1: int
        Depth of the random prefix attacheded to first common paths

    prefix_depth2: int
        Depth of the random prefix attacheded to second common paths

    labels: int or collection
        Number of or collection of tokens that can be used as node labels

    sep: str
        path separator

    seed:
        Random state or seed


    Examples
    --------
    >>> paths1, paths2 = random_paths(size=5, max_depth=3, common=6, prefix_depth1=3, prefix_depth2=3, seed=0, labels=2 ** 64)
    >>> from networkx.algorithms.isomorphism._embeddinghelpers import path_embedding
    >>> from networkx.algorithms.isomorphism._embeddinghelpers import tree_embedding
    >>> tree = path_embedding.paths_to_otree(paths1)
    >>> seq, open_to_close, toks = tree_embedding.tree_to_seq(tree, mode='chr')
    >>> seq, open_to_close, toks = tree_embedding.tree_to_seq(tree, mode='number')
    >>> seq, open_to_close, toks = tree_embedding.tree_to_seq(tree, mode='tuple')
    >>> # xdoctest: +REQUIRES(module:ubelt)
    >>> import ubelt as ub
    >>> print('paths1 = {}'.format(ub.repr2(paths1, nl=1)))
    >>> print('paths2 = {}'.format(ub.repr2(paths2, nl=1)))
    """
    from networkx.utils import create_py_random_state
    rng = create_py_random_state(seed)

    if isinstance(labels, int):
        def _convert_digit_base(digit, alphabet):
            """
            Parameters
            ----------
            digit : int
                number in base 10 to convert

            alphabet : list
                symbols of the conversion base
            """
            baselen = len(alphabet)
            x = digit
            if x == 0:
                return alphabet[0]
            sign = 1 if x > 0 else -1
            x *= sign
            digits = []
            while x:
                digits.append(alphabet[x % baselen])
                x //= baselen
            if sign < 0:
                digits.append('-')
            digits.reverse()
            newbase_str = ''.join(digits)
            return newbase_str

        alphabet = list(map(chr, range(ord('a'), ord('z'))))

        def random_label():
            digit = rng.randint(0, labels)
            label = _convert_digit_base(digit, alphabet)
            return label
    else:
        from functools import partial
        random_label = partial(rng.choice, labels)

    def random_path(rng, max_depth):
        depth = rng.randint(1, max_depth)
        parts = [str(random_label()) for _ in range(depth)]
        path = sep.join(parts)
        return path

    # These paths might be shared (but usually not)
    iid_paths1 = {random_path(rng, max_depth) for _ in range(size)}
    iid_paths2 = {random_path(rng, max_depth) for _ in range(size)}

    # These paths will be shared
    common_paths = {random_path(rng, max_depth) for _ in range(common)}

    if prefix_depth1 > 0:
        prefix1 = random_path(rng, prefix_depth1)
        common1 = {sep.join([prefix1, suff]) for suff in common_paths}
    else:
        common1 = common_paths

    if prefix_depth2 > 0:
        prefix2 = random_path(rng, prefix_depth2)
        common2 = {sep.join([prefix2, suff]) for suff in common_paths}
    else:
        common2 = common_paths

    paths1 = sorted(common1 | iid_paths1)
    paths2 = sorted(common2 | iid_paths2)

    return paths1, paths2


def random_ordered_tree(n, seed=None):
    """
    Creates a random ordered tree

    Parameters
    ----------
    n : int
        A positive integer representing the number of nodes in the tree.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    networkx.OrderedDiGraph

    Example
    --------
    >>> assert len(random_ordered_tree(n=1, seed=0).nodes) == 1
    >>> assert len(random_ordered_tree(n=2, seed=0).nodes) == 2
    >>> assert len(random_ordered_tree(n=3, seed=0).nodes) == 3
    """
    import networkx as nx
    tree = nx.dfs_tree(nx.random_tree(n, seed=seed))
    otree = nx.OrderedDiGraph()
    otree.add_nodes_from(tree.nodes)
    otree.add_edges_from(tree.edges)
    return otree


def simple_sequences(size=80, **kw):
    n = size
    paths1 = ['{}/{}'.format(i, j) for i in range(0, (n // 10) + 1) for j in range(0, n)]
    paths2 = ['q/r/sp/' + p for p in paths1]
    len(paths1)
    return paths1, paths2
