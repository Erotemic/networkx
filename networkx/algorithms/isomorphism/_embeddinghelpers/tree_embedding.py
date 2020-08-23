import networkx as nx
from collections import OrderedDict, defaultdict
from .balanced_sequence import longest_common_balanced_sequence


def maximum_common_ordered_tree_embedding(
        tree1, tree2, node_affinity='auto', impl='iter-alt2', mode='number'):
    """

    Example
    -------
    >>> from netharn.initializers._nx_extensions import *  # NOQA
    >>> from netharn.initializers._nx_extensions import _lcs, _print_forest
    >>> def random_ordered_tree(n, seed=None):
    >>>     tree = nx.dfs_tree(nx.random_tree(n, seed=seed))
    >>>     otree = nx.OrderedDiGraph()
    >>>     otree.add_edges_from(tree.edges)
    >>>     return otree
    >>> tree1 = random_ordered_tree(10, seed=1)
    >>> tree2 = random_ordered_tree(10, seed=2)
    >>> print('tree1')
    >>> _print_forest(tree1)
    >>> print('tree2')
    >>> _print_forest(tree2)

    >>> embedding1, embedding2 = maximum_common_ordered_tree_embedding(tree1, tree2 )
    >>> print('embedding1')
    >>> _print_forest(embedding1)
    >>> print('embedding2')
    >>> _print_forest(embedding2)
    """
    if not (isinstance(tree1, nx.OrderedDiGraph) and nx.is_forest(tree1)):
        raise nx.NetworkXNotImplemented('only implemented for directed ordered trees')
    if not (isinstance(tree1, nx.OrderedDiGraph) and nx.is_forest(tree2)):
        raise nx.NetworkXNotImplemented('only implemented for directed ordered trees')

    # Convert the trees to balanced sequences
    sequence1, open_to_close, toks = tree_to_seq(tree1, open_to_close=None, toks=None, mode=mode)
    sequence2, open_to_close, toks = tree_to_seq(tree2, open_to_close, toks, mode=mode)
    seq1 = sequence1
    seq2 = sequence2

    open_to_tok = invert_dict(toks)

    # Solve the longest common balanced sequence problem
    best, value = longest_common_balanced_sequence(
        seq1, seq2, open_to_close, open_to_tok=open_to_tok, node_affinity=node_affinity, impl=impl)
    subseq1, subseq2 = best

    # Convert the subsequence back into a tree
    embedding1 = seq_to_tree(subseq1, open_to_close, toks)
    embedding2 = seq_to_tree(subseq2, open_to_close, toks)
    return embedding1, embedding2


def tree_to_seq(tree, open_to_close=None, toks=None, mode='tuple', strhack=None):
    """
    Converts an ordered tree to a balanced sequence

    Example
    --------
    >>> tree = random_ordered_tree(1000)
    >>> sequence, open_to_close, toks = tree_to_seq(tree, mode='tuple')
    >>> sequence, open_to_close, toks = tree_to_seq(tree, mode='chr')
    """
    # from collections import namedtuple
    # Token = namedtuple('Token', ['action', 'value'])
    # mapping between opening and closing tokens
    sources = [n for n in tree.nodes if tree.in_degree[n] == 0]
    sequence = []

    if strhack is None:
        if mode == 'chr':
            strhack = True

    if open_to_close is None:
        open_to_close = {}
    if toks is None:
        toks = {}

    if strhack:
        if mode == 'label':
            all_labels = {n['label'] for n in list(tree.nodes.values())}
            assert all(x == 1 for x in map(len, all_labels))

    for source in sources:
        for u, v, etype in nx.dfs_labeled_edges(tree, source=source):
            if etype == 'forward':
                # u has been visited by v has not
                if v not in toks:
                    if mode == 'tuple':
                        # TODO: token encoding scheme where subdirectories
                        # are matchable via a custom operation.
                        # open_tok = '<{}>'.format(v)
                        # close_tok = '</{}>'.format(v)
                        # open_tok = Token('open', v)
                        # close_tok = Token('close', v)
                        open_tok = ('open', v)
                        close_tok = ('close', v)
                    elif mode == 'number':
                        open_tok = len(toks) + 1
                        close_tok = -open_tok
                    elif mode == 'paren':
                        open_tok = '{}('.format(v)
                        close_tok = '){}'.format(v)
                    elif mode == 'chr':
                        if not strhack:
                            open_tok = str(v)
                            close_tok = str(v) + u'\u0301'
                        else:
                            # utf8 can only encode this many chars
                            assert len(toks) < (1112064 // 2)
                            open_tok = chr(len(toks) * 2)
                            close_tok = chr(len(toks) * 2 + 1)
                    elif mode == 'label':
                        open_tok = tree.nodes[v]['label']
                        assert strhack
                        if open_tok == '{':
                            close_tok = '}'
                        if open_tok == '[':
                            close_tok = ']'
                        if open_tok == '(':
                            close_tok = ')'
                    toks[v] = open_tok
                    open_to_close[open_tok] = close_tok
                open_tok = toks[v]
                sequence.append(open_tok)
            elif etype == 'reverse':
                # Both u and v are visited and the edge is in the tree
                close_tok = open_to_close[toks[v]]
                sequence.append(close_tok)
            else:
                raise KeyError(etype)
    sequence = tuple(sequence)
    if strhack:
        sequence = ''.join(sequence)
    return sequence, open_to_close, toks


def seq_to_tree(subseq, open_to_close, toks):
    """
    Converts a balanced sequence to an ordered tree

    Example
    --------
    >>> tree = random_ordered_tree(1000)
    >>> sequence, open_to_close, toks = tree_to_seq(tree, mode='tuple')
    >>> sequence, open_to_close, toks = tree_to_seq(tree, mode='chr')
    """
    open_to_tok = invert_dict(toks)
    subtree = nx.OrderedDiGraph()
    stack = []
    for token in subseq:
        if token in open_to_close:
            node = open_to_tok[token]
            if stack:
                parent = open_to_tok[stack[-1]]
                subtree.add_edge(parent, node)
            else:
                subtree.add_node(node)
            stack.append(token)
        else:
            if not stack:
                raise Exception
            prev_open = stack.pop()
            want_close = open_to_close[prev_open]
            if token != want_close:
                raise Exception
    return subtree


def invert_dict(dict_, unique_vals=True):
    """
    Swaps the keys and values in a dictionary.

    Parameters
    ----------
    dict_ (Dict[A, B]): dictionary to invert

    unique_vals (bool, default=True): if False, the values of the new
        dictionary are sets of the original keys.

    Returns
    -------
    Dict[B, A] | Dict[B, Set[A]]:
        the inverted dictionary

    Notes
    -----
    The must values be hashable.

    If the original dictionary contains duplicate values, then only one of
    the corresponding keys will be returned and the others will be
    discarded.  This can be prevented by setting ``unique_vals=False``,
    causing the inverted keys to be returned in a set.

    Example
    -------
    >>> dict_ = {'a': 1, 'b': 2}
    >>> inverted = invert_dict(dict_)
    >>> assert inverted == {1: 'a', 2: 'b'}

    Example
    -------
    >>> dict_ = ub.odict([(2, 'a'), (1, 'b'), (0, 'c'), (None, 'd')])
    >>> inverted = invert_dict(dict_)
    >>> assert list(inverted.keys())[0] == 'a'

    Example
    -------
    >>> dict_ = {'a': 1, 'b': 0, 'c': 0, 'd': 0, 'f': 2}
    >>> inverted = ub.invert_dict(dict_, unique_vals=False)
    >>> assert inverted == {0: {'b', 'c', 'd'}, 1: {'a'}, 2: {'f'}}
    """
    if unique_vals:
        if isinstance(dict_, OrderedDict):
            inverted = OrderedDict((val, key) for key, val in dict_.items())
        else:
            inverted = {val: key for key, val in dict_.items()}
    else:
        # Handle non-unique keys using groups
        inverted = defaultdict(set)
        for key, value in dict_.items():
            inverted[value].add(key)
        inverted = dict(inverted)
    return inverted


def _print_forest(graph):
    """
    Nice ascii representation of a forest

    Ignore:
        graph = nx.balanced_tree(r=2, h=3, create_using=nx.DiGraph)
        _print_forest(graph)

        graph = CategoryTree.demo('coco').graph
        _print_forest(graph)
    """
    if len(graph.nodes) == 0:
        print('--')
        return
    assert nx.is_forest(graph)

    def _recurse(node, indent='', islast=False):
        if islast:
            this_prefix = indent + '└── '
            next_prefix = indent + '    '
        else:
            this_prefix = indent + '├── '
            next_prefix = indent + '│   '
        label = graph.nodes[node].get('label', node)
        print(this_prefix + str(label))
        graph.succ[node]
        children = graph.succ[node]
        for idx, child in enumerate(children, start=1):
            islast_next = (idx == len(children))
            _recurse(child, indent=next_prefix, islast=islast_next)

    sources = [n for n in graph.nodes if graph.in_degree[n] == 0]
    for idx, node in enumerate(sources, start=1):
        islast_next = (idx == len(sources))
        _recurse(node, indent='', islast=islast_next)
