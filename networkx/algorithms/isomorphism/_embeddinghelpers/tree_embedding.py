import networkx as nx
from collections import OrderedDict, defaultdict
from .balanced_sequence import longest_common_balanced_sequence


def maximum_common_ordered_tree_embedding(
        tree1, tree2, node_affinity='auto', impl='iter-alt2', mode='number'):
    """
    Core networkx API

    Example
    -------
    >>> from networkx.algorithms.isomorphism._embeddinghelpers.tree_embedding import *  # NOQA
    >>> from networkx.algorithms.isomorphism._embeddinghelpers.demodata import random_ordered_tree  # NOQA
    >>> tree1 = random_ordered_tree(7, seed=355707353457411172772606611)
    >>> tree2 = random_ordered_tree(7, seed=1235685871331524688238689717)
    >>> print('tree1')
    >>> forest_str(tree1, eager=1)
    >>> print('tree2')
    >>> forest_str(tree2, eager=1)
    >>> embedding1, embedding2 = maximum_common_ordered_tree_embedding(tree1, tree2 )
    >>> print('embedding1')
    >>> forest_str(embedding1, eager=1)
    >>> print('embedding2')
    >>> forest_str(embedding2, eager=1)
    tree1
    └── 0
        └── 5
            └── 1
                ├── 6
                │   └── 3
                │       └── 4
                └── 2
    tree2
    └── 0
        ├── 3
        │   ├── 6
        │   │   └── 1
        │   │       └── 5
        │   └── 4
        └── 2
    embedding1
    └── 0
        ├── 3
        │   └── 4
        └── 2
    embedding2
    └── 0
        ├── 3
        │   └── 4
        └── 2
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
    -------
    >>> from networkx.algorithms.isomorphism._embeddinghelpers.tree_embedding import *  # NOQA
    >>> from networkx.algorithms.isomorphism._embeddinghelpers.demodata import random_ordered_tree  # NOQA
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
    >>> from networkx.algorithms.isomorphism._embeddinghelpers.tree_embedding import *  # NOQA
    >>> from networkx.algorithms.isomorphism._embeddinghelpers.demodata import random_ordered_tree
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
    >>> from networkx.algorithms.isomorphism._embeddinghelpers.tree_embedding import *  # NOQA
    >>> dict_ = {'a': 1, 'b': 2}
    >>> inverted = invert_dict(dict_)
    >>> assert inverted == {1: 'a', 2: 'b'}

    Example
    -------
    >>> from networkx.algorithms.isomorphism._embeddinghelpers.tree_embedding import *  # NOQA
    >>> dict_ = OrderedDict([(2, 'a'), (1, 'b'), (0, 'c'), (None, 'd')])
    >>> inverted = invert_dict(dict_)
    >>> assert list(inverted.keys())[0] == 'a'

    Example
    -------
    >>> from networkx.algorithms.isomorphism._embeddinghelpers.tree_embedding import *  # NOQA
    >>> dict_ = {'a': 1, 'b': 0, 'c': 0, 'd': 0, 'f': 2}
    >>> inverted = invert_dict(dict_, unique_vals=False)
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


def forest_str(graph, impl='iter', eager=0, write=None):
    """
    Nice utf8 representation of a forest

    Notes
    -----
    The iterative and recursive versions seem to be roughly as fast, but the
    iterative one does not have the issue of running into the call stack and
    causing a RecursionError (use params r=1, h=2 ** 14 to reproduce).

    CommandLine:
        xdoctest -m /home/joncrall/code/networkx/networkx/algorithms/isomorphism/_embeddinghelpers/tree_embedding.py forest_str --bench

    Example
    -------
    >>> from networkx.algorithms.isomorphism._embeddinghelpers.tree_embedding import *  # NOQA
    >>> from networkx.algorithms.isomorphism._embeddinghelpers.tree_embedding import forest_str
    >>> import networkx as nx
    >>> graph = nx.balanced_tree(r=2, h=3, create_using=nx.DiGraph)
    >>> print(forest_str(graph, impl='recurse'))

    Benchmark
    ---------
    >>> # xdoctest: +REQUIRES(--bench)
    >>> from networkx.algorithms.isomorphism._embeddinghelpers.tree_embedding import forest_str
    >>> import networkx as nx
    >>> # TODO: enumerate test cases
    >>> graph = nx.balanced_tree(r=1, h=int(2 ** 14), create_using=nx.DiGraph)  # causes RecursionError
    >>> graph = nx.balanced_tree(r=2, h=14, create_using=nx.DiGraph)
    >>> if len(graph.nodes) < 1000:
    >>>     forest_str(graph, eager=1, write=print)
    >>> else:
    >>>     print('graph = {!r}, {}'.format(graph, len(graph.nodes)))
    >>> import timerit
    >>> ti = timerit.Timerit(1, bestof=1, verbose=3)
    >>> ti.reset('iter-lazy').call(forest_str, graph, impl='iter', eager=0)
    >>> ti.reset('recurse-lazy').call(forest_str, graph, impl='recurse', eager=0)
    >>> # xdoctest: +REQUIRES(module:ubelt)
    >>> import ubelt as ub
    >>> print('ti.measures = {}'.format(ub.repr2(ti.measures['min'], nl=1, align=':', precision=6)))
    """
    if len(graph.nodes) == 0:
        print('--')
        return
    assert nx.is_forest(graph)

    printbuf = []
    if eager:
        if write is None:
            lazyprint = print
        else:
            lazyprint = write
    else:
        lazyprint = printbuf.append

    if impl == 'recurse':
        def _recurse(node, indent='', islast=False):
            if islast:
                this_prefix = indent + '└── '
                next_prefix = indent + '    '
            else:
                this_prefix = indent + '├── '
                next_prefix = indent + '│   '
            label = graph.nodes[node].get('label', node)
            lazyprint(this_prefix + str(label))
            graph.succ[node]
            children = graph.succ[node]
            for idx, child in enumerate(children, start=1):
                islast_next = (idx == len(children))
                _recurse(child, indent=next_prefix, islast=islast_next)

        sources = [n for n in graph.nodes if graph.in_degree[n] == 0]
        for idx, node in enumerate(sources, start=1):
            islast_next = (idx == len(sources))
            _recurse(node, indent='', islast=islast_next)

    elif impl == 'iter':
        sources = [n for n in graph.nodes if graph.in_degree[n] == 0]

        stack = []
        for idx, node in enumerate(sources):
            # islast_next = (idx == len(sources))
            islast_next = (idx <= 1)
            stack.append((node, '', islast_next))

        while stack:
            node, indent, islast = stack.pop()
            if islast:
                this_prefix = indent + '└── '
                next_prefix = indent + '    '
            else:
                this_prefix = indent + '├── '
                next_prefix = indent + '│   '
            label = graph.nodes[node].get('label', node)

            lazyprint(this_prefix + str(label))
            graph.succ[node]
            children = graph.succ[node]
            for idx, child in enumerate(children, start=1):
                # islast_next = (idx == len(children))
                islast_next = (idx <= 1)
                try_frame = (child, next_prefix, islast_next)
                stack.append(try_frame)

    else:
        raise KeyError(impl)

    if printbuf:
        return '\n'.join(printbuf)
    else:
        return ''
