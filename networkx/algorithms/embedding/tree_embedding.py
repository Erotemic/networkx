"""
Algorithm for computing tree embeddings

Issues
------
- [ ] Should we return which edges were contracted in each tree to create the
  embeddings? That seems useful (but maybe not equivalent to the embeddings
  themselves?) Note, I had an initial attempt to do this, but I haven't found
  an efficient way to modify the dynamic program yet.

- [ ] How to deal with cython + networkx? Do we need to fix that skbuild with
  pypy?

- [ ] The open_to_node problem:
        Note, we may be able to simply use the position of each opening token
        as a proxy for unique tokens. Pass in an ordered list of nodes, then
        just use their indexes.

"""
import networkx as nx
from collections import OrderedDict, defaultdict
from networkx.algorithms.string import balanced_sequence


def maximum_common_ordered_tree_embedding(
        tree1, tree2, node_affinity='auto', impl='auto', mode='auto'):
    """
    Finds the maximum common subtree-embedding between two ordered trees.

    A tree S is an embedded subtree of T if it can be obtained from T by a
    series of edge contractions.

    Note this produces a subtree embedding, which is not necessarilly a
    subgraph isomorphism (although a subgraph isomorphism is also an
    embedding.)

    The maximum common embedded subtree problem can be solved in in
    `O(n1 * n2 * min(d1, l1) * min(d2, l2))` time on ordered trees with n1 and
    n2 nodes, of depth d1 and d2 and with l1 and l2 leaves, respectively.

    Implements algorithm described in [1]_, which introduces the problem as
    follows:

    "An important generalization of tree and subtree isomorphism, known as
    minor containment, is the problem of determining whether a tree is
    isomorphic to an embedded subtree of another tree, where an embedded
    subtree of a tree is obtained by contracting some of the edges in the tree.
    A further generalization of minor containment on trees, known as maximum
    common embedded subtree, is the problem of finding or determining the size
    of a largest common embedded subtree of two trees. The latter also
    generalizes the maximum common subtree isomorphism problem, in which a
    common subtree of largest size is contained as a subtree, not only
    embedded, in the two trees."

    Parameters
    ----------
    tree1, tree2 : nx.OrderedDiGraph
        Trees to find the maximum embedding between

    node_affinity : None | str | callable
        Function for to determine if two nodes can be matched. The return is
        interpreted as a weight that is used to break ties. If None then any
        node can match any other node and only the topology is important.
        The default is "eq", which is the same as ``operator.eq``.

    impl : str
        Determines the backend implementation. Defaults to "auto".

    mode : str
        Determines the backend representation. Defaults to "auto".

    References
    ----------
    .. [1] Lozano, Antoni, and Gabriel Valiente.
        "On the maximum common embedded subtree problem for ordered trees."
        String Algorithmics (2004): 155-170.
        https://pdfs.semanticscholar.org/0b6e/061af02353f7d9b887f9a378be70be64d165.pdf

    Returns
    -------
    Tuple[nx.OrderedDiGraph, nx.OrderedDiGraph] :
        The maximum value common embedding for each tree with respect to the
        chosen ``node_affinity`` function. The topology of both graphs will
        always be the same, the only difference is that the node labels in the
        first and second embeddings will correspond to ``tree1`` and `tree2``
        respectively. When ``node_affinity='eq'`` then embeddings should be
        identical.

    Example
    -------
    >>> from networkx.algorithms.embedding.tree_embedding import *  # NOQA
    >>> from networkx.generators.random_graphs import random_ordered_tree
    >>> tree1 = random_ordered_tree(7, seed=3257073545741117277206611, directed=True)
    >>> tree2 = random_ordered_tree(7, seed=123568587133124688238689717, directed=True)
    >>> print(forest_str(tree1))
    └── 1
        ├── 6
        │   ├── 4
        │   └── 3
        └── 0
            └── 5
                └── 2
    >>> print(forest_str(tree2))
    └── 4
        └── 1
            ├── 2
            │   ├── 6
            │   └── 0
            └── 3
                └── 5
    >>> embedding1, embedding2 = maximum_common_ordered_tree_embedding(tree1, tree2 )
    >>> print(forest_str(embedding1))
    └── 1
        ├── 6
        └── 5
    >>> print(forest_str(embedding2))
    └── 1
        ├── 6
        └── 5
    """
    if not (isinstance(tree1, nx.OrderedDiGraph) and nx.is_forest(tree1)):
        raise nx.NetworkXNotImplemented('only implemented for directed ordered trees')
    if not (isinstance(tree1, nx.OrderedDiGraph) and nx.is_forest(tree2)):
        raise nx.NetworkXNotImplemented('only implemented for directed ordered trees')

    # Convert the trees to balanced sequences
    sequence1, open_to_close, node_to_open = tree_to_seq(
        tree1, open_to_close=None, node_to_open=None, mode=mode)
    sequence2, open_to_close, node_to_open = tree_to_seq(
        tree2, open_to_close, node_to_open, mode=mode)
    seq1 = sequence1
    seq2 = sequence2

    # FIXME: I think this may cause bugs in two cases, which may or may not be
    # possible, but I need to look into it and provide a fix or justification
    # as to why these cases wont be hit:
    # (1) when the two trees share nodes that have different open tokens
    # (2) when the mapping between nodes to opening tokens is not unique.
    #     I'm not sure if this second case can happen when we are converting
    #     from a tree to a sequence, there are certainly sequences where the
    #     same opening token might share multiple tree nodes.
    open_to_node = invert_dict(node_to_open)

    # Solve the longest common balanced sequence problem
    best, value = balanced_sequence.longest_common_balanced_sequence(
        seq1, seq2, open_to_close, open_to_node=open_to_node,
        node_affinity=node_affinity, impl=impl)
    subseq1, subseq2 = best

    # Convert the subsequence back into a tree
    embedding1 = seq_to_tree(subseq1, open_to_close, open_to_node)
    embedding2 = seq_to_tree(subseq2, open_to_close, open_to_node)
    return embedding1, embedding2


def tree_to_seq(tree, open_to_close=None, node_to_open=None, mode='auto',
                container='auto'):
    r"""
    Converts an ordered tree to a balanced sequence for use in algorithm
    reductions.

    Parameters
    ----------
    open_to_close : Dict | None
        Dictionary of opening to closing tokens to be updated for problems
        where multiple trees are converted to sequences.

    open_to_node : Dict | None
        Dictionary of opening tokens to nodes to be updated for problems where
        multiple trees are converted to sequences.

    mode : str
        Determines the item type of the sequence.  Can be 'number', 'chr'.
        Default is 'auto', which will choose 'chr' unless the graph is too big.

    container : str
        Determines the container type. Can be "list" or "str". If "auto"
        tries to choose the best.

    Example
    -------
    >>> from networkx.algorithms.embedding.tree_embedding import *  # NOQA
    >>> tree = nx.path_graph(3, nx.OrderedDiGraph)
    >>> print(forest_str(tree))
    └── 0
        └── 1
            └── 2
    >>> sequence, open_to_close, node_to_open = tree_to_seq(tree, mode='number')
    >>> print('sequence = {!r}'.format(sequence))
    sequence = (1, 2, 3, -3, -2, -1)

    >>> tree = nx.balanced_tree(2, 2, nx.OrderedDiGraph)
    >>> print(forest_str(tree))
    └── 0
        ├── 2
        │   ├── 6
        │   └── 5
        └── 1
            ├── 4
            └── 3
    >>> sequence, open_to_close, node_to_open = tree_to_seq(tree, mode='number')
    >>> print('sequence = {!r}'.format(sequence))
    sequence = (1, 2, 3, -3, 4, -4, -2, 5, 6, -6, 7, -7, -5, -1)

    >>> from networkx.generators.random_graphs import random_ordered_tree
    >>> tree = random_ordered_tree(2, seed=1, directed=True)
    >>> sequence, open_to_close, node_to_open = tree_to_seq(tree, mode='chr')
    >>> print('sequence = {!r}'.format(sequence))
    sequence = '\x00\x02\x03\x01'
    >>> sequence, open_to_close, node_to_open = tree_to_seq(tree, mode='number')
    >>> print('sequence = {!r}'.format(sequence))
    sequence = (1, 2, -2, -1)
    """
    # mapping between opening and closing tokens
    sources = [n for n in tree.nodes if tree.in_degree[n] == 0]
    sequence = []

    if mode == 'auto':
        mode = 'chr' if len(tree) < 1112064 // 2 else 'number'

    if mode == 'paren':
        all_labels = {n['label'] for n in list(tree.nodes.values())}
        assert all(x == 1 for x in map(len, all_labels))

    if container == 'auto':
        container = 'str' if mode == 'chr' else 'list'

    seq_is_str = (container == 'str')

    if open_to_close is None:
        open_to_close = {}
    if node_to_open is None:
        node_to_open = {}

    for source in sources:
        for u, v, etype in nx.dfs_labeled_edges(tree, source=source):
            if etype == 'forward':
                # u has been visited by v has not
                if v not in node_to_open:
                    if mode == 'number':
                        open_tok = len(node_to_open) + 1
                        close_tok = -open_tok
                    elif mode == 'chr':
                        if seq_is_str:
                            # utf8 can only encode this many chars
                            assert len(node_to_open) < (1112064 // 2)
                            open_tok = chr(len(node_to_open) * 2)
                            close_tok = chr(len(node_to_open) * 2 + 1)
                        else:
                            # note ussing the accent mark wont work in string
                            # mode even though the close tok renders as a
                            # single character.
                            open_tok = str(v)
                            close_tok = str(v) + u'\u0301'
                    elif mode == 'paren':
                        open_tok = tree.nodes[v]['label']
                        if open_tok == '{':
                            close_tok = '}'
                        elif open_tok == '[':
                            close_tok = ']'
                        elif open_tok == '(':
                            close_tok = ')'
                        else:
                            raise KeyError(open_tok)
                    else:
                        raise KeyError(mode)
                    node_to_open[v] = open_tok
                    open_to_close[open_tok] = close_tok
                open_tok = node_to_open[v]
                sequence.append(open_tok)
            elif etype == 'reverse':
                # Both u and v are visited and the edge is in the tree
                close_tok = open_to_close[node_to_open[v]]
                sequence.append(close_tok)
            else:
                raise KeyError(etype)
    sequence = tuple(sequence)

    if seq_is_str:
        sequence = ''.join(sequence)
    return sequence, open_to_close, node_to_open


def seq_to_tree(subseq, open_to_close, open_to_node):
    """
    Converts a balanced sequence to an ordered tree

    Parameters
    ----------
    subseq : Tuple | str
        a balanced sequence of hashable items as a string or tuple

    open_to_close : Dict
        a dictionary that maps opening tokens to closing tokens in the balanced
        sequence problem.

    open_to_node : Dict
        a dictionary that maps a sequence token to a node corresponding to an
        original problem (e.g. a tree node). Must be unique. If unspecified new
        nodes will be generated and the opening sequence token will be used as
        a node label.

    Example
    --------
    >>> open_to_close = {'{': '}', '(': ')', '[': ']'}
    >>> open_to_node = None
    >>> subseq = '({[[]]})[[][]]{{}}'
    >>> subtree = seq_to_tree(subseq, open_to_close, open_to_node)
    >>> print(forest_str(subtree))
    ├── {
    │   └── {
    ├── [
    │   ├── [
    │   └── [
    └── (
        └── {
            └── [
                └── [
    """
    nextnode = 0  # only used if open_to_node is not specified
    subtree = nx.OrderedDiGraph()
    stack = []
    for token in subseq:
        if token in open_to_close:
            if open_to_node is None:
                node = nextnode
                nextnode += 1
            else:
                node = open_to_node[token]
            if stack:
                parent_tok, parent_node = stack[-1]
                subtree.add_edge(parent_node, node)
            else:
                subtree.add_node(node)
            if open_to_node is None:
                subtree.nodes[node]['label'] = token
            stack.append((token, node))
        else:
            if not stack:
                raise balanced_sequence.UnbalancedException
            prev_open, prev_node = stack.pop()
            want_close = open_to_close[prev_open]
            if token != want_close:
                raise balanced_sequence.UnbalancedException
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
    >>> from networkx.algorithms.embedding.tree_embedding import *  # NOQA
    >>> dict_ = {'a': 1, 'b': 2}
    >>> inverted = invert_dict(dict_)
    >>> assert inverted == {1: 'a', 2: 'b'}

    Example
    -------
    >>> from networkx.algorithms.embedding.tree_embedding import *  # NOQA
    >>> dict_ = OrderedDict([(2, 'a'), (1, 'b'), (0, 'c'), (None, 'd')])
    >>> inverted = invert_dict(dict_)
    >>> assert list(inverted.keys())[0] == 'a'

    Example
    -------
    >>> from networkx.algorithms.embedding.tree_embedding import *  # NOQA
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


def forest_str(graph, eager=False, write=None, use_labels=True, sources=None):
    """
    Creates a nice utf8 representation of a directed forest

    Parameters
    ----------
    graph : nx.DiGraph | nx.Graph
        graph to represent (must be a tree, forest, or the empty graph)

    eager : bool
        if True, the text will be written directly to stdout or the write
        function if specified

    write : callable
        function to use to write to, if None new lines are appended to
        a list and returned

    use_labels : bool
        if True will use the "label" attribute of a node to display if it
        exists otherwise it will use the node value itself.

    sources : List
        Only relevant for undirected graphs, specifies which nodes to list
        first.

    Returns
    -------
    str :
        utf8 representation of the tree / forest

    TODO
    ----
    - [ ] Is this useful? If so, should this move to networkx.drawing.text

    Example
    -------
    >>> import networkx as nx
    >>> graph = nx.balanced_tree(r=2, h=3, create_using=nx.DiGraph)
    >>> print(forest_str(graph))
    └── 0
        ├── 2
        │   ├── 6
        │   │   ├── 14
        │   │   └── 13
        │   └── 5
        │       ├── 12
        │       └── 11
        └── 1
            ├── 4
            │   ├── 10
            │   └── 9
            └── 3
                ├── 8
                └── 7
    >>> graph = nx.balanced_tree(r=1, h=2, create_using=nx.Graph)
    >>> print(forest_str(graph))
    ├── 1
    │   ├── 2
    │   └── 0
    """
    printbuf = []
    if eager:
        if write is None:
            lazyprint = print
        else:
            lazyprint = write
    else:
        lazyprint = printbuf.append

    if len(graph.nodes) == 0:
        lazyprint('<empty graph>')
    else:
        assert nx.is_forest(graph)

        if graph.is_directed():
            sources = [n for n in graph.nodes if graph.in_degree[n] == 0]
            succ = graph.succ
        else:
            # use arbitrary sources for undirected trees
            sources = sorted(graph.nodes, key=lambda n: graph.degree[n])
            succ = graph.adj

        seen = set()
        stack = []
        for idx, node in enumerate(sources):
            islast_next = (idx == 0)
            stack.append((node, '', islast_next))

        while stack:
            node, indent, islast = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            if islast:
                this_prefix = indent + '└── '
                next_prefix = indent + '    '
            else:
                this_prefix = indent + '├── '
                next_prefix = indent + '│   '
            label = graph.nodes[node].get('label', node)
            lazyprint(this_prefix + str(label))

            children = [child for child in succ[node] if child not in seen]
            for idx, child in enumerate(children, start=1):
                islast_next = (idx <= 1)
                try_frame = (child, next_prefix, islast_next)
                stack.append(try_frame)

    if printbuf:
        return '\n'.join(printbuf)
    else:
        return ''
