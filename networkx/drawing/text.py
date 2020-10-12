"""
Text-based visual representations of graphs
"""


def forest_str(graph, use_labels=True, sources=None, write=None):
    """
    Creates a nice utf8 representation of a directed forest

    Parameters
    ----------
    graph : nx.DiGraph | nx.Graph
        Graph to represent (must be a tree, forest, or the empty graph)

    use_labels : bool
        If True will use the "label" attribute of a node to display if it
        exists otherwise it will use the node value itself. Defaults to True.

    sources : List
        Mainly relevant for undirected forests, specifies which nodes to list
        first. If unspecified the root nodes of each tree will be used for
        directed forests; for undirected forests this defaults to the nodes
        with the smallest degree.

    write : callable
        Function to use to write to, if None new lines are appended to
        a list and returned. If set to the `print` function, lines will
        be written to stdout as they are generated. If specified,
        this function will return None. Defaults to None.

    Returns
    -------
    str | None :
        utf8 representation of the tree / forest

    Example
    -------
    >>> import networkx as nx
    >>> graph = nx.balanced_tree(r=2, h=3, create_using=nx.DiGraph)
    >>> print(nx.forest_str(graph))
    ╙── 0
        ├─➤ 2
        │   ├─➤ 6
        │   │   ├─➤ 14
        │   │   └─➤ 13
        │   └─➤ 5
        │       ├─➤ 12
        │       └─➤ 11
        └─➤ 1
            ├─➤ 4
            │   ├─➤ 10
            │   └─➤ 9
            └─➤ 3
                ├─➤ 8
                └─➤ 7

    >>> graph = nx.balanced_tree(r=1, h=2, create_using=nx.Graph)
    >>> print(nx.forest_str(graph))
    ╟── 1
    ║   ├── 2
    ║   └── 0
    """
    import networkx as nx

    printbuf = []
    if write is None:
        _write = printbuf.append
    else:
        _write = write

    if len(graph.nodes) == 0:
        _write("<empty graph>")
    else:
        if not nx.is_forest(graph):
            raise nx.NetworkXNotImplemented(
                "input must be a forest or the empty graph")

        is_directed = graph.is_directed()
        succ = graph.succ if is_directed else graph.adj

        if sources is None:
            if is_directed:
                # use real source nodes for directed trees
                sources = [n for n in graph.nodes if graph.in_degree[n] == 0]
            else:
                # use arbitrary sources for undirected trees
                sources = sorted(graph.nodes, key=lambda n: graph.degree[n])

        seen = set()
        stack = []
        for idx, node in enumerate(sources):
            islast_next = idx == 0
            stack.append((node, "", islast_next))

        while stack:
            node, indent, islast = stack.pop()
            if node in seen:
                continue
            seen.add(node)

            # https://en.wikipedia.org/wiki/Box-drawing_character
            # https://stackoverflow.com/questions/2701192/triangle-arrow
            # # should we use arrows for directed cases?
            # Candidate utf8 characters:
            # ╼ → ► ⟶ ➙ ➝
            # shortlist: ➙ ➤
            if 0:
                candidates = (
                    '→'
                    'ᐅ ⇢ ⇀ →'
                    '╼ → ► ⟶'
                    '➙ ➝ ➝ ➞'
                    '➟ ➠ ➡ ➢'
                    '➣ ➤ ➥ ➦'
                    '➧').split(' ')
                for c in candidates:
                    if len(c) == 1:
                        print('─' + c)

            if not indent:
                # Top level items (i.e. trees in the forest) get different
                # glyphs to indicate they are not actually connected
                if islast:
                    this_prefix = indent + "╙── "
                    next_prefix = indent + "    "
                else:
                    this_prefix = indent + "╟── "
                    next_prefix = indent + "║   "

            else:
                if is_directed:
                    if islast:
                        this_prefix = indent + "└─➤ "
                        next_prefix = indent + "    "
                    else:
                        this_prefix = indent + "├─➤ "
                        next_prefix = indent + "│   "
                else:
                    if islast:
                        this_prefix = indent + "└── "
                        next_prefix = indent + "    "
                    else:
                        this_prefix = indent + "├── "
                        next_prefix = indent + "│   "

            if use_labels:
                label = graph.nodes[node].get("label", node)
            else:
                label = node

            _write(this_prefix + str(label))

            children = [child for child in succ[node] if child not in seen]
            for idx, child in enumerate(children, start=1):
                islast_next = idx <= 1
                try_frame = (child, next_prefix, islast_next)
                stack.append(try_frame)

    if write is None:
        # Only return a string if the custom write function was not specified
        return "\n".join(printbuf)
