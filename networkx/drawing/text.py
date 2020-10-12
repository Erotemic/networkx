"""
Text-based visual representations of graphs
"""


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
    import networkx as nx
    printbuf = []
    if eager:
        if write is None:
            lazyprint = print
        else:
            lazyprint = write
    else:
        lazyprint = printbuf.append

    if len(graph.nodes) == 0:
        lazyprint("<empty graph>")
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
            islast_next = idx == 0
            stack.append((node, "", islast_next))

        while stack:
            node, indent, islast = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            if islast:
                this_prefix = indent + "└── "
                next_prefix = indent + "    "
            else:
                this_prefix = indent + "├── "
                next_prefix = indent + "│   "
            label = graph.nodes[node].get("label", node)
            lazyprint(this_prefix + str(label))

            children = [child for child in succ[node] if child not in seen]
            for idx, child in enumerate(children, start=1):
                islast_next = idx <= 1
                try_frame = (child, next_prefix, islast_next)
                stack.append(try_frame)

    if printbuf:
        return "\n".join(printbuf)
    else:
        return ""
