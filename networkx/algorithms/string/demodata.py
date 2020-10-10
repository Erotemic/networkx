"""
Random data generator for the balanced sequence embedding problem.
"""


def random_balanced_sequence(n, seed=None, mode='chr', open_to_close=None):
    r"""
    Creates a random balanced sequence for testing / benchmarks

    Parameters
    ----------
    n : int
        A positive integer representing the number of nodes in the tree.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    open_to_close : dict | None
        if specified, updates existing open_to_close with tokens from this
        sequence.

    mode: str
        the type of sequence returned (see :func:`tree_to_seq` for details)

    Returns
    -------
    Tuple[(str | List), Dict[str, str]]
        The first item is the sequence itself
        the second item is the open_to_close mappings.

    Example
    -------
    >>> # Demo the various sequence encodings that we might use
    >>> seq, open_to_close = random_balanced_sequence(2, seed=1, mode='tuple')
    >>> print('seq = {!r}'.format(seq))
    >>> seq, open_to_close = random_balanced_sequence(4, seed=1, mode='chr')
    >>> print('seq = {!r}'.format(seq))
    >>> seq, open_to_close = random_balanced_sequence(4, seed=1, mode='number')
    >>> print('seq = {!r}'.format(seq))
    >>> seq, open_to_close = random_balanced_sequence(4, seed=1, mode='str')
    >>> print('seq = {!r}'.format(seq))
    >>> seq, open_to_close = random_balanced_sequence(10, seed=1, mode='paren')
    >>> print('seq = {!r}'.format(seq))
    seq = (('open', 0), ('open', 1), ('close', 1), ('close', 0))
    seq = '\x00\x02\x04\x06\x07\x05\x03\x01'
    seq = (1, 2, 3, 4, -4, -3, -2, -1)
    seq = ('2(', '1(', '0(', '3(', ')3', ')0', ')1', ')2')
    seq = '([[[]{{}}](){{[]}}])'
    """
    from networkx.algorithms.embedding.tree_embedding import tree_to_seq
    from networkx.generators.random_graphs import random_ordered_tree
    from networkx.utils import create_py_random_state
    # Create a random otree and then convert it to a balanced sequence
    rng = create_py_random_state(seed)
    tree = random_ordered_tree(n, seed=rng)
    if mode == 'paren':
        pool = '[{('
        for node in tree.nodes:
            tree.nodes[node]['label'] = rng.choice(pool)
        seq, open_to_close, _ = tree_to_seq(
            tree, mode=mode, open_to_close=open_to_close, strhack=1)
    else:
        seq, open_to_close, _ = tree_to_seq(
            tree, mode=mode, open_to_close=open_to_close)
    return seq, open_to_close
