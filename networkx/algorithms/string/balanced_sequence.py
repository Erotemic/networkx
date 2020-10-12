"""
Core python implementations for the longest common balanced sequence
subproblem, which is used by
:module:`networkx.algorithms.embedding.tree_embedding`.
"""
import operator

__all__ = [
    'available_impls_longest_common_balanced_sequence',
    'longest_common_balanced_sequence',
    'random_balanced_sequence',
]


def longest_common_balanced_sequence(
        seq1, seq2, open_to_close, open_to_node=None,
        node_affinity='auto', impl='auto'):
    """
    Finds the longest common balanced sequence between two sequences

    Parameters
    ----------
    seq1, seq2: Iterable
        two input balanced sequences

    open_to_close : Dict
        a mapping from opening to closing tokens in the balanced sequence

    open_to_node : Dict | None
        a dictionary that maps a sequence token to a token corresponding to an
        original problem (e.g. a tree node), if unspecified an identity mapping
        is assumed. FIXME: see outstanding issues.
        WILL LIKELY CHANGE IN THE FUTURE

    node_affinity : None | str | callable
        Function for to determine if two nodes can be matched. The return is
        interpreted as a weight that is used to break ties. If None then any
        node can match any other node and only the topology is important.
        The default is "eq", which is the same as ``operator.eq``.

    impl : str
        Determines the backend implementation. There are currently 3 different
        backend implementations:  "iter", and "iter-cython". The
        default is "auto", which choose "iter-cython" if available, otherwise
        "iter".

    Example
    -------
    >>> # extremely simple case
    >>> seq1 = '[][[]][]'
    >>> seq2 = '[[]][[]]'
    >>> open_to_close = {'[': ']'}
    >>> best, value = longest_common_balanced_sequence(seq1, seq2, open_to_close)
    >>> subseq1, subseq2 = best
    >>> print('subseq1 = {!r}'.format(subseq1))
    subseq1 = '[][[]]'

    >>> # 1-label case from the paper (see Example 5)
    >>> # https://pdfs.semanticscholar.org/0b6e/061af02353f7d9b887f9a378be70be64d165.pdf
    >>> seq1 = '0010010010111100001011011011'
    >>> seq2 = '001000101101110001000100101110111011'
    >>> open_to_close = {'0': '1'}
    >>> best, value = longest_common_balanced_sequence(seq1, seq2, open_to_close)
    >>> subseq1, subseq2 = best
    >>> print('subseq1 = {!r}'.format(subseq1))
    subseq1 = '00100101011100001011011011'
    >>> assert value == 13

    >>> # 3-label case
    >>> seq1 = '{({})([[]([]){(()(({()[]({}{})}))){}}])}'
    >>> seq2 = '{[({{}}{{[][{}]}(()[(({()})){[]()}])})]}'
    >>> open_to_close = {'{': '}', '(': ')', '[': ']'}
    >>> best, value = longest_common_balanced_sequence(seq1, seq2, open_to_close)
    >>> subseq1, subseq2 = best
    >>> print('subseq1 = {!r}'.format(subseq1))
    subseq1 = '{{}[][]()(({()})){}}'
    >>> assert value == 10
    """
    if node_affinity == 'auto' or node_affinity == 'eq':
        node_affinity = operator.eq
    if node_affinity is None:
        def _matchany(a, b):
            return True
        node_affinity = _matchany
    if open_to_node is None:
        open_to_node = IdentityDict()
    full_seq1 = seq1
    full_seq2 = seq2
    if impl == 'auto':
        if _cython_lcs_backend(error='ignore'):
            impl = 'iter-cython'
        else:
            impl = 'iter'

    if impl == 'iter':
        best, value = _lcs_iter(
            full_seq1, full_seq2, open_to_close, node_affinity, open_to_node)
    elif impl == 'iter-cython':
        balanced_sequence_cython = _cython_lcs_backend(error='raise')
        best, value = balanced_sequence_cython._lcs_iter_cython(
            full_seq1, full_seq2, open_to_close, node_affinity, open_to_node)
    else:
        raise KeyError(impl)
    return best, value


def available_impls_longest_common_balanced_sequence():
    """
    Returns all available implementations for
    :func:`longest_common_balanced_sequence`.
    """
    impls = []
    if _cython_lcs_backend():
        impls += [
            'iter-cython',
        ]

    # Pure python backends
    impls += [
        'iter',
    ]
    return impls


def _cython_lcs_backend(error='ignore'):
    """
    Returns the cython backend if available, otherwise None
    """
    try:
        from networkx.algorithms.string import balanced_sequence_cython
    except Exception:
        if error == 'ignore':
            return None
        elif error == 'raise':
            raise
        else:
            raise KeyError(error)
    else:
        return balanced_sequence_cython


def _lcs_iter(full_seq1, full_seq2, open_to_close, node_affinity, open_to_node):
    """
    Depth first stack trajectory and replace try except statements with ifs

    This is the current best pure-python algorithm candidate

    Converts _lcs_recursive to an iterative algorithm using a fairly
    straightforward method that effectivly simulates callstacks.
    Uses a breadth-first trajectory and try-except to catch missing
    memoized results (which seems to be slightly slower than if statements).

    Example
    -------
    >>> full_seq1 = '{({})([[]([]){(()(({()[]({}{})}))){}}])}'
    >>> full_seq2 = '{[({{}}{{[][{}]}(()[(({()})){[]()}])})]}'
    >>> open_to_close = {'{': '}', '(': ')', '[': ']'}
    >>> full_seq1 = '[][[]][]'
    >>> full_seq2 = '[[]][[]]'
    >>> open_to_close = {'[': ']'}
    >>> import operator as op
    >>> node_affinity = op.eq
    >>> open_to_node = IdentityDict()
    >>> res = _lcs_iter(full_seq1, full_seq2, open_to_close, node_affinity,
    ...                 open_to_node)
    >>> val, embeddings = res
    """
    all_decomp1 = generate_all_decomp(full_seq1, open_to_close, open_to_node)
    all_decomp2 = generate_all_decomp(full_seq2, open_to_close, open_to_node)

    key0 = (full_seq1, full_seq2)
    frame0 = key0
    stack = [frame0]

    # Memoize mapping (seq1, seq2) -> best size, embeddings, deleted edges
    _results = {}

    # Populate base cases
    empty1 = type(next(iter(all_decomp1.keys())))()
    empty2 = type(next(iter(all_decomp2.keys())))()
    best = (empty1, empty2)
    base_result = (0, best)
    for seq1 in all_decomp1.keys():
        key1 = seq1
        t1, a1, b1, head1, tail1, head_tail1 = all_decomp1[key1]
        _results[(seq1, empty2)] = base_result
        _results[(head1, empty2)] = base_result
        _results[(tail1, empty2)] = base_result
        _results[(head_tail1, empty2)] = base_result

    for seq2 in all_decomp2.keys():
        key2 = seq2
        t2, a2, b2, head2, tail2, head_tail2 = all_decomp2[key2]
        _results[(empty1, seq2)] = base_result
        _results[(empty1, head2)] = base_result
        _results[(empty1, tail2)] = base_result
        _results[(empty1, head_tail2)] = base_result

    del frame0
    del empty1
    del empty2
    del best
    del base_result

    while stack:
        key = stack[-1]
        if key not in _results:
            seq1, seq2 = key

            t1, a1, b1, head1, tail1, head_tail1 = all_decomp1[seq1]
            t2, a2, b2, head2, tail2, head_tail2 = all_decomp2[seq2]

            # Case 2: The current edge in sequence1 is deleted
            try_key = (head_tail1, seq2)
            if try_key in _results:
                cand1 = _results[try_key]
            else:
                # stack.append(key)
                stack.append(try_key)
                continue

            # Case 3: The current edge in sequence2 is deleted
            try_key = (seq1, head_tail2)
            if try_key in _results:
                cand2 = _results[try_key]
            else:
                # stack.append(key)
                stack.append(try_key)
                continue

            # Case 1: The LCS involves this edge
            affinity = node_affinity(t1, t2)
            if affinity:
                try_key = (head1, head2)
                if try_key in _results:
                    pval_h, new_heads = _results[try_key]
                else:
                    # stack.append(key)
                    stack.append(try_key)
                    continue

                try_key = (tail1, tail2)
                if try_key in _results:
                    pval_t, new_tails = _results[try_key]
                else:
                    # stack.append(key)
                    stack.append(try_key)
                    continue

                new_head1, new_head2 = new_heads
                new_tail1, new_tail2 = new_tails

                subseq1 = a1 + new_head1 + b1 + new_tail1
                subseq2 = a2 + new_head2 + b2 + new_tail2

                res3 = (subseq1, subseq2)
                val3 = pval_h + pval_t + affinity
                cand3 = (val3, res3)
            else:
                cand3 = (-1, None)

            # We solved the frame
            _results[key] = max(cand1, cand2, cand3)
        stack.pop()

    val, best = _results[key0]
    found = (best, val)
    return found


class UnbalancedException(Exception):
    """
    Denotes that a sequence was unbalanced
    """
    pass


class IdentityDict:
    """
    Used when ``open_to_node`` is unspecified
    """
    def __getitem__(self, key):
        return key


def generate_all_decomp(seq, open_to_close, open_to_node=None):
    """
    Generates all decompositions of a single balanced sequence by
    recursive decomposition of the head, tail, and head|tail.

    Parameters
    ----------
    seq : Tuple | str
        a tuple of hashable items or a string where each character is an item

    open_to_close : Dict
        a dictionary that maps opening tokens to closing tokens in the balanced
        sequence problem.

    open_to_node : Dict
        a dictionary that maps a sequence token to a token corresponding to an
        original problem (e.g. a tree node)

    Returns
    -------
    Dict : mapping from a sub-sequence to its decomposition

    Notes
    -----
    In the paper: See Definition 2, 4, Lemma, 1, 2, 3, 4.

    Example
    -------
    >>> # Example 2 in the paper (one from each column)
    >>> seq = '00100100101111'
    >>> open_to_close = {'0': '1'}
    >>> all_decomp = generate_all_decomp(seq, open_to_close)
    >>> assert len(all_decomp) == len(seq) // 2
    >>> import pprint
    >>> pprint.pprint(all_decomp)
    {'00100100101111': ('0', '0', '1', '010010010111', '', '010010010111'),
     '0010010111': ('0', '0', '1', '01001011', '', '01001011'),
     '001011': ('0', '0', '1', '0101', '', '0101'),
     '01': ('0', '0', '1', '', '', ''),
     '010010010111': ('0', '0', '1', '', '0010010111', '0010010111'),
     '01001011': ('0', '0', '1', '', '001011', '001011'),
     '0101': ('0', '0', '1', '', '01', '01')}

    Example
    -------
    >>> open_to_close = {'{': '}', '(': ')', '[': ']'}
    >>> seq = '({[[]]})[[][]]{{}}'
    >>> all_decomp = generate_all_decomp(seq, open_to_close)
    >>> node, *decomp = all_decomp[seq]
    >>> pop_open, pop_close, head, tail, head_tail = decomp
    >>> print('node = {!r}'.format(node))
    node = '('
    >>> print('pop_open = {!r}'.format(pop_open))
    pop_open = '('
    >>> print('pop_close = {!r}'.format(pop_close))
    pop_close = ')'
    >>> print('head = {!r}'.format(head))
    head = '{[[]]}'
    >>> print('tail = {!r}'.format(tail))
    tail = '[[][]]{{}}'
    >>> print('head_tail = {!r}'.format(head_tail))
    head_tail = '{[[]]}[[][]]{{}}'
    >>> decomp_alt = balanced_decomp(seq, open_to_close)
    >>> assert decomp_alt == tuple(decomp)

    Example
    -------
    >>> seq, open_to_close = random_balanced_sequence(10)
    >>> all_decomp = generate_all_decomp(seq, open_to_close)
    """
    if open_to_node is None:
        open_to_node = IdentityDict()
    all_decomp = {}
    stack = [seq]
    while stack:
        seq = stack.pop()
        if seq not in all_decomp and seq:
            (pop_open, pop_close,
             head, tail, head_tail) = balanced_decomp(seq, open_to_close)
            node = open_to_node[pop_open[0]]
            all_decomp[seq] = (node, pop_open, pop_close,
                               head, tail, head_tail)
            if head:
                if tail:
                    stack.append(head_tail)
                    stack.append(tail)
                stack.append(head)
            elif tail:
                stack.append(tail)
    return all_decomp


def balanced_decomp(sequence, open_to_close):
    """
    Generates a decomposition of a balanced sequence.

    Parameters
    ----------
    sequence : str
        balanced sequence to be decomposed

    open_to_close: dict
        a dictionary that maps opening tokens to closing tokens in the balanced
             sequence problem.

    Returns
    -------
    : tuple[T, T, T, T, T]
        where ``T = type(sequence)``
        Contents of this tuple are:

            0. a1 - a sequence of len(1) containing the current opening token
            1. b1 - a sequence of len(1) containing the current closing token
            2. head - head of the sequence
            3. tail - tail of the sequence
            4. head_tail - the concatanted head and tail

    Example
    -------
    >>> # Example 3 from the paper
    >>> sequence = '001000101101110001000100101110111011'
    >>> open_to_close = {'0': '1'}
    >>> a1, b1, head, tail, head_tail = balanced_decomp(sequence, open_to_close)
    >>> print('head = {!r}'.format(head))
    head = '010001011011'
    >>> print('tail = {!r}'.format(tail))
    tail = '0001000100101110111011'

    Example
    -------
    >>> open_to_close = {0: 1}
    >>> sequence = [0, 0, 0, 1, 1, 1, 0, 1]
    >>> a1, b1, head, tail, head_tail = balanced_decomp(sequence, open_to_close)
    >>> print('a1 = {!r}'.format(a1))
    a1 = [0]
    >>> print('b1 = {!r}'.format(b1))
    b1 = [1]
    >>> print('head = {!r}'.format(head))
    head = [0, 0, 1, 1]
    >>> print('tail = {!r}'.format(tail))
    tail = [0, 1]
    >>> print('head_tail = {!r}'.format(head_tail))
    head_tail = [0, 0, 1, 1, 0, 1]
    >>> a2, b2, tail1, tail2, head_tail2 = balanced_decomp(tail, open_to_close)

    Example
    -------
    >>> open_to_close = {'{': '}', '(': ')', '[': ']'}
    >>> sequence = '({[[]]})[[][]]'
    >>> a1, b1, head, tail, head_tail = balanced_decomp(sequence, open_to_close)
    >>> print('a1 = {!r}'.format(a1))
    a1 = '('
    >>> print('b1 = {!r}'.format(b1))
    b1 = ')'
    >>> print('head = {!r}'.format(head))
    head = '{[[]]}'
    >>> print('tail = {!r}'.format(tail))
    tail = '[[][]]'
    >>> print('head_tail = {!r}'.format(head_tail))
    head_tail = '{[[]]}[[][]]'
    >>> a2, b2, tail1, tail2, head_tail2 = balanced_decomp(tail, open_to_close)
    >>> print('a2 = {!r}'.format(a2))
    a2 = '['
    >>> print('b2 = {!r}'.format(b2))
    b2 = ']'
    >>> print('tail1 = {!r}'.format(tail1))
    tail1 = '[][]'
    >>> print('tail2 = {!r}'.format(tail2))
    tail2 = ''
    >>> print('head_tail2 = {!r}'.format(head_tail2))
    head_tail2 = '[][]'
    """
    gen = generate_balance(sequence, open_to_close)

    bal_curr, tok_curr = next(gen)
    pop_open = sequence[0:1]
    want_close = open_to_close[tok_curr]

    head_stop = 1
    for head_stop, (bal_curr, tok_curr) in enumerate(gen, start=1):
        if tok_curr is None:
            break
        elif bal_curr and tok_curr == want_close:
            pop_close = sequence[head_stop:head_stop + 1]
            break
    head = sequence[1:head_stop]
    tail = sequence[head_stop + 1:]
    head_tail = head + tail
    return pop_open, pop_close, head, tail, head_tail


def generate_balance(sequence, open_to_close):
    r"""
    Iterates through a balanced sequence and reports if the sequence-so-far
    is balanced at that position or not.

    Parameters
    ----------
    sequence: List[Tuple] | str:
        an input balanced sequence

    open_to_close : Dict
        a mapping from opening to closing tokens in the balanced sequence

    Raises
    ------
    UnbalancedException - if the input sequence is not balanced

    Yields
    ------
    Tuple[bool, T]:
        boolean indicating if the sequence is balanced at this index,
        and the current token

    Example
    -------
    >>> open_to_close = {0: 1}
    >>> sequence = [0, 0, 0, 1, 1, 1]
    >>> gen = list(generate_balance(sequence, open_to_close))
    >>> for flag, token in gen:
    ...     print('flag={:d}, token={}'.format(flag, token))
    flag=0, token=0
    flag=0, token=0
    flag=0, token=0
    flag=0, token=1
    flag=0, token=1
    flag=1, token=1

    Example
    -------
    >>> sequence, open_to_close = random_balanced_sequence(4, seed=0)
    >>> gen = list(generate_balance(sequence, open_to_close))
    """
    stack = []
    # Traversing the Expression
    for token in sequence:

        if token in open_to_close:
            # Push opening elements onto the stack
            stack.append(token)
        else:
            # Check that closing elements
            if not stack:
                raise UnbalancedException
            prev_open = stack.pop()
            want_close = open_to_close[prev_open]

            if token != want_close:
                raise UnbalancedException

        # If the stack is empty the sequence is currently balanced
        currently_balanced = not bool(stack)
        yield currently_balanced, token

    if stack:
        raise UnbalancedException


def balanced_decomp_unsafe(sequence, open_to_close):
    """
    Same as :func:`balanced_decomp` but assumes that ``sequence`` is valid
    balanced sequence in order to execute faster.
    """
    gen = generate_balance_unsafe(sequence, open_to_close)

    bal_curr, tok_curr = next(gen)
    pop_open = sequence[0:1]
    want_close = open_to_close[tok_curr]

    head_stop = 1
    for head_stop, (bal_curr, tok_curr) in enumerate(gen, start=1):
        if bal_curr and tok_curr == want_close:
            pop_close = sequence[head_stop:head_stop + 1]
            break
    head = sequence[1:head_stop]
    tail = sequence[head_stop + 1:]
    head_tail = head + tail
    return pop_open, pop_close, head, tail, head_tail


def generate_balance_unsafe(sequence, open_to_close):
    """
    Same as :func:`generate_balance` but assumes that ``sequence`` is valid
    balanced sequence in order to execute faster.
    """
    stacklen = 0
    for token in sequence:
        if token in open_to_close:
            stacklen += 1
        else:
            stacklen -= 1
        yield stacklen == 0, token


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
        the type of sequence returned (see :func:`tree_to_seq` for details) can
        also be "param", which is a special case that returns a nested set of
        parenthesis.

    Returns
    -------
    Tuple[(str | List), Dict[str, str]]
        The first item is the sequence itself
        the second item is the open_to_close mappings.

    Example
    -------
    >>> # Demo the various sequence encodings that we might use
    >>> seq, open_to_close = random_balanced_sequence(4, seed=1, mode='chr')
    >>> print('seq = {!r}'.format(seq))
    seq = '\x00\x02\x04\x06\x07\x05\x03\x01'
    >>> seq, open_to_close = random_balanced_sequence(4, seed=1, mode='number')
    >>> print('seq = {!r}'.format(seq))
    seq = (1, 2, 3, 4, -4, -3, -2, -1)
    >>> seq, open_to_close = random_balanced_sequence(10, seed=1, mode='paren')
    >>> print('seq = {!r}'.format(seq))
    seq = '([[[]{{}}](){{[]}}])'
    """
    from networkx.algorithms.embedding.tree_embedding import tree_to_seq
    from networkx.generators.random_graphs import random_ordered_tree
    from networkx.utils import create_py_random_state
    # Create a random otree and then convert it to a balanced sequence
    rng = create_py_random_state(seed)

    # To create a random balanced sequences we simply create a random ordered
    # tree and convert it to a sequence
    tree = random_ordered_tree(n, seed=rng, directed=True)
    if mode == 'paren':
        # special case
        pool = '[{('
        for node in tree.nodes:
            tree.nodes[node]['label'] = rng.choice(pool)
        seq, open_to_close, _ = tree_to_seq(
            tree, mode=mode, open_to_close=open_to_close, container='str')
    else:
        seq, open_to_close, _ = tree_to_seq(
            tree, mode=mode, open_to_close=open_to_close)
    return seq, open_to_close
