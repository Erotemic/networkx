"""
Balanced sequences are used via reduction to solve the maximum common subtree
embedding problem.
"""
import operator


# @profile
def longest_common_balanced_sequence(seq1, seq2, open_to_close, open_to_node=None, node_affinity='auto', impl='iter'):
    """
    CommandLine
    -----------
    xdoctest -m ~/code/networkx/networkx/algorithms/isomorphism/_embeddinghelpers/balanced_sequence.py longest_common_balanced_sequence:0 --profile && cat profile_output.txt


    Example
    -------
    >>> from networkx.algorithms.isomorphism._embeddinghelpers.demodata import random_ordered_tree
    >>> from networkx.algorithms.isomorphism._embeddinghelpers.balanced_sequence import *  # NOQA
    >>> from networkx.algorithms.isomorphism._embeddinghelpers.balanced_sequence import _lcs_iter_simple, _lcs_iter_simple_alt1, _lcs_iter_simple_alt2, _lcs_iter_prehash, _lcs_iter_prehash2, _lcs_recurse
    >>> from networkx.algorithms.isomorphism._embeddinghelpers.tree_embedding import tree_to_seq
    >>> from networkx.algorithms.isomorphism._embeddinghelpers.tree_embedding import forest_str
    >>> from networkx.algorithms.isomorphism._embeddinghelpers.tree_embedding import invert_dict
    >>> tree1 = random_ordered_tree(5, seed=10)
    >>> tree2 = random_ordered_tree(5, seed=3)
    >>> import kwarray
    >>> rng = kwarray.ensure_rng(3432432, 'python')
    >>> #tree1 = random_ordered_tree(300, seed=rng, pool='[{(')
    >>> #tree2 = random_ordered_tree(300, seed=rng, pool='[{(')
    >>> if len(tree1.nodes) < 20:
    >>>     forest_str(tree1, eager=1)
    >>>     forest_str(tree2, eager=1)
    >>> seq1, open_to_close, node_to_open = tree_to_seq(tree1, mode='chr', strhack=1)
    >>> seq2, open_to_close, node_to_open = tree_to_seq(tree2, open_to_close, node_to_open, mode='chr', strhack=1)
    >>> full_seq1 = seq1
    >>> full_seq2 = seq2
    >>> print('seq1 = {!r}'.format(seq1))
    >>> print('seq2 = {!r}'.format(seq2))
    >>> open_to_node = invert_dict(node_to_open)
    >>> best1, val1 = longest_common_balanced_sequence(seq1, seq2, open_to_close, open_to_node, impl='iter-alt2')
    >>> #
    >>> # xdoctest: +REQUIRES(module:ubelt)
    >>> import ubelt as ub
    >>> node_affinity = operator.eq
    >>> with ub.Timer('iterative-alt2'):
    >>>     best1, val1 = longest_common_balanced_sequence(seq1, seq2, open_to_close, open_to_node, impl='iter-alt2')
    >>>     #print('val1, best1 = {}, {!r}'.format(val1, best1))
    >>> with ub.Timer('iterative-alt1'):
    >>>     best1, val1 = longest_common_balanced_sequence(seq1, seq2, open_to_close, open_to_node, impl='iter-alt1')
    >>>     #print('val1, best1 = {}, {!r}'.format(val1, best1))
    >>> with ub.Timer('iterative'):
    >>>     best1, val1 = longest_common_balanced_sequence(seq1, seq2, open_to_close, open_to_node, impl='iter')
    >>>     #print('val1, best1 = {}, {!r}'.format(val1, best1))
    >>> with ub.Timer('recursive'):
    >>>     best2, val2 = longest_common_balanced_sequence(seq1, seq2, open_to_close, open_to_node, impl='recurse')
    >>>     #print('val2, best2 = {}, {!r}'.format(val2, best2))
    >>> with ub.Timer('iterative-prehash'):
    >>>     best1, val1 = longest_common_balanced_sequence(seq1, seq2, open_to_close, open_to_node, impl='iter-prehash')
    >>>     #print('val1, best1 = {}, {!r}'.format(val1, best1))
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
    if impl == 'recurse':
        _memo = {}
        _seq_memo = {}
        best, value = _lcs_recurse(full_seq1, full_seq2, open_to_close, node_affinity, open_to_node, _memo, _seq_memo)
    elif impl == 'iter':
        best, value = _lcs_iter_simple(full_seq1, full_seq2, open_to_close, node_affinity, open_to_node)
    elif impl == 'iter-prehash':
        best, value = _lcs_iter_prehash(full_seq1, full_seq2, open_to_close, node_affinity, open_to_node)
    elif impl == 'iter-prehash2':
        best, value = _lcs_iter_prehash2(full_seq1, full_seq2, open_to_close, node_affinity, open_to_node)
    elif impl == 'iter-alt1':
        best, value = _lcs_iter_simple_alt1(full_seq1, full_seq2, open_to_close, node_affinity, open_to_node)
    elif impl == 'iter-alt2':
        best, value = _lcs_iter_simple_alt2(full_seq1, full_seq2, open_to_close, node_affinity, open_to_node)
    elif impl == 'iter-alt2-cython':
        from networkx.algorithms.isomorphism.balanced_sequence_cython import _lcs_iter_simple_alt2_cython
        best, value = _lcs_iter_simple_alt2_cython(full_seq1, full_seq2, open_to_close, node_affinity, open_to_node)
    elif impl == 'iter-prehash2-cython':
        from networkx.algorithms.isomorphism.balanced_sequence_cython import _lcs_iter_prehash2_cython
        best, value = _lcs_iter_prehash2_cython(full_seq1, full_seq2, open_to_close, node_affinity, open_to_node)
    else:
        raise KeyError(impl)
    return best, value


def _lcs_iter_simple(full_seq1, full_seq2, open_to_close, node_affinity, open_to_node):
    """
    Converts _lcs_recursive to an iterative algorithm using a fairly
    straightforward method that effectivly simulates callstacks
    """
    all_decomp1 = generate_all_decomp(full_seq1, open_to_close, open_to_node)
    all_decomp2 = generate_all_decomp(full_seq2, open_to_close, open_to_node)

    args0 = (full_seq1, full_seq2)
    frame0 = args0
    stack = [frame0]

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

    del args0
    del frame0
    del empty1
    del empty2
    del best
    del base_result

    missing_frames = []
    while stack:
        key = stack.pop()
        if key not in _results:
            seq1, seq2 = key
            missing_frames.clear()

            # try:
            t1, a1, b1, head1, tail1, head_tail1 = all_decomp1[seq1]
            # except KeyError:
            #     a1, b1, head1, tail1 = balanced_decomp_unsafe(seq1, open_to_close)
            #     head_tail1 = head1 + tail1
            #     all_decomp1[seq1] = a1, b1, head1, tail1, head_tail1

            # try:
            t2, a2, b2, head2, tail2, head_tail2 = all_decomp2[seq2]
            # except KeyError:
            #     a2, b2, head2, tail2 = balanced_decomp_unsafe(seq2, open_to_close)
            #     head_tail2 = head2 + tail2
            #     all_decomp2[seq2] = a2, b2, head2, tail2, head_tail2

            # Case 2: The current edge in sequence1 is deleted
            try:
                try_key = (head_tail1, seq2)
                cand1 = _results[try_key]
            except KeyError:
                missing_frames.append(try_key)

            # Case 3: The current edge in sequence2 is deleted
            try:
                try_key = (seq1, head_tail2)
                cand2 = _results[try_key]
            except KeyError:
                missing_frames.append(try_key)

            # Case 1: The LCS involves this edge
            affinity = node_affinity(t1, t2)
            if affinity:
                try:
                    try_key = (head1, head2)
                    pval_h, new_heads = _results[try_key]
                except KeyError:
                    missing_frames.append(try_key)

                try:
                    try_key = (tail1, tail2)
                    pval_t, new_tails = _results[try_key]
                except KeyError:
                    missing_frames.append(try_key)

                if not missing_frames:
                    new_head1, new_head2 = new_heads
                    new_tail1, new_tail2 = new_tails

                    subseq1 = a1 + new_head1 + b1 + new_tail1
                    subseq2 = a2 + new_head2 + b2 + new_tail2

                    res3 = (subseq1, subseq2)
                    val3 = pval_h + pval_t + affinity
                    cand3 = (val3, res3)
            else:
                cand3 = (-1, None)

            if missing_frames:
                # We did not solve this frame yet
                stack.append(key)
                stack.extend(missing_frames)
                # stack.extend(missing_frames[::-1])
            else:
                # We solved the frame
                _results[key] = max(cand1, cand2, cand3)

    val, best = _results[key]
    found = (best, val)
    return found


def _lcs_iter_simple_alt1(full_seq1, full_seq2, open_to_close, node_affinity, open_to_node):
    """
    Depth first stack trajectory
    """
    all_decomp1 = generate_all_decomp(full_seq1, open_to_close, open_to_node)
    all_decomp2 = generate_all_decomp(full_seq2, open_to_close, open_to_node)

    args0 = (full_seq1, full_seq2)
    frame0 = args0
    stack = [frame0]

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

    del args0
    del frame0
    del empty1
    del empty2
    del best
    del base_result

    while stack:
        key = stack.pop()
        if key not in _results:
            seq1, seq2 = key

            t1, a1, b1, head1, tail1, head_tail1 = all_decomp1[seq1]

            t2, a2, b2, head2, tail2, head_tail2 = all_decomp2[seq2]

            # Case 2: The current edge in sequence1 is deleted
            try:
                try_key = (head_tail1, seq2)
                cand1 = _results[try_key]
            except KeyError:
                stack.append(key)
                stack.append(try_key)
                continue

            # Case 3: The current edge in sequence2 is deleted
            try:
                try_key = (seq1, head_tail2)
                cand2 = _results[try_key]
            except KeyError:
                stack.append(key)
                stack.append(try_key)
                continue

            # Case 1: The LCS involves this edge
            affinity = node_affinity(t1, t2)
            if affinity:
                try:
                    try_key = (head1, head2)
                    pval_h, new_heads = _results[try_key]
                except KeyError:
                    stack.append(key)
                    stack.append(try_key)
                    continue

                try:
                    try_key = (tail1, tail2)
                    pval_t, new_tails = _results[try_key]
                except KeyError:
                    stack.append(key)
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

    val, best = _results[key]
    found = (best, val)
    return found


def _lcs_iter_simple_alt2(full_seq1, full_seq2, open_to_close, node_affinity, open_to_node):
    """
    Depth first stack trajectory and replace try except statements with ifs
    """
    all_decomp1 = generate_all_decomp(full_seq1, open_to_close, open_to_node)
    all_decomp2 = generate_all_decomp(full_seq2, open_to_close, open_to_node)

    key0 = (full_seq1, full_seq2)
    frame0 = key0
    stack = [frame0]

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


def _lcs_iter_prehash(full_seq1, full_seq2, open_to_close, node_affinity, open_to_node):
    """
    Version of the lcs iterative algorithm where we precompute hash values
    """
    all_decomp1 = generate_all_decomp_prehash(full_seq1, open_to_close, open_to_node)
    all_decomp2 = generate_all_decomp_prehash(full_seq2, open_to_close, open_to_node)

    key_decomp1 = {}
    key_decomp2 = {}
    _results = {}
    # Populate base cases
    empty1 = type(next(iter(all_decomp1.keys())))()
    empty2 = type(next(iter(all_decomp2.keys())))()
    empty1_key = hash(empty1)
    empty2_key = hash(empty2)
    best = (empty1, empty2)
    base_result = (0, best)
    for seq1, info1 in all_decomp1.items():
        seq1_key = hash(seq1)
        head1_key, tail1_key, head_tail1_key = all_decomp1[seq1][5:8]
        _results[(seq1_key, empty2_key)] = base_result
        _results[(head1_key, empty2_key)] = base_result
        _results[(tail1_key, empty2_key)] = base_result
        _results[(head_tail1_key, empty2_key)] = base_result
        key_decomp1[seq1_key] = info1

    for seq2, info2 in all_decomp2.items():
        seq2_key = hash(seq2)
        head2_key, tail2_key, head_tail2_key = all_decomp2[seq2][5:8]
        _results[(empty1_key, seq2_key)] = base_result
        _results[(empty1_key, head2_key)] = base_result
        _results[(empty1_key, tail2_key)] = base_result
        _results[(empty1_key, head_tail2_key)] = base_result
        key_decomp2[seq2_key] = info2

    full_seq1_key = hash(full_seq1)
    full_seq2_key = hash(full_seq2)
    key0 = (full_seq1_key, full_seq2_key)
    frame0 = key0, full_seq1, full_seq2
    stack = [frame0]
    missing_frames = []
    while stack:
        frame = stack.pop()
        key, seq1, seq2 = frame
        seq1_key, seq2_key = key
        if key not in _results:
            missing_frames.clear()

            try:
                info1 = key_decomp1[seq1_key]
            except KeyError:
                info1 = balanced_decomp_prehash(seq1, open_to_close)
                key_decomp1[seq1_key] = info1
            tok1, seq1, head1, tail1, head_tail1, head1_key, tail1_key, head_tail1_key, a1, b1 = info1

            try:
                info2 = key_decomp2[seq2_key]
            except KeyError:
                info2 = balanced_decomp_prehash(seq2, open_to_close)
                key_decomp2[seq2_key] = info2
            tok2, seq2, head2, tail2, head_tail2, head2_key, tail2_key, head_tail2_key, a2, b2 = info2

            affinity = node_affinity(tok1, tok2)

            # Case 2: The current edge in sequence1 is deleted
            try:
                try_key = (head_tail1_key, seq2_key)
                cand1 = _results[try_key]
            except KeyError:
                miss_frame = try_key, head_tail1, seq2
                missing_frames.append(miss_frame)

            # Case 3: The current edge in sequence2 is deleted
            try:
                try_key = (seq1_key, head_tail2_key)
                cand2 = _results[try_key]
            except KeyError:
                miss_frame = try_key, seq1, head_tail2
                missing_frames.append(miss_frame)

            # Case 1: The LCS involves this edge
            if affinity:
                try:
                    try_key = (head1_key, head2_key)
                    pval_h, new_heads = _results[try_key]
                except KeyError:
                    miss_frame = try_key, head1, head2
                    missing_frames.append(miss_frame)

                try:
                    try_key = (tail1_key, tail2_key)
                    pval_t, new_tails = _results[try_key]
                except KeyError:
                    miss_frame = try_key, tail1, tail2
                    missing_frames.append(miss_frame)

                if not missing_frames:
                    new_head1, new_head2 = new_heads
                    new_tail1, new_tail2 = new_tails

                    subseq1 = a1 + new_head1 + b1 + new_tail1
                    subseq2 = a2 + new_head2 + b2 + new_tail2

                    res3 = (subseq1, subseq2)
                    val3 = pval_h + pval_t + affinity
                    cand3 = (val3, res3)
            else:
                cand3 = (-1, None)

            if missing_frames:
                # We did not solve this frame yet
                stack.append(frame)
                stack.extend(missing_frames[::-1])
            else:
                # We solved the frame
                _results[key] = max(cand1, cand2, cand3)

    # The stack pop is our solution
    (val, best) = _results[key]
    found = (best, val)
    return found


def _lcs_iter_prehash2(full_seq1, full_seq2, open_to_close, node_affinity, open_to_node):
    """
    Version of the lcs iterative algorithm where we precompute hash values
    """

    all_decomp1 = generate_all_decomp_prehash(full_seq1, open_to_close, open_to_node)
    all_decomp2 = generate_all_decomp_prehash(full_seq2, open_to_close, open_to_node)

    key_decomp1 = {}
    key_decomp2 = {}
    _results = {}
    # Populate base cases
    empty1 = type(next(iter(all_decomp1.keys())))()
    empty2 = type(next(iter(all_decomp2.keys())))()
    empty1_key = hash(empty1)
    empty2_key = hash(empty2)
    best = (empty1, empty2)
    base_result = (0, best)
    for seq1, info1 in all_decomp1.items():
        seq1_key = hash(seq1)
        head1_key, tail1_key, head_tail1_key = all_decomp1[seq1][5:8]
        _results[(seq1_key, empty2_key)] = base_result
        _results[(head1_key, empty2_key)] = base_result
        _results[(tail1_key, empty2_key)] = base_result
        _results[(head_tail1_key, empty2_key)] = base_result
        key_decomp1[seq1_key] = info1

    for seq2, info2 in all_decomp2.items():
        seq2_key = hash(seq2)
        head2_key, tail2_key, head_tail2_key = all_decomp2[seq2][5:8]
        _results[(empty1_key, seq2_key)] = base_result
        _results[(empty1_key, head2_key)] = base_result
        _results[(empty1_key, tail2_key)] = base_result
        _results[(empty1_key, head_tail2_key)] = base_result
        key_decomp2[seq2_key] = info2

    full_seq1_key = hash(full_seq1)
    full_seq2_key = hash(full_seq2)
    key0 = (full_seq1_key, full_seq2_key)
    frame0 = key0, full_seq1, full_seq2
    stack = [frame0]
    missing_frames = []
    while stack:
        frame = stack[-1]
        key, seq1, seq2 = frame
        seq1_key, seq2_key = key
        if key not in _results:
            missing_frames.clear()

            # if seq1_key not in key_decomp1:
            info1 = key_decomp1[seq1_key]
            # else:
            #     info1 = balanced_decomp_prehash(seq1, open_to_close)
            #     key_decomp1[seq1_key] = info1
            tok1, seq1, head1, tail1, head_tail1, head1_key, tail1_key, head_tail1_key, a1, b1 = info1

            # if seq2_key not in key_decomp2:
            info2 = key_decomp2[seq2_key]
            # else:
            #     info2 = balanced_decomp_prehash(seq2, open_to_close)
            #     key_decomp2[seq2_key] = info2
            tok2, seq2, head2, tail2, head_tail2, head2_key, tail2_key, head_tail2_key, a2, b2 = info2

            affinity = node_affinity(tok1, tok2)

            # Case 2: The current edge in sequence1 is deleted
            try_key = (head_tail1_key, seq2_key)
            if try_key in _results:
                cand1 = _results[try_key]
            else:
                miss_frame = try_key, head_tail1, seq2
                stack.append(miss_frame)
                continue

            # Case 3: The current edge in sequence2 is deleted
            try_key = (seq1_key, head_tail2_key)
            if try_key in _results:
                cand2 = _results[try_key]
            else:
                miss_frame = try_key, seq1, head_tail2
                stack.append(miss_frame)
                continue

            # Case 1: The LCS involves this edge
            if affinity:
                try_key = (head1_key, head2_key)
                if try_key in _results:
                    pval_h, new_heads = _results[try_key]
                else:
                    miss_frame = try_key, head1, head2
                    stack.append(miss_frame)
                    continue

                try_key = (tail1_key, tail2_key)
                if try_key in _results:
                    pval_t, new_tails = _results[try_key]
                else:
                    miss_frame = try_key, tail1, tail2
                    stack.append(miss_frame)
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

    # The stack pop is our solution
    (val, best) = _results[key0]
    found = (best, val)
    return found


def _lcs_recurse(seq1, seq2, open_to_close, node_affinity, open_to_node, _memo, _seq_memo):
    if not seq1:
        return (seq1, seq1), 0
    elif not seq2:
        return (seq2, seq2), 0
    else:
        key1 = hash(seq1)  # using hash(seq) is faster than seq itself
        key2 = hash(seq2)
        key = hash((key1, key2))
        if key in _memo:
            return _memo[key]

        if key1 in _seq_memo:
            a1, b1, head1, tail1, head1_tail1 = _seq_memo[key1]
        else:
            a1, b1, head1, tail1, head1_tail1 = balanced_decomp_unsafe(seq1, open_to_close)
            _seq_memo[key1] = a1, b1, head1, tail1, head1_tail1

        if key2 in _seq_memo:
            a2, b2, head2, tail2, head2_tail2 = _seq_memo[key2]
        else:
            a2, b2, head2, tail2, head2_tail2 = balanced_decomp_unsafe(seq2, open_to_close)
            _seq_memo[key2] = a2, b2, head2, tail2, head2_tail2

        # Case 2: The current edge in sequence1 is deleted
        best, val = _lcs_recurse(head1_tail1, seq2, open_to_close, node_affinity, open_to_node, _memo, _seq_memo)

        # Case 3: The current edge in sequence2 is deleted
        cand, val_alt = _lcs_recurse(seq1, head2_tail2, open_to_close, node_affinity, open_to_node, _memo, _seq_memo)
        if val_alt > val:
            best = cand
            val = val_alt

        # Case 1: The LCS involves this edge
        t1 = open_to_node[a1[0]]
        t2 = open_to_node[a2[0]]
        affinity = node_affinity(t1, t2)
        if affinity:
            new_heads, pval_h = _lcs_recurse(head1, head2, open_to_close, node_affinity, open_to_node, _memo, _seq_memo)
            new_tails, pval_t = _lcs_recurse(tail1, tail2, open_to_close, node_affinity, open_to_node, _memo, _seq_memo)

            new_head1, new_head2 = new_heads
            new_tail1, new_tail2 = new_tails

            subseq1 = a1 + new_head1 + b1 + new_tail1
            subseq2 = a2 + new_head2 + b2 + new_tail2

            cand = (subseq1, subseq2)
            val_alt = pval_h + pval_t + affinity
            if val_alt > val:
                best = cand
                val = val_alt

        found = (best, val)
        _memo[key] = found
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


def generate_all_decomp_prehash(seq, open_to_close, open_to_node):
    """
    Like :func:`generate_all_decomp` but additionally returns the
    precomputed hashes of the sequences.
    """
    all_decomp = {}
    stack = [seq]
    while stack:
        seq = stack.pop()
        if seq:
            # key = hash(seq)
            key = seq
            if key not in all_decomp:
                info = balanced_decomp_prehash(seq, open_to_close, open_to_node)
                head, tail, head_tail = info[2:5]
                all_decomp[key] = info
                stack.append(head_tail)
                stack.append(head)
                stack.append(tail)
    return all_decomp


def generate_all_decomp(seq, open_to_close, open_to_node=None):
    """
    Generates all possible decompositions of a single balanced sequence

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

    Example
    -------
    >>> open_to_close = {'{': '}', '(': ')', '[': ']'}
    >>> seq = '({[[]]})[[][]]{{}}'
    >>> all_decomp = generate_all_decomp(seq, open_to_close)
    >>> node, *decomp = all_decomp[seq]
    >>> pop_open, pop_close, head, tail, head_tail = decomp
    >>> print('node = {!r}'.format(node))
    >>> print('pop_open = {!r}'.format(pop_open))
    >>> print('pop_close = {!r}'.format(pop_close))
    >>> print('head = {!r}'.format(head))
    >>> print('tail = {!r}'.format(tail))
    >>> print('head_tail = {!r}'.format(head_tail))
    node = '('
    pop_open = '('
    pop_close = ')'
    head = '{[[]]}'
    tail = '[[][]]{{}}'
    head_tail = '{[[]]}[[][]]{{}}'
    >>> decomp_alt = balanced_decomp(seq, open_to_close)
    >>> assert decomp_alt == tuple(decomp)

    Example
    -------
    >>> from networkx.algorithms.isomorphism._embeddinghelpers.demodata import random_ordered_tree
    >>> from networkx.algorithms.isomorphism._embeddinghelpers.tree_embedding import tree_to_seq
    >>> tree = random_ordered_tree(10)
    >>> seq, open_to_close, node_to_open = tree_to_seq(tree, mode='chr', strhack=True)
    >>> all_decomp = generate_all_decomp(seq, open_to_close)
    """
    if open_to_node is None:
        open_to_node = IdentityDict()
    all_decomp = {}
    stack = [seq]
    while stack:
        seq = stack.pop()
        if seq not in all_decomp and seq:
            pop_open, pop_close, head, tail, head_tail = balanced_decomp(seq, open_to_close)
            node = open_to_node[pop_open[0]]
            all_decomp[seq] = (node, pop_open, pop_close, head, tail, head_tail)
            stack.append(head_tail)
            stack.append(head)
            stack.append(tail)
    return all_decomp


def balanced_decomp(sequence, open_to_close):
    """
    Generates a decomposition of a balanced sequence.

    Parameters
    ----------
    open_to_close: dict
        a dictionary that maps opening tokens to closing tokens in the balanced
        sequence problem.

    Returns
    -------
    : tuple[T, T, T, T, T]
        where ``T = type(sequence)``

    Example
    -------
    >>> open_to_close = {0: 1}
    >>> sequence = [0, 0, 0, 1, 1, 1, 0, 1]
    >>> a1, b1, head, tail, head_tail = balanced_decomp(sequence, open_to_close)
    >>> print('a1 = {!r}'.format(a1))
    >>> print('b1 = {!r}'.format(b1))
    >>> print('head = {!r}'.format(head))
    >>> print('tail = {!r}'.format(tail))
    >>> print('head_tail = {!r}'.format(head_tail))
    a1 = [0]
    b1 = [1]
    head = [0, 0, 1, 1]
    tail = [0, 1]
    head_tail = [0, 0, 1, 1, 0, 1]
    >>> a2, b2, tail1, tail2, head_tail2 = balanced_decomp(tail, open_to_close)

    Example
    -------
    >>> open_to_close = {'{': '}', '(': ')', '[': ']'}
    >>> sequence = '({[[]]})[[][]]'
    >>> a1, b1, head, tail, head_tail = balanced_decomp(sequence, open_to_close)
    >>> print('a1 = {}'.format(ub.repr2(a1, nl=1)))
    >>> print('b1 = {}'.format(ub.repr2(b1, nl=1)))
    >>> print('head = {}'.format(ub.repr2(head, nl=1)))
    >>> print('tail = {}'.format(ub.repr2(tail, nl=1)))
    >>> print('head_tail = {}'.format(ub.repr2(head_tail, nl=1)))
    a1 = '('
    b1 = ')'
    head = '{[[]]}'
    tail = '[[][]]'
    head_tail = '{[[]]}[[][]]'
    >>> a2, b2, tail1, tail2, head_tail2 = balanced_decomp(tail, open_to_close)
    >>> print('a2 = {}'.format(ub.repr2(a2, nl=1)))
    >>> print('b2 = {}'.format(ub.repr2(b2, nl=1)))
    >>> print('tail1 = {}'.format(ub.repr2(tail1, nl=1)))
    >>> print('tail2 = {}'.format(ub.repr2(tail2, nl=1)))
    >>> print('head_tail2 = {}'.format(ub.repr2(head_tail2, nl=1)))
    a2 = '['
    b2 = ']'
    tail1 = '[][]'
    tail2 = ''
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


def balanced_decomp_unsafe(sequence, open_to_close):
    """
    Same as :func:`balanced_decomp` but assumes that ``sequence`` is valid
    balanced sequence in order to execute faster.

    Example
    -------
    >>> open_to_close = {'{': '}', '(': ')', '[': ']'}
    >>> sequence = '({[[]]})[[][]]'
    >>> print('sequence = {!r}'.format(sequence))
    >>> a1, b1, head, tail, head_tail = balanced_decomp(sequence, open_to_close)
    >>> print('a1 = {!r}'.format(a1))
    >>> print('tail = {!r}'.format(tail))
    >>> print('head = {!r}'.format(head))
    >>> a2, b2, tail1, tail2, head_tail2 = balanced_decomp(tail, open_to_close)
    >>> print('a2 = {!r}'.format(a2))
    >>> print('tail1 = {!r}'.format(tail1))
    >>> print('tail2 = {!r}'.format(tail2))
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


def balanced_decomp_prehash(seq, open_to_close, open_to_node):
    """
    Like :func:`balanced_decomp` but additionally returns the
    precomputed hashes of the sequences.
    """
    pop_open, pop_close, head, tail, head_tail = balanced_decomp_unsafe(seq, open_to_close)
    head_key = hash(head)
    tail_key = hash(tail)
    head_tail_key = hash(head_tail)
    node = open_to_node[pop_open[0]]
    a = pop_open
    b = pop_close
    info = (node, seq, head, tail, head_tail, head_key, tail_key, head_tail_key, a, b)
    return info


def generate_balance(sequence, open_to_close):
    """
    Iterates through a balanced sequence and reports if the sequence-so-far
    is balanced at that position or not.

    Raises
    ------
    UnbalancedException - if the sequence is not balanced

    Example
    -------
    >>> open_to_close = {0: 1}
    >>> sequence = [0, 0, 0, 1, 1, 1]
    >>> gen = list(generate_balance(sequence, open_to_close))
    >>> for flag, token in gen:
    >>>     print('flag={:d}, token={}'.format(flag, token))

    Example
    -------
    >>> from networkx.algorithms.isomorphism._embeddinghelpers.demodata import random_ordered_tree
    >>> from networkx.algorithms.isomorphism._embeddinghelpers.tree_embedding import tree_to_seq
    >>> tree = random_ordered_tree(1000)
    >>> sequence, open_to_close, node_to_open = tree_to_seq(tree)
    >>> gen = list(generate_balance(sequence, open_to_close))
    >>> for flag, token in gen:
    >>>     print('flag={:d}, token={}'.format(flag, token))
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


def generate_balance_unsafe(sequence, open_to_close):
    """
    Same as :func:`generate_balance` but assumes that ``sequence`` is valid
    balanced sequence in order to execute faster.

    Benchmark
    ---------
    >>> # xdoctest: +REQUIRES(--benchmark)
    >>> from networkx.algorithms.isomorphism._embeddinghelpers.demodata import random_ordered_tree
    >>> from networkx.algorithms.isomorphism._embeddinghelpers.tree_embedding import tree_to_seq
    >>> tree = random_ordered_tree(1000)
    >>> sequence, open_to_close, node_to_open = tree_to_seq(tree, mode='tuple')
    >>> sequence, open_to_close, node_to_open = tree_to_seq(tree, mode='number')
    >>> import timerit
    >>> ti = timerit.Timerit(100, bestof=10, verbose=2)
    >>> for timer in ti.reset('time'):
    >>>     with timer:
    >>>         list(generate_balance_unsafe(sequence, open_to_close))
    >>> import timerit
    >>> ti = timerit.Timerit(100, bestof=10, verbose=2)
    >>> for timer in ti.reset('time'):
    >>>     with timer:
    >>>         list(generate_balance_unsafe_cython(sequence, open_to_close))
    """
    stacklen = 0
    for token in sequence:
        if token in open_to_close:
            stacklen += 1
        else:
            stacklen -= 1
        yield stacklen == 0, token
