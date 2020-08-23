# distutils: language = c++
"""
cythonize -a -i ~/code/networkx/networkx/algorithms/isomorphism/_embeddinghelpers/balanced_sequence_cython.pyx

Examples
--------
>>> from networkx.algorithms.isomorphism.balanced_sequence import *  # NOQA
>>> from networkx.algorithms.isomorphism.balanced_sequence import _lcs_iter_simple, _lcs_iter_simple_alt1, _lcs_iter_simple_alt2, _lcs_iter_prehash, _lcs_recurse, _print_forest
>>> tree1 = random_ordered_tree(5, seed=10, pool='[{(')
>>> tree2 = random_ordered_tree(5, seed=3, pool='[{(')

>>> import kwarray
>>> rng = kwarray.ensure_rng(3432432, 'python')
>>> tree1 = random_ordered_tree(300, seed=rng, pool='[{(')
>>> tree2 = random_ordered_tree(300, seed=rng, pool='[{(')
>>> if len(tree1.nodes) < 20:
>>>     _print_forest(tree1)
>>>     _print_forest(tree2)
>>> seq1, open_to_close, toks = tree_to_seq(tree1, mode='label', strhack=1)
>>> seq2, open_to_close, toks = tree_to_seq(tree2, open_to_close, toks, mode='label', strhack=1)
>>> full_seq1 = seq1
>>> full_seq2 = seq2

>>> node_affinity = operator.eq
>>> open_to_tok = ub.invert_dict(toks)
>>> from networkx.algorithms.isomorphism.balanced_sequence_cython import _lcs_iter_simple_alt2_cython  # NOQA

with ub.Timer('cython'):
    best, value = _lcs_iter_simple_alt2_cython(full_seq1, full_seq2, open_to_close, node_affinity, open_to_tok)

with ub.Timer('python'):
    best, value = _lcs_iter_simple_alt2(full_seq1, full_seq2, open_to_close, node_affinity, open_to_tok)

References
----------
https://stackoverflow.com/questions/41909486/returning-a-complex-object-containing-pyobject-from-c-function-cython
"""

# from libcpp.map cimport map as cmap
# from libcpp.string cimport string
# from libcpp.vector cimport vector
        

def _lcs_iter_simple_alt2_cython(full_seq1, full_seq2, open_to_close, node_affinity, open_to_tok):
    """
    Depth first stack trajectory and replace try except statements with ifs
    """
    if open_to_tok is None:
        open_to_tok = IdentityDict()
    all_decomp1 = generate_all_decomp_cython(full_seq1, open_to_close, open_to_tok)
    all_decomp2 = generate_all_decomp_cython(full_seq2, open_to_close, open_to_tok)

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


def _lcs_iter_prehash2_cython(full_seq1, full_seq2, open_to_close, node_affinity, open_to_tok):
    """
    Version of the lcs iterative algorithm where we precompute hash values
    """
    cdef dict all_decomp1 = generate_all_decomp_prehash_cython(full_seq1, open_to_close, open_to_tok)
    cdef dict all_decomp2 = generate_all_decomp_prehash_cython(full_seq2, open_to_close, open_to_tok)
    cdef dict key_decomp1 = {}
    cdef dict key_decomp2 = {}

    cdef dict _results = {}
    # cdef cmap[Py_hash_t, tuple] _results3
    # = {}
    # Populate base cases
    empty1 = type(next(iter(all_decomp1.keys())))()
    empty2 = type(next(iter(all_decomp2.keys())))()
    cdef Py_hash_t empty1_key = hash(empty1)
    cdef Py_hash_t empty2_key = hash(empty2)
    cdef tuple best = (empty1, empty2)

    cdef tuple info1, info2
    cdef tuple try_key, key
    cdef Py_hash_t seq1_key, seq2_key
    cdef Py_hash_t head1_key, tail1_key, head_tail1_key
    cdef Py_hash_t head2_key, tail2_key, head_tail2_key
    cdef tuple frame
    cdef tuple miss_frame

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

    cdef Py_hash_t full_seq1_key = hash(full_seq1)
    cdef Py_hash_t full_seq2_key = hash(full_seq2)

    cdef tuple key0 = (full_seq1_key, full_seq2_key)
    cdef tuple frame0 = (key0, full_seq1, full_seq2)
    cdef list stack = [frame0]
    # cdef vector[PyObjectPtr] stack2
    # stack2.push_back(&frame0)

    while stack:
        frame = stack[-1]
        key, seq1, seq2 = frame
        seq1_key, seq2_key = key
        if key not in _results:
            # if seq1_key not in key_decomp1:
            info1 = key_decomp1[seq1_key]
            # else:
            #     info1 = decomp_info(seq1, open_to_close)
            #     key_decomp1[seq1_key] = info1
            tok1, seq1, head1, tail1, head_tail1, head1_key, tail1_key, head_tail1_key, a1, b1 = info1

            # if seq2_key not in key_decomp2:
            info2 = key_decomp2[seq2_key]
            # else:
            #     info2 = decomp_info(seq2, open_to_close)
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


cdef tuple balanced_decomp_unsafe_cython(sequence, dict open_to_close):
    cdef int stacklen = 1  # always +1 in the first iteration
    cdef int head_stop = 1

    tok_curr = sequence[0]
    want_close = open_to_close[tok_curr]

    # for tok_curr in sequence[1:]:
    for head_stop in range(1, len(sequence)):
        tok_curr = sequence[head_stop]
        stacklen += 1 if tok_curr in open_to_close else -1
        if stacklen == 0 and tok_curr == want_close:
            pop_close = sequence[head_stop:head_stop + 1]
            break

    pop_open = sequence[0:1]
    head = sequence[1:head_stop]
    tail = sequence[head_stop + 1:]
    head_tail = head + tail
    return pop_open, pop_close, head, tail, head_tail


def generate_all_decomp_cython(seq, open_to_close, open_to_tok=None):
    """
    >>> tree = random_ordered_tree(10)
    >>> seq, open_to_close, toks = tree_to_seq(tree, mode='chr', strhack=True)
    >>> all_decomp = generate_all_decomp_cython(seq, open_to_close)
    """
    all_decomp = {}
    stack = [seq]
    while stack:
        seq = stack.pop()
        if seq not in all_decomp and seq:
            pop_open, pop_close, head, tail, head_tail = balanced_decomp_unsafe_cython(seq, open_to_close)
            tok = open_to_tok[pop_open[0]]
            all_decomp[seq] = (tok, pop_open, pop_close, head, tail, head_tail)
            stack.append(head_tail)
            stack.append(head)
            stack.append(tail)
    return all_decomp


cdef tuple balanced_decomp_prehash_cython(seq, dict open_to_close, open_to_tok):
    cdef tuple info
    pop_open, pop_close, head, tail, head_tail = balanced_decomp_unsafe_cython(seq, open_to_close)
    cdef Py_hash_t head_key = hash(head)
    cdef Py_hash_t tail_key = hash(tail)
    cdef Py_hash_t head_tail_key = hash(head_tail)
    tok = open_to_tok[pop_open[0]]
    a = pop_open
    b = pop_close
    info = (tok, seq, head, tail, head_tail, head_key, tail_key, head_tail_key, a, b)
    return info


cdef dict generate_all_decomp_prehash_cython(seq, dict open_to_close, open_to_tok):
    cdef dict all_decomp = {}
    cdef list stack = [seq]
    cdef tuple info
    while stack:
        seq = stack.pop()
        if seq:
            # key = hash(seq)
            key = seq
            if key not in all_decomp:
                info = balanced_decomp_prehash_cython(seq, open_to_close, open_to_tok)
                head, tail, head_tail = info[2:5]
                all_decomp[key] = info
                stack.append(head_tail)
                stack.append(head)
                stack.append(tail)
    return all_decomp


class IdentityDict:
    """ Used when ``open_to_tok`` is unspecified """
    def __getitem__(self, key):
        return key
