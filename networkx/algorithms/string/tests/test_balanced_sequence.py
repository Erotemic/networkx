"""
Tests for balanced sequences

Command Line
------------
pytest networkx/algorithms/string/tests/test_balanced_sequence.py
"""


def test_all_implementations_are_same():
    """
    Tests several random sequences
    """
    from networkx.algorithms.string import balanced_sequence
    from networkx.algorithms.string import random_balanced_sequence
    from networkx.utils import create_py_random_state

    seed = 93024896892223032652928827097264
    rng = create_py_random_state(seed)

    maxsize = 20
    num_trials = 5

    for _ in range(num_trials):
        n1 = rng.randint(1, maxsize)
        n2 = rng.randint(1, maxsize)

        seq1, open_to_close = random_balanced_sequence(n1, seed=rng)
        seq2, open_to_close = random_balanced_sequence(n2, open_to_close=open_to_close, seed=rng)
        longest_common_balanced_sequence = balanced_sequence.longest_common_balanced_sequence

        # Note: the returned sequences may be different (maximum embeddings may not
        # be unique), but the values should all be the same.
        results = {}
        impls = balanced_sequence.available_impls_longest_common_balanced_sequence()
        for impl in impls:
            best, val = longest_common_balanced_sequence(
                seq1, seq2, open_to_close, node_affinity=None, impl=impl)
            results[impl] = val


def test_paper_case():
    # 1-label case from the paper (see Example 5)
    # https://pdfs.semanticscholar.org/0b6e/061af02353f7d9b887f9a378be70be64d165.pdf
    from networkx.algorithms.string import balanced_sequence
    seq1 = '0010010010111100001011011011'
    seq2 = '001000101101110001000100101110111011'
    open_to_close = {'0': '1'}
    best, value = balanced_sequence.longest_common_balanced_sequence(
        seq1, seq2, open_to_close)
    subseq1, subseq2 = best
    print('subseq1 = {!r}'.format(subseq1))
    assert value == 13
    assert subseq1 == '00100101011100001011011011'
