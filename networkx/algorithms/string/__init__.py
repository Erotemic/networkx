"""
Subpackages for helpers and such related to string based problems.

balanced_sequence.py - core python implementations for the longest common
balanced sequence subproblem, this is used by
:module:`networkx.algorithms.embedding.tree_embedding`.

balanced_sequence_cython.pyx -
faster alternative implementsions for balanced_sequence.py

demodata.py - Contains sting-based random and demonstration data for for
docstrings, benchmarks, and synthetic problems.

Regen Command:
    mkinit ~/code/networkx/networkx/algorithms/string/__init__.py

Test Command:
    xdoctest -m networkx.algorithms.string all
    pytest networkx/algorithms/string
"""
from networkx.algorithms.string import balanced_sequence
from networkx.algorithms.string import demodata

from networkx.algorithms.string.balanced_sequence import (
    available_impls_longest_common_balanced_sequence,
    longest_common_balanced_sequence,)
from networkx.algorithms.string.demodata import (random_balanced_sequence,)

__all__ = ['available_impls_longest_common_balanced_sequence',
           'balanced_sequence', 'demodata', 'longest_common_balanced_sequence',
           'random_balanced_sequence']
