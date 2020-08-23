"""
Subpackages for helpers and such related to the ordered subtree embedding /
isomorphism problems.

Contains routines for solving balanced sequence and path subproblems. Only the
final graph-based API is exposed, but modification to the internals (is / will
be) available via keyword arguments.

balanced_sequence.py - core python implementations for the longest common
balanced sequence subproblem.

balanced_sequence_cython.pyx -
faster alternative implementsions for balanced_sequence.py

tree_embedding.py - defines reduction from tree problem to balanced sequence
problems.

path_embedding.py - defines reduction from path problem to tree problem (not
core, this is just useful for testing among other things).

demodata.py - Contains data for docstrings, benchmarks, and synthetic problems


CommandLine
-----------
xdoctest -m /home/joncrall/code/networkx/networkx/algorithms/isomorphism/_embeddinghelpers list
xdoctest -m /home/joncrall/code/networkx/networkx/algorithms/isomorphism/_embeddinghelpers all

mkinit ~/code/networkx/networkx/algorithms/isomorphism/_embeddinghelpers/__init__.py -w
"""

__submodules__ = [
    'tree_embedding',
]

# from networkx.algorithms.isomorphism._embeddinghelpers import tree_embedding
from networkx.algorithms.isomorphism._embeddinghelpers.tree_embedding import (
    maximum_common_ordered_tree_embedding)

__all__ = ['maximum_common_ordered_tree_embedding']
