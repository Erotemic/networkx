"""
Subpackages related to embedding problems.

tree_embedding.py - defines reduction from tree problem to balanced sequence
problems.


CommandLine
-----------
# Run all tests in this subpackage
pytest networkx/algorithms/embedding --doctest-modules

# Autogenerate the `__init__.py` file for this subpackage with `mkinit`.
mkinit ~/code/networkx/networkx/algorithms/embedding/__init__.py -w
"""

__submodules__ = [
    'tree_embedding',
]

from networkx.algorithms.embedding import tree_embedding

from networkx.algorithms.embedding.tree_embedding import (
    forest_str, invert_dict, maximum_common_ordered_tree_embedding,
    seq_to_tree, tree_to_seq,)

__all__ = ['forest_str', 'invert_dict',
           'maximum_common_ordered_tree_embedding', 'seq_to_tree',
           'tree_embedding', 'tree_to_seq']
