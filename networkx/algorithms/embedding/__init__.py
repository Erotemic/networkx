"""
Subpackages related to the ordered subtree embedding problem.

tree_embedding.py - defines reduction from tree problem to balanced sequence
problems.


Outstanding Issues
------------------
- [ ] Multiple implementations of the algorithm backend / data structure
  reduction, need to reduce the impelmentation and / or determine a better
  mechansim for allowing the user to switch between them.



CommandLine
-----------
xdoctest -m networkx.algorithms.embedding all
pytest networkx/algorithms/embedding

# The mkinit tool helps autogenerate explicit `__init__.py` files
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
