"""
Subpackages related to minor (i.e. embedding) problems.
"""

__devnotes__ = """


tree_embedding.py - defines reduction from tree problem to balanced sequence
problems.

CommandLine
-----------
# Run all tests in this subpackage
pytest networkx/algorithms/minors --doctest-modules

# Autogenerate the `__init__.py` file for this subpackage with `mkinit`.
mkinit ~/code/networkx/networkx/algorithms/minors/__init__.py -w
"""

__submodules__ = [
    "tree_embedding",
    "contraction",
]

from networkx.algorithms.minors import tree_embedding
from networkx.algorithms.minors import contraction

from networkx.algorithms.minors.tree_embedding import (
    maximum_common_ordered_tree_embedding,
)
from networkx.algorithms.minors.contraction import (
    contracted_edge,
    contracted_nodes,
    equivalence_classes,
    identified_nodes,
    quotient_graph,
)

__all__ = [
    "contracted_edge",
    "contracted_nodes",
    "contraction",
    "equivalence_classes",
    "identified_nodes",
    "maximum_common_ordered_tree_embedding",
    "quotient_graph",
    "tree_embedding",
]
