"""
Examples for how to use :func:`maximum_common_ordered_tree_embedding` to
solve the maximum common path embedding problem.

Also contains benchmarks demonstrating runtime differences for different
backend modes of the :func:`maximum_common_ordered_tree_embedding` algorithm.

Command Line
------------
# Run tests and doctests
pytest examples/applications/filesystem_embedding.py -s -v --doctest-modules

# Run benchmark (requires timerit and ubelt module)
xdoctest -m examples/applications/filesystem_embedding.py bench_maximum_common_path_embedding
"""
import networkx as nx
from networkx.algorithms.embedding import maximum_common_ordered_tree_embedding
import operator


def maximum_common_path_embedding(paths1, paths2, sep='/', impl='auto',
                                  mode='auto'):
    """
    Finds the maximum path embedding common between two sets of paths

    Parameters
    ----------
    paths1, paths2: List[str]
        a list of paths

    sep: str
        path separator character

    impl: str
        backend runtime to use

    mode: str
        backend representation to use

    Returns
    -------
    :tuple
    corresponding lists subpaths1 and subpaths2 which are subsets of
    paths1 and path2 respectively

    Examples
    --------
    >>> paths1 = [
    ...     '/usr/bin/python',
    ...     '/usr/bin/python3.6.1',
    ...     '/usr/lib/python3.6/dist-packages/networkx',
    ...     '/usr/lib/python3.6/dist-packages/numpy',
    ...     '/usr/include/python3.6/Python.h',
    ... ]
    >>> paths2 = [
    ...     '/usr/local/bin/python',
    ...     '/usr/bin/python3.6.2',
    ...     '/usr/local/lib/python3.6/dist-packages/networkx',
    ...     '/usr/local/lib/python3.6/dist-packages/scipy',
    ...     '/usr/local/include/python3.6/Python.h',
    ... ]
    >>> subpaths1, subpaths2 = maximum_common_path_embedding(paths1, paths2)
    >>> import pprint
    >>> print('subpaths1 = {}'.format(pprint.pformat(subpaths1)))
    subpaths1 = ['/usr/bin/python',
     '/usr/include/python3.6/Python.h',
     '/usr/lib/python3.6/dist-packages/networkx']
    >>> print('subpaths2 = {}'.format(pprint.pformat(subpaths2)))
    subpaths2 = ['/usr/local/bin/python',
     '/usr/local/include/python3.6/Python.h',
     '/usr/local/lib/python3.6/dist-packages/networkx']
    """
    # the longest common balanced sequence problem
    def _affinity(node1, node2):
        score = 0
        for t1, t2 in zip(node1[::-1], node2[::-1]):
            if t1 == t2:
                score += 1
            else:
                break
        return score
    node_affinity = _affinity

    tree1 = paths_to_otree(paths1, sep=sep)
    tree2 = paths_to_otree(paths2, sep=sep)

    subtree1, subtree2 = maximum_common_ordered_tree_embedding(
            tree1, tree2, node_affinity=node_affinity, impl=impl, mode=mode)

    subpaths1 = [sep.join(node) for node in subtree1.nodes if subtree1.out_degree[node] == 0]
    subpaths2 = [sep.join(node) for node in subtree2.nodes if subtree2.out_degree[node] == 0]
    return subpaths1, subpaths2


def paths_to_otree(paths, sep='/'):
    """
    Generates an ordered tree from a list of path strings

    Parameters
    ----------
    paths: List[str]
        a list of paths

    sep : str
        path separation character. defaults to "/"

    Returns
    -------
    nx.OrderedDiGraph

    Example
    -------
    >>> from networkx.algorithms.embedding.tree_embedding import forest_str
    >>> paths = [
    ...     '/etc/ld.so.conf',
    ...     '/usr/bin/python3.6',
    ...     '/usr/include/python3.6/Python.h',
    ...     '/usr/lib/python3.6/config-3.6m-x86_64-linux-gnu/libpython3.6.so',
    ...     '/usr/local/bin/gnumake.h',
    ...     '/usr/local/etc',
    ...     '/usr/local/lib/python3.6/dist-packages',
    ... ]
    >>> otree = paths_to_otree(paths)
    >>> print(forest_str(otree))
    └── /
        ├── usr
        │   ├── local
        │   │   ├── lib
        │   │   │   └── python3.6
        │   │   │       └── dist-packages
        │   │   ├── etc
        │   │   └── bin
        │   │       └── gnumake.h
        │   ├── lib
        │   │   └── python3.6
        │   │       └── config-3.6m-x86_64-linux-gnu
        │   │           └── libpython3.6.so
        │   ├── include
        │   │   └── python3.6
        │   │       └── Python.h
        │   └── bin
        │       └── python3.6
        └── etc
            └── ld.so.conf
    """
    otree = nx.OrderedDiGraph()
    for path in sorted(paths):
        parts = tuple(path.split(sep))
        node_path = []
        for i in range(1, len(parts) + 1):
            node = parts[0:i]
            otree.add_node(node)
            otree.nodes[node]['label'] = node[-1]
            node_path.append(node)
        for u, v in zip(node_path[:-1], node_path[1:]):
            otree.add_edge(u, v)
    if ('',) in otree.nodes:
        otree.nodes[('',)]['label'] = sep
    return otree


def random_paths(
        size=10, max_depth=10, common=0, prefix_depth1=0, prefix_depth2=0,
        sep='/', labels=26, seed=None):
    """
    Returns two randomly created paths (as in directory structures) for use in
    testing and benchmarking :func:`maximum_common_path_embedding`.

    Parameters
    ----------
    size : int
        The number of independant random paths

    max_depth : int
        Maximum depth for the independant random paths

    common : int
        The number of shared common paths

    prefix_depth1: int
        Depth of the random prefix attacheded to first common paths

    prefix_depth2: int
        Depth of the random prefix attacheded to second common paths

    labels: int or collection
        Number of or collection of tokens that can be used as node labels

    sep: str
        path separator

    seed:
        Random state or seed

    Examples
    --------
    >>> from networkx.algorithms.embedding.tree_embedding import tree_to_seq
    >>> paths1, paths2 = random_paths(
    ...     size=3, max_depth=3, common=3,
    ...     prefix_depth1=3, prefix_depth2=3, labels=2 ** 16,
    ...     seed=0)
    >>> tree = paths_to_otree(paths1)
    >>> seq, open_to_close, node_to_open = tree_to_seq(tree, mode='chr')
    >>> seq, open_to_close, node_to_open = tree_to_seq(tree, mode='number')
    >>> import pprint
    >>> print('paths1 = {}'.format(pprint.pformat(paths1)))
    paths1 = ['brwb/eaaw/druy/ctge/dyaj/vcy',
     'brwb/eaaw/druy/dqbh/cqht',
     'brwb/eaaw/druy/plp',
     'dnfa/img',
     'dyxs/dacf',
     'ebwq/djxa']
    >>> print('paths2 = {}'.format(pprint.pformat(paths2)))
    paths2 = ['buug/befe/cjcq',
     'ccnj/bfum/cpbb',
     'ceps/nbn/cxp/ctge/dyaj/vcy',
     'ceps/nbn/cxp/dqbh/cqht',
     'ceps/nbn/cxp/plp',
     'twe']
    """
    from networkx.utils import create_py_random_state
    rng = create_py_random_state(seed)

    if isinstance(labels, int):
        alphabet = list(map(chr, range(ord('a'), ord('z'))))

        def random_label():
            digit = rng.randint(0, labels)
            label = _convert_digit_base(digit, alphabet)
            return label
    else:
        from functools import partial
        random_label = partial(rng.choice, labels)

    def random_path(rng, max_depth):
        depth = rng.randint(1, max_depth)
        parts = [str(random_label()) for _ in range(depth)]
        path = sep.join(parts)
        return path

    # These paths might be shared (but usually not)
    iid_paths1 = {random_path(rng, max_depth) for _ in range(size)}
    iid_paths2 = {random_path(rng, max_depth) for _ in range(size)}

    # These paths will be shared
    common_paths = {random_path(rng, max_depth) for _ in range(common)}

    if prefix_depth1 > 0:
        prefix1 = random_path(rng, prefix_depth1)
        common1 = {sep.join([prefix1, suff]) for suff in common_paths}
    else:
        common1 = common_paths

    if prefix_depth2 > 0:
        prefix2 = random_path(rng, prefix_depth2)
        common2 = {sep.join([prefix2, suff]) for suff in common_paths}
    else:
        common2 = common_paths

    paths1 = sorted(common1 | iid_paths1)
    paths2 = sorted(common2 | iid_paths2)

    return paths1, paths2


def _convert_digit_base(digit, alphabet):
    """
    Parameters
    ----------
    digit : int
        number in base 10 to convert

    alphabet : list
        symbols of the conversion base
    """
    baselen = len(alphabet)
    x = digit
    if x == 0:
        return alphabet[0]
    sign = 1 if x > 0 else -1
    x *= sign
    digits = []
    while x:
        digits.append(alphabet[x % baselen])
        x //= baselen
    if sign < 0:
        digits.append('-')
    digits.reverse()
    newbase_str = ''.join(digits)
    return newbase_str


def bench_maximum_common_path_embedding():
    """
    Runs algorithm benchmarks over a range of parameters.

    Running this benchmark does require some external libraries

    Requirements
    ------------
    timerit >= 0.3.0
    ubelt >= 0.9.2

    Command Line
    ------------
    xdoctest -m examples/applications/filesystem_embedding.py bench_maximum_common_path_embedding
    """
    import itertools as it
    import ubelt as ub
    import timerit
    from networkx.algorithms.string import balanced_sequence

    data_modes = []

    available_impls = balanced_sequence.available_impls_longest_common_balanced_sequence()

    # Define which implementations we are going to test
    run_basis = {
        'mode': [
            'chr',
            'number'
        ],
        'impl': available_impls,
    }

    # Define the properties of the random data we are going to test on
    data_basis = {
        'size': [20, 50],
        'max_depth': [8, 16],
        'common': [8, 16],
        'prefix_depth1': [0, 4],
        'prefix_depth2': [0, 4],
        # 'labels': [26 ** 1, 26 ** 8]
        'labels': [1, 26]
    }

    # run_basis['impl'] = set(run_basis['impl']) & {
    #     'iter-cython',
    #     'iter',
    # }

    # TODO: parametarize demo names
    # BENCH_MODE = None
    # BENCH_MODE = 'small'
    BENCH_MODE = 'medium'
    # BENCH_MODE = 'large'

    if BENCH_MODE == 'small':
        data_basis = {
            'size': [30],
            'max_depth': [8, 2],
            'common': [2, 8],
            'prefix_depth1': [0, 4],
            'prefix_depth2': [0],
            'labels': [4]
        }
        run_basis['impl'] = ub.oset(available_impls) - {'recurse'}
        run_basis['mode'] = ['number', 'chr']
        # runparam_to_time = {
        #     ('chr', 'iter-cython')       : {'mean': 0.036, 'max': 0.094},
        #     ('chr', 'iter')              : {'mean': 0.049, 'max': 0.125},
        #     ('number', 'iter-cython')    : {'mean': 0.133, 'max': 0.363},
        #     ('number', 'iter')           : {'mean': 0.149, 'max': 0.408},
        # }

    if BENCH_MODE == 'medium':
        data_basis = {
            'size': [30, 40],
            'max_depth': [4, 8],
            'common': [8, 50],
            'prefix_depth1': [0, 4],
            'prefix_depth2': [2],
            'labels': [8, 1]
        }
        # Results
        # runparam_to_time = {
        #     ('chr', 'iter-cython')    : {'mean': 0.112, 'max': 0.467},
        #     ('chr', 'iter')           : {'mean': 0.155, 'max': 0.661},
        # }

    if BENCH_MODE == 'large':
        data_basis = {
            'size': [30, 40],
            'max_depth': [4, 12],  # 64000
            'common': [8, 32],
            'prefix_depth1': [0, 4],
            'prefix_depth2': [2],
            'labels': [8]
        }
        run_basis['impl'] = available_impls
        # runparam_to_time = {
        #     ('chr', 'iter-cython')    : {'mean': 0.282, 'max': 0.923},
        #     ('chr', 'iter')           : {'mean': 0.409, 'max': 1.328},
        # }

    elif BENCH_MODE == 'too-big':
        data_basis = {
            'size': [100],
            'max_depth': [8],
            'common': [80],
            'prefix_depth1': [4],
            'prefix_depth2': [2],
            'labels': [8]
        }

    data_modes = [
        dict(zip(data_basis.keys(), vals))
        for vals in it.product(*data_basis.values())]
    run_modes = [
        dict(zip(run_basis.keys(), vals))
        for vals in it.product(*run_basis.values())]

    print('len(data_modes) = {!r}'.format(len(data_modes)))
    print('len(run_modes) = {!r}'.format(len(run_modes)))
    print('total = {}'.format(len(data_modes) * len(run_modes)))

    seed = 0
    for idx, datakw in enumerate(data_modes):
        print('datakw = {}'.format(ub.repr2(datakw, nl=1)))
        _datakw = ub.dict_diff(datakw, {'complexity'})
        paths1, paths2 = random_paths(seed=seed, **_datakw)
        tree1 = paths_to_otree(paths1)
        tree2 = paths_to_otree(paths2)
        stats1 = {
            'npaths': len(paths1),
            'n_nodes': len(tree1.nodes),
            'n_edges': len(tree1.edges),
            'n_leafs': len([n for n in tree1.nodes if len(tree1.succ[n]) == 0]),
            'depth': max(len(p.split('/')) for p in paths1),
        }
        stats2 = {
            'npaths': len(paths2),
            'n_nodes': len(tree2.nodes),
            'n_edges': len(tree2.edges),
            'n_leafs': len([n for n in tree2.nodes if len(tree2.succ[n]) == 0]),
            'depth': max(len(p.split('/')) for p in paths2),
        }
        complexity = (
            stats1['n_nodes'] * min(stats1['n_leafs'], stats1['depth']) *
            stats2['n_nodes'] * min(stats2['n_leafs'], stats2['depth'])) ** .25

        datakw['complexity'] = complexity
        print('datakw = {}'.format(ub.repr2(datakw, nl=0, precision=2)))

        print('stats1 = {}'.format(ub.repr2(stats1, nl=0)))
        print('stats2 = {}'.format(ub.repr2(stats2, nl=0)))

    total = len(data_modes) * len(run_modes)
    print('len(data_modes) = {!r}'.format(len(data_modes)))
    print('len(run_modes) = {!r}'.format(len(run_modes)))
    print('total = {!r}'.format(total))
    seed = 0

    prog = ub.ProgIter(total=total, verbose=3)
    prog.begin()
    results = []
    ti = timerit.Timerit(1, bestof=1, verbose=1, unit='s')
    for datakw in data_modes:
        _datakw = ub.dict_diff(datakw, {'complexity'})
        paths1, paths2 = random_paths(seed=seed, **_datakw)
        print('---')
        prog.step(4)
        tree1 = paths_to_otree(paths1)
        tree2 = paths_to_otree(paths2)
        stats1 = {
            'npaths': len(paths1),
            'n_nodes': len(tree1.nodes),
            'n_edges': len(tree1.edges),
            'n_leafs': len([n for n in tree1.nodes if len(tree1.succ[n]) == 0]),
            'depth': max(len(p.split('/')) for p in paths1),
        }
        stats2 = {
            'npaths': len(paths2),
            'n_nodes': len(tree2.nodes),
            'n_edges': len(tree2.edges),
            'n_leafs': len([n for n in tree2.nodes if len(tree2.succ[n]) == 0]),
            'depth': max(len(p.split('/')) for p in paths2),
        }
        complexity = (
            stats1['n_nodes'] * min(stats1['n_leafs'], stats1['depth']) *
            stats2['n_nodes'] * min(stats2['n_leafs'], stats2['depth'])) ** .25

        datakw['complexity'] = complexity
        print('datakw = {}'.format(ub.repr2(datakw, nl=0, precision=2)))

        if True:
            # idx + 4 > len(data_modes):
            print('stats1 = {}'.format(ub.repr2(stats1, nl=0)))
            print('stats2 = {}'.format(ub.repr2(stats2, nl=0)))
        for runkw in run_modes:
            paramkw = {**datakw, **runkw}
            run_key = ub.repr2(
                paramkw, sep='', itemsep='', kvsep='',
                explicit=1, nobr=1, nl=0, precision=1)
            try:
                for timer in ti.reset(run_key):
                    with timer:
                        maximum_common_path_embedding(paths1, paths2, **runkw)
            except RecursionError as ex:
                print('ex = {!r}'.format(ex))
                row = paramkw.copy()
                row['time'] = float('nan')
            else:
                row = paramkw.copy()
                row['time'] = ti.min()
            results.append(row)
    prog.end()

    print(ub.repr2(
        ub.sorted_vals(ti.measures['min']), nl=1, align=':', precision=6))

    import pandas as pd
    import kwarray
    df = pd.DataFrame.from_dict(results)

    dataparam_to_time = {}
    for mode, subdf in df.groupby(['complexity'] + list(data_basis.keys())):
        stats = kwarray.stats_dict(subdf['time'])
        stats.pop('min', None)
        stats.pop('std', None)
        stats.pop('shape', None)
        dataparam_to_time[mode] = stats
    dataparam_to_time = ub.sorted_vals(dataparam_to_time, key=lambda x: x['max'])
    print('dataparam_to_time = {}'.format(
        ub.repr2(dataparam_to_time, nl=1, precision=3, align=':')))
    print(list(data_basis.keys()))

    runparam_to_time = {}
    for mode, subdf in df.groupby(['mode', 'impl']):
        stats = kwarray.stats_dict(subdf['time'])
        stats.pop('min', None)
        stats.pop('std', None)
        stats.pop('shape', None)
        runparam_to_time[mode] = stats
    runparam_to_time = ub.sorted_vals(runparam_to_time, key=lambda x: x['max'])
    print('runparam_to_time = {}'.format(
        ub.repr2(runparam_to_time, nl=1, precision=3, align=':')))


def allsame(iterable, eq=operator.eq):
    """
    Determine if all items in a sequence are the same

    Args:
        iterable (Iterable[A]):
            items to determine if they are all the same

        eq (Callable[[A, A], bool], default=operator.eq):
            function used to test for equality

    Returns:
        bool: True if all items are equal, otherwise False

    Example:
        >>> allsame([1, 1, 1, 1])
        True
        >>> allsame([])
        True
        >>> allsame([0, 1])
        False
        >>> iterable = iter([0, 1, 1, 1])
        >>> _ = next(iterable)
        >>> allsame(iterable)
        True
        >>> allsame(range(10))
        False
        >>> allsame(range(10), lambda a, b: True)
        True
    """
    iter_ = iter(iterable)
    try:
        first = next(iter_)
    except StopIteration:
        return True
    return all(eq(first, item) for item in iter_)


# -- tests


def test_not_compatable():
    paths1 = [
        'foo/bar'
    ]
    paths2 = [
        'baz/biz'
    ]
    embedding1, embedding2 = maximum_common_path_embedding(paths1, paths2)
    assert len(embedding1) == 0
    assert len(embedding2) == 0


def test_compatable():
    paths1 = [
        'root/suffix1'
    ]
    paths2 = [
        'root/suffix2'
    ]
    embedding1, embedding2 = maximum_common_path_embedding(paths1, paths2)
    assert embedding1 == ['root']
    assert embedding2 == ['root']

    paths1 = [
        'root/suffix1'
    ]
    paths2 = [
        'root'
    ]
    embedding1, embedding2 = maximum_common_path_embedding(paths1, paths2)
    assert embedding1 == ['root']
    assert embedding2 == ['root']


def test_prefixed():
    paths1 = [
        'prefix1/root/suffix1'
    ]
    paths2 = [
        'root/suffix2'
    ]
    embedding1, embedding2 = maximum_common_path_embedding(paths1, paths2)
    assert embedding1 == ['prefix1/root']
    assert embedding2 == ['root']

    paths1 = [
        'prefix1/root/suffix1'
    ]
    paths2 = [
        'prefix1/root/suffix2'
    ]
    embedding1, embedding2 = maximum_common_path_embedding(paths1, paths2)
    assert embedding1 == ['prefix1/root']
    assert embedding2 == ['prefix1/root']


def test_simple1():
    paths1 = [
        'root/file1',
        'root/file2',
        'root/file3',
    ]
    paths2 = [
        'prefix1/root/file1',
        'prefix1/root/file2',
        'root/file3',
    ]
    embedding1, embedding2 = maximum_common_path_embedding(paths1, paths2)
    assert embedding1 == paths1
    assert embedding2 == paths2

    paths1 = [
        'root/file1',
        'root/file2',
        'root/file3',
    ]
    paths2 = [
        'prefix1/root/file1',
        'prefix1/root/file2',
        'prefix2/root/file3',
        'prefix2/root/file4',
    ]
    embedding1, embedding2 = maximum_common_path_embedding(paths1, paths2)
    assert embedding1 == paths1


def test_random1():
    paths1, paths2 = random_paths(10, seed=321)
    embedding1, embedding2 = maximum_common_path_embedding(paths1, paths2)


def _demodata_resnet_module_state(arch):
    """
    Construct paths corresponding to resnet convnet state keys to
    simulate a real world use case for path-embeddings.

    Ignore
    ------
    # Check to make sure the demodata agrees with real data
    import torchvision
    paths_true = list(torchvision.models.resnet50().state_dict().keys())
    paths_demo = _demodata_resnet_module_state('resnet50')
    print(ub.hzcat([ub.repr2(paths_true, nl=2), ub.repr2(paths_demo)]))
    assert paths_demo == paths_true

    paths_true = list(torchvision.models.resnet18().state_dict().keys())
    paths_demo = _demodata_resnet_module_state('resnet18')
    print(ub.hzcat([ub.repr2(paths_true, nl=2), ub.repr2(paths_demo)]))
    assert paths_demo == paths_true

    paths_true = list(torchvision.models.resnet152().state_dict().keys())
    paths_demo = _demodata_resnet_module_state('resnet152')
    print(ub.hzcat([ub.repr2(paths_true, nl=2), ub.repr2(paths_demo)]))
    assert paths_demo == paths_true
    """
    if arch == 'resnet18':
        block_type = 'basic'
        layer_blocks = [2, 2, 2, 2]
    elif arch == 'resnet50':
        block_type = 'bottleneck'
        layer_blocks = [3, 4, 6, 3]
    elif arch == 'resnet152':
        block_type = 'bottleneck'
        layer_blocks = [3, 8, 36, 3]
    else:
        raise KeyError(arch)
    paths = []
    paths += [
        'conv1.weight',
        'bn1.weight',
        'bn1.bias',
        'bn1.running_mean',
        'bn1.running_var',
        'bn1.num_batches_tracked',
    ]
    if block_type == 'bottleneck':
        num_convs = 3
    elif block_type == 'basic':
        num_convs = 2
    else:
        raise KeyError(block_type)

    for layer_idx, nblocks in enumerate(layer_blocks, start=1):
        for block_idx in range(0, nblocks):
            prefix = 'layer{}.{}.'.format(layer_idx, block_idx)

            for conv_idx in range(1, num_convs + 1):
                paths += [
                    prefix + 'conv{}.weight'.format(conv_idx),
                    prefix + 'bn{}.weight'.format(conv_idx),
                    prefix + 'bn{}.bias'.format(conv_idx),
                    prefix + 'bn{}.running_mean'.format(conv_idx),
                    prefix + 'bn{}.running_var'.format(conv_idx),
                    prefix + 'bn{}.num_batches_tracked'.format(conv_idx),
                ]
            if block_idx == 0 and layer_idx > 0:
                if block_type != 'basic' or layer_idx > 1:
                    paths += [
                        prefix + 'downsample.0.weight',
                        prefix + 'downsample.1.weight',
                        prefix + 'downsample.1.bias',
                        prefix + 'downsample.1.running_mean',
                        prefix + 'downsample.1.running_var',
                        prefix + 'downsample.1.num_batches_tracked',
                    ]
    paths += [
        'fc.weight',
        'fc.bias',
    ]
    return paths


def test_realworld_case1():
    """
    import torchvision
    paths1 = list(torchvision.models.resnet50().state_dict().keys())

    print(ub.hzcat(['paths1 = {}'.format(ub.repr2(paths1, nl=2)), ub.repr2(paths)]))
    len(paths1)
    """
    # times: resnet18:  0.16 seconds
    # times: resnet50:  0.93 seconds
    # times: resnet152: 9.83 seconds
    paths1 = _demodata_resnet_module_state('resnet50')
    paths2 = ['module.' + p for p in paths1]

    embedding1, embedding2 = maximum_common_path_embedding(
            paths1, paths2, sep='.')
    assert [p[len('module.'):] for p in embedding2] == embedding1


def test_realworld_case2():
    """
    Ignore
    ------
    import torchvision
    paths1 = list(torchvision.models.resnet152().state_dict().keys())
    print('paths1 = {}'.format(ub.repr2(paths1, nl=2)))
    """
    backbone = _demodata_resnet_module_state('resnet18')

    # Detector strips of prefix and suffix of the backbone net
    subpaths = ['detector.' + p for p in backbone[6:-2]]
    paths1 = [
        'detector.conv1.weight',
        'detector.bn1.weight',
        'detector.bn1.bias',
    ] + subpaths + [
        'detector.head1.conv1.weight',
        'detector.head1.conv2.weight',
        'detector.head1.conv3.weight',
        'detector.head1.fc.weight',
        'detector.head1.fc.bias',
        'detector.head2.conv1.weight',
        'detector.head2.conv2.weight',
        'detector.head2.conv3.weight',
        'detector.head2.fc.weight',
        'detector.head2.fc.bias',
    ]

    paths2 = ['module.' + p for p in backbone]

    embedding1, embedding2 = maximum_common_path_embedding(
            paths1, paths2, sep='.')

    mapping = dict(zip(embedding1, embedding2))

    # Note in the embedding case there may be superfluous assignments
    # but they can either be discarded in post-processing or they wont
    # be in the solution if we use isomorphisms instead of embeddings
    assert len(subpaths) < len(mapping), (
        'all subpaths should be in the mapping')

    non_common1 = set(paths1) - set(embedding1)
    non_common2 = set(paths2) - set(embedding2)

    assert non_common2 == {
            'module.bn1.num_batches_tracked',
            'module.bn1.running_mean',
            'module.bn1.running_var',
            }

    assert non_common1 == {
        'detector.conv1.weight',
        'detector.head1.conv1.weight',
        'detector.head1.conv2.weight',
        'detector.head1.conv3.weight',
        'detector.head1.fc.bias',
        'detector.head1.fc.weight',
        'detector.head2.conv2.weight',
        'detector.head2.conv3.weight',
    }
