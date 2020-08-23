

def benchmarks():
    """
    xdoctest ~/code/networkx/networkx/algorithms/isomorphism/balanced_sequence.py benchmarks --profile
    """

    data_modes = []

    data_basis = {
        'size': [100, 2, 8, 20],
        'max_depth': [8, 2, 20],
        'common': [2, 8, 20],
        'prefix_depth1': [0, 1, 4, 8],
        'prefix_depth2': [0, 1, 4, 8],
        # 'labels': [26 ** 1, 26 ** 8]
        'labels': [1, 2]
    }

    import itertools as it
    data_modes = [
        dict(zip(data_basis.keys(), vals))
        for vals in it.product(*data_basis.values())]

    # data_modes += [
    #     {'size': size, 'max_depth': max_depth}
    #     for size in [10, 50, 100]
    #     for max_depth in [1, 3, 5, 7, 9]
    # ]

    # data_modes += [
    #     {'size': size, 'max_depth': max_depth}
    #     for size in [200, 400]
    #     for max_depth in [1, 3]
    # ]
    import ubelt as ub
    import timerit
    ti = timerit.Timerit(1, bestof=1, verbose=1, unit='s')

    impls = [
        # 'iter-alt2-cython',
        'iter-prehash2-cython',
        'iter-prehash2',
        # 'iter-alt2',
        # 'iter',
        'recurse',
    ]

    run_modes = [
        {'impl': impl, 'mode': mode}
        for impl in impls
        for mode in [
            # 'chr',
            # 'tuple',
            'number'
        ]
    ]

    for datakw in data_modes:
        data_key = ub.repr2(datakw, sep='', itemsep='', kvsep='', explicit=1,
                            nobr=1, nl=0)
        # paths1, paths2 = simple_sequences(**datakw)
        paths1, paths2 = random_paths(seed=0, **datakw)
        # paths1, paths2 = random_paths(seed=None, **datakw)

        print('---')
        for runkw in run_modes:
            # if runkw['impl'] == 'iter-alt2' and runkw['mode'] == 'number':
            #     continue
            # if runkw['impl'] == 'iter' and runkw['mode'] == 'number':
            #     continue
            run_key = ub.repr2(runkw, sep='', itemsep='', kvsep='', explicit=1,
                                nobr=1, nl=0)
            key = '{},{}'.format(data_key, run_key)
            for timer in ti.reset(key):
                with timer:
                    try:
                        maximum_common_path_embedding(paths1, paths2, **runkw)
                    except RecursionError as ex:
                        print('ex = {!r}'.format(ex))

    print(ub.repr2(ub.sorted_vals(ti.measures['min']), nl=1, align=':', precision=6))
