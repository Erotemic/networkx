from networkx.algorithms.isomorphism._embeddinghelpers.path_embedding import maximum_common_path_embedding
from networkx.algorithms.isomorphism._embeddinghelpers.demodata import random_paths


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
    print('embedding1 = {}'.format(ub.repr2(embedding1, nl=1)))
    print('embedding2 = {}'.format(ub.repr2(embedding2, nl=1)))
    assert embedding1 == paths1
    assert embedding2 == paths2


def test_random1():
    paths1, paths2 = random_paths(10, seed=321)
    print('paths1 = {}'.format(ub.repr2(paths1, nl=1)))
    print('paths2 = {}'.format(ub.repr2(paths2, nl=1)))
    embedding1, embedding2 = maximum_common_path_embedding(paths1, paths2)
    print('embedding1 = {}'.format(ub.repr2(embedding1, nl=1)))
    print('embedding2 = {}'.format(ub.repr2(embedding2, nl=1)))
