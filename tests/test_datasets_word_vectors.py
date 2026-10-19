"""
The GloVe loaders against the v1.0 notebook code they replaced.

The models here are small stand-ins built in memory: 200,000 words like the
real vocabularies, with tied counts and non-ASCII words, but few dimensions.
"""

import pickle
import sys

import numpy as np
import pytest

gensim = pytest.importorskip('gensim')
gensim_downloader = pytest.importorskip('gensim.downloader')

from gensim.models import KeyedVectors  # noqa: E402

from pointcloudsimilarity.datasets import (  # noqa: E402
    aligned_glove_vectors,
    load_glove_twitter,
)

N_WORDS = 200_000


def _word(i):
    if i % 31 == 0:
        return f'wörd{i}'
    if i % 97 == 0:
        return f'emoji\U0001f600{i}'
    return f'w{i}'


@pytest.fixture(scope='module')
def models():
    """Stand-ins for the GloVe Twitter models, keyed by dimension."""
    rng = np.random.default_rng(0)
    words = [_word(i) for i in range(N_WORDS)]
    # Few distinct counts, in no particular order: many ties to break.
    counts = rng.integers(1, 1000, size=N_WORDS)
    models = {}
    for dim, size in [(200, 4), (100, 3), (50, 2), (25, 1)]:
        kv = KeyedVectors(vector_size=size)
        kv.add_vectors(words, rng.standard_normal((N_WORDS, size)).astype(np.float32))
        for word, count in zip(words, counts):
            kv.set_vecattr(word, 'count', int(count))
        models[dim] = kv
    return models


# v1.0 notebooks/01d-w2v_glove.py; reads the cache 00-load_gensim_glove.py wrote.
def _v1_glove_extract():
    with open('glove_models.pkl', 'rb') as f:
        models = pickle.load(f)
    model200 = models['model200']
    model100 = models['model100']
    model50 = models['model50']
    model25 = models['model25']
    print('Models loaded from glove_models.pkl')

    model100.sort_by_descending_frequency()
    all_vocab = list(model100.key_to_index)
    # all_vocab = [ w for w in all_vocab if w.isalnum() ]
    all_vocab = [w for w in all_vocab if w.isascii()]

    for w in all_vocab[190000:191000]:
        v200 = model200[w]
        v100 = model100[w]
        v50 = model50[w]
        v25 = model25[w]
        yield w, v200, v100, v50, v25


def test_matches_v1_glove_extract(models, tmp_path, monkeypatch):
    with open(tmp_path / 'glove_models.pkl', 'wb') as f:
        pickle.dump({f'model{dim}': kv for dim, kv in models.items()}, f)
    monkeypatch.chdir(tmp_path)
    w, v200, v100, v50, v25 = zip(*_v1_glove_extract())

    vocab_before = list(models[100].index_to_key)
    words, vectors = aligned_glove_vectors(
        models, start=190000, n_words=1000, reference=100
    )

    assert words == list(w)
    for dim, expected in [(200, v200), (100, v100), (50, v50), (25, v25)]:
        np.testing.assert_array_equal(vectors[dim], np.array(expected))
        assert vectors[dim].dtype == np.float32
    # v1.0 sorted model100 in place; the library leaves the models alone.
    assert models[100].index_to_key == vocab_before


def test_words_follow_the_reference_frequencies(models):
    words, vectors = aligned_glove_vectors(models, n_words=500)
    counts = [models[200].get_vecattr(word, 'count') for word in words]
    assert counts == sorted(counts, reverse=True)
    assert all(word.isascii() for word in words)
    assert {dim: v.shape for dim, v in vectors.items()} == {
        200: (500, 4),
        100: (500, 3),
        50: (500, 2),
        25: (500, 1),
    }
    np.testing.assert_array_equal(vectors[50][7], models[50][words[7]])


@pytest.fixture
def downloader(monkeypatch):
    """Replace gensim's downloader; return the names it is asked for."""
    calls = []

    def load(name):
        calls.append(name)
        kv = KeyedVectors(vector_size=3)
        kv.add_vectors(['a', 'b', 'c'], np.arange(9, dtype=np.float32).reshape(3, 3))
        return kv

    monkeypatch.setattr(gensim_downloader, 'load', load)
    return calls


def test_cache_dir_saves_the_parsed_model(downloader, tmp_path):
    first = load_glove_twitter(50, cache_dir=tmp_path / 'cache')
    second = load_glove_twitter(50, cache_dir=tmp_path / 'cache')
    assert downloader == ['glove-twitter-50']
    assert second.index_to_key == first.index_to_key
    np.testing.assert_array_equal(second.vectors, first.vectors)


def test_without_cache_dir_every_call_parses(downloader):
    load_glove_twitter(25)
    load_glove_twitter(25)
    assert downloader == ['glove-twitter-25', 'glove-twitter-25']


def test_unknown_dims_are_rejected():
    with pytest.raises(ValueError, match='25, 50, 100, 200'):
        load_glove_twitter(300)


def test_missing_gensim_explains_install(monkeypatch):
    for module in ['gensim', 'gensim.downloader', 'gensim.models']:
        monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ImportError, match="'gensim' package.*uv sync --extra data"):
        load_glove_twitter(25)
