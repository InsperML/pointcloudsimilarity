"""
GloVe Twitter word vectors, as aligned point clouds of the same words.
"""

from pathlib import Path

import numpy as np

__all__ = ['GLOVE_TWITTER_DIMS', 'aligned_glove_vectors', 'load_glove_twitter']

GLOVE_TWITTER_DIMS = (25, 50, 100, 200)


def load_glove_twitter(dim, cache_dir=None):
    """
    GloVe Twitter word vectors of one dimensionality, from gensim's downloader.

    The first call downloads the model into ~/gensim-data and parses it, which
    takes minutes for the larger models. With cache_dir, the parsed model is
    saved there in gensim's format and later calls memory-map it instead.

    Parameters:
    dim (int): 25, 50, 100 or 200.
    cache_dir (str or Path): Directory for parsed models, or None to parse the
        download on every call.

    Returns:
    gensim.models.KeyedVectors: The word vectors, read-only if from cache_dir.
    """
    if dim not in GLOVE_TWITTER_DIMS:
        raise ValueError(f'dim must be one of {GLOVE_TWITTER_DIMS}, got {dim}.')
    name = f'glove-twitter-{dim}'
    path = None if cache_dir is None else Path(cache_dir) / f'{name}.kv'
    if path is not None and path.exists():
        from gensim import models

        return models.KeyedVectors.load(str(path), mmap='r')

    from gensim import downloader

    vectors = downloader.load(name)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        vectors.save(str(path))
    return vectors


def aligned_glove_vectors(models, start=0, n_words=1000, reference=None):
    """
    The vectors of the same words in several models: one point cloud each.

    Words are ranked by descending frequency in the reference model, non-ASCII
    words are dropped, and the n_words from rank start on are kept.

    Parameters:
    models (dict): gensim KeyedVectors by key, e.g. {25: ..., 200: ...}.
    start (int): Rank of the first word.
    n_words (int): Number of words.
    reference: Key of the model whose frequencies rank the words. Defaults to
        the first key.

    Returns:
    tuple: The words (list of str), and a dict mapping each key of models to
    an array of shape (n_words, vector size) whose rows follow the words.
    """
    ranking = models[next(iter(models)) if reference is None else reference]
    counts = np.array(
        [ranking.get_vecattr(word, 'count') for word in ranking.index_to_key],
        dtype=np.int64,
    )
    indices = np.arange(len(counts))
    # The order KeyedVectors.sort_by_descending_frequency() gives, without
    # reordering the model in place.
    by_frequency = np.lexsort((-indices, -counts))
    vocab = [ranking.index_to_key[i] for i in by_frequency]
    words = [word for word in vocab if word.isascii()][start : start + n_words]
    vectors = {
        key: np.array([model[word] for word in words]) for key, model in models.items()
    }
    return words, vectors
