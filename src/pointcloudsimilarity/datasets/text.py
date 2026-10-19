"""
Texts from Hugging Face datasets, sampled reproducibly.
"""

import numpy as np

from ._optional import import_optional

__all__ = ['load_text_dataset']


def load_text_dataset(
    name,
    split='train',
    text_column='text',
    label_column=None,
    n_samples=1000,
    seed=0,
    config=None,
    streaming=False,
    pool_size=None,
    max_chars=None,
):
    """
    Draw n_samples texts, without replacement, from a Hugging Face dataset.

    Parameters:
    name (str): Dataset on the Hugging Face Hub, e.g. 'imdb'.
    split (str): Dataset split.
    text_column (str): Column holding the texts.
    label_column (str): Column holding the labels, or None to skip them.
    n_samples (int): Number of texts. If the pool has no more rows than this,
        all of them are returned, in order.
    seed (int): Seed of np.random.default_rng, which draws the sample.
    config (str): Dataset configuration, e.g. '20231101.en' for Wikipedia.
    streaming (bool): Stream the dataset instead of downloading all of it.
    pool_size (int): Sample from the first pool_size rows only. With
        streaming=True, the rest of the dataset is never read.
    max_chars (int): Truncate each text to this many characters.

    Returns:
    tuple: The texts (list of str) and their labels (list), or None as labels
    if label_column is None.
    """
    datasets = import_optional('datasets', 'load_text_dataset')
    ds = datasets.load_dataset(name, config, split=split, streaming=streaming)
    if pool_size is not None:
        if not streaming:
            # Unlike a streamed one, a downloaded dataset rejects taking more
            # rows than it has.
            pool_size = min(pool_size, len(ds))
        ds = ds.take(pool_size)

    texts, labels = [], []
    for row in ds:
        texts.append(row[text_column][:max_chars])
        if label_column is not None:
            labels.append(row[label_column])

    if len(texts) > n_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(texts), size=n_samples, replace=False)
        texts = [texts[i] for i in idx]
        if label_column is not None:
            labels = [labels[i] for i in idx]

    if label_column is None:
        labels = None
    return texts, labels
