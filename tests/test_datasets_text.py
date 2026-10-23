"""
load_text_dataset against the v1.0 notebook code it replaced.

Both run on the same fake Hub: datasets.load_dataset is replaced by a function
serving a small in-memory table, so no network is needed. The _v1_* functions
are copied from v1.0 and changed only to call datasets.load_dataset through
the module, so that the fake applies to them too.
"""

import numpy as np
import pytest

datasets = pytest.importorskip('datasets')

from pointcloudsimilarity.datasets import load_text_dataset  # noqa: E402

N_ROWS = 3000


@pytest.fixture
def hub(monkeypatch):
    """Replace datasets.load_dataset; return the list of calls it receives."""
    table = {
        'text': [f'text {i} ' + 'word ' * (i % 97) for i in range(N_ROWS)],
        'sentence': [f'sentence {i} ' + 'x ' * (i % 151) for i in range(N_ROWS)],
        'label': [i % 3 for i in range(N_ROWS)],
    }
    calls = []

    def load_dataset(path, name=None, split=None, streaming=False):
        calls.append((path, name, split, streaming))
        ds = datasets.Dataset.from_dict(table)
        return ds.to_iterable_dataset() if streaming else ds

    monkeypatch.setattr(datasets, 'load_dataset', load_dataset)
    return calls


# v1.0 notebooks/ijcnn/02-finetuning.py
def _v1_get_dataset(
    dataset_name, split='train', column_X='text', column_Y='label', n_samples=1000
):
    print(f'Loading dataset {dataset_name}...')
    ds = datasets.load_dataset(dataset_name, split=split, streaming=False)
    print(f'Downloaded dataset {dataset_name} with {len(ds)} samples.')
    texts = [article[column_X] for article in ds]
    ans = [article[column_Y] for article in ds]

    rng = np.random.default_rng(12345)
    print(f'Loaded {len(texts)} samples.')
    if len(texts) > n_samples:
        idx = rng.choice(len(texts), size=n_samples, replace=False)
        texts = [texts[i] for i in idx]
        ans = [ans[i] for i in idx]
    else:
        texts = texts[:n_samples]
        ans = ans[:n_samples]
    return texts, ans


# v1.0 notebooks/01c-bert_vs_distillbert.py, model_vs_model_experiment()
def _v1_01c_texts(N_SAMPLES):
    ds = datasets.load_dataset(
        'wikimedia/wikipedia', '20231101.en', split='train', streaming=True
    )
    texts = []
    for i, article in enumerate(ds.take(N_SAMPLES * 10)):
        texts.append(article['text'])

    # Randomly select N_SAMPLES texts
    rng = np.random.default_rng(1234)
    if len(texts) > N_SAMPLES:
        idx = rng.choice(len(texts), size=N_SAMPLES, replace=False)
        texts = [texts[i] for i in idx]
    else:
        texts = texts[:N_SAMPLES]
    return texts


# v1.0 notebooks/01e-bert_initialization_and_finetuning.py, model_vs_model_experiment()
def _v1_01e_texts(N_SAMPLES):
    ds = datasets.load_dataset('sst2', split='train', streaming=False)
    texts = [article['sentence'] for article in ds]
    # Randomly select N_SAMPLES texts
    rng = np.random.default_rng(1234)
    if len(texts) > N_SAMPLES:
        idx = rng.choice(len(texts), size=N_SAMPLES, replace=False)
        texts = [texts[i] for i in idx]
    else:
        texts = texts[:N_SAMPLES]
    return texts


# v1.0 notebooks/01f-gpts.py, model_vs_model_experiment()
def _v1_01f_texts(N_SAMPLES):
    ds = datasets.load_dataset('sst2', split='train', streaming=False)
    texts = []
    for i, article in enumerate(ds.take(N_SAMPLES * 10)):
        texts.append(article['sentence'][:200])
    # Randomly select N_SAMPLES texts
    rng = np.random.default_rng(1234)
    if len(texts) > N_SAMPLES:
        idx = rng.choice(len(texts), size=N_SAMPLES, replace=False)
        texts = [texts[i] for i in idx]
    else:
        texts = texts[:N_SAMPLES]
    return texts


# v1.0 notebooks/04-quantization.py, model_vs_quantized_model_similarities()
def _v1_04_texts(N_SAMPLES):
    ds = datasets.load_dataset('imdb', split='train', streaming=False)
    texts = [article['text'] for article in ds]
    # Randomly select N_SAMPLES texts
    rng = np.random.default_rng(1234)
    if len(texts) > N_SAMPLES:
        idx = rng.choice(len(texts), size=N_SAMPLES, replace=False)
        texts = [texts[i] for i in idx]
    else:
        texts = texts[:N_SAMPLES]
    return texts


# n_samples above and below N_ROWS: sampled, and everything in order.
@pytest.mark.parametrize('n_samples', [2000, 5000])
@pytest.mark.parametrize('split', ['test', 'train'])
def test_matches_ijcnn_02_get_dataset(hub, n_samples, split):
    expected = _v1_get_dataset('sst2', split, 'sentence', 'label', n_samples)
    actual = load_text_dataset(
        'sst2',
        split=split,
        text_column='sentence',
        label_column='label',
        n_samples=n_samples,
        seed=12345,
    )
    assert actual == expected
    assert hub[0] == hub[1]


# Pools of 10 * N_SAMPLES rows: 1000 (part of the table) and 5000 (all of it).
@pytest.mark.parametrize('N_SAMPLES', [100, 500])
def test_matches_01c_wikipedia(hub, N_SAMPLES):
    texts, labels = load_text_dataset(
        'wikimedia/wikipedia',
        config='20231101.en',
        streaming=True,
        pool_size=N_SAMPLES * 10,
        n_samples=N_SAMPLES,
        seed=1234,
    )
    assert texts == _v1_01c_texts(N_SAMPLES)
    assert labels is None
    assert hub[0] == hub[1] == ('wikimedia/wikipedia', '20231101.en', 'train', True)


def test_matches_01e_sst2(hub):
    texts, _ = load_text_dataset(
        'sst2', text_column='sentence', n_samples=1000, seed=1234
    )
    assert texts == _v1_01e_texts(1000)
    assert hub[0] == hub[1]


# v1.0 fails if 10 * N_SAMPLES exceeds the rows of a downloaded dataset.
@pytest.mark.parametrize('N_SAMPLES', [100, 250])
def test_matches_01f_truncated_sst2(hub, N_SAMPLES):
    texts, _ = load_text_dataset(
        'sst2',
        text_column='sentence',
        n_samples=N_SAMPLES,
        seed=1234,
        pool_size=N_SAMPLES * 10,
        max_chars=200,
    )
    expected = _v1_01f_texts(N_SAMPLES)
    assert texts == expected
    assert max(len(t) for t in expected) == 200
    assert hub[0] == hub[1]


def test_matches_04_imdb(hub):
    texts, _ = load_text_dataset('imdb', n_samples=1000, seed=1234)
    assert texts == _v1_04_texts(1000)
    assert hub[0] == hub[1]


@pytest.mark.parametrize('streaming', [False, True])
def test_pool_larger_than_the_dataset_takes_it_all(hub, streaming):
    texts, _ = load_text_dataset(
        'x', n_samples=N_ROWS, pool_size=2 * N_ROWS, streaming=streaming
    )
    assert texts == load_text_dataset('x', n_samples=N_ROWS)[0]
    assert len(texts) == N_ROWS


def test_labels_follow_their_texts(hub):
    texts, labels = load_text_dataset(
        'x', text_column='sentence', label_column='label', n_samples=50
    )
    assert [int(t.split()[1]) % 3 for t in texts] == labels


def test_seed_sets_the_sample(hub):
    first = load_text_dataset('x', n_samples=50, seed=1)
    assert load_text_dataset('x', n_samples=50, seed=1) == first
    assert load_text_dataset('x', n_samples=50, seed=2) != first
