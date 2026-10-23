"""
load_fma against the v1.0 vae.py code it replaced, on a small fake FMA directory.
"""

import sys
from functools import partial

import numpy as np
import pytest
import torch

fma = pytest.importorskip('insperdatasets.audio.fma')
audio_utils = pytest.importorskip('insperdatasets.audio.utils')
sf = pytest.importorskip('soundfile')

from pointcloudsimilarity.datasets import load_fma  # noqa: E402

TRACK_IDS = [2, 3, 5, 10, 20, 26, 30, 46, 48, 134, 1001, 1002, 2005]
TRACK_IDS += [2010, 3000, 3001, 3002, 3003, 4040, 5050, 6060, 7070, 8080, 9090, 10010]
MISSING = {3, 26, 1002, 3001}  # tracks without a wav file, as in a partial download
EMPTY = {48, 4040}  # empty wav files, as a failed conversion leaves
UNREADABLE = MISSING | EMPTY
SAMPLE_RATE = 16000


def _wav_path(root, track_id):
    return root / 'fma_wav16k' / f'{track_id // 1000:03d}' / f'{track_id:06d}.wav'


@pytest.fixture(scope='module')
def fma_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp('fma')
    (root / 'fma_metadata').mkdir()
    # tracks.csv has three header rows, which insperdatasets joins into column
    # names: here track_id, track_genre_top and set_subset.
    lines = [',track,set', ',genre_top,subset', 'track_id,,']
    lines += [f'{t},{["Rock", "Pop", ""][t % 3]},small' for t in TRACK_IDS]
    (root / 'fma_metadata' / 'tracks.csv').write_text('\n'.join(lines) + '\n')
    rng = np.random.default_rng(0)
    for t in TRACK_IDS:
        if t in MISSING:
            continue
        path = _wav_path(root, t)
        path.parent.mkdir(parents=True, exist_ok=True)
        if t in EMPTY:
            path.touch()
            continue
        n_samples = int(rng.integers(SAMPLE_RATE // 2, 3 * SAMPLE_RATE))
        sf.write(path, rng.uniform(-0.5, 0.5, n_samples), SAMPLE_RATE)
    return root


# v1.0 notebooks/experiments_audio/vae.py, main(), with the data directory and
# the debug flag as arguments and its prints left out.
def _v1_datasets(data_dir, N_SAMPLES, debug, num_samples_debug=10):
    dataset = fma.FMADataset(
        data_dir=data_dir,
        loader_func=partial(
            audio_utils.load_audio,
            crop='random',
            t_len=N_SAMPLES,
        ),
    )

    # Downsample to 1/100 for faster tests
    if debug:
        num_samples = min(num_samples_debug, len(dataset))
        dataset = torch.utils.data.Subset(dataset, range(0, num_samples))

    # Split dataset into train (80%) and validation (20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
    )
    return train_dataset, val_dataset


def _rows(dataset):
    """The rows of tracks.csv that the items of nested Subsets come from."""
    if isinstance(dataset, torch.utils.data.Subset):
        rows = _rows(dataset.dataset)
        return [rows[i] for i in dataset.indices]
    return list(range(len(dataset)))


def _read_all(*datasets):
    """Every item of the datasets, read in order."""
    return [item for dataset in datasets for item in dataset]


def _assert_same_items(expected, actual):
    assert len(actual) == len(expected)
    for (audio, genre), (expected_audio, expected_genre) in zip(actual, expected):
        assert str(genre) == str(expected_genre)
        if expected_audio is None:
            assert audio is None
        else:
            assert torch.equal(audio[0], expected_audio[0])
            assert audio[1] == expected_audio[1]


@pytest.mark.parametrize('debug', [False, True])
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_matches_v1_vae_datasets(fma_dir, debug, seed):
    torch.manual_seed(seed)
    expected = _v1_datasets(fma_dir, SAMPLE_RATE, debug)
    expected_items = _read_all(*expected)
    expected_state = torch.get_rng_state()

    torch.manual_seed(seed)
    actual = load_fma(
        fma_dir,
        clip_samples=SAMPLE_RATE,
        max_tracks=10 if debug else None,
        skip_missing=False,
    )
    assert [_rows(d) for d in actual] == [_rows(d) for d in expected]
    _assert_same_items(expected_items, _read_all(*actual))
    assert torch.equal(torch.get_rng_state(), expected_state)


AVAILABLE_ROWS = [row for row, t in enumerate(TRACK_IDS) if t not in UNREADABLE]


def test_lists_only_tracks_with_a_file(fma_dir):
    train, val = load_fma(fma_dir, seed=0)
    assert sorted(_rows(train) + _rows(val)) == AVAILABLE_ROWS
    assert all(audio is not None for audio, _ in _read_all(train, val))


def test_without_skip_missing_tracks_give_none_which_collate_drops(fma_dir):
    train, val = load_fma(fma_dir, seed=0, skip_missing=False)
    items = _read_all(train, val)
    rows = _rows(train) + _rows(val)
    assert sorted(rows) == list(range(len(TRACK_IDS)))
    missing = [audio is None for audio, _ in items]
    assert missing == [TRACK_IDS[row] in UNREADABLE for row in rows]
    audio, mask, labels = audio_utils.collate_audio(items)
    assert audio.shape[0] == len(labels) == len(TRACK_IDS) - len(UNREADABLE)


@pytest.mark.parametrize('clip_samples', [None, SAMPLE_RATE])
def test_clips_are_cut_from_longer_tracks_only(fma_dir, clip_samples):
    train, val = load_fma(fma_dir, clip_samples=clip_samples, seed=0)
    for row, (audio, _) in zip(_rows(train) + _rows(val), _read_all(train, val)):
        if audio is None:
            continue
        waveform, sample_rate = audio
        n_samples = sf.info(str(_wav_path(fma_dir, TRACK_IDS[row]))).frames
        expected = n_samples if clip_samples is None else min(n_samples, clip_samples)
        assert waveform.shape == (1, expected)
        assert sample_rate == SAMPLE_RATE


@pytest.mark.parametrize('skip_missing', [True, False])
@pytest.mark.parametrize('max_tracks', [2, 10, 100])
def test_max_tracks_keeps_the_first_tracks(fma_dir, max_tracks, skip_missing):
    train, val = load_fma(
        fma_dir, max_tracks=max_tracks, seed=0, skip_missing=skip_missing
    )
    listed = AVAILABLE_ROWS if skip_missing else list(range(len(TRACK_IDS)))
    kept = listed[:max_tracks]
    assert sorted(_rows(train) + _rows(val)) == kept
    assert len(train) == int(0.8 * len(kept))


def test_seed_fixes_the_split_without_global_state(fma_dir):
    before = torch.get_rng_state()
    first = [_rows(d) for d in load_fma(fma_dir, seed=3)]
    assert torch.equal(torch.get_rng_state(), before)
    assert [_rows(d) for d in load_fma(fma_dir, seed=3)] == first
    assert [_rows(d) for d in load_fma(fma_dir, seed=4)] != first


def test_missing_insperdatasets_explains_install(monkeypatch, fma_dir):
    for module in [
        'insperdatasets',
        'insperdatasets.audio',
        'insperdatasets.audio.fma',
    ]:
        monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ImportError, match="'insperdatasets' package.*--extra audio"):
        load_fma(fma_dir)
