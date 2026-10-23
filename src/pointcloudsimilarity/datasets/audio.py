"""
Audio from the Free Music Archive (FMA), as torch datasets for training.
"""

from functools import partial

import torch

from ._optional import import_optional

__all__ = ['load_fma']


def load_fma(
    data_dir,
    clip_samples=None,
    max_tracks=None,
    train_fraction=0.8,
    seed=None,
    skip_missing=True,
):
    """
    The FMA tracks, split at random into training and validation datasets.

    Lists the tracks in data_dir/fma_metadata/tracks.csv and reads their audio
    from data_dir/fma_wav16k, with insperdatasets. An item is
    ((audio, sample_rate), genre), audio being a float32 tensor of shape
    (channels, samples). A track whose file cannot be read gives (None, genre)
    instead, which insperdatasets.audio.utils.collate_audio drops from its batch.

    Parameters:
    data_dir (str or Path): FMA directory, holding fma_metadata/ and fma_wav16k/.
    clip_samples (int): Read a random clip of this many samples (the whole
        track if shorter), drawn with torch's global random state each time an
        item is read. None reads whole tracks.
    max_tracks (int): Keep only the first max_tracks tracks, for quick tests.
    train_fraction (float): Fraction of the tracks used for training.
    seed (int): Seed of the split. None splits with torch's global random state.
    skip_missing (bool): List only the tracks whose file exists and is not
        empty: a partial download holds a fraction of the catalogue, and a
        failed conversion can leave an empty file. False lists every track,
        as v1.0 did, and those give (None, genre).

    Returns:
    tuple of torch.utils.data.Dataset: The training and validation tracks.
    """
    fma = import_optional('insperdatasets.audio.fma', 'load_fma', extra='audio')
    audio_utils = import_optional(
        'insperdatasets.audio.utils', 'load_fma', extra='audio'
    )
    load_audio = audio_utils.load_audio
    if clip_samples is not None:
        load_audio = partial(load_audio, crop='random', t_len=clip_samples)

    dataset = fma.FMADataset(data_dir=data_dir, loader_func=load_audio)
    rows = range(len(dataset))
    if skip_missing:
        paths = dataset.data['file_path']
        rows = [row for row, path in enumerate(paths) if _has_audio(path)]
    if max_tracks is not None:
        rows = rows[:max_tracks]
    if skip_missing or max_tracks is not None:
        dataset = torch.utils.data.Subset(dataset, rows)

    train_size = int(train_fraction * len(dataset))
    generator = (
        torch.default_generator if seed is None else torch.Generator().manual_seed(seed)
    )
    train, val = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size], generator=generator
    )
    return train, val


def _has_audio(path):
    try:
        return path.stat().st_size > 0
    except FileNotFoundError:
        return False
