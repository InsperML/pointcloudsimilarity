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
):
    """
    The FMA tracks, split at random into training and validation datasets.

    Lists every track in data_dir/fma_metadata/tracks.csv and reads its audio
    from data_dir/fma_wav16k, with insperdatasets. An item is
    ((audio, sample_rate), genre), audio being a float32 tensor of shape
    (channels, samples). A track whose file is missing or unreadable gives
    (None, genre) instead, which insperdatasets.audio.utils.collate_audio drops
    from its batch.

    Parameters:
    data_dir (str or Path): FMA directory, holding fma_metadata/ and fma_wav16k/.
    clip_samples (int): Read a random clip of this many samples (the whole
        track if shorter), drawn with torch's global random state each time an
        item is read. None reads whole tracks.
    max_tracks (int): Keep only the first max_tracks tracks of the catalogue,
        for quick tests.
    train_fraction (float): Fraction of the tracks used for training.
    seed (int): Seed of the split. None splits with torch's global random state.

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
    if max_tracks is not None:
        dataset = torch.utils.data.Subset(
            dataset, range(0, min(max_tracks, len(dataset)))
        )

    train_size = int(train_fraction * len(dataset))
    generator = (
        torch.default_generator if seed is None else torch.Generator().manual_seed(seed)
    )
    train, val = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size], generator=generator
    )
    return train, val
