# Point Cloud Similarity

## Project Overview

This repository accompanies our IJCNN 2026 paper, *“Diagnosing Neural Convergence with Topological Alignment Spectra”*.

We introduce the **Topological Alignment Spectrum (TAS)**, a multi-scale diagnostic framework for analyzing representational similarity in neural networks. Unlike traditional scalar metrics such as CKA or Procrustes—which compress alignment into a single value—TAS evaluates similarity **across a continuum of neighborhood scales**, revealing how models align locally, globally, and everything in between.

By normalizing neighborhood overlap against a principled random baseline, TAS produces a **dimension-invariant spectrum** where:
- **1** indicates perfect structural alignment  
- **0** corresponds to chance-level similarity  
- **< 0** reveals active anti-alignment at specific scales  

This enables TAS to disentangle distinct geometric phenomena—such as **local noise (jitter)** versus **global semantic reorganization**—that are typically indistinguishable under standard methods.

---

## Why this matters

Modern neural networks are often **underspecified**: different models can achieve identical accuracy while encoding fundamentally different internal structures. This has practical implications:

- **Risk:** Two models with similar performance may behave inconsistently under distribution shift.  
- **Limitation of current tools:** Scalar similarity metrics frequently report high similarity even when structural differences are substantial (e.g., jitter vs. cluster reshuffling).  
- **Opportunity:** TAS provides a **diagnostic lens** to identify *where* and *how* representations diverge, improving interpretability, debugging, and model auditing.

In our experiments (e.g., MultiBERTs), TAS reveals that fine-tuning induces **global topological reorganization**, challenging the common assumption that adaptation is mostly local.

---

## What you’ll find in this repo

- Implementation of the **Topological Alignment Spectrum (TAS)**
- Reproducible experiments on synthetic data and language models
- Tools to analyze representational similarity across scales
- Examples illustrating how TAS distinguishes different geometric distortions
- Synthetic point clouds with known distortions, and loaders for the text, word-vector and audio data of the experiments

## Papers

    @misc{tavares2025ijcnn,
        title={Diagnosing Neural Convergence with Topological Alignment Spectra}, 
        author={Tiago F. Tavares and Fabio Ayres and Paris Smaragdis},
        year={2025},
        url={https://arxiv.org/abs/2411.08687}, 
    }

## Quick start

### Installing directly from git

```bash
pip install git+ssh://git@github.com/InsperML/pointcloudsimilarity.git
```

Some parts need extra dependencies, installed with an extra:

| Extra | Needed for |
|---|---|
| `faiss` | `TASSimilarityFaiss` |
| `rtd` | `RTDSimilarity` |
| `data` | `load_text_dataset` and the GloVe loaders in `pointcloudsimilarity.datasets` |
| `audio` | `load_fma` in `pointcloudsimilarity.datasets` |
| `notebooks` | the experiment scripts in `notebooks/` |

```bash
pip install "pointcloudsimilarity[data] @ git+ssh://git@github.com/InsperML/pointcloudsimilarity.git"
```

### Example code

```python
from pointcloudsimilarity.similarities import (
    CKASimilarity,
    GULPSimilarity,
    GWSimilarity,
    TASSimilarityTorch,
    ProcrustesSimilarity,
    PWCCASimilarity,
    RTDSimilarity
)

my_similarity = TASSimilarityTorch(k=3) # or choose other similarities?
sim = my_similarity(pc1, pc2) # point cloud 1, point cloud 2 - paired point clouds
```

To get the full spectrum across neighborhood sizes, use `tas_spectrum`. It runs a
single nearest-neighbor search for the largest `k`:

```python
import numpy as np
from pointcloudsimilarity.similarities import tas_spectrum

ks = np.arange(1, pc1.shape[0] - 1, 10)
spectrum = tas_spectrum(pc1, pc2, ks)  # TAS value for each k
```

## Datasets

`pointcloudsimilarity.datasets` builds the point clouds the experiments compare.

### Synthetic point clouds

Each generator returns a pair `(X, Y)`: the same points before and after a known change.

| Function | How `Y` differs from `X` |
|---|---|
| `jittered_blobs` | Every point moves by Gaussian noise: local jitter, the clusters stay. |
| `shuffled_blobs` | The blobs trade places, each keeping its own shape: global reorganization. |
| `power_distortion` | Each coordinate is pushed away from zero by a power of its size. |
| `noise_pair` | `Y` is independent noise: the chance-level baseline. |
| `mix_with_noise` | Given `X`, interpolates it toward independent noise. |

`equidistant_blobs` gives labelled blobs around `simplex_centers`, the vertices of a regular simplex, so that all classes are equally far apart.

Their spectra separate local jitter, global reorganization and chance:

```python
import numpy as np
from pointcloudsimilarity.datasets import jittered_blobs, noise_pair, shuffled_blobs
from pointcloudsimilarity.similarities import tas_spectrum

centers = np.sign(np.random.default_rng(0).normal(size=(4, 10)))  # 4 blobs of 250 points
ks = np.arange(1, 999)

X, Y = jittered_blobs(1000, 10, centers, cluster_std=0.1, noise_std=0.1, random_state=0)
jitter = tas_spectrum(X, Y, ks)  # about 0 at small k, 1 at k = 249 (one whole blob)

X, Y = shuffled_blobs(1000, centers, cluster_std=0.1, random_state=0)
shuffle = tas_spectrum(X, Y, ks)  # 1 up to k = 249, then 0 or below

X, Y = noise_pair(1000, 10, random_state=0)
noise = tas_spectrum(X, Y, ks)  # about 0 at every k
```

The generators take `random_state` as scikit-learn does: an int for reproducible data, or `None` (the default) to draw from NumPy's global random state, as the paper scripts do after `np.random.seed()`.

### Real data

The loaders import their dependencies only when called, so `import pointcloudsimilarity.datasets` stays light.

| Loader | Returns | Extra |
|---|---|---|
| `load_text_dataset` | Texts, and optionally labels, sampled reproducibly from a Hugging Face dataset | `data` |
| `load_glove_twitter`, `aligned_glove_vectors` | GloVe Twitter word vectors; the same words' vectors from several models, one point cloud per model | `data` |
| `load_fma` | Free Music Archive tracks, as torch training and validation datasets | `audio` |

```python
from pointcloudsimilarity.datasets import (
    aligned_glove_vectors,
    load_fma,
    load_glove_twitter,
    load_text_dataset,
)

# 2000 IMDB reviews and their labels, to embed with the models you compare
texts, labels = load_text_dataset(
    'imdb', split='test', label_column='label', n_samples=2000, seed=0
)

# 1000 words as seen by two GloVe models of different sizes
models = {dim: load_glove_twitter(dim, cache_dir='glove_models') for dim in (25, 200)}
words, vectors = aligned_glove_vectors(models, n_words=1000)
spectrum = tas_spectrum(vectors[25], vectors[200], ks)

# Random 3 s clips of FMA tracks, for training an audio model
train, val = load_fma('/path/to/fma', clip_samples=48000, seed=0)
```

`load_glove_twitter` downloads a model on first use and parses it, which can take from minutes (25 dimensions) to half an hour (200 dimensions); with `cache_dir`, later calls load the parsed model in seconds. `load_fma` expects `fma_metadata/tracks.csv` and the tracks as 16 kHz WAV files in `fma_wav16k/`, and lists only tracks whose file is there. Its items are `((audio, sample_rate), genre)`; batch them with `insperdatasets.audio.utils.collate_audio`.

## Cloning and working with the repo

```bash
git clone git@github.com:InsperML/pointcloudsimilarity.git
cd pointcloudsimilarity
uv sync  # add --extra data, --extra audio, ... as needed, or --all-extras
source .venv/bin/activate
```

## IJCNN 2026 paper experiments

The scripts need the `notebooks` extra, and save their figures in `notebooks/ijcnn/figures/`:

```bash
uv sync --extra notebooks
mkdir -p notebooks/ijcnn/figures
python notebooks/ijcnn/00-downsampling.py
python notebooks/ijcnn/01-toy_case_sweeps.py
python notebooks/ijcnn/02-finetuning.py
```

