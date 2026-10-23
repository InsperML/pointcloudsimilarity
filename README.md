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

- Implementation of the **Topological Alignment Spectrum (TAS)** and other point cloud similarity measures
- Tools to analyze representational similarity across scales

## Papers

    @misc{tavares2026ijcnn,
        title={Diagnosing Neural Convergence with Topological Alignment Spectra}, 
        author={Tiago F. Tavares and Fabio Ayres and Paris Smaragdis},
        year={2026},
        booktitle={Published at IJCNN 2026},
        url={https://arxiv.org/abs/2411.08687}, 
    }

## Quick start

### Installing

```bash
pip install pointcloudsimilarity
```

This pulls in PyTorch. On Linux, pip installs its CUDA build by default; for a
smaller CPU-only install, install torch first from
https://download.pytorch.org/whl/cpu.

### Example code

```python
from pointcloudsimilarity.similarities import (
    CKASimilarity,
    GULPSimilarity,
    GWSimilarity,
    TASSimilarityTorch,
    ProcrustesSimilarity,
    PWCCASimilarity,
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

## Implemented similarity measures

All measures take two point clouds with the same number of rows, where row `i`
of each cloud is the same input (for example, the same sentence embedded by two
models). Each class is also available through
`compute_similarity(X, Y, method=...)`.

| Measure | Class (`method`) | How it works | Reference |
|---|---|---|---|
| TAS | `TASSimilarityTorch` (`tas_torch`), `TASSimilarityCPU` (`tas_cpu`) | For every point, finds its `k` nearest neighbors in each cloud and measures how much the two neighbor sets overlap (Jaccard index). The mean overlap is normalized against what random neighbor sets would give, so 1 is identical neighborhoods, 0 is chance and negative values are worse than chance. Sweeping `k` gives the spectrum, from local to global structure. | Tavares, Ayres & Smaragdis, *Diagnosing Neural Convergence with Topological Alignment Spectra*, IJCNN 2026. [arXiv:2411.08687](https://arxiv.org/abs/2411.08687) |
| CKA | `CKASimilarity` (`cka`) | Centered Kernel Alignment. Builds a kernel (Gram) matrix for each cloud, linear or RBF, and computes their normalized alignment (HSIC). Linear CKA is `‖YᵀX‖²_F / (‖XᵀX‖_F ‖YᵀY‖_F)` on centered data, which does not change under rotations or uniform scaling of either cloud. With the RBF kernel, each cloud's bandwidth comes from its median pairwise distance. | Kornblith, Norouzi, Lee & Hinton, *Similarity of Neural Network Representations Revisited*, ICML 2019. [arXiv:1905.00414](https://arxiv.org/abs/1905.00414) |
| GULP | `GULPSimilarity` (`gulp`) | Compares two representations by how differently they predict linear functions of the inputs when ridge regression (regularization `lambda_`) is fit on top of each. The distance is computed in closed form from the covariance and cross-covariance matrices. Here the data are centered and each point is scaled to unit norm first, and the similarity is `1 − d²`. | Boix-Adserà, Lawrence, Stepaniants & Rigollet, *GULP: a prediction-based metric between representations*, NeurIPS 2022. [arXiv:2210.06545](https://arxiv.org/abs/2210.06545) |
| Gromov–Wasserstein | `GWSimilarity` (`gw_sim`) | Computes the matrix of Euclidean distances within each cloud, then finds the soft matching of points (a transport plan with uniform weights) that makes the two distance matrices agree best, using [POT](https://pythonot.github.io/). It does not use the row pairing, so it only compares the shapes. Similarity is `1 / (1 + GW)`. | Mémoli, *Gromov–Wasserstein Distances and the Metric Approach to Object Matching*, Foundations of Computational Mathematics, 2011. [doi:10.1007/s10208-011-9093-5](https://doi.org/10.1007/s10208-011-9093-5) |
| Procrustes | `ProcrustesSimilarity` (`procrustes`) | Centers both clouds and scales them to unit norm, then rotates (or reflects) and scales the second to fit the first as closely as possible. The disparity is the remaining sum of squared differences, and the similarity is `1 − disparity`. Uses `scipy.spatial.procrustes`, so both clouds need the same dimension. | Gower, *Generalized Procrustes Analysis*, Psychometrika, 1975. [doi:10.1007/BF02291478](https://doi.org/10.1007/BF02291478) |
| PWCCA | `PWCCASimilarity` (`pwcca`) | Canonical Correlation Analysis finds pairs of directions, one in each cloud, whose projections are maximally correlated. PWCCA averages those canonical correlations, weighting each by how much of the first cloud it accounts for; `symmetric=True` averages both directions. It needs more points than dimensions. The code is adapted from Google's [SVCCA](https://github.com/google/svcca) repository. | Morcos, Raghu & Bengio, *Insights on representational similarity in neural networks with canonical correlation*, NeurIPS 2018. [arXiv:1806.05759](https://arxiv.org/abs/1806.05759) |

## Cloning and working with the repo

```bash
git clone git@github.com:InsperML/pointcloudsimilarity.git
cd pointcloudsimilarity
uv sync
source .venv/bin/activate
```

## License

Apache-2.0, see [LICENSE](LICENSE). `legacy/pwcca.py` is adapted from Google's
[SVCCA](https://github.com/google/svcca), also under Apache-2.0.
