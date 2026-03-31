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

### Example code

```python
from pointcloudsimilarity.similarities import (
    CKASimilarity,
    GULPSimilarity,
    GWSimilarity,
    NNGSSimilarityTorch,
    ProcrustesSimilarity,
    PWCCASimilarity,
    RTDSimilarity
)

my_similarity = NNGSSimilarityTorch(k=3) # or choose other similarities?
sim = my_similarity(pc1, pc2) # point cloud 1, point cloud 2 - paired point clouds
```


## Cloning and working with the repo

```bash
git clone git@github.com:InsperML/pointcloudsimilarity.git
cd pointcloudsimilarity
uv sync
source .venv/bin/activate
```

## IJCNN 2026 paper experiments

```bash 
python notebooks/ijcnn/00-downsampling.py
python notebooks/ijcnn/01-toy_case_sweeps.py
python notebooks/ijcnn/02-finetuning.py
```

