"""
Record the data each notebook feeds to the metrics, to check the datasets refactor.

    python harness.py TARGET OUT_DIR --root ROOT

Runs TARGET from ROOT's notebooks (ROOT/src must be first on PYTHONPATH, see
run_baseline.sh) and writes OUT_DIR/TARGET.json. ROOT must be a disposable copy:
the scripts write figures next to themselves.

Every metric call and tas_spectrum call is recorded with a hash of each input
array (dtype, shape and bytes) and its result, so two runs agree only if the
scripts built exactly the same point clouds. Text targets stop each script at
its first model call and record a hash of the texts it would embed.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import random
import runpy
import sys
import tomllib
from itertools import product
from pathlib import Path

import numpy as np
import torch

RECORDS = []


class Captured(Exception):
    """Raised at a script's first model call, once its texts are recorded."""


def digest(a):
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    a = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256()
    h.update(f'{a.dtype}|{a.shape}|'.encode())
    h.update(a.tobytes())
    return h.hexdigest()[:16]


def digest_json(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False).encode()).hexdigest()[:16]


def jsonable(result):
    if result is None:
        return None
    if isinstance(result, (np.ndarray, torch.Tensor, list, tuple)):
        return [float(v) for v in np.asarray(result, dtype=np.float64).ravel()]
    return float(result)


def scalar_params(obj):
    return {
        k: v
        for k, v in vars(obj).items()
        if isinstance(v, (bool, int, float, str, type(None)))
    }


def instrument_metrics():
    from pointcloudsimilarity import similarities as sims

    for cls in set(sims.available_metrics.values()):

        def wrapped(self, pc1, pc2, _orig=cls.__call__, **kwargs):
            result = _orig(self, pc1, pc2, **kwargs)
            RECORDS.append(
                {
                    'call': type(self).__name__,
                    'params': scalar_params(self),
                    'x': digest(pc1),
                    'y': digest(pc2),
                    'shapes': [list(pc1.shape), list(pc2.shape)],
                    'result': jsonable(result),
                }
            )
            return result

        cls.__call__ = wrapped

    original_spectrum = sims.tas_spectrum

    def spectrum(pc1, pc2, ks, *args, **kwargs):
        ks = list(ks)
        result = original_spectrum(pc1, pc2, ks, *args, **kwargs)
        RECORDS.append(
            {
                'call': 'tas_spectrum',
                'params': {k: v for k, v in kwargs.items() if k != 'device'},
                'x': digest(pc1),
                'y': digest(pc2),
                'shapes': [list(pc1.shape), list(pc2.shape)],
                'ks': digest(np.asarray(ks, dtype=np.float64)),
                'result': jsonable(result),
            }
        )
        return result

    sims.tas_spectrum = spectrum


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_script(path, seed=0):
    """Run a notebook as `python path` would, after seeding every global RNG."""
    sys.path.insert(0, str(path.parent))
    os.chdir(path.parent)
    seed_everything(seed)
    runpy.run_path(str(path), run_name='__main__')


def load_module(path):
    sys.path.insert(0, str(path.parent))
    os.chdir(path.parent)
    name = 'nb_' + path.stem.replace('-', '_')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def capture_texts(*args, **kwargs):
    texts = kwargs.get('texts')
    if texts is None:
        texts = next(a for a in args if isinstance(a, list))
    RECORDS.append(
        {
            'call': 'embed_texts',
            'model': args[0],
            'n': len(texts),
            'texts': digest_json(texts),
        }
    )
    raise Captured


EMBEDDING_FUNCTIONS = [
    'get_bert_embeddings',
    'get_finetuned_bert_embeddings',
    'get_gpt_embeddings',
    'get_sbert_embeddings',
    'get_gptx_embeddings',
    'get_finetuned_gptx_embeddings',
]


def texts_of_script(path):
    """Run a script's main() until it first embeds texts, recording them."""
    module = load_module(path)
    for name in EMBEDDING_FUNCTIONS:
        if hasattr(module, name):
            setattr(module, name, capture_texts)
    try:
        module.main()
    except Captured:
        pass


def texts_of_ijcnn_02(path):
    """ijcnn/02 runs three datasets; record each get_dataset result."""
    module = load_module(path)
    original_get_dataset = module.get_dataset

    def get_dataset(*args, **kwargs):
        texts, labels = original_get_dataset(*args, **kwargs)
        RECORDS.append(
            {
                'call': 'get_dataset',
                'args': [list(args), kwargs],
                'n': len(texts),
                'texts': digest_json(texts),
                'labels': digest_json([int(v) for v in labels]),
            }
        )
        return texts, labels

    module.get_dataset = get_dataset
    sys.modules['lib_get_embeddings'].get_bert_embeddings = capture_texts
    original_experiment = module.similarities_model_vs_finetuning

    def experiment(*args, **kwargs):
        try:
            original_experiment(*args, **kwargs)
        except Captured:
            pass

    module.similarities_model_vs_finetuning = experiment
    module.main()


def experiments_data(root):
    """
    The tensors experiments/training.py trains on, for each sweep configuration.

    Only make_dataloader exists on both sides of the refactor; the tests in
    tests/test_datasets_synthetic.py compare the float64 data it is built from.
    """
    exp_dir = root / 'notebooks' / 'experiments'
    dataset_ = load_module(exp_dir / 'dataset_.py')
    with open(exp_dir / 'experiments.toml', 'rb') as f:
        sweep = tomllib.load(f)['sweep']
    keys = ['n_samples', 'n_features', 'n_classes', 'cluster_std', 'center_scale']
    for values in product(*(sweep[k] for k in keys)):
        config = dict(zip(keys, values))
        X, y, _ = dataset_.make_dataloader(**config, device=torch.device('cpu'))
        RECORDS.append(
            {
                'call': 'make_dataloader',
                'params': config,
                'x': digest(X),
                'y': digest(y),
            }
        )


def targets(root):
    nb = root / 'notebooks'
    return {
        'ijcnn_00': lambda: run_script(nb / 'ijcnn' / '00-downsampling.py'),
        'ijcnn_01': lambda: run_script(nb / 'ijcnn' / '01-toy_case_sweeps.py'),
        '01_whitenoise': lambda: run_script(nb / '01-whitenoise.py'),
        '01a_sweeps': lambda: run_script(nb / '01a-sweeps.py'),
        'experiments_data': lambda: experiments_data(root),
        'ijcnn_02_texts': lambda: texts_of_ijcnn_02(nb / 'ijcnn' / '02-finetuning.py'),
        '01c_texts': lambda: texts_of_script(nb / '01c-bert_vs_distillbert.py'),
        '01e_texts': lambda: texts_of_script(
            nb / '01e-bert_initialization_and_finetuning.py'
        ),
        '01f_texts': lambda: texts_of_script(nb / '01f-gpts.py'),
        '04_texts': lambda: texts_of_script(nb / '04-quantization.py'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('target')
    parser.add_argument('out_dir', type=Path)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root, out_dir = args.root.resolve(), args.out_dir.resolve()

    import pointcloudsimilarity

    library = Path(pointcloudsimilarity.__file__).resolve()
    if not library.is_relative_to(root / 'src'):
        sys.exit(f'pointcloudsimilarity comes from {library}, not {root / "src"}')

    instrument_metrics()
    targets(root)[args.target]()

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f'{args.target}.json', 'w') as f:
        json.dump({'library': str(library), 'records': RECORDS}, f, indent=1)

    # Streaming datasets leave threads that can abort interpreter shutdown
    # (seen with 01c_texts). The results are written, so skip the shutdown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
