from pathlib import Path

from pointcloudsimilarity.datasets import load_glove_twitter

# Parsing a downloaded GloVe model takes minutes; 01d-w2v_glove.py memory-maps
# the parsed models from this cache instead. 01d also fills it on its first run.
GLOVE_CACHE = Path(__file__).parent / 'glove_models'

for dim in (200, 100, 50, 25):
    print(f"Loading model {dim}")
    load_glove_twitter(dim, cache_dir=GLOVE_CACHE)

print(f"Models saved to {GLOVE_CACHE}")
