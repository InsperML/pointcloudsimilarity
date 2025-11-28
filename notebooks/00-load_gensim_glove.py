import gensim.downloader as api
import pickle
import torch
from pointcloudsimilarity.similarities import (CKASimilarity, GULPSimilarity,
                                               GWSimilarity,
                                               NNGSSimilarityTorch,
                                               ProcrustesSimilarity,
                                               PWCCASimilarity)
import torch.nn.functional as F
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm

# info = api.info()  # show info about available models/datasets
print("Loading model 200")
model200 = api.load("glove-twitter-200")  # download the model and return as object ready for use

print("Loading model 100")
model100 = api.load("glove-twitter-100")  # download the model and return as object ready for use

print("Loading model 50")
model50 = api.load("glove-twitter-50")  # download the model and return as object ready for use


print("Loading model 25")
model25 = api.load("glove-twitter-25")  # download the model and return as object ready for use

with open("glove_models.pkl", "wb") as f:
    pickle.dump({"model200": model200, "model100": model100, "model50": model50, "model25": model25}, f)
print("Models saved to glove_models.pkl")