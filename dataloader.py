from spektral.data import Dataset
import networkx as nx
from os import listdir
from os.path import join as joinpath
import json
import pandas as pd
from networkx.convert_matrix import to_scipy_sparse_matrix
from spektral.data.graph import Graph
import numpy as np
from sklearn.model_selection import train_test_split
from spektral.utils import one_hot
from spektral.data.loaders import BatchLoader
import numpy as np
import tensorflow as tf

def idx_to_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return mask.astype(bool)

class Twitch(Dataset):
    def __init__(self, language, transforms=None, **kwargs):
        path = "./data/twitch/"

        available_languages = list(filter(lambda file: "txt" not in file, listdir(path)))
        assert(language in available_languages)
        self.custom_path = joinpath(path, language)
        self.language = language
        super().__init__(transforms, **kwargs)

    def read(self):
        prefix = f"musae_{self.language}"

        # load adjacency matrix
        df = pd.read_csv(joinpath(self.custom_path, f"{prefix}_edges.csv"))
        edge_list = df.values.tolist()
        G = nx.from_edgelist(edge_list)
        A = to_scipy_sparse_matrix(G, dtype=float)

        # load target labels
        df = pd.read_csv(joinpath(self.custom_path, f"{prefix}_target.csv"))
        labels = df["mature"].values.astype(int)

        # load node features
        with open(joinpath(self.custom_path, f"{prefix}_features.json")) as f:
            data = json.load(f)
        
        n_node_features = max([max(v) for v in data.values()])
        features = np.zeros((len(G.nodes), n_node_features))
        
        for k,v in data.items():
            idx = np.array(v) - 1
            features[int(k)][idx] = 1
        
        n = len(G.nodes)
        indices = np.arange(n)
        idx_tr, idx_te, _, y_te = train_test_split(indices, labels, train_size=0.6, stratify=labels)
        idx_va, idx_te = train_test_split(idx_te, train_size=0.5, stratify=y_te)
        
        self.mask_tr = idx_to_mask(idx_tr, n)
        self.mask_va = idx_to_mask(idx_va, n)
        self.mask_te = idx_to_mask(idx_te, n)

        return [Graph(a=A, x=features, y=one_hot(labels, 2))]
    