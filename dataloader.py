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
import numpy as np
from sklearn.decomposition import PCA

def idx_to_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return mask.astype(bool)

class Twitch(Dataset):
    def __init__(self, language, path="./data/twitch/", features_dim=None, transforms=None, **kwargs):
        available_languages = list(filter(lambda file: "txt" not in file, listdir(path)))
        assert(language in available_languages)
        self.custom_path = joinpath(path, language)
        self.language = language
        self.features_dim = features_dim
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
        df = df.drop_duplicates("new_id", keep="first")
        labels = df.sort_values("new_id")["mature"].astype(int)

        # load node features
        with open(joinpath(self.custom_path, f"{prefix}_features.json")) as f:
            data = json.load(f)
        
        n_node_features = max([max(v) for v in data.values()])
        features = np.zeros((len(G.nodes), n_node_features))
        
        for k,v in data.items():
            idx = np.array(v) - 1
            features[int(k)][idx] = 1
        
        if self.features_dim is not None:
            pca = PCA(n_components=self.features_dim)
            features = pca.fit_transform(features)
        
        n = len(G.nodes)
        indices = np.arange(n)
        idx_tr, idx_te, _, y_te = train_test_split(indices, labels, train_size=0.6, stratify=labels)
        idx_va, idx_te = train_test_split(idx_te, train_size=0.5, stratify=y_te)
        
        self.mask_tr = idx_to_mask(idx_tr, n)
        self.mask_va = idx_to_mask(idx_va, n)
        self.mask_te = idx_to_mask(idx_te, n)

        return [Graph(a=A, x=features, y=one_hot(labels, 2))]
    
class Git(Dataset):
    def __init__(self, path="./data/git_web_ml/", features_dim=None, transforms=None, **kwargs):
        self.custom_path = path
        self.features_dim = features_dim
        super().__init__(transforms, **kwargs)

    def read(self):
        prefix = "musae_git"

        # load adjacency matrix
        df = pd.read_csv(joinpath(self.custom_path, f"{prefix}_edges.csv"))
        edge_list = df.values.tolist()
        G = nx.from_edgelist(edge_list)
        A = to_scipy_sparse_matrix(G, dtype=float)

        # load target labels
        df = pd.read_csv(joinpath(self.custom_path, f"{prefix}_target.csv"))
        labels = df["ml_target"].values.tolist()

        # load node features
        with open(joinpath(self.custom_path, f"{prefix}_features.json")) as f:
            data = json.load(f)
        
        n_node_features = max([max(v) for v in data.values()])
        features = np.zeros((len(G.nodes), n_node_features))
        
        for k,v in data.items():
            idx = np.array(v) - 1
            features[int(k)][idx] = 1
        
        if self.features_dim is not None:
            pca = PCA(n_components=self.features_dim)
            features = pca.fit_transform(features)
        
        n = len(G.nodes)
        indices = np.arange(n)
        idx_tr, idx_te, _, y_te = train_test_split(indices, labels, train_size=0.6, stratify=labels)
        idx_va, idx_te = train_test_split(idx_te, train_size=0.5, stratify=y_te)
        
        self.mask_tr = idx_to_mask(idx_tr, n)
        self.mask_va = idx_to_mask(idx_va, n)
        self.mask_te = idx_to_mask(idx_te, n)

        return [Graph(a=A, x=features, y=one_hot(labels, 2))]