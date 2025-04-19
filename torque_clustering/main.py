from typing import List, Tuple, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

from torque_clustering.utils import _prune_by_torque_gap, final_partition_from_tree, _nearest_neighbors, \
    _connected_components


def torque_clustering(dist_mat: np.ndarray) -> Tuple[np.ndarray, Dict[Tuple[int, int], float]]:
    n = dist_mat.shape[0]
    clusters = [[i] for i in range(n)]
    masses = np.ones(n, dtype=int)
    torque = {}  # τ of every connection across whole tree

    while len(clusters) > 1:
        idx_of = {}
        for cid, members in enumerate(clusters):
            for m in members:
                idx_of[m] = cid

        conns = _nearest_neighbors(dist_mat, masses)

        for i, j, d2 in conns:
            MiMj = masses[idx_of[i]] * masses[idx_of[j]]
            torque[(idx_of[i], idx_of[j])] = MiMj * d2

        comps = _connected_components(conns, n)
        if len(comps) == len(clusters):
            break
        clusters = comps
        masses = np.array([len(c) for c in clusters])

    labels = np.empty(n, dtype=int)
    for cid, members in enumerate(clusters):
        labels[members] = cid
    return labels, torque


def torque_cluster_text(corpus: List[str], vectorizer: TfidfVectorizer = None) -> np.ndarray:
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=50_000,
                                     ngram_range=(1, 2),
                                     stop_words='english')
    X = vectorizer.fit_transform(corpus)
    dist_mat = cosine_distances(X)
    labels_tree, tau = torque_clustering(dist_mat)
    L = _prune_by_torque_gap(tau)
    labels_final = final_partition_from_tree(labels_tree, tau, L)
    return labels_final


if __name__ == "__main__":
    docs = [
        "Deep learning with convolutional neural networks",
        "Back‑propagation and gradient descent for CNNs",
        "The capital city of France is Paris",
        "Eiffel tower is located in Paris",
        "Transformers and attention in natural language processing",
        "BERT and GPT are transformer‑based language models"
    ]
    labels = torque_cluster_text(docs)
    for doc, lab in zip(docs, labels):
        print(f"[{lab}] {doc}")
