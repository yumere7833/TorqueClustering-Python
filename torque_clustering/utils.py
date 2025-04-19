from typing import Dict, Tuple, List

import networkx as nx
import numpy as np


def _prune_by_torque_gap(torque: Dict[Tuple[int, int], float]) -> int:
    tau = np.array(sorted(torque.values(), reverse=True))
    if len(tau) <= 1:
        return 0
    gaps = tau[:-1] - tau[1:]
    L = int(np.argmax(gaps) + 1)
    return L


def final_partition_from_tree(labels_init: np.ndarray, torque: Dict[Tuple[int, int], float], L: int) -> np.ndarray:
    g = nx.Graph()
    n = len(labels_init)
    g.add_nodes_from(range(n))
    sorted_edges = sorted(torque.items(), key=lambda kv: kv[1], reverse=True)
    bad_edges = {kv[0] for kv in sorted_edges[:L]}
    for (u, v), _ in torque.items():
        if (u, v) not in bad_edges:
            g.add_edge(u, v)
    comps = list(nx.connected_components(g))
    final_labels = np.empty(n, dtype=int)
    for cid, comp in enumerate(comps):
        final_labels[list(comp)] = cid
    return final_labels


def _nearest_neighbors(dist_mat: np.ndarray, masses: np.ndarray) -> List[Tuple[int, int, float]]:
    n = len(masses)
    conns = []
    for i in range(n):
        candidates = np.argsort(dist_mat[i])
        for j in candidates:
            if j != i and masses[i] <= masses[j]:
                conns.append((i, j, dist_mat[i, j] ** 2))
                break
    return conns


def _connected_components(conns: List[Tuple[int, int, float]], n: int) -> List[List[int]]:
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from([(i, j) for i, j, _ in conns])
    return [list(c) for c in nx.connected_components(g)]
