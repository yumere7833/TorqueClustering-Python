# TorqueClustering-Python
## Overview

This repository provides the **Python implementation** of the Torque Clustering algorithm, originally implemented in MATLAB. Torque Clustering is an autonomous clustering algorithm that identifies clusters by finding mass and distance peaks. This Python version reproduces the functionality of the original algorithm while leveraging Python's ecosystem for ease of use and integration.

---

## Usage and Implementation

### TorqueClustering Function Usage

The `torque_clustering` function performs clustering based on a distance matrix. It supports **automatic cluster number determination**.

#### Function Signature:
```python
def torque_clustering(dist_mat: np.ndarray) -> tuple[np.ndarray, dict[tuple[int, int], float]]:
```

#### Input Arguments:
- **`dist_mat`** *(n × n matrix)*:  
  The distance matrix of `n` data samples.

#### Output:
- **`labels`** *(n × 1 array)*:  
  The final cluster labels for each data point.

- **`torque`** *(dictionary)*:  
  The torque values for all connections in the tree.

---

### Example Usage

#### Clustering Text Data
```python
from torque_clustering.main import torque_cluster_text

docs = [
    "Deep learning with convolutional neural networks",
    "Back‑propagation and gradient descent for CNNs",
    "The capital city of France is Paris",
    "Eiffel tower is located in Paris",
    "Transformers and attention in natural language processing",
    "BERT and GPT are transformer‑based language models"
]

labels = torque_cluster_text(docs)
for doc, label in zip(docs, labels):
    print(f"[{label}] {doc}")
```

#### Clustering with a Distance Matrix
```python
from torque_clustering.main import torque_clustering
import numpy as np

# Example distance matrix
dist_mat = np.array([
    [0.0, 0.5, 0.8],
    [0.5, 0.0, 0.6],
    [0.8, 0.6, 0.0]
])

labels, torque = torque_clustering(dist_mat)
print("Cluster Labels:", labels)
```

---

## Features

- **Automatic Cluster Number Determination**: Automatically determines the optimal number of clusters using the torque gap method.
- **Text Clustering**: Supports clustering of text data using TF-IDF vectorization and cosine distance.
- **Reproducibility**: Implements the same algorithmic principles as the original MATLAB version.

---

## Citation

If you find this repository useful, please cite the original paper:  
**Jie Yang and Chin-Teng Lin, “Autonomous clustering by fast find of mass and distance peaks,” IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), DOI: 10.1109/TPAMI.2025.3535743**  
The pre-print version of this paper is available [here](https://www.computer.org/csdl/journal/tp/5555/01/10856563/23Saifm0vLy) and [here](https://www.techrxiv.org/users/686426/articles/679723-autonomous-clustering-by-fast-find-of-mass-and-distance-peaks).

---

## License

This repository is licensed under the **MIT License**.  
You are free to use, modify, and distribute this code for both personal and commercial purposes.

For more details, refer to the `LICENSE` file in this repository.
