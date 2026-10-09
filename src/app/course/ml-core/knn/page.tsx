"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import numpy as np
from collections import Counter

class KDNode:
    """
    K-Dimensional Tree (KD-Tree) node for spatial partitioning.
    Accelerates nearest neighbor search from brute-force O(N) to average O(log N).
    """
    def __init__(self, point: np.ndarray, label: int, axis: int, left=None, right=None):
        self.point = point
        self.label = label
        self.axis = axis
        self.left = left
        self.right = right

class KDTree:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.k_dims = X.shape[1]
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int):
        if len(X) == 0:
            return None

        axis = depth % self.k_dims
        # Sort along current splitting dimension
        sorted_indices = np.argsort(X[:, axis])
        median_idx = len(X) // 2
        median_pos = sorted_indices[median_idx]

        return KDNode(
            point=X[median_pos],
            label=y[median_pos],
            axis=axis,
            left=self._build_tree(X[sorted_indices[:median_idx]], y[sorted_indices[:median_idx]], depth + 1),
            right=self._build_tree(X[sorted_indices[median_idx + 1:]], y[sorted_indices[median_idx + 1:]], depth + 1)
        )

class KNearestNeighbors:
    """
    K-Nearest Neighbors classifier supporting:
    - Minkowski metric: D(x, y) = (Σ |x_i - y_i|^p)^(1/p)
      (p=1: Manhattan / L1, p=2: Euclidean / L2, p=∞: Chebyshev)
    - Inverse Distance Weighting: w_i = 1 / (d(x, x_i) + ε)
    """
    def __init__(self, k: int = 5, p: float = 2.0, weights: str = "uniform"):
        self.k = k
        self.p = p
        self.weights = weights
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Lazy learner: stores training data matrix in memory
        self.X_train = X.copy()
        self.y_train = y.copy()

    def _minkowski_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        if self.p == np.inf:
            return np.max(np.abs(x1 - x2))
        return np.sum(np.abs(x1 - x2) ** self.p) ** (1.0 / self.p)

    def predict_sample(self, x: np.ndarray) -> int:
        # Compute distances to all training points
        distances = np.array([self._minkowski_distance(x, x_tr) for x_tr in self.X_train])
        # Find indices of k smallest distances
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]

        if self.weights == "uniform":
            return Counter(k_labels).most_common(1)[0][0]
        elif self.weights == "distance":
            # Weighted vote: w_i = 1 / (dist + 1e-6)
            k_dists = distances[k_indices]
            weights = 1.0 / (k_dists + 1e-6)
            unique_labels = np.unique(k_labels)
            label_scores = {l: np.sum(weights[k_labels == l]) for l in unique_labels}
            return max(label_scores, key=label_scores.get)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict_sample(x) for x in X])

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    data = load_iris()
    # Feature scaling is mandatory for metric-based learning!
    X = StandardScaler().fit_transform(data.data)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    knn = KNearestNeighbors(k=5, p=2.0, weights="distance")
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)

    print(f"Scratch Distance-Weighted KNN Accuracy: {accuracy_score(y_test, preds):.4f}")
`;

export default function KNNPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-100 text-emerald-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
          Track 1 · Classical Machine Learning
        </span>
      </div>

      <h1>K-Nearest Neighbors (KNN): Metric Geometry, KD-Trees, and the Curse of Dimensionality</h1>

      <blockquote>
        <p>
          <em>"KNN is the prototypical non-parametric, instance-based lazy learner: it makes zero distributional assumptions about the hypothesis space, deferring all computation until query time."</em> — Evelyn Fix &amp; Joseph Hodges (1951)
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Derive generalized Minkowski distance metrics (Manhattan, Euclidean, Chebyshev) and their geometric unit balls.</li>
        <li>Explain why feature standardization is mandatory for metric-based classifiers.</li>
        <li>Understand Cover and Hart&apos;s 1-NN asymptotic error theorem (error rate at most twice Bayes error).</li>
        <li>Analyze KD-Tree and Ball Tree indexing structures to accelerate nearest neighbor queries from <code>O(N)</code> to <code>O(log N)</code>.</li>
        <li>Implement a generalized Minkowski KNN classifier with inverse distance weighting from scratch in NumPy.</li>
      </ul>

      <h2>1 · Generalized Minkowski Metric Spaces</h2>
      <p>
        The decision boundary of KNN is determined exclusively by the choice of metric function <code>D: ℝ^d × ℝ^d → ℝ^+</code>. The <strong>Minkowski Distance</strong> provides a generalized <code>L_p</code> norm:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>D(x, y) = ( Σ_(i=1)^d |x_i - y_i|^p )^(1/p)</p>
        <br />
        <p>Special Cases:</p>
        <p>  p = 1:  Manhattan Distance (L1 norm)   = Σ |x_i - y_i|</p>
        <p>  p = 2:  Euclidean Distance (L2 norm)   = √( Σ (x_i - y_i)² )</p>
        <p>  p → ∞:  Chebyshev Distance (L_∞ norm)  = max_i |x_i - y_i|</p>
      </div>

      <h3>Why Feature Standardization is Mandatory</h3>
      <p>
        Suppose feature 1 is income in dollars (range <code>[20,000, 200,000]</code>) and feature 2 is age in years (range <code>[18, 80]</code>). In an unscaled Euclidean calculation:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>D² = (Income_1 - Income_2)² + (Age_1 - Age_2)²</p>
        <p>   ≈ (10,000)² + (5)² = 100,000,000 + 25</p>
      </div>

      <p>
        The age difference has zero statistical impact on the neighbor ranking. Without standardizing features (Z-score <code>(x - μ)/σ</code>), the feature with the largest raw numerical scale dominates all distances completely.
      </p>

      <h2>2 · Theoretical Guarantees: Cover and Hart&apos;s Theorem (1967)</h2>
      <p>
        Thomas Cover and Peter Hart established one of the foundational asymptotic theorems of machine learning:
      </p>

      <blockquote>
        <p>
          <strong>Theorem:</strong> Let <code>R*</code> denote the minimal theoretical error rate achievable by any classifier (the <strong>Bayes Optimal Error Rate</strong>). As the number of training samples approaches infinity (<code>N → ∞</code>), the error rate <code>R_(1NN)</code> of the simple 1-Nearest Neighbor classifier is bounded by:
        </p>
      </blockquote>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>R* ≤ R_(1NN) ≤ 2 R* · ( 1 - R* / (C - 1) ) ≤ 2 R*</p>
        <br />
        <p>where C is the number of classes.</p>
      </div>

      <p>
        This proves that even without learning explicit parametric weights, an asymptotic 1-NN classifier captures at least half the information present in the underlying data distribution.
      </p>

      <h2>3 · The Bias-Variance Tradeoff in KNN</h2>
      <ul>
        <li><strong>k = 1:</strong> Fits training data perfectly (zero training error). Complex, jagged Voronoi tessellation decision boundaries. Highly sensitive to noise (<strong>High Variance, Low Bias</strong>).</li>
        <li><strong>Large k:</strong> Decision boundary smooths out into coarse linear approximations. If <code>k = N</code>, the model predicts the overall majority class everywhere (<strong>Low Variance, High Bias</strong>).</li>
      </ul>

      <h2>4 · Indexing Data Structures: KD-Trees vs. Ball Trees</h2>
      <p>
        Brute-force nearest neighbor search computes distances to all <code>N</code> training samples, costing <code>O(N · d)</code> per query.
      </p>

      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Indexing Structure</th>
              <th className="px-4 py-3 text-left">Partitioning Geometry</th>
              <th className="px-4 py-3 text-left">Query Complexity</th>
              <th className="px-4 py-3 text-left">Failure Regimes</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Brute Force</td>
              <td className="px-4 py-3">None (exhaustive linear scan)</td>
              <td className="px-4 py-3"><code>O(N · d)</code></td>
              <td className="px-4 py-3">Slow for large N.</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">KD-Tree (Bentley 1975)</td>
              <td className="px-4 py-3">Axis-aligned orthogonal hyperplanes splitting median values cycling through dimensions.</td>
              <td className="px-4 py-3"><code>O(d · log N)</code> for small d</td>
              <td className="px-4 py-3">Breaks down for <code>d &gt; 20</code> (curse of dimensionality causes backtracking to inspect all branches).</td>
            </tr>
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Ball Tree (Omohundro 1989)</td>
              <td className="px-4 py-3">Nested bounding hyperspheres utilizing triangle inequality <code>|d(x, c) - d(y, c)| ≤ d(x, y)</code>.</td>
              <td className="px-4 py-3"><code>O(d · log N)</code></td>
              <td className="px-4 py-3">Handles higher dimensions and complex distance metrics better than KD-Trees.</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>5 · Python Implementation from First Principles</h2>
      <TerminalBlock language="python" filename="knn_from_scratch.py" code={code} />

      <h2>6 · Further Reading</h2>
      <ul>
        <li>Cover, T., &amp; Hart, P. (1967). <em>Nearest neighbor pattern classification</em>. IEEE Transactions on Information Theory, 13(1), 21-27.</li>
        <li>Bentley, J. L. (1975). <em>Multidimensional binary search trees used for associative searching</em>. Communications of the ACM, 18(9), 509-517.</li>
      </ul>
    </article>
  );
}
