"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import numpy as np

class KMeansClustering:
    """
    K-Means Clustering implementing:
    - Lloyd's alternating minimization algorithm.
    - K-Means++ probabilistic distance-squared initialization (Arthur & Vassilvitskii, 2007).
    - Distortion objective (Inertia): J = Σ_(k=1)^K Σ_(x_i ∈ C_k) ||x_i - μ_k||²
    """
    def __init__(self, n_clusters: int = 3, max_iter: int = 300, tol: float = 1e-4, init: str = "k-means++"):
        self.k = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.centroids = None
        self.inertia_ = 0.0

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        if self.init != "k-means++":
            # Random uniform initialization
            idx = np.random.choice(n_samples, self.k, replace=False)
            return X[idx].copy()

        # K-Means++ Initialization
        # 1. Choose first center uniformly at random
        centroids = [X[np.random.randint(0, n_samples)]]

        # 2. Choose remaining centers with probability proportional to D(x)²
        for _ in range(1, self.k):
            # Compute distance of each point to nearest already chosen centroid
            dists = np.array([
                min(np.sum((x - c) ** 2) for c in centroids) for x in X
            ])
            # Probability distribution proportional to squared distance D(x)²
            probs = dists / np.sum(dists)
            cumprobs = np.cumsum(probs)
            r = np.random.rand()
            chosen_idx = np.searchsorted(cumprobs, r)
            centroids.append(X[chosen_idx])

        return np.array(centroids)

    def fit(self, X: np.ndarray):
        n_samples, n_features = X.shape
        self.centroids = self._init_centroids(X)

        for iteration in range(self.max_iter):
            # Step 1: Expectation / Assignment Step
            # Assign each x_i to closest centroid μ_k
            # Euclidean pairwise distance matrix (N, K)
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            # Step 2: Maximization / Update Step
            # Recompute centroids as the mean of assigned points
            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.k):
                cluster_points = X[labels == k]
                if len(cluster_points) > 0:
                    new_centroids[k] = np.mean(cluster_points, axis=0)
                else:
                    # Handle empty cluster: re-initialize with random sample
                    new_centroids[k] = X[np.random.randint(0, n_samples)]

            # Check convergence tolerance
            shift = np.linalg.norm(self.centroids - new_centroids)
            self.centroids = new_centroids
            if shift < self.tol:
                break

        # Calculate final inertia (distortion)
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        self.inertia_ = np.sum([np.sum((X[labels == k] - self.centroids[k]) ** 2) for k in range(self.k)])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes Mean Silhouette Coefficient: s(i) = (b(i) - a(i)) / max(a(i), b(i))
    where a(i) is mean intra-cluster distance, b(i) is mean nearest-cluster distance.
    """
    n_samples = len(X)
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1 or len(unique_labels) >= n_samples:
        return 0.0

    s_values = []
    for i in range(n_samples):
        own_cluster = labels[i]
        own_mask = (labels == own_cluster)
        
        # Intra-cluster mean distance a(i)
        if np.sum(own_mask) > 1:
            a_i = np.mean(np.linalg.norm(X[own_mask] - X[i], axis=1))
        else:
            a_i = 0.0

        # Nearest-cluster mean distance b(i)
        b_candidates = []
        for other_cluster in unique_labels:
            if other_cluster == own_cluster:
                continue
            other_mask = (labels == other_cluster)
            b_candidates.append(np.mean(np.linalg.norm(X[other_mask] - X[i], axis=1)))
        b_i = min(b_candidates)

        denom = max(a_i, b_i)
        s_i = (b_i - a_i) / denom if denom > 0 else 0.0
        s_values.append(s_i)

    return float(np.mean(s_values))

if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)
    kmeans = KMeansClustering(n_clusters=4, init="k-means++")
    kmeans.fit(X)
    labels = kmeans.predict(X)

    score = silhouette_score(X, labels)
    print(f"K-Means (k=4) Inertia:         {kmeans.inertia_:.2f}")
    print(f"Mean Silhouette Coefficient:   {score:.4f} [Scale -1 to +1]")
`;

export default function ClusteringPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-100 text-emerald-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
          Track 1 · Classical Machine Learning
        </span>
      </div>

      <h1>Clustering Algorithms: K-Means++, DBSCAN, and Intrinsic Validation Metrics</h1>

      <blockquote>
        <p>
          <em>"Clustering seeks to partition unlabeled observations into disjoint subsets such that points within the same group exhibit minimal dispersion while points across different groups exhibit maximal separation."</em>
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Formulate the K-Means distortion objective (Inertia) and prove why Lloyd&apos;s algorithm guarantees convergence to a local minimum.</li>
        <li>Derive the <code>O(log k)</code> competitive approximation guarantee of Arthur &amp; Vassilvitskii&apos;s K-Means++ initialization.</li>
        <li>Explain density-based clustering (DBSCAN), core points, density reachability, and arbitrary-shape cluster discovery.</li>
        <li>Derive intrinsic validation metrics: Silhouette Coefficient, Davies-Bouldin Index, and the Elbow method.</li>
        <li>Implement K-Means++ and the Silhouette metric from scratch in NumPy.</li>
      </ul>

      <h2>1 · K-Means and Lloyd&apos;s Alternating Minimization Algorithm</h2>
      <p>
        Given <code>n</code> unlabeled observations <code>&#123;x_1, ..., x_n&#125; ⊂ ℝ^d</code>, K-Means seeks to partition them into <code>K</code> clusters <code>C = &#123;C_1, ..., C_K&#125;</code> to minimize the within-cluster sum of squares (<strong>Inertia / Distortion Objective</strong>):
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>J(C, μ) = Σ_(k=1)^K Σ_(x_i ∈ C_k) ‖x_i - μ_k‖₂²</p>
        <br />
        <p>where μ_k = (1 / |C_k|) Σ_(x_i ∈ C_k) x_i   is the centroid of cluster C_k.</p>
      </div>

      <p>
        Because finding the global minimum across all <code>K^n / K!</code> partitions is an <strong>NP-hard problem</strong> in <code>d ≥ 2</code> dimensions, Stuart Lloyd (1957) formulated an alternating expectation-maximization heuristic:
      </p>

      <ol>
        <li>
          <strong>Assignment Step (Expectation):</strong> Fix centroids <code>μ</code>, assign each point to its nearest centroid:
          <div className="not-prose bg-slate-900 rounded-lg p-3 font-mono text-slate-100 text-xs my-2">
            C_k = &#123;x_i : ‖x_i - μ_k‖ ≤ ‖x_i - μ_j‖, ∀ j ∈ &#123;1, ..., K&#125;&#125;
          </div>
        </li>
        <li>
          <strong>Update Step (Maximization):</strong> Fix cluster assignments <code>C_k</code>, update centroids:
          <div className="not-prose bg-slate-900 rounded-lg p-3 font-mono text-slate-100 text-xs my-2">
            μ_k = (1 / |C_k|) Σ_(x_i ∈ C_k) x_i
          </div>
        </li>
      </ol>

      <p>
        <strong>Convergence Proof:</strong> At each step, <code>J</code> strictly decreases or remains constant. Because there are a finite number of partitions (<code>K^n</code>), Lloyd&apos;s algorithm is guaranteed to converge to a local minimum in finite steps.
      </p>

      <h2>2 · K-Means++ Initialization (Arthur &amp; Vassilvitskii, 2007)</h2>
      <p>
        Standard uniform random initialization frequently traps K-Means in catastrophic local minima (e.g. placing two centroids in one true cluster while merging two distant clusters).
      </p>
      <p>
        <strong>K-Means++</strong> resolves this by spacing initial centroids apart using a probabilistic distance-squared weighting:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>1. Choose the first center c_1 uniformly at random from X.</p>
        <p>2. For each point x, compute D(x) = min_(j) ‖x - c_j‖ (shortest distance to an existing center).</p>
        <p>3. Choose the next center c_i with probability:</p>
        <p>   P(c_i = x) = D(x)² / Σ_(x&apos; ∈ X) D(x&apos;)²</p>
        <p>4. Repeat steps 2-3 until K centers are chosen.</p>
      </div>

      <p>
        <strong>Theoretical Guarantee:</strong> K-Means++ proves that the expected distortion is bounded by <code>E[J] ≤ 8(ln K + 2) · J_opt</code>, guaranteeing an <code>O(log K)</code> competitive ratio over optimal within polynomial time.
      </p>

      <h2>3 · Density-Based Clustering: DBSCAN (Ester et al., 1996)</h2>
      <p>
        K-Means assumes clusters are convex, isotropic (spherical) Gaussians of comparable variance. When clusters form arbitrary non-linear manifolds (e.g. concentric circles, elongated spirals) or contain noise outliers, K-Means fails.
      </p>
      <p>
        <strong>DBSCAN (Density-Based Spatial Clustering of Applications with Noise)</strong> defines clusters as continuous connected regions of high point density separated by low-density regions, governed by two hyperparameters:
      </p>
      <ul>
        <li><code>ε (Eps):</code> Neighborhood radius distance.</li>
        <li><code>MinPts:</code> Minimum number of points required within the <code>ε</code>-neighborhood to form a dense region.</li>
      </ul>

      <h3>DBSCAN Point Taxonomy</h3>
      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Category</th>
              <th className="px-4 py-3 text-left">Mathematical Condition</th>
              <th className="px-4 py-3 text-left">Role in Cluster</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Core Point</td>
              <td className="px-4 py-3"><code>|N_ε(p)| ≥ MinPts</code></td>
              <td className="px-4 py-3">Initiates a new cluster; dense interior nucleus.</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Border Point</td>
              <td className="px-4 py-3"><code>|N_ε(p)| &lt; MinPts</code>, but <code>p ∈ N_ε(q)</code> for core point <code>q</code></td>
              <td className="px-4 py-3">Appended to the cluster of core point <code>q</code>; outer boundary.</td>
            </tr>
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Noise (Outlier)</td>
              <td className="px-4 py-3">Neither a Core Point nor reachable from any Core Point</td>
              <td className="px-4 py-3">Explicitly assigned label <code>-1</code>; immune to forcing into arbitrary clusters.</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>4 · Intrinsic Validation: The Silhouette Coefficient</h2>
      <p>
        Because clustering is unsupervised, we evaluate partitions using cluster separation and compactness:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p className="text-slate-400">// For observation i in cluster A:</p>
        <p>a(i) = (1 / (|A| - 1)) Σ_(j ∈ A, j ≠ i) ‖x_i - x_j‖   (Mean intra-cluster distance)</p>
        <br />
        <p className="text-slate-400">// For any other cluster C ≠ A:</p>
        <p>d(i, C) = (1 / |C|) Σ_(j ∈ C) ‖x_i - x_j‖</p>
        <p>b(i) = min_(C ≠ A) d(i, C)                          (Mean nearest-cluster distance)</p>
        <br />
        <p className="text-slate-400">// Silhouette Coefficient:</p>
        <p>s(i) = ( b(i) - a(i) ) / max( a(i), b(i) )</p>
      </div>

      <p>
        Properties of <code>s(i) ∈ [-1, +1]</code>:
      </p>
      <ul>
        <li><code>s(i) ≈ +1:</code> <code>a(i) ≪ b(i)</code>. Point is tightly clustered with its peers and far from neighbor clusters.</li>
        <li><code>s(i) ≈ 0:</code> Point lies directly on the decision boundary between two clusters.</li>
        <li><code>s(i) ≈ -1:</code> Point is closer to neighbor cluster than its assigned group (misclustered).</li>
      </ul>

      <h2>5 · Python Implementation from First Principles</h2>
      <TerminalBlock language="python" filename="kmeans_plusplus_scratch.py" code={code} />

      <h2>6 · Further Reading</h2>
      <ul>
        <li>Arthur, D., &amp; Vassilvitskii, S. (2007). <em>k-means++: The advantages of careful seeding</em>. SODA &apos;07, 1027–1035.</li>
        <li>Ester, M., Kriegel, H. P., Sander, J., &amp; Xu, X. (1996). <em>A density-based algorithm for discovering clusters in large spatial databases with noise</em>. KDD-96, 226–231.</li>
        <li>Rousseeuw, P. J. (1987). <em>Silhouettes: a graphical aid to the interpretation and validation of cluster analysis</em>. Journal of Computational and Applied Mathematics, 20, 53-65.</li>
      </ul>
    </article>
  );
}
