"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import numpy as np

class PrincipalComponentAnalysis:
    """
    Principal Component Analysis (PCA) implementing:
    1. Centering the data matrix: X̃ = X - μ
    2. Sample Covariance Matrix: Σ = (1 / (N - 1)) X̃^T X̃
    3. Eigendecomposition: Σ v_i = λ_i v_i
    4. SVD alternative: X̃ = U Σ V^T (numerically superior)
    5. Projection to top-k principal subspace: Z = X̃ · W_k
    """
    def __init__(self, n_components: int = 2):
        self.k = n_components
        self.components = None
        self.explained_variance_ratio_ = None
        self.mean = None

    def fit(self, X: np.ndarray):
        n_samples, n_features = X.shape

        # Step 1: Center the data (mean = 0)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Step 2: Compute sample covariance matrix
        cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)

        # Step 3: Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Step 4: Sort eigenvalues and eigenvectors descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Step 5: Store top-k components and explained variance ratio
        self.components = eigenvectors[:, :self.k]
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues[:self.k] / total_var

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Project centered data onto principal eigenvectors: Z = (X - μ) · W
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        # Reconstruct high-dimensional approximations: X̂ = Z · W^T + μ
        return np.dot(Z, self.components.T) + self.mean

if __name__ == "__main__":
    from sklearn.datasets import load_iris

    data = load_iris()
    X = data.data

    pca = PrincipalComponentAnalysis(n_components=2)
    pca.fit(X)
    Z = pca.transform(X)

    print(f"Original shape:              {X.shape}")
    print(f"Reduced 2D subspace shape:   {Z.shape}")
    print(f"Explained Variance Ratio:    {pca.explained_variance_ratio_}")
    print(f"Total variance preserved:    {np.sum(pca.explained_variance_ratio_):.2%}")
`;

export default function DimReductionPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-100 text-emerald-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
          Track 1 · Classical Machine Learning
        </span>
      </div>

      <h1>Dimensionality Reduction: PCA, t-SNE, and Manifold Learning</h1>

      <blockquote>
        <p>
          <em>"Principal Component Analysis identifies the orthogonal axes that simultaneously maximize projected variance and minimize reconstruction mean squared error."</em> — Karl Pearson (1901) &amp; Harold Hotelling (1933)
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Derive Principal Component Analysis from the variance maximization objective using Lagrange multipliers.</li>
        <li>Prove the equivalence between covariance eigendecomposition and Singular Value Decomposition (SVD) of the centered data matrix.</li>
        <li>Explain the Curse of Dimensionality (exponential volume growth, distance concentration phenomenon).</li>
        <li>Understand non-linear manifold learning: t-SNE (Student-t distribution, KL divergence minimization) and UMAP.</li>
        <li>Implement PCA from scratch in NumPy with projection and reconstruction methods.</li>
      </ul>

      <h2>1 · The Curse of Dimensionality</h2>
      <p>
        As dimensionality <code>d</code> increases, data behavior becomes counterintuitive:
      </p>
      <ul>
        <li><strong>Exponential Volume:</strong> A unit hypercube <code>[0, 1]^d</code> has volume 1. The volume of an inscribed hypersphere of radius <code>0.5</code> shrinks to zero: <code>lim_(d → ∞) V_sphere(d) = 0</code>. Almost all mass concentrates in the outer corners.</li>
        <li><strong>Distance Concentration:</strong> For random points uniformly distributed in high dimensions, the ratio between the maximum pairwise distance and minimum pairwise distance converges to 1:
          <div className="not-prose bg-slate-900 rounded-lg p-3 font-mono text-slate-100 text-xs my-2">
            lim_(d → ∞) [ dist_max - dist_min ] / dist_min = 0
          </div>
          Every point becomes roughly equidistant from every other point, causing distance-based metrics (kNN, Euclidean clustering) to degrade.
        </li>
      </ul>

      <h2>2 · PCA: Variance Maximization Derivation</h2>
      <p>
        Let <code>X̃ ∈ ℝ^(n × d)</code> denote the centered data matrix (mean zero). We seek a unit vector <code>w ∈ ℝ^d</code> (<code>‖w‖₂ = 1</code>) such that the projected coordinates <code>z = X̃ w</code> have maximum empirical sample variance:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>Var(z) = (1 / (n - 1)) z^T z</p>
        <p>       = (1 / (n - 1)) (X̃ w)^T (X̃ w)</p>
        <p>       = w^T [ (X̃^T X̃) / (n - 1) ] w</p>
        <p>       = w^T Σ w</p>
        <br />
        <p>where Σ is the sample covariance matrix.</p>
      </div>

      <p>
        We formulate the constrained optimization problem via Lagrange multipliers:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>max_w  L(w, λ) = w^T Σ w - λ (w^T w - 1)</p>
        <br />
        <p>Setting the partial derivative with respect to w to zero:</p>
        <p>∇_w L = 2 Σ w - 2 λ w = 0</p>
        <br />
        <p>Σ w = λ w</p>
      </div>

      <p>
        <strong>The Fundamental Theorem of PCA:</strong> The optimal projection vector <code>w</code> is an <strong>eigenvector</strong> of the covariance matrix <code>Σ</code>, and the variance along that axis equals its corresponding <strong>eigenvalue</strong> <code>λ</code>.
      </p>

      <h2>3 · Equivalence to Singular Value Decomposition (SVD)</h2>
      <p>
        Computing <code>Σ = (1/(n-1)) X̃^T X̃</code> explicitly requires <code>O(n · d²)</code> operations and can square the matrix condition number, worsening numerical instability.
      </p>
      <p>
        Modern libraries compute PCA directly via SVD of the centered data matrix:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>X̃ = U · S · V^T</p>
        <br />
        <p>Then the covariance matrix is:</p>
        <p>Σ = (1 / (n - 1)) X̃^T X̃ = (1 / (n - 1)) (V S U^T) (U S V^T) = V [ S² / (n - 1) ] V^T</p>
      </div>

      <p>
        The right singular vectors <code>V</code> are exactly the principal eigenvectors, and the eigenvalues relate to singular values by <code>λ_i = s_i² / (n - 1)</code>.
      </p>

      <h2>4 · Non-Linear Manifold Learning: t-SNE</h2>
      <p>
        PCA is a linear projection. If the data lies on a curved, non-linear manifold (e.g. Swiss roll), PCA cannot unroll it.
      </p>
      <p>
        <strong>t-SNE (van der Maaten &amp; Hinton, 2008)</strong> converts Euclidean distances into conditional probabilities:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p className="text-slate-400">// High-dimensional similarity (Gaussian distribution):</p>
        <p>p_(j|i) = exp( - ‖x_i - x_j‖² / 2σ_i² ) / Σ_(k ≠ i) exp( - ‖x_i - x_k‖² / 2σ_i² )</p>
        <br />
        <p className="text-slate-400">// Low-dimensional 2D similarity (Student-t distribution with 1 DOF):</p>
        <p>q_ij = (1 + ‖y_i - y_j‖²)⁻¹ / Σ_(k ≠ l) (1 + ‖y_k - y_l‖²)⁻¹</p>
        <br />
        <p className="text-slate-400">// Optimization: Minimize Kullback-Leibler (KL) Divergence via gradient descent</p>
        <p>KL(P || Q) = Σ_i Σ_j p_ij · ln( p_ij / q_ij )</p>
      </div>

      <p>
        The heavy tails of the Student-t distribution in the low-dimensional map solve the <strong>crowding problem</strong>, allowing distinct clusters to separate cleanly in 2D visualizations.
      </p>

      <h2>5 · Python Implementation of PCA</h2>
      <TerminalBlock language="python" filename="pca_from_scratch.py" code={code} />

      <h2>6 · Further Reading</h2>
      <ul>
        <li>Pearson, K. (1901). <em>On lines and planes of closest fit to systems of points in space</em>. Philosophical Magazine, 2(11), 559–572.</li>
        <li>van der Maaten, L., &amp; Hinton, G. (2008). <em>Visualizing data using t-SNE</em>. Journal of Machine Learning Research, 9, 2579–2605.</li>
        <li>McInnes, L., Healy, J., &amp; Melville, J. (2018). <em>UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction</em>. arXiv:1802.03426.</li>
      </ul>
    </article>
  );
}
