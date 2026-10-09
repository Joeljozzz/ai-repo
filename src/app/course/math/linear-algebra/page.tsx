"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import numpy as np

# ── Vectors & Dot Product ────────────────────────────────
a = np.array([2.0, 3.0, 1.0])
b = np.array([1.0, 0.0, 4.0])

dot_product = np.dot(a, b)          # 2·1 + 3·0 + 1·4 = 6
cosine_sim  = dot_product / (np.linalg.norm(a) * np.linalg.norm(b))
print(f"Dot product:     {dot_product}")
print(f"Cosine similarity: {cosine_sim:.4f}")

# ── Matrix Multiplication ─────────────────────────────────
A = np.array([[1, 2], [3, 4], [5, 6]])   # (3×2)
B = np.array([[7, 8, 9], [10, 11, 12]])  # (2×3)
C = A @ B                                # (3×3)
print(f"\\nA @ B =\\n{C}")

# ── Eigendecomposition ────────────────────────────────────
# For a symmetric matrix, eigenvectors are orthogonal
M = np.array([[4.0, 2.0], [2.0, 3.0]])
eigenvalues, eigenvectors = np.linalg.eig(M)
print(f"\\nEigenvalues:  {eigenvalues}")
print(f"Eigenvectors:\\n{eigenvectors}")

# ── Singular Value Decomposition (SVD) ────────────────────
# Every matrix A can be decomposed as U Σ Vᵀ
X = np.random.randn(100, 5)           # 100 samples, 5 features
U, S, Vt = np.linalg.svd(X, full_matrices=False)
print(f"\\nSVD shapes: U={U.shape}, S={S.shape}, Vt={Vt.shape}")

# Reconstruct with top-2 singular values (dimensionality reduction)
k = 2
X_reduced = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
print(f"Rank-{k} approximation shape: {X_reduced.shape}")

# Explained variance (how much info is preserved)
explained_var = np.sum(S[:k]**2) / np.sum(S**2)
print(f"Variance explained by top-{k} components: {explained_var:.2%}")`;

export default function LinearAlgebraPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">

      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-sky-100 text-sky-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-sky-500"></span>
          Track 0 · Mathematical Foundations
        </span>
      </div>

      <h1>Linear Algebra for Machine Learning</h1>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Represent data, weights, and transformations as vectors and matrices.</li>
        <li>Compute and interpret dot products, matrix products, and their geometric meaning.</li>
        <li>Perform and interpret eigendecomposition and SVD — the two most important matrix factorisations in ML.</li>
        <li>Understand how PCA is derived directly from SVD.</li>
        <li>Apply these operations in NumPy.</li>
      </ul>

      <h2>1 · Vectors</h2>
      <p>
        In machine learning, a single data point (one row of your dataset) is represented as a
        <strong> column vector</strong> x ∈ ℝⁿ. Every feature is one dimension. A model&apos;s
        weight vector w ∈ ℝⁿ lives in exactly the same space.
      </p>
      <p>
        The <strong>dot product</strong> (inner product) of two vectors a and b is:
      </p>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 space-y-1">
        <p>a · b = Σᵢ aᵢbᵢ = |a| |b| cos θ</p>
      </div>
      <p>
        The geometric interpretation — |a| |b| cos θ — is fundamental to understanding how neural
        networks work. When a and b are unit vectors, the dot product equals the cosine similarity,
        which measures the angle between them. This is the exact operation used in:
      </p>
      <ul>
        <li>Attention mechanisms (Q · Kᵀ)</li>
        <li>Vector database similarity search</li>
        <li>The forward pass of every linear layer: z = w·x + b</li>
      </ul>

      <h2>2 · Matrices as Linear Transformations</h2>
      <p>
        A matrix A ∈ ℝᵐˣⁿ represents a linear map from ℝⁿ → ℝᵐ. Multiplying a data matrix
        X ∈ ℝᴺˣⁿ (N samples, n features) by a weight matrix W ∈ ℝⁿˣᵐ produces a new matrix
        Z = XW ∈ ℝᴺˣᵐ (N samples, m output dimensions).
      </p>
      <p>This is exactly what a single dense (linear) layer of a neural network computes.</p>

      <h3>Important Matrix Properties</h3>
      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Property</th>
              <th className="px-4 py-3 text-left">Definition</th>
              <th className="px-4 py-3 text-left">ML Relevance</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white"><td className="px-4 py-3 font-semibold">Transpose Aᵀ</td><td className="px-4 py-3">(Aᵀ)ᵢⱼ = Aⱼᵢ</td><td className="px-4 py-3">Switching rows/cols; backprop gradients</td></tr>
            <tr className="bg-slate-50"><td className="px-4 py-3 font-semibold">Inverse A⁻¹</td><td className="px-4 py-3">AA⁻¹ = I</td><td className="px-4 py-3">Closed-form OLS: θ = (XᵀX)⁻¹Xᵀy</td></tr>
            <tr className="bg-white"><td className="px-4 py-3 font-semibold">Rank</td><td className="px-4 py-3">Dim of column space</td><td className="px-4 py-3">Rank deficiency → multicollinearity issues</td></tr>
            <tr className="bg-slate-50"><td className="px-4 py-3 font-semibold">Trace</td><td className="px-4 py-3">Tr(A) = Σᵢ Aᵢᵢ</td><td className="px-4 py-3">Sum of eigenvalues; used in regularisation proofs</td></tr>
            <tr className="bg-white"><td className="px-4 py-3 font-semibold">Determinant</td><td className="px-4 py-3">det(A)</td><td className="px-4 py-3">Volume scaling factor; zero → singular matrix</td></tr>
          </tbody>
        </table>
      </div>

      <h2>3 · Eigendecomposition</h2>
      <p>
        A non-zero vector v is an <strong>eigenvector</strong> of square matrix A if:
      </p>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4">
        <p>A v = λ v</p>
      </div>
      <p>
        λ is the corresponding <strong>eigenvalue</strong>. The eigenvector direction is unchanged
        by the transformation A — it is only scaled by λ. If A is real and symmetric (like a
        covariance matrix), all eigenvalues are real and all eigenvectors are mutually orthogonal.
      </p>
      <p>
        This leads to the <strong>eigendecomposition</strong>:
      </p>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4">
        <p>A = Q Λ Qᵀ</p>
        <p className="text-slate-400 text-xs mt-2">Q = matrix of eigenvectors (orthogonal), Λ = diagonal matrix of eigenvalues</p>
      </div>
      <p>
        <strong>Why it matters:</strong> Principal Component Analysis (PCA) computes the
        eigendecomposition of the data covariance matrix. The eigenvectors are the principal
        components (directions of maximum variance); the eigenvalues tell you how much variance
        each component explains.
      </p>

      <h2>4 · Singular Value Decomposition (SVD)</h2>
      <p>
        SVD generalises eigendecomposition to any matrix (not just square symmetric ones):
      </p>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 space-y-1">
        <p>A = U Σ Vᵀ</p>
        <p className="text-slate-400 text-xs mt-2">U ∈ ℝᵐˣᵐ: left singular vectors (orthogonal)</p>
        <p className="text-slate-400 text-xs">Σ ∈ ℝᵐˣⁿ: singular values on diagonal, sorted descending</p>
        <p className="text-slate-400 text-xs">Vᵀ ∈ ℝⁿˣⁿ: right singular vectors (orthogonal)</p>
      </div>
      <p>
        <strong>Truncated SVD</strong> keeps only the top-k singular values and their vectors,
        giving the best rank-k approximation of A in the Frobenius norm sense (Eckart–Young
        theorem). This is the engine behind:
      </p>
      <ul>
        <li><strong>PCA</strong> (SVD of the centred data matrix)</li>
        <li><strong>Latent Semantic Analysis</strong> (document-term matrices)</li>
        <li><strong>LoRA</strong> (Low-Rank Adaptation for LLMs — decompose weight updates into two low-rank matrices)</li>
        <li><strong>Recommender systems</strong> (matrix factorisation on user-item matrices)</li>
      </ul>

      <h2>5 · Norms</h2>
      <p>
        A norm ‖x‖ measures the "size" or "length" of a vector. The p-norm is:
      </p>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 space-y-1">
        <p>‖x‖_p = (Σᵢ |xᵢ|ᵖ)^(1/p)</p>
        <p></p>
        <p>‖x‖₁  (L1 / Manhattan):  Σᵢ |xᵢ|          → Lasso regularisation</p>
        <p>‖x‖₂  (L2 / Euclidean):  √(Σᵢ xᵢ²)        → Ridge regularisation, KNN</p>
        <p>‖x‖₀  (L0):              count of non-zeros → sparsity (NP-hard to optimise)</p>
      </div>

      <h2>6 · Python Implementation</h2>
      <TerminalBlock language="python" filename="linear_algebra.py" code={code} />

      <h2>7 · Common Pitfalls</h2>
      <ul>
        <li><strong>Forgetting to centre data before PCA.</strong> PCA decomposes variance. If you don&apos;t subtract the column means first, the first PC will simply point toward the data centroid, capturing mean offset rather than variance structure.</li>
        <li><strong>Confusing eigendecomposition with SVD.</strong> Eigendecomposition requires a square matrix; SVD works on any shape. For a symmetric positive semi-definite matrix (like a covariance matrix), the singular values equal the eigenvalues.</li>
        <li><strong>Using the matrix inverse naively.</strong> Computing A⁻¹ is numerically unstable and O(n³). For solving Ax = b, always use <code>np.linalg.solve(A, b)</code> instead.</li>
      </ul>

      <h2>8 · Further Reading</h2>
      <ul>
        <li>Strang, G. (2016). <em>Introduction to Linear Algebra</em>, 5th ed. Wellesley-Cambridge Press.</li>
        <li>Goodfellow et al. (2016). <em>Deep Learning</em>, Chapter 2: Linear Algebra.</li>
        <li>NumPy <code>np.linalg</code> module documentation.</li>
      </ul>

    </article>
  );
}
