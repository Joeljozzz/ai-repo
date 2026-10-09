"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import numpy as np

class LinearSVM:
    """
    Support Vector Machine binary classifier using Hinge Loss and Stochastic Gradient Descent.
    Finds the maximum-margin hyperplane w·x - b = 0 separating classes y ∈ {-1, +1}.
    """
    def __init__(self, learning_rate: float = 0.001, lambda_param: float = 0.01, n_iters: int = 1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param  # Regularization parameter (1 / C)
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Format labels into {-1, +1}
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Functional margin condition: y_i * (w·x_i - b) >= 1
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    # Point correctly outside the margin: only L2 regularizer contributes to gradient
                    dw = 2 * self.lambda_param * self.w
                    db = 0.0
                else:
                    # Margin violation or misclassification: hinge loss penalty active
                    dw = 2 * self.lambda_param * self.w - y_[idx] * x_i
                    db = y_[idx]

                # Parameter update
                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw_output = np.dot(X, self.w) - self.b
        return np.where(raw_output >= 0, 1, 0)

# Verification with scikit-learn SVC
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score

    X, y = make_blobs(n_samples=200, centers=2, random_state=42, cluster_std=1.2)
    
    svm = LinearSVM(learning_rate=0.001, lambda_param=0.01, n_iters=1500)
    svm.fit(X, y)
    preds = svm.predict(X)
    print(f"Scratch Linear SVM Accuracy: {accuracy_score(y, preds):.4f}")

    sk_svm = SVC(kernel="linear", C=1.0 / (2 * 0.01))
    sk_svm.fit(X, y)
    print(f"Scikit-learn Linear SVC Accuracy: {accuracy_score(y, sk_svm.predict(X)):.4f}")
`;

export default function SVMPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-100 text-emerald-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
          Track 1 · Classical Machine Learning
        </span>
      </div>

      <h1>Support Vector Machines (SVM)</h1>

      <blockquote>
        <p>
          <em>"The goal of Support Vector Machines is not just to find any separating boundary, but the unique hyperplane that maximizes the generalization margin."</em> — Vladimir Vapnik
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Derive the hard-margin and soft-margin primal optimization objectives from geometric distance principles.</li>
        <li>Formulate the Lagrangian dual using KKT multipliers and understand why sparsity emerges in support vectors.</li>
        <li>State Mercer&apos;s Theorem and explain how kernel functions compute inner products in infinite-dimensional Hilbert spaces.</li>
        <li>Understand the trade-offs of the regularization parameter C and kernel coefficient gamma.</li>
        <li>Implement a functional linear SVM from first principles with stochastic subgradient descent.</li>
      </ul>

      <h2>1 · The Geometry of Maximum Margin Separation</h2>
      <p>
        Consider a binary classification problem with linearly separable data points <code>(x_i, y_i)</code> where <code>x_i ∈ ℝ^d</code> and <code>y_i ∈ &#123;-1, +1&#125;</code>. An infinite number of hyperplanes can separate these classes. The Perceptron algorithm finds an arbitrary separating hyperplane, which often generalizes poorly if it passes dangerously close to training examples.
      </p>
      <p>
        Support Vector Machines identify the hyperplane <code>w · x + b = 0</code> that maximizes the <strong>geometric margin</strong> <code>γ</code>, defined as the shortest orthogonal distance between the hyperplane and any training sample:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p className="text-slate-400">// Geometric distance from a point x_i to hyperplane w·x + b = 0</p>
        <p>γ_i = y_i · (w · x_i + b) / ‖w‖₂</p>
        <br />
        <p className="text-slate-400">// Normalizing such that the closest point satisfies y_i · (w · x_i + b) = 1:</p>
        <p>Total Margin = 2 / ‖w‖₂</p>
      </div>

      <p>
        Maximizing <code>2 / ‖w‖</code> is mathematically equivalent to minimizing <code>½ ‖w‖²</code>. This gives the classical <strong>Hard-Margin Primal Quadratic Program</strong>:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>min_(w, b)  ½ ‖w‖²</p>
        <p>subject to: y_i · (w · x_i + b) ≥ 1,   ∀ i ∈ &#123;1, ..., n&#125;</p>
      </div>

      <h2>2 · The Soft-Margin Formulation and Slack Variables</h2>
      <p>
        In real-world data, classes overlap or contain noise, making exact linear separability impossible. Corinna Cortes and Vladimir Vapnik introduced non-negative slack variables <code>ξ_i ≥ 0</code> to penalize margin violations:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>min_(w, b, ξ)  ½ ‖w‖² + C · Σ_(i=1)^n ξ_i</p>
        <p>subject to:   y_i · (w · x_i + b) ≥ 1 - ξ_i,   ∀ i</p>
        <p>              ξ_i ≥ 0,                         ∀ i</p>
      </div>

      <p>
        The hyperparameter <strong>C &gt; 0</strong> governs the regularization trade-off:
      </p>
      <ul>
        <li><strong>High C:</strong> Heavy penalty on misclassifications. Forces narrower margins, risking overfitting and high variance.</li>
        <li><strong>Low C:</strong> Tolerates margin violations. Encourages wider margins, improving generalization at the expense of potential bias.</li>
      </ul>

      <h2>3 · Lagrangian Duality and KKT Conditions</h2>
      <p>
        By introducing Lagrange multipliers <code>α_i ≥ 0</code> and <code>r_i ≥ 0</code>, we construct the Lagrangian:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>L(w, b, ξ, α, r) = ½ ‖w‖² + C Σ ξ_i - Σ α_i [y_i(w · x_i + b) - 1 + ξ_i] - Σ r_i ξ_i</p>
      </div>

      <p>
        Setting the partial derivatives with respect to primal variables <code>w</code>, <code>b</code>, and <code>ξ</code> to zero:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>∂L/∂w = 0  ⟹  w = Σ_(i=1)^n α_i y_i x_i</p>
        <p>∂L/∂b = 0  ⟹  Σ_(i=1)^n α_i y_i = 0</p>
        <p>∂L/∂ξ_i = 0 ⟹  C - α_i - r_i = 0  ⟹  0 ≤ α_i ≤ C</p>
      </div>

      <p>
        Substituting these relations back yields the <strong>Dual Optimization Problem</strong>:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>max_α  Σ_(i=1)^n α_i - ½ Σ_(i=1)^n Σ_(j=1)^n α_i α_j y_i y_j (x_i · x_j)</p>
        <p>subject to: 0 ≤ α_i ≤ C,   ∀ i</p>
        <p>            Σ_(i=1)^n α_i y_i = 0</p>
      </div>

      <p>
        The Karush-Kuhn-Tucker (KKT) complementary slackness conditions state:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>α_i · [y_i(w · x_i + b) - 1 + ξ_i] = 0</p>
      </div>

      <p>
        This establishes the core sparsity property of SVMs:
      </p>
      <ul>
        <li>For points strictly beyond the margin, <code>y_i(w · x_i + b) &gt; 1</code>, which forces <code>α_i = 0</code>. These points have zero influence on the decision boundary.</li>
        <li>Only points on or inside the margin satisfy <code>α_i &gt; 0</code>. These are the <strong>Support Vectors</strong>.</li>
      </ul>

      <h2>4 · The Kernel Trick and Mercer&apos;s Theorem</h2>
      <p>
        Notice that in the dual problem and inference formula <code>f(x) = sign(Σ α_i y_i (x_i · x) + b)</code>, the feature vectors appear strictly as <strong>inner products</strong> <code>x_i · x_j</code>.
      </p>
      <p>
        If a dataset requires a non-linear mapping <code>Φ: ℝ^d → ℋ</code> into a higher-dimensional Hilbert space, computing <code>Φ(x_i) · Φ(x_j)</code> directly could be computationally prohibitive or impossible (if <code>dim(ℋ) = ∞</code>).
      </p>
      <p>
        <strong>Mercer&apos;s Theorem</strong> guarantees that any continuous, symmetric, positive semi-definite function <code>K(x_i, x_j)</code> corresponds to an inner product in some reproducing kernel Hilbert space:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>K(x_i, x_j) = ⟨Φ(x_i), Φ(x_j)⟩_ℋ</p>
      </div>

      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Kernel</th>
              <th className="px-4 py-3 text-left">Formula</th>
              <th className="px-4 py-3 text-left">Hyperparameters</th>
              <th className="px-4 py-3 text-left">Characteristic Space</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Linear</td>
              <td className="px-4 py-3 font-mono">x_i · x_j</td>
              <td className="px-4 py-3">None</td>
              <td className="px-4 py-3">Original d dimensions (fastest)</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Polynomial</td>
              <td className="px-4 py-3 font-mono">(γ x_i · x_j + c)^d</td>
              <td className="px-4 py-3">degree d, scale γ, intercept c</td>
              <td className="px-4 py-3">Combinatorial monomial features</td>
            </tr>
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Radial Basis Function (RBF)</td>
              <td className="px-4 py-3 font-mono">exp(-γ ‖x_i - x_j‖²)</td>
              <td className="px-4 py-3">bandwidth γ = 1 / (2σ²)</td>
              <td className="px-4 py-3">Infinite-dimensional Hilbert space</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Sigmoid</td>
              <td className="px-4 py-3 font-mono">tanh(γ x_i · x_j + c)</td>
              <td className="px-4 py-3">scale γ, intercept c</td>
              <td className="px-4 py-3">Equated to two-layer MLP</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>5 · Python Implementation from First Principles</h2>
      <TerminalBlock language="python" filename="svm_implementation.py" code={code} />

      <h2>6 · Common Pitfalls in Practice</h2>
      <ul>
        <li>
          <strong>Omitting feature standardization:</strong> SVM boundaries depend on Euclidean distances. If one feature has scale [0, 1000] and another [0, 1], the unscaled feature dominates the margin calculation entirely.
        </li>
        <li>
          <strong>Training with massive sample sizes (N &gt; 100,000):</strong> Solving the dual quadratic program via sequential minimal optimization (SMO) scales between <code>O(N²)</code> and <code>O(N³)</code>. For huge datasets, prefer LinearSVC (LibLinear) or SGDClassifier with hinge loss.
        </li>
        <li>
          <strong>Overfitting via large gamma in RBF kernels:</strong> When gamma is excessively high, the Gaussian bell curve becomes a tight spike around individual training points, resulting in islands of decision regions with poor generalization.
        </li>
      </ul>

      <h2>7 · Further Reading</h2>
      <ul>
        <li>Cortes, C., &amp; Vapnik, V. (1995). <em>Support-Vector Networks</em>. Machine Learning, 20(3), 273-297.</li>
        <li>Schölkopf, B., &amp; Smola, A. J. (2002). <em>Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond</em>. MIT Press.</li>
        <li>Platt, J. (1998). <em>Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines</em>. Microsoft Research Technical Report.</li>
      </ul>
    </article>
  );
}
