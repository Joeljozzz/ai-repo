"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import numpy as np

class XGBoostRegressor:
    """
    Simplified XGBoost Regressor implementing:
    - 2nd-order Taylor expansion objective: obj = Σ [g_i*f_t(x_i) + 1/2*h_i*f_t(x_i)²] + γ*T + 1/2*λ*Σ w_j²
    - Optimal leaf weight: w_j* = - G_j / (H_j + λ)
    - Optimal split score: Gain = 1/2 * [ G_L²/(H_L+λ) + G_R²/(H_R+λ) - G²/(H+λ) ] - γ
    """
    def __init__(self, n_estimators: int = 50, max_depth: int = 3, lr: float = 0.1, reg_lambda: float = 1.0, gamma: float = 0.0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.trees = []
        self.base_pred = 0.0

    class Tree:
        def __init__(self, depth=0, max_depth=3, reg_lambda=1.0, gamma=0.0):
            self.depth = depth
            self.max_depth = max_depth
            self.reg_lambda = reg_lambda
            self.gamma = gamma
            self.feature = None
            self.threshold = None
            self.weight = None
            self.left = None
            self.right = None

        def fit(self, X: np.ndarray, g: np.ndarray, h: np.ndarray):
            G = np.sum(g)
            H = np.sum(h)

            # Stopping condition: compute leaf weight w* = -G / (H + λ)
            if self.depth >= self.max_depth or len(g) <= 2:
                self.weight = - G / (H + self.reg_lambda)
                return

            best_gain = 0.0
            best_feat = None
            best_thresh = None

            current_score = (G ** 2) / (H + self.reg_lambda)

            n_samples, n_features = X.shape
            for feat in range(n_features):
                vals = np.unique(X[:, feat])
                for thr in vals:
                    left_mask = X[:, feat] <= thr
                    right_mask = ~left_mask

                    if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                        continue

                    G_L, H_L = np.sum(g[left_mask]), np.sum(h[left_mask])
                    G_R, H_R = np.sum(g[right_mask]), np.sum(h[right_mask])

                    # XGBoost split gain formula
                    gain = 0.5 * (
                        (G_L ** 2) / (H_L + self.reg_lambda) +
                        (G_R ** 2) / (H_R + self.reg_lambda) -
                        current_score
                    ) - self.gamma

                    if gain > best_gain:
                        best_gain = gain
                        best_feat = feat
                        best_thresh = thr

            if best_gain <= 0.0:
                self.weight = - G / (H + self.reg_lambda)
                return

            self.feature = best_feat
            self.threshold = best_thresh

            left_mask = X[:, self.feature] <= self.threshold
            self.left = XGBoostRegressor.Tree(self.depth + 1, self.max_depth, self.reg_lambda, self.gamma)
            self.left.fit(X[left_mask], g[left_mask], h[left_mask])

            self.right = XGBoostRegressor.Tree(self.depth + 1, self.max_depth, self.reg_lambda, self.gamma)
            self.right.fit(X[~left_mask], g[~left_mask], h[~left_mask])

        def predict_row(self, x: np.ndarray) -> float:
            if self.weight is not None:
                return self.weight
            if x[self.feature] <= self.threshold:
                return self.left.predict_row(x)
            return self.right.predict_row(x)

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Initial constant prediction
        self.base_pred = np.mean(y)
        y_pred = np.full(shape=y.shape, fill_value=self.base_pred)

        for _ in range(self.n_estimators):
            # For Squared Error L(y, y_hat) = 1/2 (y - y_hat)²:
            # First derivative (gradient): g_i = y_pred - y
            # Second derivative (hessian):  h_i = 1.0
            g = y_pred - y
            h = np.ones_like(y)

            tree = self.Tree(max_depth=self.max_depth, reg_lambda=self.reg_lambda, gamma=self.gamma)
            tree.fit(X, g, h)

            tree_preds = np.array([tree.predict_row(x) for x in X])
            y_pred += self.lr * tree_preds
            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = np.full(shape=(X.shape[0],), fill_value=self.base_pred)
        for tree in self.trees:
            tree_preds = np.array([tree.predict_row(x) for x in X])
            y_pred += self.lr * tree_preds
        return y_pred

# Verification
if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.metrics import r2_score

    X, y = make_regression(n_samples=300, n_features=6, noise=5.0, random_state=42)
    model = XGBoostRegressor(n_estimators=30, lr=0.1, reg_lambda=1.5, max_depth=3)
    model.fit(X, y)
    preds = model.predict(X)
    print(f"Scratch XGBoost R² score: {r2_score(y, preds):.4f}")
`;

export default function XGBoostPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-100 text-emerald-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
          Track 1 · Classical Machine Learning
        </span>
      </div>

      <h1>XGBoost &amp; LightGBM: System Architecture and Mathematical Foundations</h1>

      <blockquote>
        <p>
          <em>"XGBoost is not just an algorithmic improvement on Gradient Boosting—it is a co-design of second-order numerical optimization with cache-aware block memory systems."</em> — Tianqi Chen &amp; Carlos Guestrin (2016)
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Derive the 2nd-order Taylor expansion objective of XGBoost using gradients and Hessians.</li>
        <li>Derive the exact closed-form optimal leaf weight <code>w*</code> and split gain formulas.</li>
        <li>Understand the role of complexity penalization hyperparameters (<code>γ</code> for leaf count and <code>λ</code> for L2 leaf regularization).</li>
        <li>Explain LightGBM&apos;s two core innovations: Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB).</li>
        <li>Implement a functional, second-order tree booster from scratch in NumPy.</li>
      </ul>

      <h2>1 · Second-Order Taylor Objective Approximation</h2>
      <p>
        Standard Gradient Boosting relies on the first derivative of the loss function (pseudo-residuals). Tianqi Chen formulated XGBoost by performing a <strong>second-order Taylor expansion</strong> around the prediction at step <code>t-1</code>:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p className="text-slate-400">// Exact objective at step t with regularization Ω(f_t)</p>
        <p>Obj^(t) = Σ_(i=1)^n L(y_i, ŷ_i^(t-1) + f_t(x_i)) + Ω(f_t)</p>
        <br />
        <p className="text-slate-400">// Taylor series: f(x + Δx) ≈ f(x) + f&apos;(x)Δx + ½ f&apos;&apos;(x)Δx²</p>
        <p>Obj^(t) ≈ Σ_(i=1)^n [ L(y_i, ŷ_i^(t-1)) + g_i · f_t(x_i) + ½ h_i · f_t(x_i)² ] + Ω(f_t)</p>
        <br />
        <p className="text-slate-400">// where:</p>
        <p>g_i = ∂ L(y_i, ŷ_i^(t-1)) / ∂ ŷ_i^(t-1)   (Gradient)</p>
        <p>h_i = ∂² L(y_i, ŷ_i^(t-1)) / ∂ (ŷ_i^(t-1))² (Hessian)</p>
      </div>

      <p>
        Dropping the constant term <code>L(y_i, ŷ_i^(t-1))</code> yields the canonical simplified objective:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>Obj̃^(t) = Σ_(i=1)^n [ g_i · f_t(x_i) + ½ h_i · f_t(x_i)² ] + γ · T + ½ λ · Σ_(j=1)^T w_j²</p>
      </div>

      <h2>2 · Closed-Form Optimal Leaf Weights and Split Gain</h2>
      <p>
        Let <code>I_j = &#123;i | q(x_i) = j&#125;</code> represent the index set of data points assigned to leaf <code>j</code>, and define the sums:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>G_j = Σ_(i ∈ I_j) g_i,      H_j = Σ_(i ∈ I_j) h_i</p>
      </div>

      <p>
        The objective can be rewritten by grouping by leaf nodes:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>Obj̃^(t) = Σ_(j=1)^T [ G_j · w_j + ½ (H_j + λ) · w_j² ] + γ · T</p>
      </div>

      <p>
        Taking the derivative with respect to leaf weight <code>w_j</code> and setting it to zero yields the <strong>optimal leaf weight</strong>:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>w_j* = - G_j / (H_j + λ)</p>
      </div>

      <p>
        Substituting <code>w_j*</code> back into the objective yields the <strong>minimal profile score</strong> of tree structure <code>q</code>:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>Score = - ½ · Σ_(j=1)^T [ G_j² / (H_j + λ) ] + γ · T</p>
      </div>

      <p>
        When considering a split of node <code>I</code> into <code>I_L</code> and <code>I_R</code>, the gain is the score of the split children minus the parent node score:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>Gain = ½ [ G_L² / (H_L + λ) + G_R² / (H_R + λ) - (G_L + G_R)² / (H_L + H_R + λ) ] - γ</p>
      </div>

      <p>
        This formula demonstrates the elegance of XGBoost:
      </p>
      <ul>
        <li><strong>γ (Gamma):</strong> The pseudo-regularizer acting as a pruning threshold. If the computed gain is less than <code>γ</code>, the tree refuses to create the split.</li>
        <li><strong>λ (Lambda):</strong> Penalizes excessive leaf weights, attenuating sensitivity to sparse or outlier samples where <code>H_j</code> is small.</li>
      </ul>

      <h2>3 · LightGBM Algorithmic Innovations</h2>
      <p>
        While XGBoost optimized exact split finding with pre-sorted feature bins and sparsity-aware routing, Ke et al. (Microsoft Research, 2017) created LightGBM to scale to millions of rows via two key paradigms:
      </p>

      <h3>3.1 Gradient-based One-Side Sampling (GOSS)</h3>
      <p>
        Samples with small gradients have already been trained well (small error). Dropping them entirely changes the data distribution. GOSS keeps all samples with large gradients (top <code>a × 100%</code>), and randomly samples a fraction <code>b</code> from the remaining small-gradient pool, scaling the small-gradient group by <code>(1 - a) / b</code> to restore the original distribution without bias.
      </p>

      <h3>3.2 Exclusive Feature Bundling (EFB)</h3>
      <p>
        High-dimensional datasets (e.g., text, one-hot encodings) are predominantly sparse. Non-zero values rarely overlap across unrelated features. EFB groups mutually exclusive sparse features into dense composite bundles, reducing the effective feature dimension without information loss.
      </p>

      <h2>4 · Python Implementation of Second-Order Booster</h2>
      <TerminalBlock language="python" filename="xgboost_from_scratch.py" code={code} />

      <h2>5 · Comparison Table</h2>
      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Framework</th>
              <th className="px-4 py-3 text-left">Tree Growth Strategy</th>
              <th className="px-4 py-3 text-left">Split Finding Algorithm</th>
              <th className="px-4 py-3 text-left">Memory Efficiency</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Standard GBM</td>
              <td className="px-4 py-3">Level-wise (balanced)</td>
              <td className="px-4 py-3">Exact greedy search</td>
              <td className="px-4 py-3">Low (stores continuous floats)</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">XGBoost</td>
              <td className="px-4 py-3">Level-wise (with pruning)</td>
              <td className="px-4 py-3">Weighted Quantile Sketch + Histogram</td>
              <td className="px-4 py-3">High (compressed column blocks)</td>
            </tr>
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">LightGBM</td>
              <td className="px-4 py-3">Leaf-wise (best-first)</td>
              <td className="px-4 py-3">Histogram-based + GOSS + EFB</td>
              <td className="px-4 py-3">Extreme (discrete integer bins)</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>6 · Further Reading</h2>
      <ul>
        <li>Chen, T., &amp; Guestrin, C. (2016). <em>XGBoost: A Scalable Tree Boosting System</em>. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785–794).</li>
        <li>Ke, G., et al. (2017). <em>LightGBM: A Highly Efficient Gradient Boosting Decision Tree</em>. Advances in Neural Information Processing Systems (NeurIPS 30), 3146–3154.</li>
      </ul>
    </article>
  );
}
