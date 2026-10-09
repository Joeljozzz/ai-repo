"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import numpy as np
from collections import Counter

class DecisionStumpOrTree:
    """
    Lightweight CART tree for Random Forest ensemble.
    Implements feature subsampling (m = sqrt(p)) at each split.
    """
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2, max_features: int = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree = None

    def _gini(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None, 0.0

        # Random feature subsampling: select random subset of feature indices
        m = self.max_features if self.max_features else int(np.sqrt(n_features))
        feature_indices = np.random.choice(n_features, size=m, replace=False)

        best_gini = self._gini(y)
        best_gain = 0.0
        best_feat, best_thresh = None, None

        for feat in feature_indices:
            thresholds = np.unique(X[:, feat])
            for thr in thresholds:
                left_mask = X[:, feat] <= thr
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                gini_left = self._gini(y[left_mask])
                gini_right = self._gini(y[right_mask])
                weighted_gini = (np.sum(left_mask) / n_samples) * gini_left + (np.sum(right_mask) / n_samples) * gini_right
                gain = best_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thr

        return best_feat, best_thresh, best_gain

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split:
            return {"leaf": True, "value": Counter(y).most_common(1)[0][0]}

        feat, thresh, gain = self._best_split(X, y)
        if feat is None or gain <= 1e-7:
            return {"leaf": True, "value": Counter(y).most_common(1)[0][0]}

        left_mask = X[:, feat] <= thresh
        return {
            "leaf": False,
            "feature": feat,
            "threshold": thresh,
            "left": self._build(X[left_mask], y[left_mask], depth + 1),
            "right": self._build(X[~left_mask], y[~left_mask], depth + 1)
        }

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.tree = self._build(X, y)

    def _predict_sample(self, node: dict, x: np.ndarray):
        if node["leaf"]:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_sample(node["left"], x)
        return self._predict_sample(node["right"], x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_sample(self.tree, x) for x in X])

class RandomForestClassifier:
    """
    Random Forest ensemble algorithm (Leo Breiman, 2001).
    - Bootstrap Aggregation (Bagging): samples N observations with replacement.
    - Random Feature Subspaces: considers m = sqrt(p) random features per split.
    - Out-Of-Bag (OOB) error computation without a separate validation set.
    """
    def __init__(self, n_estimators: int = 50, max_depth: int = 8, min_samples_split: int = 2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.oob_score_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.trees = []
        
        # Track out-of-bag predictions: dict mapping sample index -> list of predictions
        oob_predictions = {i: [] for i in range(n_samples)}

        for _ in range(self.n_estimators):
            # 1. Bootstrap sample (with replacement)
            boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            oob_idx = list(set(range(n_samples)) - set(boot_idx))

            X_boot, y_boot = X[boot_idx], y[boot_idx]

            tree = DecisionStumpOrTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=int(np.sqrt(n_features))
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

            # Record predictions for out-of-bag instances
            if len(oob_idx) > 0:
                preds = tree.predict(X[oob_idx])
                for idx, pred in zip(oob_idx, preds):
                    oob_predictions[idx].append(pred)

        # 2. Compute Out-Of-Bag (OOB) accuracy
        correct = 0
        total_eval = 0
        for i in range(n_samples):
            if len(oob_predictions[i]) > 0:
                majority = Counter(oob_predictions[i]).most_common(1)[0][0]
                if majority == y[i]:
                    correct += 1
                total_eval += 1

        self.oob_score_ = correct / total_eval if total_eval > 0 else 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        all_tree_preds = np.array([tree.predict(X) for tree in self.trees]) # (B, N)
        # Majority voting across all B trees
        final_preds = []
        for j in range(X.shape[0]):
            votes = all_tree_preds[:, j]
            final_preds.append(Counter(votes).most_common(1)[0][0])
        return np.array(final_preds)

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X, y = make_classification(n_samples=500, n_features=12, n_informative=8, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=30, max_depth=6)
    rf.fit(X_train, y_train)

    test_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"Random Forest Out-of-Bag (OOB) Score: {rf.oob_score_:.4f}")
    print(f"Random Forest Test Set Accuracy:      {test_acc:.4f}")
`;

export default function RandomForestsPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-100 text-emerald-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
          Track 1 · Classical Machine Learning
        </span>
      </div>

      <h1>Random Forests &amp; Bagging: Ensemble Theory, Decorrelation, and OOB Analysis</h1>

      <blockquote>
        <p>
          <em>"Random Forests combine bootstrap aggregation with random feature selection to construct an ensemble of decorrelated decision trees, dramatically shrinking variance without inflating bias."</em> — Leo Breiman (2001)
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Derive the variance reduction formula for an average of <code>B</code> identically distributed, correlated random variables.</li>
        <li>Explain why Bootstrap Aggregation (Bagging) alone cannot eliminate correlation when dominant features exist.</li>
        <li>Understand the random subspace projection heuristic (<code>m = √p</code> for classification, <code>m = p/3</code> for regression).</li>
        <li>Prove that each bootstrap sample leaves approximately <code>36.8%</code> of samples out-of-bag (OOB).</li>
        <li>Implement a full Random Forest classifier from scratch in NumPy with built-in OOB error evaluation.</li>
      </ul>

      <h2>1 · Variance of Correlated Ensembles: The Mathematical Foundation</h2>
      <p>
        Consider an ensemble of <code>B</code> decision trees, each trained on a bootstrap sample of the dataset. Let each tree make prediction <code>T_b(x)</code> with individual variance <code>σ²</code> and pairwise correlation <code>ρ = Corr(T_i(x), T_j(x))</code> for <code>i ≠ j</code>.
      </p>
      <p>
        The variance of the average ensemble prediction <code>T̄(x) = (1/B) Σ_(b=1)^B T_b(x)</code> is:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>Var( T̄(x) ) = Var( (1/B) Σ_(b=1)^B T_b(x) )</p>
        <p>            = (1 / B²) [ Σ_(b=1)^B Var(T_b(x)) + Σ_(i ≠ j) Cov(T_i(x), T_j(x)) ]</p>
        <p>            = (1 / B²) [ B · σ² + B(B - 1) · ρ · σ² ]</p>
        <br />
        <p>Var( T̄(x) ) = ρ · σ² + [ (1 - ρ) / B ] · σ²</p>
      </div>

      <p>
        Analyzing the limit as the number of trees grows large:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>lim_(B → ∞) Var( T̄(x) ) = ρ · σ²</p>
      </div>

      <p>
        <strong>The Key Insight:</strong> As <code>B → ∞</code>, the second term vanishes entirely. However, the irreducible floor of ensemble variance is bounded by <strong><code>ρ · σ²</code></strong>.
      </p>
      <ul>
        <li>Standard Bagging (Breiman 1996) samples rows with replacement, but all trees still have access to all <code>p</code> features. If one feature is overwhelmingly strong (e.g. high mutual information), almost every tree will split on it at the root, leading to high pairwise correlation <code>ρ &gt; 0.7</code>.</li>
        <li><strong>Random Forests (Breiman 2001)</strong> intentionally restrict candidate features at each split to a random subset of size <code>m ≪ p</code>. This decorrelates the trees, drastically reducing <code>ρ</code> and lowering overall ensemble variance.</li>
      </ul>

      <h2>2 · The 63.2% / 36.8% Rule: Out-of-Bag (OOB) Mathematics</h2>
      <p>
        Suppose our training set contains <code>N</code> distinct observations. Each bootstrap draw picks an observation uniformly at random with probability <code>1/N</code>.
      </p>
      <p>
        The probability that a specific observation <code>x_i</code> is <strong>not</strong> chosen in a single draw is <code>1 - 1/N</code>. In a bootstrap sample of size <code>N</code>, the probability that <code>x_i</code> is never chosen is:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>P(x_i ∉ Bootstrap Sample) = ( 1 - 1/N )^N</p>
        <br />
        <p>Using the fundamental limit identity lim_(n → ∞) (1 - 1/n)^n = e^(-1):</p>
        <p>lim_(N → ∞) ( 1 - 1/N )^N = 1 / e ≈ 0.367879... (≈ 36.8%)</p>
      </div>

      <p>
        Therefore, each bootstrap sample uses approximately <strong>63.2%</strong> unique observations and leaves <strong>36.8%</strong> out-of-bag.
      </p>
      <p>
        <strong>Practical Consequence:</strong> For every observation <code>x_i</code>, approximately <code>B / e</code> trees were never trained on it. By aggregating predictions for <code>x_i</code> only from the trees that excluded it, Random Forests provide an unbiased estimate of generalization error (the <strong>OOB Score</strong>) without needing cross-validation or a separate validation split.
      </p>

      <h2>3 · Feature Importance Metrics</h2>
      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Method</th>
              <th className="px-4 py-3 text-left">Mechanism</th>
              <th className="px-4 py-3 text-left">Pros &amp; Cons</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Mean Decrease in Impurity (MDI)</td>
              <td className="px-4 py-3">Total Gini impurity reduction weighted by samples across all splits where feature is used.</td>
              <td className="px-4 py-3">Fast to compute. Biased toward high-cardinality numerical/categorical features with many unique values.</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Permutation Importance (MDA)</td>
              <td className="px-4 py-3">Shuffle values of feature <code>j</code> in OOB data; measure decrease in OOB classification accuracy.</td>
              <td className="px-4 py-3">Unbiased and robust. Computationally slower; sensitive to collinearity among correlated features.</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>4 · Python Implementation of Random Forest Classifier</h2>
      <TerminalBlock language="python" filename="random_forest_scratch.py" code={code} />

      <h2>5 · Comparison: Random Forests vs. Gradient Boosting</h2>
      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Dimension</th>
              <th className="px-4 py-3 text-left">Random Forests (Bagging)</th>
              <th className="px-4 py-3 text-left">Gradient Boosting (Boosting)</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Primary Goal</td>
              <td className="px-4 py-3">Reduce <strong>Variance</strong></td>
              <td className="px-4 py-3">Reduce <strong>Bias</strong></td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Base Learners</td>
              <td className="px-4 py-3">Deep, low-bias, high-variance trees</td>
              <td className="px-4 py-3">Shallow, high-bias, low-variance trees (stumps)</td>
            </tr>
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Parallelization</td>
              <td className="px-4 py-3">Embarrassingly parallel (trees are independent)</td>
              <td className="px-4 py-3">Inherently sequential (tree <code>t</code> depends on residuals of <code>t-1</code>)</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Overfitting Risk</td>
              <td className="px-4 py-3">Adding more trees <em>never</em> causes overfitting</td>
              <td className="px-4 py-3">Excessive trees can overfit noise in residuals</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>6 · Further Reading</h2>
      <ul>
        <li>Breiman, L. (2001). <em>Random Forests</em>. Machine Learning, 45(1), 5-32.</li>
        <li>Breiman, L. (1996). <em>Bagging predictors</em>. Machine Learning, 24(2), 123-140.</li>
        <li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <em>The Elements of Statistical Learning</em>, Chapter 15: Random Forests.</li>
      </ul>
    </article>
  );
}
