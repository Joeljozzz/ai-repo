"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import numpy as np
from collections import Counter
from typing import Optional, List

class Node:
    """
    Representation of an individual decision or leaf node in the CART tree.
    """
    def __init__(
        self,
        feature: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional['Node'] = None,
        right: Optional['Node'] = None,
        *,
        value: Optional[int] = None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self) -> bool:
        return self.value is not None

class DecisionTree:
    """
    Classification and Regression Tree (CART) implementation supporting
    both Shannon Entropy and Gini Impurity split criteria.
    """
    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 1e-7
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.root: Optional[Node] = None

    def _impurity(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)

        if self.criterion == "gini":
            # Gini Impurity: 1 - Σ p_i²
            return 1.0 - np.sum(probs ** 2)
        elif self.criterion == "entropy":
            # Shannon Entropy: - Σ p_i log2(p_i)
            return -np.sum(probs * np.log2(probs + 1e-15))
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        best_gain = -1.0
        split_feature, split_threshold = None, None
        parent_impurity = self._impurity(y)
        n_samples, n_features = X.shape

        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                left_mask = X_column <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # Weighted average child impurity
                n_l, n_r = np.sum(left_mask), np.sum(right_mask)
                imp_l = self._impurity(y[left_mask])
                imp_r = self._impurity(y[right_mask])
                child_impurity = (n_l / n_samples) * imp_l + (n_r / n_samples) * imp_r

                # Information Gain
                gain = parent_impurity - child_impurity

                if gain > best_gain:
                    best_gain = gain
                    split_feature = feat_idx
                    split_threshold = threshold

        return split_feature, split_threshold, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        n_samples, _ = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria: pure node, max depth exceeded, or below sample threshold
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            most_common = Counter(y).most_common(1)[0][0]
            return Node(value=most_common)

        feature, threshold, gain = self._best_split(X, y)

        if feature is None or gain < self.min_impurity_decrease:
            most_common = Counter(y).most_common(1)[0][0]
            return Node(value=most_common)

        # Split and recurse
        left_mask = X[:, feature] <= threshold
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)

        return Node(feature=feature, threshold=threshold, left=left_child, right=right_child)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.root = self._build_tree(X, y)

    def _predict_row(self, x: np.ndarray, node: Node) -> int:
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_row(x, node.left)
        return self._predict_row(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_row(x, self.root) for x in X])

# Validation on the Wine multiclass dataset
if __name__ == "__main__":
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

    tree = DecisionTree(criterion="entropy", max_depth=4)
    tree.fit(X_train, y_train)
    preds = tree.predict(X_test)

    print(f"Scratch CART Accuracy (Entropy): {accuracy_score(y_test, preds):.4f}")
`;

export default function DecisionTreesPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-100 text-emerald-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
          Track 1 · Classical Machine Learning
        </span>
      </div>

      <h1>Decision Trees &amp; Information Theory (CART)</h1>

      <blockquote>
        <p>
          <em>"Decision trees approximate complex high-dimensional response surfaces by recursively partitioning the feature space into hyper-rectangles using greedy information-theoretic criteria."</em>
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Derive Shannon Entropy and Gini Impurity from first principles.</li>
        <li>Understand the Information Gain (Kullback-Leibler divergence reduction) objective in CART.</li>
        <li>Analyze how non-parametric recursive binary splitting partitions feature space into orthogonal hyperplanes.</li>
        <li>Contrast pre-pruning heuristics with minimal cost-complexity post-pruning.</li>
        <li>Implement a full CART tree classifier from scratch in NumPy.</li>
      </ul>

      <h2>1 · Information Theoretic Foundations of Split Criteria</h2>
      <p>
        Consider a node containing training samples with discrete labels <code>y ∈ &#123;1, ..., K&#125;</code> where <code>p_k</code> denotes the empirical fraction of class <code>k</code>. We quantify the uncertainty or heterogeneity using two principal measures:
      </p>

      <h3>1.1 Shannon Entropy</h3>
      <p>
        Derived by Claude Shannon in 1948, entropy measures the expected information content (in bits) required to describe a randomly sampled label:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>H(S) = - Σ_(k=1)^K p_k · log₂(p_k)</p>
      </div>

      <p>
        For binary classification where <code>p</code> is the positive fraction:
      </p>
      <ul>
        <li>When <code>p = 0.5</code> (maximum chaos), <code>H(S) = - (0.5 log₂ 0.5 + 0.5 log₂ 0.5) = 1.0</code> bit.</li>
        <li>When <code>p = 1.0</code> or <code>0.0</code> (pure node), <code>H(S) = 0.0</code> bits.</li>
      </ul>

      <h3>1.2 Gini Impurity</h3>
      <p>
        Used by the CART algorithm (Breiman et al.), Gini Impurity calculates the probability that a randomly chosen element from the set would be incorrectly labeled if it were randomly labeled according to the class distribution:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>Gini(S) = 1 - Σ_(k=1)^K p_k²</p>
      </div>

      <p>
        For a binary split, maximum Gini is <code>1 - (0.5² + 0.5²) = 0.5</code>. Gini impurity is computationally faster because it avoids evaluating transcendental logarithm functions on thousands of potential split candidate thresholds.
      </p>

      <h2>2 · The Information Gain Objective</h2>
      <p>
        When evaluating candidate feature <code>X_j</code> and threshold <code>θ</code>, the dataset <code>D</code> is partitioned into left and right subsets <code>D_L = &#123;x | x_j ≤ θ&#125;</code> and <code>D_R = &#123;x | x_j &gt; θ&#125;</code>.
      </p>
      <p>
        The <strong>Information Gain (IG)</strong> represents the reduction in entropy resulting from this split:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>IG(D, X_j, θ) = Impurity(D) - [ (|D_L| / |D|) · Impurity(D_L) + (|D_R| / |D|) · Impurity(D_R) ]</p>
      </div>

      <p>
        The CART algorithm acts greedily by selecting the parameter pair <code>(j*, θ*) = argmax IG(D, X_j, θ)</code> at every internal branch.
      </p>

      <h2>3 · Overfitting and Cost-Complexity Pruning</h2>
      <p>
        An unconstrained decision tree will split until every leaf node contains a single sample (0 impurity), achieving 100% training accuracy but memorizing noise (astronomical variance).
      </p>
      <p>
        In Minimal Cost-Complexity Pruning (Breiman et al.), we minimize the penalized cost function:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>R_α(T) = R(T) + α · |T|</p>
        <br />
        <p>where:</p>
        <p>  R(T) = Misclassification rate of tree T</p>
        <p>  |T|  = Number of terminal leaf nodes</p>
        <p>  α    = Complexity cost parameter (tuned via cross-validation)</p>
      </div>

      <h2>4 · Python Implementation of CART</h2>
      <TerminalBlock language="python" filename="cart_decision_tree.py" code={code} />

      <h2>5 · Key Structural Advantages and Pitfalls</h2>
      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Property</th>
              <th className="px-4 py-3 text-left">Behavior in Trees</th>
              <th className="px-4 py-3 text-left">Engineering Solution</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Feature Scaling</td>
              <td className="px-4 py-3">Invariant. Monotonic transformations preserve rank orders.</td>
              <td className="px-4 py-3">No normalization or standardization required.</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Missing Values</td>
              <td className="px-4 py-3">Can handle natively via surrogate splits.</td>
              <td className="px-4 py-3">Assign missing sample down the majority branch.</td>
            </tr>
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">High Variance</td>
              <td className="px-4 py-3">Extremely sensitive to minor variations in training set.</td>
              <td className="px-4 py-3">Ensemble methods: Random Forests and Gradient Boosted Trees.</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>6 · Further Reading</h2>
      <ul>
        <li>Breiman, L., Friedman, J., Stone, C. J., &amp; Olshen, R. A. (1984). <em>Classification and Regression Trees</em>. CRC Press.</li>
        <li>Shannon, C. E. (1948). <em>A Mathematical Theory of Communication</em>. Bell System Technical Journal, 27(3), 379–423.</li>
        <li>Quinlan, J. R. (1986). <em>Induction of Decision Trees</em>. Machine Learning, 1(1), 81–106.</li>
      </ul>
    </article>
  );
}
