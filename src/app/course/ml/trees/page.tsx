"use client";
import React from 'react';
import TerminalBlock from '@/components/TerminalBlock';
import InteractiveQuiz from '@/components/InteractiveQuiz';

export default function TreesPage() {
  const treeCode = `import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None

class PhDDecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Check stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # Find the best split
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # Grow children
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # Information Gain (Shannon Entropy Reduction)
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        # Parent entropy
        parent_entropy = self._entropy(y)

        # Create children
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # Calculate weighted average entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        # Calculate IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value`;

  return (
    <div className="space-y-12 pb-24 text-slate-800 animate-in fade-in slide-in-from-bottom-4 duration-500">
      
      <header className="border-b border-slate-200 pb-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-teal-100 text-teal-700 rounded-full text-xs font-bold uppercase tracking-widest mb-4">
          <span className="w-2 h-2 rounded-full bg-teal-600 animate-pulse"></span>
          Stack 2: Classical Machine Learning
        </div>
        <h1 className="text-4xl font-extrabold tracking-tight text-slate-900 mb-6">2.3 Decision Trees</h1>
        <p className="text-xl text-slate-600 leading-relaxed max-w-3xl">
          Information Theory, Shannon Entropy, Gini Impurity, and CART split optimization.
        </p>
      </header>

      <section className="prose prose-slate max-w-none text-lg text-slate-600">
        <h2 className="text-3xl font-bold text-slate-900 mt-8 mb-4">1. Non-Parametric Partitioning</h2>
        <p>
          Unlike Linear Models or SVMs that attempt to fit a continuous geometric boundary across the entire feature space, Decision Trees partition the space into disjoint orthogonal hyper-rectangles via recursive binary splitting.
        </p>
        <p>
          Because they make no assumptions about underlying probability distributions, they are immune to non-linearities and do not require feature scaling.
        </p>

        <h2 className="text-3xl font-bold text-slate-900 mt-12 mb-4">2. Thermodynamics of Splits (Information Theory)</h2>
        <p>
          To determine the optimal feature and threshold to split on, we must measure the "purity" of the resulting child nodes. The CART (Classification and Regression Trees) algorithm uses either <strong>Gini Impurity</strong> or <strong>Shannon Entropy</strong>.
        </p>

        <div className="grid md:grid-cols-2 gap-8 my-8">
          <div className="bg-slate-50 border border-slate-200 rounded-2xl p-6 shadow-sm">
            <h3 className="text-xl font-bold text-slate-900 mt-0 mb-2">Shannon Entropy</h3>
            <p className="text-sm text-slate-600 mb-4">Derived from information theory. It measures the expected amount of information (in bits) needed to identify the class of a randomly drawn instance.</p>
            <div className="bg-white p-3 rounded border border-slate-200 font-mono text-sm shadow-sm text-slate-800">
              H(S) = - Σ (p_i * log₂(p_i))
            </div>
            <p className="text-sm text-slate-500 mt-3">Maximum entropy (chaos) is 1.0 in a 50/50 binary split. Perfect purity is 0.0.</p>
          </div>

          <div className="bg-slate-50 border border-slate-200 rounded-2xl p-6 shadow-sm">
            <h3 className="text-xl font-bold text-slate-900 mt-0 mb-2">Gini Impurity</h3>
            <p className="text-sm text-slate-600 mb-4">Measures the probability of incorrectly classifying a randomly chosen element if it was randomly labeled according to the node's class distribution.</p>
            <div className="bg-white p-3 rounded border border-slate-200 font-mono text-sm shadow-sm text-slate-800">
              G(S) = 1 - Σ (p_i)²
            </div>
            <p className="text-sm text-slate-500 mt-3">Max impurity is 0.5 for a 50/50 split. Gini is computationally faster (no logarithms).</p>
          </div>
        </div>

        <h2 className="text-3xl font-bold text-slate-900 mt-12 mb-4">3. Information Gain (IG)</h2>
        <p>
          The algorithm evaluates every possible feature and threshold by calculating the <strong>Information Gain</strong>: the reduction in Entropy after a split.
        </p>
        <div className="bg-slate-50 border border-slate-200 rounded-2xl p-6 my-6 shadow-sm overflow-x-auto font-mono text-sm text-slate-800">
          IG(Dp, A) = H(Dp) - Σ ( |Dv| / |Dp| ) * H(Dv)
        </div>
        <p>
          Where <code>Dp</code> is the parent dataset, and <code>Dv</code> are the child partitions. The split that maximizes IG is chosen.
        </p>

        <div className="bg-white border border-slate-200 p-6 rounded-2xl my-8 shadow-lg not-prose">
            <h3 className="text-slate-900 font-bold mt-0 text-xl mb-4">Implementation: CART Tree with Shannon Entropy</h3>
            <TerminalBlock 
                language="python" 
                filename="decision_tree_cart.py" 
                code={treeCode}
            />
        </div>

        <InteractiveQuiz 
            question="Decision trees are prone to severe overfitting. Which of the following is NOT a mathematically sound method for regularizing a CART decision tree?"
            options={[
                "Setting a hard cap on the maximum tree depth (max_depth).",
                "Applying L2 (Ridge) Penalty to the Information Gain calculation.",
                "Cost-Complexity Pruning (Weakest Link Pruning) using an alpha parameter.",
                "Enforcing a minimum number of samples required to split an internal node."
            ]}
            correctIndex={1}
            explanation="L2 (Ridge) penalty is a regularization technique for parametric models (Linear Regression, NNs) where weight vectors exist. Decision Trees do not have weights to penalize. They are regularized through structural constraints (depth limits, min samples) or post-pruning algorithms."
        />
      </section>
    </div>
  );
}
