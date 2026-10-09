"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import numpy as np
from collections import defaultdict

class NaiveBayesClassifier:
    """
    Naive Bayes Classifier supporting:
    1. Gaussian Naive Bayes (Continuous features): P(x_i | y) ~ N(μ_iy, σ_iy²)
    2. Multinomial Naive Bayes (Discrete count features) with Laplace Smoothing (α):
       P(x_i | y) = (N_yi + α) / (N_y + α · D)
    """
    def __init__(self, model_type: str = "gaussian", alpha: float = 1.0):
        self.model_type = model_type
        self.alpha = alpha  # Laplace smoothing parameter
        self.classes = None
        self.priors = {}
        # Parameters for Gaussian
        self.means = {}
        self.variances = {}
        # Parameters for Multinomial
        self.feature_probs = {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)

        for c in self.classes:
            X_c = X[y == c]
            # Prior P(Y = c) = N_c / N
            self.priors[c] = X_c.shape[0] / float(n_samples)

            if self.model_type == "gaussian":
                # Compute empirical mean and variance (with epsilon to avoid division by zero)
                self.means[c] = np.mean(X_c, axis=0)
                self.variances[c] = np.var(X_c, axis=0) + 1e-9

            elif self.model_type == "multinomial":
                # Laplace-smoothed feature likelihood: P(x_i | c) = (count_i + α) / (total_count + α * d)
                total_feature_counts = np.sum(X_c, axis=0)
                total_count = np.sum(total_feature_counts)
                self.feature_probs[c] = (total_feature_counts + self.alpha) / (
                    total_count + self.alpha * n_features
                )

    def _gaussian_log_likelihood(self, x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
        # ln P(x | μ, σ²) = - 1/2 ln(2π σ²) - (x - μ)² / (2 σ²)
        log_prob = -0.5 * np.sum(np.log(2.0 * np.pi * var)) - 0.5 * np.sum(((x - mean) ** 2) / var)
        return log_prob

    def predict_sample(self, x: np.ndarray):
        posteriors = {}

        for c in self.classes:
            # log posterior: ln P(Y=c) + Σ ln P(x_i | Y=c)
            log_prior = np.log(self.priors[c])

            if self.model_type == "gaussian":
                log_likelihood = self._gaussian_log_likelihood(x, self.means[c], self.variances[c])
            elif self.model_type == "multinomial":
                log_likelihood = np.sum(x * np.log(self.feature_probs[c]))

            posteriors[c] = log_prior + log_likelihood

        # Return argmax_c [ log P(Y=c|X) ]
        return max(posteriors, key=posteriors.get)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict_sample(x) for x in X])

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

    gnb = NaiveBayesClassifier(model_type="gaussian")
    gnb.fit(X_train, y_train)
    preds = gnb.predict(X_test)

    print(f"Scratch Gaussian Naive Bayes Accuracy: {accuracy_score(y_test, preds):.4f}")
`;

export default function NaiveBayesPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-100 text-emerald-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
          Track 1 · Classical Machine Learning
        </span>
      </div>

      <h1>Naive Bayes Classifiers: Bayes Theorem, Conditional Independence, and Laplace Smoothing</h1>

      <blockquote>
        <p>
          <em>"The Naive Bayes assumption of conditional feature independence is almost never true in practice, yet the classifier consistently achieves competitive generalization accuracy because classification boundaries depend on ranking rather than exact probability calibration."</em> — Pedro Domingos &amp; Michael Pazzani (1997)
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Derive the Maximum A Posteriori (MAP) classification rule from Bayes&apos; theorem.</li>
        <li>Understand the conditional independence assumption and how it reduces parameter complexity from <code>O(K · 2^d)</code> to <code>O(K · d)</code>.</li>
        <li>Derive Gaussian, Multinomial, and Bernoulli likelihood models.</li>
        <li>Formulate Laplace (additive) smoothing to resolve the zero-frequency probability trap.</li>
        <li>Implement a full Gaussian and Multinomial Naive Bayes classifier from scratch in NumPy.</li>
      </ul>

      <h2>1 · Maximum A Posteriori (MAP) Decision Rule</h2>
      <p>
        Given a feature vector <code>x = (x_1, x_2, ..., x_d)</code> and discrete class <code>y ∈ &#123;1, ..., K&#125;</code>, Bayes&apos; theorem defines the posterior probability:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>P(Y = y | x) = [ P(Y = y) · P(x | Y = y) ] / P(x)</p>
        <br />
        <p>where:</p>
        <p>  P(Y = y)  = Class Prior</p>
        <p>  P(x | Y)  = Class Likelihood</p>
        <p>  P(x)      = Evidence (marginal probability, constant across classes)</p>
      </div>

      <p>
        The MAP decision rule chooses the class maximizing the posterior:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>ŷ = argmax_(y) P(Y = y) · P(x_1, ..., x_d | Y = y)</p>
      </div>

      <h2>2 · The "Naive" Conditional Independence Assumption</h2>
      <p>
        To compute the joint likelihood <code>P(x_1, ..., x_d | y)</code> without simplifying assumptions on <code>d</code> binary features requires estimating <code>2^d - 1</code> parameters per class—an intractable sample complexity.
      </p>
      <p>
        The <strong>Naive Bayes assumption</strong> asserts that all features are mutually conditionally independent given the class label:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>P(x_1, ..., x_d | Y = y) = ∏_(i=1)^d P(x_i | Y = y)</p>
      </div>

      <p>
        Working in log-space prevents numerical underflow when multiplying hundreds of small decimal probabilities:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>log P(Y = y | x) ∝ ln P(Y = y) + Σ_(i=1)^d ln P(x_i | Y = y)</p>
      </div>

      <h2>3 · Likelihood Models: Gaussian vs. Multinomial</h2>
      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Variant</th>
              <th className="px-4 py-3 text-left">Feature Domain</th>
              <th className="px-4 py-3 text-left">Likelihood Formula P(x_i | y)</th>
              <th className="px-4 py-3 text-left">Canonical Applications</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Gaussian Naive Bayes</td>
              <td className="px-4 py-3">Continuous real values <code>x_i ∈ ℝ</code></td>
              <td className="px-4 py-3 font-mono">(1 / √(2πσ_iy²)) · exp( - (x_i - μ_iy)² / 2σ_iy² )</td>
              <td className="px-4 py-3">Sensor telemetry, medical measurements, biometric metrics</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Multinomial Naive Bayes</td>
              <td className="px-4 py-3">Discrete word counts / frequencies</td>
              <td className="px-4 py-3 font-mono">θ_yi = (N_yi + α) / (N_y + α · D)</td>
              <td className="px-4 py-3">Spam filtering, sentiment analysis, document classification</td>
            </tr>
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Bernoulli Naive Bayes</td>
              <td className="px-4 py-3">Binary indicator presence/absence <code>&#123;0, 1&#125;</code></td>
              <td className="px-4 py-3 font-mono">p_yi^(x_i) · (1 - p_yi)^(1 - x_i)</td>
              <td className="px-4 py-3">Short-text categorization, keyword presence checking</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>4 · Laplace (Additive) Smoothing: The Zero-Frequency Problem</h2>
      <p>
        If word <code>w_k</code> never appears in class <code>y</code> during training, maximum likelihood estimation assigns <code>P(w_k | y) = 0</code>. Because probabilities are multiplied across features, a single zero probability completely destroys the entire product:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>P(x | y) = P(x_1 | y) · ... · 0 · ... · P(x_d | y) = 0</p>
      </div>

      <p>
        <strong>Laplace Smoothing (α = 1)</strong> resolves this by applying a uniform Dirichlet prior over feature counts:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>P(x_i | y) = ( count(x_i, y) + α ) / ( Σ_(j) count(x_j, y) + α · d )</p>
      </div>

      <h2>5 · Python Implementation from First Principles</h2>
      <TerminalBlock language="python" filename="naive_bayes_scratch.py" code={code} />

      <h2>6 · Further Reading</h2>
      <ul>
        <li>Domingos, P., &amp; Pazzani, M. (1997). <em>On the optimality of the simple Bayesian classifier under zero-one loss</em>. Machine Learning, 29(2), 103-130.</li>
        <li>McCallum, A., &amp; Nigam, K. (1998). <em>A comparison of event models for naive bayes text classification</em>. AAAI-98 Workshop on Learning for Text Categorization.</li>
      </ul>
    </article>
  );
}
