"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import numpy as np

class GradientBoostingRegressor:
    """
    Functional Gradient Boosting Machine for regression optimizing Squared Error Loss:
    L(y, F(x)) = 1/2 (y - F(x))²
    Negative gradient (pseudo-residuals): r_im = y_i - F_{m-1}(x_i)
    """
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.f0 = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        from sklearn.tree import DecisionTreeRegressor
        
        # Step 1: Initialize model with optimal constant minimizing L:
        # F_0(x) = argmin_c Σ L(y_i, c) = mean(y)
        self.f0 = np.mean(y)
        f_m = np.full(shape=y.shape, fill_value=self.f0)

        for m in range(self.n_estimators):
            # Step 2a: Compute pseudo-residuals (negative gradient)
            residuals = y - f_m

            # Step 2b: Fit weak base learner h_m(x) to residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
            tree.fit(X, residuals)

            # Step 2c: Update additive model with shrinkage:
            # F_m(x) = F_{m-1}(x) + ν · h_m(x)
            f_m += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # F_M(x) = F_0(x) + ν · Σ_(m=1)^M h_m(x)
        y_pred = np.full(shape=(X.shape[0],), fill_value=self.f0)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

# Verification on California Housing dataset
if __name__ == "__main__":
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(data.data[:2000], data.target[:2000], test_size=0.2, random_state=42)

    gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=3)
    gbm.fit(X_train, y_train)
    preds = gbm.predict(X_test)

    print(f"Scratch GBM Test RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.4f}")
    print(f"Scratch GBM Test R²:   {r2_score(y_test, preds):.4f}")
`;

export default function BoostingPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-100 text-emerald-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
          Track 1 · Classical Machine Learning
        </span>
      </div>

      <h1>Boosting: AdaBoost &amp; Gradient Boosted Machines (GBM)</h1>

      <blockquote>
        <p>
          <em>"Boosting transforms an ensemble of weak learners—models performing only marginally better than random guessing—into an arbitrarily strong predictor by optimizing functional residuals in an additive manner."</em> — Jerome Friedman
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Understand the theoretical progression from PAC learning (Valiant, Kearns) to AdaBoost (Freund &amp; Schapire).</li>
        <li>Derive AdaBoost sample weight updates and classifier voting coefficients from exponential loss minimization.</li>
        <li>Formulate Gradient Boosting as Functional Gradient Descent in function space rather than parameter space.</li>
        <li>Derive pseudo-residuals for common loss functions (Squared Error, Absolute Error, Negative Log-Likelihood).</li>
        <li>Implement a functional Gradient Boosted Regressor from scratch with shrinkage regularization.</li>
      </ul>

      <h2>1 · From PAC Learning to AdaBoost</h2>
      <p>
        In Probably Approximately Correct (PAC) learning theory, Leslie Valiant and Michael Kearns posed an open challenge: <em>Can a set of weak learners (each guaranteeing accuracy slightly above 50%) be combined to produce a strong learner with arbitrarily high accuracy?</em>
      </p>
      <p>
        Yoav Freund and Robert Schapire answered affirmatively in 1995 with <strong>AdaBoost (Adaptive Boosting)</strong>. AdaBoost trains decision stumps sequentially. At step <code>m</code>:
      </p>
      <ol>
        <li>Samples misclassified by model <code>m-1</code> are assigned higher weights.</li>
        <li>Correctly classified samples are assigned lower weights.</li>
        <li>The subsequent base learner is forced to focus its capacity on the hardest examples.</li>
      </ol>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p className="text-slate-400">// AdaBoost Exponential Loss Formulation</p>
        <p>L(y, F(x)) = exp( -y · F(x) ),   where y ∈ &#123;-1, +1&#125;</p>
        <br />
        <p className="text-slate-400">// Weighted error rate of base classifier h_m:</p>
        <p>ε_m = Σ_(i: y_i ≠ h_m(x_i)) w_i^(m) / Σ_(i=1)^n w_i^(m)</p>
        <br />
        <p className="text-slate-400">// Classifier stage voting weight:</p>
        <p>α_m = ½ ln( (1 - ε_m) / ε_m )</p>
        <br />
        <p className="text-slate-400">// Sample weight update for next iteration:</p>
        <p>w_i^(m+1) = w_i^(m) · exp( -α_m · y_i · h_m(x_i) )</p>
      </div>

      <h2>2 · Gradient Boosting as Functional Gradient Descent</h2>
      <p>
        In 1999, Jerome Friedman realized that AdaBoost was a specific instance of gradient descent where the loss is exponential. He generalized this to <strong>Gradient Boosted Machines (GBM)</strong> for any differentiable loss function.
      </p>
      <p>
        In conventional gradient descent, we update numerical parameters <code>θ ∈ ℝ^p</code>:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>θ_(m) = θ_(m-1) - η · ∇_θ L(θ)</p>
      </div>

      <p>
        In Gradient Boosting, we treat the predictions <code>F(x_1), ..., F(x_n)</code> as the parameters themselves and optimize in <strong>function space</strong>:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>F_m(x) = F_(m-1)(x) + ν · h_m(x)</p>
      </div>

      <p>
        To move in the direction of steepest descent, the new weak learner <code>h_m(x)</code> is fitted to the <strong>negative gradient of the loss</strong> with respect to the model&apos;s current predictions:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>r_im = - [ ∂L(y_i, F(x_i)) / ∂F(x_i) ]_(F(x) = F_(m-1)(x))</p>
      </div>

      <h2>3 · Mathematical Derivations of Pseudo-Residuals</h2>
      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Loss Function</th>
              <th className="px-4 py-3 text-left">Objective L(y, F)</th>
              <th className="px-4 py-3 text-left">Negative Gradient (Pseudo-Residual r_i)</th>
              <th className="px-4 py-3 text-left">Properties</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Squared Error (L2)</td>
              <td className="px-4 py-3 font-mono">½ (y - F)²</td>
              <td className="px-4 py-3 font-mono">y_i - F(x_i)</td>
              <td className="px-4 py-3">Matches standard residuals; sensitive to outliers</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Absolute Error (L1)</td>
              <td className="px-4 py-3 font-mono">|y - F|</td>
              <td className="px-4 py-3 font-mono">sign(y_i - F(x_i))</td>
              <td className="px-4 py-3">Median regression; highly robust to extreme noise</td>
            </tr>
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Bernoulli Log-Loss</td>
              <td className="px-4 py-3 font-mono">- y log(p) - (1-y) log(1-p)</td>
              <td className="px-4 py-3 font-mono">y_i - σ(F(x_i))</td>
              <td className="px-4 py-3">Binary classification probabilities</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>4 · Regularization: Shrinkage and Subsampling</h2>
      <p>
        Unconstrained gradient boosting rapidly overfits training data because sequential trees continuously focus on edge-case residuals. Two primary regularization mechanisms prevent this:
      </p>
      <ul>
        <li>
          <strong>Shrinkage (Learning Rate ν ∈ (0, 1]):</strong> Scales the contribution of each tree: <code>F_m(x) = F_(m-1)(x) + ν · h_m(x)</code>. Values between <code>0.01</code> and <code>0.1</code> vastly improve generalisation at the cost of requiring more estimators.
        </li>
        <li>
          <strong>Stochastic Gradient Boosting:</strong> Jerome Friedman introduced randomly subsampling without replacement a fraction <code>η ∈ (0.5, 0.8)</code> of training data for each iteration, decorrelating base trees.
        </li>
      </ul>

      <h2>5 · Python Implementation from First Principles</h2>
      <TerminalBlock language="python" filename="gbm_regressor.py" code={code} />

      <h2>6 · Common Pitfalls</h2>
      <ul>
        <li>
          <strong>Selecting learning rate without scaling estimators:</strong> When halving the learning rate <code>ν</code>, you generally need to roughly double the number of trees <code>M</code> to reach the same level of convergence.
        </li>
        <li>
          <strong>Deep base learners:</strong> Unlike Random Forests where deep trees (depth &gt; 15) are averaged to lower variance, Boosting iteratively decreases bias. Deep trees in boosting lead to catastrophic overfitting within dozens of rounds. Use shallow trees (depth 3 to 6).
        </li>
      </ul>

      <h2>7 · Further Reading</h2>
      <ul>
        <li>Friedman, J. H. (2001). <em>Greedy Function Approximation: A Gradient Boosting Machine</em>. The Annals of Statistics, 29(5), 1189–1232.</li>
        <li>Freund, Y., &amp; Schapire, R. E. (1997). <em>A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting</em>. Journal of Computer and System Sciences, 55(1), 119–139.</li>
      </ul>
    </article>
  );
}
