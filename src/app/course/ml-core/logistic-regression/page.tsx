"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import numpy as np

class LogisticRegression:
    """
    Binary Logistic Regression with Maximum Likelihood Estimation (MLE)
    optimized via Vectorized Batch Gradient Descent with L2 Regularization.
    """
    def __init__(self, learning_rate: float = 0.05, n_iters: int = 1000, lambda_reg: float = 0.01):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid function
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            # Forward pass: log-odds linear transformation
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            # Compute Binary Cross-Entropy Loss with L2 penalty
            epsilon = 1e-15  # Guard against log(0)
            y_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
            bce_loss = - (1 / n_samples) * np.sum(
                y * np.log(y_clipped) + (1 - y) * np.log(1 - y_clipped)
            ) + (self.lambda_reg / (2 * n_samples)) * np.sum(self.weights ** 2)
            self.loss_history.append(bce_loss)

            # Gradient calculation via Chain Rule: ∂J/∂w = (1/m) X^T (y_pred - y) + (λ/m)w
            error = y_pred - y
            dw = (1 / n_samples) * np.dot(X.T, error) + (self.lambda_reg / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(error)

            # Parameter updates
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

# Verification using synthetic classification data
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

    X, y = make_classification(n_samples=1000, n_features=8, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(learning_rate=0.1, n_iters=1500, lambda_reg=0.01)
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)
    preds = clf.predict(X_test)

    print(f"Test Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"Log Loss:      {log_loss(y_test, probs):.4f}")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")
`;

export default function LogisticRegressionPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-100 text-emerald-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
          Track 1 · Classical Machine Learning
        </span>
      </div>

      <h1>Logistic Regression &amp; Maximum Likelihood Estimation</h1>

      <blockquote>
        <p>
          <em>"Logistic regression is not a regression model; it is a probability machine for binary and multinomial categorization derived strictly from the log-odds formulation."</em>
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Derive the sigmoid activation function from the logit (log-odds) representation.</li>
        <li>Derive the negative log-likelihood (binary cross-entropy loss) using Maximum Likelihood Estimation (MLE).</li>
        <li>Derive the gradient of cross-entropy with respect to weights and show why it matches the linear regression gradient structure.</li>
        <li>Generalize from binary classification to multi-class classification using the Softmax function.</li>
        <li>Implement vectorised, numerically stable gradient descent in Python with L2 shrinkage.</li>
      </ul>

      <h2>1 · The Logit Link Function and Odds Ratio</h2>
      <p>
        In linear regression, the model estimates <code>y = w · x + b</code> where <code>y ∈ ℝ</code>. For classification, predicting continuous numbers violates probability constraints (which must reside in <code>[0, 1]</code>).
      </p>
      <p>
        Instead, we model the <strong>probability</strong> <code>p = P(Y = 1 | X = x)</code>. We start by taking the <strong>odds</strong>, defined as the ratio of success probability to failure probability:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>Odds = p / (1 - p)</p>
      </div>

      <p>
        While probability is bounded in <code>[0, 1]</code>, the odds span <code>[0, ∞)</code>. Taking the natural logarithm produces the <strong>logit (log-odds)</strong>, which maps smoothly to the real line <code>(-∞, +∞)</code>:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>logit(p) = ln( p / (1 - p) ) = w · x + b</p>
      </div>

      <p>
        Inverting this transformation recovers the <strong>Standard Logistic (Sigmoid) Function</strong> <code>σ(z)</code>:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>p / (1 - p) = e^(w·x + b)</p>
        <p>p = e^(w·x + b) / (1 + e^(w·x + b))</p>
        <p>p = σ(z) = 1 / (1 + e^(-z))    where z = w · x + b</p>
      </div>

      <h2>2 · Derivative of the Sigmoid Function</h2>
      <p>
        The sigmoid function has a mathematical identity that simplifies gradient calculation:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>d/dz [σ(z)] = d/dz [(1 + e^(-z))^(-1)]</p>
        <p>           = - (1 + e^(-z))^(-2) · (-e^(-z))</p>
        <p>           = [ 1 / (1 + e^(-z)) ] · [ e^(-z) / (1 + e^(-z)) ]</p>
        <p>           = σ(z) · (1 - σ(z))</p>
      </div>

      <h2>3 · Maximum Likelihood Estimation (MLE) and Cross-Entropy</h2>
      <p>
        Assuming our training observations are independently and identically distributed (i.i.d.), each outcome follows a Bernoulli distribution where <code>P(y_i | x_i) = p_i^(y_i) · (1 - p_i)^(1 - y_i)</code> with <code>y_i ∈ &#123;0, 1&#125;</code>.
      </p>
      <p>
        The joint probability across the training set defines the <strong>Likelihood Function</strong> <code>L(w, b)</code>:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>L(w, b) = ∏_(i=1)^n p_i^(y_i) · (1 - p_i)^(1 - y_i)</p>
      </div>

      <p>
        Maximizing products of decimals leads to arithmetic underflow in floating-point operations. We take the natural log to yield the <strong>Log-Likelihood</strong> <code>ℓ(w, b)</code>:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>ℓ(w, b) = Σ_(i=1)^n [ y_i ln(p_i) + (1 - y_i) ln(1 - p_i) ]</p>
      </div>

      <p>
        In optimization literature, we minimize losses rather than maximize objectives. Negating the log-likelihood and scaling by <code>1/n</code> produces the <strong>Binary Cross-Entropy Loss (Log Loss)</strong> <code>J(w, b)</code>:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>J(w, b) = - (1/n) Σ_(i=1)^n [ y_i ln(σ(w·x_i + b)) + (1 - y_i) ln(1 - σ(w·x_i + b)) ]</p>
      </div>

      <h2>4 · Deriving the Gradient</h2>
      <p>
        We apply the multivariable chain rule to find <code>∂J / ∂w_j</code> for an arbitrary parameter:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>∂J / ∂w_j = (∂J / ∂p_i) · (∂p_i / ∂z_i) · (∂z_i / ∂w_j)</p>
        <br />
        <p>1. ∂J / ∂p_i = - [ y_i / p_i - (1 - y_i) / (1 - p_i) ] = (p_i - y_i) / [ p_i (1 - p_i) ]</p>
        <p>2. ∂p_i / ∂z_i = p_i (1 - p_i)</p>
        <p>3. ∂z_i / ∂w_j = x_ij</p>
        <br />
        <p>Multiplying yields:</p>
        <p>∂J / ∂w_j = (1/n) Σ_(i=1)^n (p_i - y_i) · x_ij</p>
        <br />
        <p>In vector form:</p>
        <p>∇_w J = (1/n) X^T (σ(Xw + b) - y)</p>
      </div>

      <p>
        This reveals why logistic regression converges efficiently: the gradient matches Ordinary Least Squares, substituting linear predictions with the logistic probabilities.
      </p>

      <h2>5 · Python Implementation with L2 Shrinkage</h2>
      <TerminalBlock language="python" filename="logistic_regression_mle.py" code={code} />

      <h2>6 · Multi-Class Extension: The Softmax Formulation</h2>
      <p>
        When targets belong to <code>K &gt; 2</code> mutually exclusive categories, the sigmoid generalises to the <strong>Softmax</strong> operator:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>P(Y = k | x) = e^(w_k · x + b_k) / [ Σ_(j=1)^K e^(w_j · x + b_j) ]</p>
      </div>

      <h2>7 · Practical Pitfalls in Industry</h2>
      <ul>
        <li>
          <strong>Perfect Separability (Hauck-Donner Effect):</strong> When classes are completely linearly separable, the MLE weights diverge to positive/negative infinity because <code>σ(∞) = 1.0</code>. Always incorporate L2 regularization to keep weight norms bounded.
        </li>
        <li>
          <strong>Interpreting Coefficients with Collinear Features:</strong> Multicollinearity inflates coefficient variance without necessarily destroying predictive accuracy. Use Variance Inflation Factors (VIF) or L1/L2 penalties before making business inferences from raw weights.
        </li>
      </ul>

      <h2>8 · Further Reading</h2>
      <ul>
        <li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <em>The Elements of Statistical Learning</em> (2nd ed.), Section 4.4: Logistic Regression.</li>
        <li>McCullagh, P., &amp; Nelder, J. A. (1989). <em>Generalized Linear Models</em> (2nd ed.). Chapman &amp; Hall/CRC.</li>
      </ul>
    </article>
  );
}
