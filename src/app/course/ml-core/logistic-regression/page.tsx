"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";
import MathBlock from "@/components/MathBlock";
import ConceptCard from "@/components/ConceptCard";
import ComparisonTable from "@/components/ComparisonTable";
import AlgorithmBlock from "@/components/AlgorithmBlock";

const fullLogisticCode = `import numpy as np
from typing import Tuple, Dict, Optional, Literal

class AdvancedLogisticRegression:
    """
    Production-grade Logistic Regression Engine supporting:
    1. Vectorized Batch Gradient Descent with Nesterov Accelerated Momentum
    2. Newton-Raphson / Iteratively Reweighted Least Squares (IRLS) exact solver
    3. Regularization options:
       - 'none': Vanilla Maximum Likelihood Estimation
       - 'l2' (Ridge): (lambda / 2) * ||w||_2^2
       - 'l1' (Lasso): Proximal gradient / Soft-thresholding operator
       - 'elasticnet': Convex combination of L1 and L2 penalties
    4. Multinomial Softmax Regression for K-class classification
    """
    def __init__(
        self,
        solver: Literal["gd", "newton", "proximal"] = "gd",
        penalty: Literal["none", "l2", "l1", "elasticnet"] = "l2",
        C: float = 1.0,               # Inverse regularization strength (1 / lambda)
        l1_ratio: float = 0.5,        # Elastic Net mixing parameter: alpha in [0, 1]
        learning_rate: float = 0.05,
        max_iter: int = 1000,
        tol: float = 1e-6
    ):
        self.solver = solver
        self.penalty = penalty
        self.C = C
        self.lambda_reg = 1.0 / C if C > 0 else 0.0
        self.l1_ratio = l1_ratio
        self.lr = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.classes_: Optional[np.ndarray] = None
        self.loss_history: list = []

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid preventing exp(-z) floating-point overflow."""
        z_safe = np.clip(z, -500, 500)
        return np.where(z_safe >= 0, 
                        1.0 / (1.0 + np.exp(-z_safe)), 
                        np.exp(z_safe) / (1.0 + np.exp(z_safe)))

    @staticmethod
    def soft_threshold(v: np.ndarray, threshold: float) -> np.ndarray:
        """Proximal operator for L1 Lasso norm: sign(v) * max(0, |v| - threshold)."""
        return np.sign(v) * np.maximum(0.0, np.abs(v) - threshold)

    def _compute_loss_and_grad(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> Tuple[float, np.ndarray, float]:
        n_samples = X.shape[0]
        z = np.dot(X, w) + b
        p = self.sigmoid(z)
        
        # Binary Cross-Entropy Loss with epsilon guard
        eps = 1e-15
        p_clipped = np.clip(p, eps, 1.0 - eps)
        bce_loss = -np.mean(y * np.log(p_clipped) + (1.0 - y) * np.log(1.0 - p_clipped))
        
        # Calculate regularization cost
        reg_cost = 0.0
        reg_grad_w = np.zeros_like(w)
        
        if self.penalty == "l2":
            reg_cost = 0.5 * self.lambda_reg * np.sum(w ** 2)
            reg_grad_w = self.lambda_reg * w
        elif self.penalty == "l1":
            reg_cost = self.lambda_reg * np.sum(np.abs(w))
            # Subgradient for L1
            reg_grad_w = self.lambda_reg * np.sign(w)
        elif self.penalty == "elasticnet":
            l1_cost = self.l1_ratio * self.lambda_reg * np.sum(np.abs(w))
            l2_cost = 0.5 * (1.0 - self.l1_ratio) * self.lambda_reg * np.sum(w ** 2)
            reg_cost = l1_cost + l2_cost
            reg_grad_w = (self.l1_ratio * self.lambda_reg * np.sign(w) + 
                          (1.0 - self.l1_ratio) * self.lambda_reg * w)
            
        total_loss = bce_loss + (reg_cost / n_samples)
        
        # Analytical gradients via multivariable chain rule
        error = p - y
        dw = (np.dot(X.T, error) / n_samples) + (reg_grad_w / n_samples)
        db = float(np.mean(error))
        
        return total_loss, dw, db

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        if self.solver == "newton":
            # Newton-Raphson / IRLS Solver: w^(t+1) = w^t - H^(-1) * grad
            for it in range(self.max_iter):
                loss, dw, db = self._compute_loss_and_grad(X, y, self.weights, self.bias)
                self.loss_history.append(loss)
                
                # Pre-activation probabilities
                p = self.sigmoid(np.dot(X, self.weights) + self.bias)
                # Diagonal weighting matrix S = diag(p * (1 - p))
                s = p * (1.0 - p)
                
                # Hessian matrix: H = (1/n) * X^T * S * X + lambda * I
                X_weighted = X * s[:, np.newaxis]
                H = (np.dot(X.T, X_weighted) / n_samples)
                if self.penalty == "l2":
                    H += (self.lambda_reg / n_samples) * np.eye(n_features)
                    
                # Solve linear system H * delta_w = dw for numerical stability (avoid direct inversion)
                try:
                    delta_w = np.linalg.solve(H + 1e-6 * np.eye(n_features), dw)
                except np.linalg.LinAlgError:
                    delta_w = np.linalg.pinv(H).dot(dw)
                    
                self.weights -= delta_w
                self.bias -= db * 0.1  # Damped bias update
                
                if np.linalg.norm(delta_w) < self.tol:
                    break
                    
        elif self.solver == "proximal" and self.penalty in ["l1", "elasticnet"]:
            # Proximal Gradient Descent for exact sparse feature selection
            for it in range(self.max_iter):
                # Gradient of smooth unpenalized BCE loss
                p = self.sigmoid(np.dot(X, self.weights) + self.bias)
                error = p - y
                dw_smooth = np.dot(X.T, error) / n_samples
                if self.penalty == "elasticnet":
                    dw_smooth += (1.0 - self.l1_ratio) * (self.lambda_reg / n_samples) * self.weights
                    
                db = float(np.mean(error))
                
                # Gradient step
                w_intermediate = self.weights - self.lr * dw_smooth
                self.bias -= self.lr * db
                
                # Proximal step (soft-thresholding)
                l1_penalty_effective = (self.lambda_reg if self.penalty == "l1" 
                                        else self.l1_ratio * self.lambda_reg)
                gamma = self.lr * (l1_penalty_effective / n_samples)
                w_new = self.soft_threshold(w_intermediate, threshold=gamma)
                
                if np.linalg.norm(w_new - self.weights) < self.tol:
                    self.weights = w_new
                    break
                self.weights = w_new
                
        else:
            # Standard Batch Gradient Descent with Nesterov momentum
            v_w = np.zeros_like(self.weights)
            v_b = 0.0
            momentum = 0.9
            
            for it in range(self.max_iter):
                # Look-ahead point
                w_lookahead = self.weights - momentum * v_w
                b_lookahead = self.bias - momentum * v_b
                
                loss, dw, db = self._compute_loss_and_grad(X, y, w_lookahead, b_lookahead)
                self.loss_history.append(loss)
                
                v_w = momentum * v_w + self.lr * dw
                v_b = momentum * v_b + self.lr * db
                
                self.weights -= v_w
                self.bias -= v_b
                
                if it > 1 and abs(self.loss_history[-2] - self.loss_history[-1]) < self.tol:
                    break

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        linear_val = np.dot(X, self.weights) + self.bias
        p1 = self.sigmoid(linear_val)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

# ==============================================================================
# VERIFICATION DEMO: RIDGE VS LASSO VS NEWTON SOLVER
# ==============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    # Generate linearly separable synthetic data with 5 true signals and 15 noise features
    N, D = 400, 20
    X_syn = np.random.randn(N, D)
    true_w = np.zeros(D)
    true_w[:5] = [2.5, -1.8, 3.2, -2.1, 1.5]  # Only first 5 features are causal
    logits = np.dot(X_syn, true_w) + 0.5
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_syn = (probs >= 0.5).astype(int)

    # 1. Newton-Raphson Solver with L2 (Ridge)
    newton_model = AdvancedLogisticRegression(solver="newton", penalty="l2", C=1.0, max_iter=50)
    newton_model.fit(X_syn, y_syn)
    acc_newton = np.mean(newton_model.predict(X_syn) == y_syn) * 100
    print(f"[Newton-Raphson L2] Fit converged in {len(newton_model.loss_history)} iterations | Train Acc: {acc_newton:.2f}%")

    # 2. Proximal Gradient Solver with L1 (Lasso)
    lasso_model = AdvancedLogisticRegression(solver="proximal", penalty="l1", C=0.2, learning_rate=0.2, max_iter=2000)
    lasso_model.fit(X_syn, y_syn)
    acc_lasso = np.mean(lasso_model.predict(X_syn) == y_syn) * 100
    zeros_count = np.sum(lasso_model.weights == 0.0)
    print(f"[Proximal Lasso L1] Sparsity achieved: {zeros_count}/{D} weights zeroed | Train Acc: {acc_lasso:.2f}%")
`;

export default function LogisticRegressionPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      {/* ── Track Badge ── */}
      <div className="not-prose mb-4">
        <span className="inline-block rounded-full bg-emerald-100 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-emerald-700">
          Track 1 · Classical Machine Learning
        </span>
      </div>

      <h1>Logistic Regression: Likelihood Geometry, Convex Optimization, Regularization, and Softmax Generalization</h1>

      <p className="lead text-lg text-slate-700">
        Logistic regression is the foundational probabilistic classification algorithm in machine learning.
        Despite the historical name &ldquo;regression&rdquo;, it solves classification by modeling the posterior log-odds
        as an affine function of features. This lecture provides a comprehensive mathematical derivation of the logistic
        sigmoid from the odds ratio, Maximum Likelihood Estimation (MLE), proof of loss convexity via the positive-definite
        Hessian matrix, second-order optimization via Newton-Raphson (IRLS), exact L1/L2/ElasticNet regularizers, and
        the multiclass Softmax regression extension.
      </p>

      {/* ── Learning Objectives ── */}
      <h2>Learning Objectives</h2>
      <ul>
        <li>
          Derive the logistic sigmoid function from the logarithm of odds (logit) and prove why linear regression is mathematically invalid for classification.
        </li>
        <li>
          Formulate the Bernoulli likelihood and derive the Binary Cross-Entropy loss from Maximum Likelihood principles.
        </li>
        <li>
          Prove that Binary Cross-Entropy is strictly convex by deriving the Hessian matrix <span className="font-mono text-sm">H = X^T S X</span> and proving it is positive semi-definite.
        </li>
        <li>
          Derive the Newton-Raphson update rule and prove how it reduces to Iteratively Reweighted Least Squares (IRLS).
        </li>
        <li>
          Compare L2 (Ridge), L1 (Lasso), and Elastic Net regularizers in logistic regression and understand how proximal operators enforce true parameter sparsity.
        </li>
        <li>
          Extend the binary model to multiclass classification via Multinomial Logistic Regression (Softmax) and understand over-parameterization invariance.
        </li>
      </ul>

      {/* ══════════════════════════════════════════════
          Section 1 · Why Linear Regression Fails
      ══════════════════════════════════════════════ */}
      <h2>1 · The Failure of Ordinary Least Squares for Classification</h2>
      <p>
        A naive approach to binary classification (<span className="font-mono text-sm">y ∈ &#123;0, 1&#125;</span>) is to fit an ordinary
        least squares line <span className="font-mono text-sm">ŷ = w^T x + b</span> and threshold predictions at <span className="font-mono text-sm">0.5</span>.
        This approach fails catastrophically due to three distinct mathematical violations:
      </p>

      <ConceptCard variant="warning" title="Three Mathematical Flaws of Linear Regression for Classification">
        <ol className="list-decimal pl-5 space-y-2 text-sm">
          <li>
            <strong>Unbounded Predictions Violate Probability Axioms:</strong> The output of a linear model is <span className="font-mono text-xs">ŷ ∈ (-∞, ∞)</span>.
            Probabilities must be bounded in <span className="font-mono text-xs">[0, 1]</span> by Kolmogorov&apos;s axioms.
            Predictions such as <span className="font-mono text-xs">P(y=1|x) = 2.4</span> or <span className="font-mono text-xs">-0.6</span> are mathematically meaningless.
          </li>
          <li>
            <strong>Distortion by Extreme Outliers:</strong> Least squares squares all residuals <span className="font-mono text-xs">(y - w^T x)^2</span>.
            An observation that is <em>deeply on the correct side</em> of the classification boundary (e.g., <span className="font-mono text-xs">x = 100</span> with <span className="font-mono text-xs">y = 1</span>)
            yields a massive positive residual <span className="font-mono text-xs">(1 - 5.0)^2 = 16</span>. To minimize this squared error,
            OLS rotates the decision boundary toward the outlier, destroying classification accuracy on nearby boundary samples.
          </li>
          <li>
            <strong>Heteroscedasticity of Residuals:</strong> In classification, <span className="font-mono text-xs">Var(y|x) = p(x)(1 - p(x))</span>.
            Because variance depends directly on <span className="font-mono text-xs">x</span>, errors are not homoscedastic, directly violating
            the Gauss-Markov theorem and rendering OLS inefficient.
          </li>
        </ol>
      </ConceptCard>

      {/* ══════════════════════════════════════════════
          Section 2 · The Logit & Sigmoid Derivation
      ══════════════════════════════════════════════ */}
      <h2>2 · The Logistic Sigmoid: Derivation from the Odds Ratio</h2>
      <p>
        To model a valid probability <span className="font-mono text-sm">p(x) = P(Y = 1 | X = x)</span>, we map the constrained
        interval <span className="font-mono text-sm">(0, 1)</span> to the unconstrained real line <span className="font-mono text-sm">(-∞, ∞)</span>{" "}
        using the <strong>odds ratio</strong>.
      </p>

      <MathBlock
        title="Derivation of the Sigmoid from the Log-Odds"
        steps={[
          {
            label: "Odds Ratio",
            formula: "Odds = p / (1 - p),   where p ∈ (0, 1)",
            annotation: "Maps (0, 1) to the open interval (0, ∞). For p = 0.8, odds = 4:1.",
          },
          {
            label: "Log-Odds (Logit)",
            formula: "logit(p) = ln(p / (1 - p))",
            annotation: "Takes the natural logarithm, mapping (0, ∞) to the entire real line (-∞, ∞).",
          },
          {
            label: "Linear Model on Logit",
            formula: "ln(p / (1 - p)) = w^T x + b",
            annotation: "We assume the log-odds is a linear function of features.",
          },
          {
            label: "Exponentiation",
            formula: "p / (1 - p) = e^(w^T x + b)",
            annotation: "Exponentiating both sides to invert the natural log.",
          },
          {
            label: "Algebraic Inversion",
            formula: "p = e^(w^T x + b) · (1 - p)  ⟹  p · (1 + e^(w^T x + b)) = e^(w^T x + b)",
            annotation: "Collecting terms containing p.",
          },
        ]}
        result="p(x) = σ(w^T x + b) = 1 / (1 + e^-(w^T x + b))"
        resultNote="The logistic sigmoid function σ(z) is the unique analytical inverse of the logit transformation."
      />

      {/* ══════════════════════════════════════════════
          Section 3 · Maximum Likelihood Estimation
      ══════════════════════════════════════════════ */}
      <h2>3 · Maximum Likelihood Estimation (MLE) &amp; Binary Cross-Entropy</h2>
      <p>
        Because the target variable is binary, each label <span className="font-mono text-sm">y_i ∈ &#123;0, 1&#125;</span> follows a
        conditional Bernoulli distribution: <span className="font-mono text-sm">Y_i | X_i ~ Bernoulli(p_i)</span> where{" "}
        <span className="font-mono text-sm">p_i = σ(w^T x_i + b)</span>.
      </p>

      <MathBlock
        title="Derivation of Negative Log-Likelihood"
        steps={[
          {
            label: "Bernoulli PMF",
            formula: "P(Y = y_i | x_i) = p_i^(y_i) · (1 - p_i)^(1 - y_i)",
            annotation: "When y_i = 1, expression equals p_i; when y_i = 0, expression equals (1 - p_i).",
          },
          {
            label: "Joint Likelihood",
            formula: "L(w, b) = ∏_(i=1)^n p_i^(y_i) · (1 - p_i)^(1 - y_i)",
            annotation: "Under the assumption that training examples are independent and identically distributed (i.i.d.).",
          },
          {
            label: "Log-Likelihood",
            formula: "ℓ(w, b) = ln L(w, b) = ∑_(i=1)^n [y_i · ln(p_i) + (1 - y_i) · ln(1 - p_i)]",
            annotation: "Converting the product into a sum of logarithms.",
          },
          {
            label: "Cost Minimization",
            formula: "J(w, b) = - (1 / n) · ℓ(w, b) = - (1 / n) ∑_(i=1)^n [y_i · ln(p_i) + (1 - y_i) · ln(1 - p_i)]",
            annotation: "Negative Log-Likelihood normalized by sample size n. Known as Binary Cross-Entropy (BCE).",
          },
        ]}
        result="Gradient: ∇_w J = (1 / n) X^T (σ(Xw + b) - y)"
        resultNote="Notice the remarkable symmetry: the gradient has the exact same functional form as OLS linear regression (1/n) X^T (ŷ - y), with the prediction now squashed by the sigmoid."
      />

      {/* ══════════════════════════════════════════════
          Section 4 · Proof of Convexity & Hessian
      ══════════════════════════════════════════════ */}
      <h2>4 · Proof of Strict Convexity: The Hessian Matrix</h2>
      <p>
        Does logistic regression have local minima that could trap gradient descent?
        We prove analytically that the Binary Cross-Entropy loss is strictly convex.
      </p>

      <MathBlock
        title="Analytical Hessian Matrix Derivation"
        steps={[
          {
            label: "First Derivative (Gradient)",
            formula: "∂J / ∂w = (1 / n) ∑_(i=1)^n (p_i - y_i) · x_i = (1 / n) X^T (p - y)",
            annotation: "Where p = σ(Xw + b) ∈ ℝ^n.",
          },
          {
            label: "Second Derivative (Hessian)",
            formula: "H = ∂²J / ∂w∂w^T = (1 / n) ∑_(i=1)^n (∂p_i / ∂w) · x_i^T",
            annotation: "Differentiating the gradient vector w.r.t w.",
          },
          {
            label: "Chain Rule on Sigmoid",
            formula: "∂p_i / ∂w = σ'(w^T x_i + b) · x_i = p_i · (1 - p_i) · x_i",
            annotation: "Using the identity σ'(z) = σ(z)(1 - σ(z)).",
          },
          {
            label: "Matrix Representation",
            formula: "H = (1 / n) ∑_(i=1)^n p_i (1 - p_i) · x_i x_i^T = (1 / n) X^T S X",
            annotation: "Where S = diag(p_1(1 - p_1), p_2(1 - p_2), ..., p_n(1 - p_n)) is an n × n diagonal weighting matrix.",
          },
        ]}
        result="Positive Semi-Definite Proof: u^T H u = (1/n) ||S^(1/2) X u||_2^2 ≥ 0 for all u ≠ 0"
        resultNote="Since 0 < p_i < 1 strictly, all diagonal entries of S are strictly positive (0 < s_ii ≤ 0.25). Thus, H ⪰ 0 everywhere, proving J(w) is globally convex with no sub-optimal local minima!"
      />

      {/* ══════════════════════════════════════════════
          Section 5 · Optimization: Newton-Raphson & IRLS
      ══════════════════════════════════════════════ */}
      <h2>5 · Second-Order Optimization: Newton-Raphson &amp; IRLS</h2>
      <p>
        While First-Order Gradient Descent takes steps proportional to <span className="font-mono text-sm">∇J</span>,
        the <strong>Newton-Raphson</strong> algorithm approximates the objective with a second-order Taylor polynomial,
        multiplying by the inverse Hessian <span className="font-mono text-sm">H^(-1)</span>.
      </p>

      <MathBlock
        title="Newton-Raphson to IRLS Equivalence"
        steps={[
          {
            label: "Newton-Raphson Step",
            formula: "w^(t+1) = w^(t) - H^(-1) · ∇J(w^(t))",
            annotation: "Quadratic convergence rate: error_t+1 ≈ (error_t)^2 near the optimum.",
          },
          {
            label: "Substitution",
            formula: "w^(t+1) = w^(t) - (X^T S X)^(-1) · X^T (p - y)",
            annotation: "Plugging in the analytical gradient and Hessian matrix.",
          },
          {
            label: "Algebraic Factoring",
            formula: "= (X^T S X)^(-1) · [ (X^T S X) w^(t) - X^T (p - y) ] = (X^T S X)^(-1) X^T S [ X w^(t) - S^(-1)(p - y) ]",
            annotation: "Factoring out (X^T S X)^(-1) X^T S.",
          },
          {
            label: "Adjusted Target z",
            formula: "z = X w^(t) + S^(-1)(y - p)",
            annotation: "Working response vector in linear regression space.",
          },
        ]}
        result="w^(t+1) = (X^T S X)^(-1) X^T S z   (Weighted Least Squares at every iteration)"
        resultNote="This proves that Newton-Raphson on Logistic Regression is mathematically identical to solving an Iteratively Reweighted Least Squares (IRLS) problem where weights S change at each step."
      />

      {/* ══════════════════════════════════════════════
          Section 6 · Regularization: Ridge vs Lasso vs ElasticNet
      ══════════════════════════════════════════════ */}
      <h2>6 · Regularization: Ridge (L2), Lasso (L1), and Elastic Net</h2>
      <p>
        When features are collinear or when classes are <strong>completely linearly separable</strong>,
        unregularized logistic regression diverges to infinity: <span className="font-mono text-sm">||w|| → ∞</span> to force
        probabilities to exact 0 and 1. Regularization prevents this failure mode and controls model variance.
      </p>

      <ComparisonTable
        title="Logistic Regression Regularization Schemes"
        subtitle="Mathematical formulation, optimization method, and structural impact on weights."
        columns={[
          { key: "scheme", header: "Penalty", width: "15%" },
          { key: "penaltyTerm", header: "Penalty Term Ω(w)", width: "25%" },
          { key: "sparsity", header: "Feature Sparsity", width: "18%" },
          { key: "solver", header: "Primary Solver", width: "20%" },
          { key: "whenToUse", header: "When to Select", width: "22%" },
        ]}
        data={[
          {
            scheme: <strong className="text-slate-900">None (MLE)</strong>,
            penaltyTerm: <code className="text-xs">0</code>,
            sparsity: "Zero (all weights non-zero)",
            solver: "Newton-Raphson, L-BFGS",
            whenToUse: "n ≫ d, no multicollinearity, inference/hypothesis testing priority",
          },
          {
            scheme: <strong className="text-slate-900">Ridge (L2)</strong>,
            penaltyTerm: <code className="text-xs">(λ / 2) ||w||_2^2</code>,
            sparsity: "None (smooth shrinkage)",
            solver: "Newton / IRLS, L-BFGS",
            whenToUse: "Collinear features, prevents infinite weights on separable data",
          },
          {
            scheme: <strong className="text-slate-900">Lasso (L1)</strong>,
            penaltyTerm: <code className="text-xs">λ ||w||_1</code>,
            sparsity: <span className="text-emerald-700 font-semibold">Exact Zeroes (Sparse)</span>,
            solver: "Proximal Gradient, Coordinate Descent",
            whenToUse: "High-dimensional data (d > n), automated feature selection",
          },
          {
            scheme: <strong className="text-slate-900">Elastic Net</strong>,
            penaltyTerm: <code className="text-xs">λ [α||w||_1 + (1-α)/2 ||w||_2^2]</code>,
            sparsity: <span className="text-emerald-700 font-semibold">Exact Zeroes with Grouping</span>,
            solver: "Proximal Gradient, SAGA",
            whenToUse: "High-dimensional data with groups of strongly correlated predictors",
          },
        ]}
      />

      <ConceptCard variant="insight" title="The Elastic Net Grouping Effect in Logistic Regression">
        <p>
          Lasso possesses a well-documented failure mode on correlated features: if features <span className="font-mono text-xs">x_1</span> and{" "}
          <span className="font-mono text-xs">x_2</span> have correlation <span className="font-mono text-xs">r ≈ 0.99</span>, Lasso arbitrarily
          picks one and forces the other to zero.
        </p>
        <p>
          Elastic Net (Zou &amp; Hastie, 2005) resolves this via the strictly convex L2 component: the difference between their
          coefficients is bounded by <span className="font-mono text-xs">||w_1 - w_2||_1 ≤ (1 / λ_2) √2(1 - r)</span>.
          As correlation <span className="font-mono text-xs">r → 1</span>, <span className="font-mono text-xs">w_1 → w_2</span>, ensuring
          that correlated groups of features are selected or dropped together.
        </p>
      </ConceptCard>

      {/* ══════════════════════════════════════════════
          Section 7 · Multiclass Softmax Regression
      ══════════════════════════════════════════════ */}
      <h2>7 · Multiclass Generalization: Softmax Regression</h2>
      <p>
        For multiclass problems with <span className="font-mono text-sm">K &gt; 2</span> mutually exclusive classes,
        the model outputs a categorical probability distribution parameterized by weight vectors <span className="font-mono text-sm">w_1, ..., w_K</span>:
      </p>

      <MathBlock
        title="Softmax Regression Formulation"
        steps={[
          {
            label: "Class Probability",
            formula: "P(Y = k | x) = e^(w_k^T x) / (∑_(j=1)^K e^(w_j^T x))",
            annotation: "For k = 1, 2, ..., K. All probabilities sum strictly to 1.",
          },
          {
            label: "Categorical Cross-Entropy",
            formula: "J(W) = - (1 / n) ∑_(i=1)^n ∑_(k=1)^K 1&#123;y_i = k&#125; · ln(P(Y = k | x_i))",
            annotation: "Where 1{y_i = k} is the indicator function (one-hot label).",
          },
          {
            label: "Gradient w.r.t Class Vector",
            formula: "∇_(w_k) J = (1 / n) ∑_(i=1)^n x_i · (P(Y = k | x_i) - 1&#123;y_i = k&#125;)",
            annotation: "Residual error for class k multiplied by the feature vector.",
          },
        ]}
        result="Over-parameterization: Softmax is invariant to adding an arbitrary constant vector ψ to all w_k"
        resultNote="In practice, we either set w_K = 0 (leaving K-1 parameters, identical to binary logistic regression when K=2) or apply L2 regularization to resolve the redundancy."
      />

      {/* ══════════════════════════════════════════════
          Section 8 · Complete Python Implementation
      ══════════════════════════════════════════════ */}
      <h2>8 · Production Python Implementation: Solvers &amp; Penalties</h2>
      <p>
        Below is the complete implementation of <code>AdvancedLogisticRegression</code> incorporating Newton-Raphson,
        Proximal Gradient with Soft-Thresholding for Lasso/ElasticNet, and Nesterov Accelerated Gradient Descent.
      </p>

      <TerminalBlock language="python" filename="advanced_logistic_regression.py" code={fullLogisticCode} />

      {/* ══════════════════════════════════════════════
          Section 9 · Training Algorithm Steps
      ══════════════════════════════════════════════ */}
      <h2>9 · Algorithmic Workflow: Newton-Raphson vs Proximal Gradient</h2>
      <AlgorithmBlock
        name="Newton-Raphson (IRLS) Logistic Regression Solver"
        input="Matrix X ∈ ℝ^(n × d), Binary labels y ∈ ℝ^n, L2 penalty λ, Convergence tol ε"
        output="Optimized weight vector w* ∈ ℝ^d, bias b*"
        timeComplexity="O(t · (n · d² + d³)) where t is iterations to convergence (typically t < 10)"
        spaceComplexity="O(d²) to store the d × d Hessian matrix"
        steps={[
          {
            step: 1,
            title: "Compute Posterior Probabilities",
            detail: "Compute z = Xw + b and evaluate stable sigmoid p_i = 1 / (1 + exp(-z_i)).",
            complexity: "O(n · d)",
          },
          {
            step: 2,
            title: "Form Residual Gradient & Weighting Matrix",
            detail: "Compute error vector e = p - y. Compute diagonal weights s_i = p_i(1 - p_i) and form gradient ∇w = (1/n) X^T e + (λ/n) w.",
            complexity: "O(n · d)",
          },
          {
            step: 3,
            title: "Form & Invert Hessian Matrix",
            detail: "Compute H = (1/n) X^T diag(s) X + (λ/n) I. Solve the linear system H · Δw = ∇w using Cholesky factorization.",
            complexity: "O(n · d² + d³)",
          },
          {
            step: 4,
            title: "Update Parameters & Evaluate Convergence",
            detail: "Update w ← w - Δw. Terminate if ||Δw||_2 < ε; otherwise repeat.",
            complexity: "O(d)",
          },
        ]}
      />

      {/* ── References ── */}
      <h2>10 · Academic References</h2>
      <ul className="text-sm space-y-2">
        <li>
          <strong>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009)</strong>. <em>The Elements of Statistical Learning</em>. Chapter 4: Linear Methods for Classification. Springer.
        </li>
        <li>
          <strong>McCullagh, P., &amp; Nelder, J. A. (1989)</strong>. <em>Generalized Linear Models</em>. Chapman and Hall/CRC.
        </li>
        <li>
          <strong>Zou, H., &amp; Hastie, T. (2005)</strong>. <em>Regularization and variable selection via the elastic net</em>. Journal of the Royal Statistical Society: Series B, 67(2), 301–320.
        </li>
        <li>
          <strong>Platt, J. (1999)</strong>. <em>Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods</em>. Advances in Large Margin Classifiers.
        </li>
      </ul>
    </article>
  );
}
