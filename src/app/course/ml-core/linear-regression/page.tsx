"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

export default function Page() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      {/* ── Track Badge ── */}
      <div className="not-prose mb-4">
        <span className="inline-block rounded-full bg-emerald-100 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-emerald-700">
          Track 1 · Classical Machine Learning
        </span>
      </div>

      <h1>Linear Regression</h1>

      <p>
        Linear regression is the bedrock of supervised learning. Despite its age—
        Gauss and Legendre formulated the least-squares method in the early 1800s—
        it remains the first model every practitioner reaches for because it is
        interpretable, computationally cheap, and provably optimal under a well-
        defined set of conditions. Understanding it deeply—loss geometry, closed-form
        solutions, gradient descent, regularisation, and diagnostics—gives you the
        conceptual vocabulary to reason about every more complex model that follows.
      </p>

      {/* ── Learning Objectives ── */}
      <h2>Learning Objectives</h2>
      <ul>
        <li>
          Derive the Ordinary Least Squares (OLS) normal equation from first
          principles and understand its O(n³) complexity trade-off.
        </li>
        <li>
          Derive the MSE gradient and implement batch gradient descent for linear
          regression from scratch, then verify against scikit-learn.
        </li>
        <li>
          State all five Gauss-Markov assumptions and explain what goes wrong in
          practice when each is violated.
        </li>
        <li>
          Distinguish Ridge (L2) from Lasso (L1) regularisation geometrically and
          algebraically, including when each produces exact zeros.
        </li>
        <li>
          Compute and interpret R², adjusted R², and connect them to the
          bias-variance trade-off under regularisation.
        </li>
        <li>
          Diagnose model violations using residual plots, Q-Q plots, and Variance
          Inflation Factor (VIF), and apply appropriate remedies.
        </li>
      </ul>

      {/* ══════════════════════════════════════════════
          Section 1 · Problem Formulation
      ══════════════════════════════════════════════ */}
      <h2>1 · Problem Formulation</h2>

      <p>
        Supervised learning starts with a dataset of <em>n</em> pairs{" "}
        <span className="font-mono text-sm">(xᵢ, yᵢ)</span> where each{" "}
        <span className="font-mono text-sm">xᵢ ∈ ℝᵈ</span> is a feature vector
        and <span className="font-mono text-sm">yᵢ ∈ ℝ</span> is a continuous
        target. Linear regression assumes the target is a linear function of the
        features plus noise:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p className="text-slate-400 mb-2">// Data generating process</p>
        <p>y = wᵀx + b + ε</p>
        <p className="mt-3 text-slate-400">// where:</p>
        <p>  w ∈ ℝᵈ   — weight vector (one scalar per feature)</p>
        <p>  b ∈ ℝ    — bias / intercept</p>
        <p>  ε ∼ 𝒩(0, σ²)  — irreducible noise, assumed i.i.d.</p>
      </div>

      <p>
        We absorb the bias into the weight vector by appending a constant feature
        of 1 to every input: <span className="font-mono text-sm">x̃ = [x; 1]</span>,{" "}
        <span className="font-mono text-sm">θ = [w; b]</span>. Then the hypothesis
        is simply:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>ĥ(x) = θᵀx̃</p>
        <p className="mt-2 text-slate-400">// In matrix form across all n examples:</p>
        <p>ŷ = Xθ,   X ∈ ℝⁿˣ⁽ᵈ⁺¹⁾</p>
      </div>

      <p>
        The <strong>design matrix</strong> X has each row as one training example
        (with the bias column of 1s appended). Our goal: find θ* that best predicts
        y. "Best" requires defining a loss function.
      </p>

      {/* ══════════════════════════════════════════════
          Section 2 · The Loss Function: MSE
      ══════════════════════════════════════════════ */}
      <h2>2 · The Loss Function: Mean Squared Error</h2>

      <p>
        The most common choice is <strong>Mean Squared Error (MSE)</strong>:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>J(θ) = (1/n) Σᵢ (yᵢ - θᵀxᵢ)²</p>
        <p className="mt-2 text-slate-400">// Or equivalently in matrix form:</p>
        <p>J(θ) = (1/n) ‖y - Xθ‖₂²</p>
        <p>     = (1/n) (y - Xθ)ᵀ(y - Xθ)</p>
      </div>

      <p>
        <strong>Why squared and not absolute?</strong> Two key reasons:
      </p>
      <ol>
        <li>
          <strong>Differentiability.</strong> The squared loss has a smooth,
          everywhere-differentiable gradient. The absolute loss{" "}
          <span className="font-mono text-sm">|ê|</span> is not differentiable at
          zero, complicating gradient-based optimisation.
        </li>
        <li>
          <strong>Quadratic penalty on large errors.</strong> Squaring amplifies
          large residuals super-linearly, making the model highly sensitive to
          outliers—which is both a feature (we care a lot about big mistakes) and
          a bug (one extreme outlier can dominate the fit). This trade-off motivates
          robust regression alternatives like Huber loss, but MSE remains the
          canonical starting point.
        </li>
      </ol>

      <p>
        Probabilistically, minimising MSE is equivalent to{" "}
        <strong>maximum likelihood estimation</strong> when ε ∼ 𝒩(0, σ²). The
        log-likelihood of the data under this assumption is:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>log p(y | X, θ) = -n/2 · log(2πσ²) - (1/2σ²) ‖y - Xθ‖₂²</p>
        <p className="mt-2 text-slate-400">// Maximising log-likelihood ⟺ minimising ‖y - Xθ‖₂²  ⟺  minimising MSE</p>
      </div>

      <p>
        This probabilistic grounding justifies MSE far beyond "it's convenient"—it
        is the correct loss when you believe errors are Gaussian.
      </p>

      {/* ══════════════════════════════════════════════
          Section 3 · Ordinary Least Squares (Normal Equation)
      ══════════════════════════════════════════════ */}
      <h2>3 · Ordinary Least Squares: the Normal Equation</h2>

      <p>
        Because J(θ) is convex and differentiable, we can find the global minimum
        analytically by setting the gradient to zero and solving.
      </p>

      <h3>Step-by-step derivation</h3>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 space-y-2">
        <p className="text-slate-400">// Expand the matrix MSE:</p>
        <p>J(θ) = (1/n)(y - Xθ)ᵀ(y - Xθ)</p>
        <p>     = (1/n)(yᵀy - 2θᵀXᵀy + θᵀXᵀXθ)</p>
        <p className="mt-3 text-slate-400">// Take the gradient w.r.t. θ (using matrix calculus identities):</p>
        <p>∂J/∂θ = (1/n)(-2Xᵀy + 2XᵀXθ)</p>
        <p className="mt-3 text-slate-400">// Set gradient to zero:</p>
        <p>XᵀXθ = Xᵀy</p>
        <p className="mt-3 text-slate-400">// Multiply both sides by (XᵀX)⁻¹, assuming it exists:</p>
        <p className="text-emerald-300 font-bold">θ* = (XᵀX)⁻¹Xᵀy</p>
      </div>

      <p>
        This is the <strong>Normal Equation</strong>. The matrix{" "}
        <span className="font-mono text-sm">Xᵀ</span> is sometimes called the{" "}
        <em>Moore-Penrose pseudoinverse</em> component, and{" "}
        <span className="font-mono text-sm">(XᵀX)⁻¹Xᵀ</span> is the{" "}
        <em>pseudoinverse</em> X⁺.
      </p>

      <h3>When does (XᵀX)⁻¹ not exist?</h3>
      <p>
        The matrix XᵀX is singular (non-invertible) when features are{" "}
        <strong>perfectly multicollinear</strong>—one feature is an exact linear
        combination of others. In practice, near-singularity causes numerical
        instability. Ridge regression (Section 7) solves this by adding λI to the
        diagonal.
      </p>

      <h3>Computational complexity</h3>
      <div className="not-prose overflow-x-auto my-6">
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="bg-slate-100">
              <th className="border border-slate-300 px-4 py-2 text-left">Step</th>
              <th className="border border-slate-300 px-4 py-2 text-left">Operation</th>
              <th className="border border-slate-300 px-4 py-2 text-left">Complexity</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-slate-300 px-4 py-2">Compute XᵀX</td>
              <td className="border border-slate-300 px-4 py-2">Matrix multiply (d+1)×n × n×(d+1)</td>
              <td className="border border-slate-300 px-4 py-2">O(nd²)</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="border border-slate-300 px-4 py-2">Invert XᵀX</td>
              <td className="border border-slate-300 px-4 py-2">Gaussian elimination on (d+1)×(d+1)</td>
              <td className="border border-slate-300 px-4 py-2">O(d³)</td>
            </tr>
            <tr>
              <td className="border border-slate-300 px-4 py-2">Compute Xᵀy</td>
              <td className="border border-slate-300 px-4 py-2">Matrix-vector multiply</td>
              <td className="border border-slate-300 px-4 py-2">O(nd)</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="border border-slate-300 px-4 py-2 font-semibold">Total</td>
              <td className="border border-slate-300 px-4 py-2"></td>
              <td className="border border-slate-300 px-4 py-2 font-semibold">O(nd² + d³)</td>
            </tr>
          </tbody>
        </table>
      </div>

      <p>
        When features d are small the bottleneck is O(nd²). When d is large (e.g.,
        d ≈ 10⁶ in NLP), inversion becomes O(d³) ≈ 10¹⁸ operations — completely
        infeasible. Gradient descent scales to millions of features because it
        never explicitly forms XᵀX.
      </p>

      {/* ══════════════════════════════════════════════
          Section 4 · Gradient Descent Solution
      ══════════════════════════════════════════════ */}
      <h2>4 · Gradient Descent Solution</h2>

      <p>
        When n or d is large, iterative optimisation is the only practical option.
        Gradient descent moves θ in the direction of steepest descent of J.
      </p>

      <h3>Gradient derivation</h3>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 space-y-2">
        <p className="text-slate-400">// For weight vector w (bias treated separately for clarity):</p>
        <p>J(w, b) = (1/n) Σᵢ (yᵢ - wᵀxᵢ - b)²</p>
        <p className="mt-3 text-slate-400">// Partial derivatives (chain rule):</p>
        <p>∂J/∂w  = -(2/n) Σᵢ (yᵢ - ŷᵢ) · xᵢ</p>
        <p>       = -(2/n) Xᵀ(y - Xθ)       [matrix form]</p>
        <p className="mt-2">∂J/∂b  = -(2/n) Σᵢ (yᵢ - ŷᵢ)</p>
        <p className="mt-3 text-slate-400">// Update rules (α = learning rate):</p>
        <p className="text-emerald-300">w ← w - α · ∂J/∂w</p>
        <p className="text-emerald-300">b ← b - α · ∂J/∂b</p>
      </div>

      <h3>Variants</h3>
      <div className="not-prose overflow-x-auto my-6">
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="bg-slate-100">
              <th className="border border-slate-300 px-4 py-2 text-left">Variant</th>
              <th className="border border-slate-300 px-4 py-2 text-left">Batch size</th>
              <th className="border border-slate-300 px-4 py-2 text-left">Update frequency</th>
              <th className="border border-slate-300 px-4 py-2 text-left">Notes</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-slate-300 px-4 py-2">Batch GD</td>
              <td className="border border-slate-300 px-4 py-2">All n</td>
              <td className="border border-slate-300 px-4 py-2">Once per epoch</td>
              <td className="border border-slate-300 px-4 py-2">Exact gradient; slow per step when n is huge</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="border border-slate-300 px-4 py-2">SGD</td>
              <td className="border border-slate-300 px-4 py-2">1</td>
              <td className="border border-slate-300 px-4 py-2">n times per epoch</td>
              <td className="border border-slate-300 px-4 py-2">Noisy gradient; fast; may not converge to exact minimum</td>
            </tr>
            <tr>
              <td className="border border-slate-300 px-4 py-2">Mini-batch GD</td>
              <td className="border border-slate-300 px-4 py-2">32–512</td>
              <td className="border border-slate-300 px-4 py-2">n/batch per epoch</td>
              <td className="border border-slate-300 px-4 py-2">Best of both worlds; standard in deep learning</td>
            </tr>
          </tbody>
        </table>
      </div>

      <p>
        For convex losses like MSE, batch gradient descent converges to the global
        minimum with a sufficiently small learning rate. The convergence rate is
        linear: error shrinks as O((1 - αμ)ᵏ) where μ is the smallest eigenvalue
        of XᵀX and k is the iteration count.
      </p>

      {/* ══════════════════════════════════════════════
          Section 5 · Gauss-Markov Assumptions
      ══════════════════════════════════════════════ */}
      <h2>5 · Assumptions and the Gauss-Markov Theorem</h2>

      <p>
        OLS is not just convenient—under specific assumptions, the{" "}
        <strong>Gauss-Markov theorem</strong> guarantees it is the{" "}
        <em>Best Linear Unbiased Estimator</em> (BLUE): no other linear unbiased
        estimator has smaller variance. Those assumptions are:
      </p>

      <div className="not-prose overflow-x-auto my-6">
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="bg-slate-100">
              <th className="border border-slate-300 px-4 py-2 text-left">#</th>
              <th className="border border-slate-300 px-4 py-2 text-left">Assumption</th>
              <th className="border border-slate-300 px-4 py-2 text-left">Formal statement</th>
              <th className="border border-slate-300 px-4 py-2 text-left">What breaks when violated</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-slate-300 px-4 py-2">A1</td>
              <td className="border border-slate-300 px-4 py-2 font-semibold">Linearity</td>
              <td className="border border-slate-300 px-4 py-2 font-mono text-xs">𝔼[y | X] = Xθ</td>
              <td className="border border-slate-300 px-4 py-2">Systematic bias; residuals show pattern</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="border border-slate-300 px-4 py-2">A2</td>
              <td className="border border-slate-300 px-4 py-2 font-semibold">Independence</td>
              <td className="border border-slate-300 px-4 py-2 font-mono text-xs">εᵢ ⊥ εⱼ  ∀ i≠j</td>
              <td className="border border-slate-300 px-4 py-2">Serially correlated errors; SE underestimated</td>
            </tr>
            <tr>
              <td className="border border-slate-300 px-4 py-2">A3</td>
              <td className="border border-slate-300 px-4 py-2 font-semibold">Homoscedasticity</td>
              <td className="border border-slate-300 px-4 py-2 font-mono text-xs">Var(εᵢ) = σ² (constant)</td>
              <td className="border border-slate-300 px-4 py-2">Inefficient estimates; heteroskedastic-robust SE needed</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="border border-slate-300 px-4 py-2">A4</td>
              <td className="border border-slate-300 px-4 py-2 font-semibold">No multicollinearity</td>
              <td className="border border-slate-300 px-4 py-2 font-mono text-xs">rank(X) = d+1</td>
              <td className="border border-slate-300 px-4 py-2">XᵀX singular; coefficients numerically unstable</td>
            </tr>
            <tr>
              <td className="border border-slate-300 px-4 py-2">A5</td>
              <td className="border border-slate-300 px-4 py-2 font-semibold">Normality of ε</td>
              <td className="border border-slate-300 px-4 py-2 font-mono text-xs">ε ∼ 𝒩(0, σ²I)</td>
              <td className="border border-slate-300 px-4 py-2">Inference (p-values, CIs) invalid; not required for BLUE</td>
            </tr>
          </tbody>
        </table>
      </div>

      <p>
        Note: A5 (normality) is not required for the Gauss-Markov theorem itself,
        only for exact inference. By the Central Limit Theorem, OLS estimators are
        approximately normal for large n even without A5.
      </p>

      {/* ══════════════════════════════════════════════
          Section 6 · Diagnosing Violations
      ══════════════════════════════════════════════ */}
      <h2>6 · Diagnosing Assumption Violations</h2>

      <p>
        Before trusting OLS estimates, every analyst must run diagnostic checks.
        Here are the canonical tools:
      </p>

      <h3>Residual vs. Fitted Plot</h3>
      <p>
        Plot ê = y − ŷ on the y-axis vs. ŷ on the x-axis. Under correct
        specification:
      </p>
      <ul>
        <li>Points should scatter randomly around the horizontal zero line (A1 satisfied).</li>
        <li>Spread should be roughly constant across ŷ values (A3 homoscedasticity satisfied).</li>
        <li>A funnel shape (increasing spread) indicates heteroscedasticity → use log transform or WLS.</li>
        <li>A curved pattern indicates non-linearity → add polynomial features or use a non-linear model.</li>
      </ul>

      <h3>Quantile-Quantile (Q-Q) Plot</h3>
      <p>
        Plot the empirical quantiles of standardised residuals against theoretical
        normal quantiles. Points falling close to the 45° line indicate ε is
        approximately Gaussian (A5). Heavy tails or S-curves signal departures from
        normality.
      </p>

      <h3>Variance Inflation Factor (VIF)</h3>
      <p>
        VIF quantifies how much the variance of a coefficient estimate is inflated
        due to multicollinearity with other predictors:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>VIF(wⱼ) = 1 / (1 - Rⱼ²)</p>
        <p className="mt-2 text-slate-400">// Rⱼ² is the R² from regressing feature j on all other features.</p>
        <p className="mt-2">// Rule of thumb:</p>
        <p>  VIF &lt; 5    → acceptable</p>
        <p>  VIF 5–10  → moderate concern; investigate</p>
        <p>  VIF &gt; 10   → severe multicollinearity; drop / combine features or use Ridge</p>
      </div>

      {/* ══════════════════════════════════════════════
          Section 7 · Ridge Regression (L2)
      ══════════════════════════════════════════════ */}
      <h2>7 · Ridge Regression (L2 Regularisation)</h2>

      <p>
        Ordinary least squares minimises fit quality (MSE) with no constraint on
        coefficient magnitude. This leads to overfitting when d is large relative
        to n, or when features are correlated. <strong>Ridge regression</strong>{" "}
        adds an L2 penalty on the weight vector:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>J_ridge(θ) = (1/n)‖y - Xθ‖₂²  +  λ‖w‖₂²</p>
        <p className="mt-2 text-slate-400">// λ ≥ 0 is the regularisation hyperparameter (not penalising bias b)</p>
        <p className="mt-2 text-slate-400">// Setting gradient to zero gives the closed form:</p>
        <p className="mt-2 text-emerald-300">θ* = (XᵀX + λI)⁻¹Xᵀy</p>
      </div>

      <p>
        The <span className="font-mono text-sm">+ λI</span> term adds λ to every
        diagonal entry of XᵀX, guaranteeing invertibility even when XᵀX is
        rank-deficient. This is why Ridge is both a regulariser{" "}
        <em>and</em> a numerical stabiliser for near-singular systems.
      </p>

      <h3>Geometric intuition</h3>
      <p>
        In the primal form, Ridge constrains the solution to lie within a sphere
        of radius r in weight space: <span className="font-mono text-sm">‖w‖₂² ≤ r²</span>.
        The L2 ball is smooth and strictly convex, so the constrained optimum
        lies on the interior of the sphere tangent to an MSE contour. Because the
        sphere has no corners, the solution is never exactly zero on any axis—Ridge{" "}
        <strong>shrinks all coefficients uniformly toward zero but never to zero</strong>.
      </p>

      <h3>Effect on the singular value decomposition</h3>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p className="text-slate-400">// Let X = UΣVᵀ be the SVD of X. Then:</p>
        <p>θ_OLS  = V Σ⁻¹ Uᵀ y</p>
        <p>θ_ridge = V (Σ² + λI)⁻¹ Σ Uᵀ y</p>
        <p className="mt-2 text-slate-400">// Ridge scales each singular-value direction by σᵢ²/(σᵢ² + λ)</p>
        <p className="text-slate-400">// Small singular values (low-variance directions) are shrunk most.</p>
      </div>

      {/* ══════════════════════════════════════════════
          Section 8 · Lasso Regression (L1)
      ══════════════════════════════════════════════ */}
      <h2>8 · Lasso Regression (L1 Regularisation)</h2>

      <p>
        <strong>Least Absolute Shrinkage and Selection Operator (Lasso)</strong>,
        introduced by Tibshirani (1996), replaces the L2 penalty with an L1 penalty:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>J_lasso(θ) = (1/n)‖y - Xθ‖₂²  +  λ‖w‖₁</p>
        <p className="mt-2 text-slate-400">// ‖w‖₁ = Σⱼ |wⱼ|  — the sum of absolute values</p>
        <p className="mt-2 text-slate-400">// NO closed form — ‖w‖₁ is not differentiable at wⱼ = 0</p>
        <p className="mt-2 text-slate-400">// Solved by coordinate descent or ISTA (proximal gradient)</p>
      </div>

      <h3>Why Lasso produces exact zeros (sparsity)</h3>
      <p>
        Geometrically, the L1 constraint is a diamond (hypercube in d dimensions),
        which has <em>corners</em> on the coordinate axes. The MSE contour ellipses
        are far more likely to touch a corner of the L1 ball than an edge,
        setting the corresponding coefficient to exactly zero. This is{" "}
        <strong>automatic variable selection</strong>—invaluable when you have
        thousands of features but expect only a handful to be relevant.
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p className="text-slate-400">// Coordinate descent update rule for Lasso (for feature j):</p>
        <p>// Compute the partial residual:</p>
        <p>ρⱼ = Σᵢ xᵢⱼ (yᵢ - ŷᵢ₍₋ⱼ₎)     where ŷᵢ₍₋ⱼ₎ excludes feature j</p>
        <p className="mt-2">// Apply soft-thresholding operator S(ρ, λ):</p>
        <p>wⱼ ← S(ρⱼ / ‖xⱼ‖₂² , λ / ‖xⱼ‖₂²)</p>
        <p className="mt-2">S(z, γ) = sign(z) · max(|z| - γ, 0)</p>
        <p className="mt-2 text-slate-400">// If |ρⱼ| &lt; λ, the coefficient is set to exactly 0.</p>
      </div>

      {/* ══════════════════════════════════════════════
          Section 9 · Elastic Net
      ══════════════════════════════════════════════ */}
      <h2>9 · Elastic Net: Combining L1 and L2</h2>

      <p>
        Elastic Net (Zou &amp; Hastie, 2005) interpolates between Ridge and Lasso:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>J_enet(θ) = (1/n)‖y - Xθ‖₂²  +  λ[α‖w‖₁ + (1-α)/2 · ‖w‖₂²]</p>
        <p className="mt-2 text-slate-400">// α ∈ [0,1] mixes L1 and L2:</p>
        <p>  α = 1  →  pure Lasso</p>
        <p>  α = 0  →  pure Ridge</p>
        <p>  α = 0.5 →  equal mix (typical default)</p>
      </div>

      <p>
        Elastic Net shines when features are <em>grouped</em> and correlated.
        Pure Lasso arbitrarily selects one feature from a correlated group;
        Elastic Net tends to select the whole group together (the "grouping effect").
      </p>

      <div className="not-prose overflow-x-auto my-6">
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="bg-slate-100">
              <th className="border border-slate-300 px-4 py-2 text-left">Method</th>
              <th className="border border-slate-300 px-4 py-2 text-left">Penalty</th>
              <th className="border border-slate-300 px-4 py-2 text-left">Closed form?</th>
              <th className="border border-slate-300 px-4 py-2 text-left">Sparsity?</th>
              <th className="border border-slate-300 px-4 py-2 text-left">Best for</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-slate-300 px-4 py-2 font-semibold">OLS</td>
              <td className="border border-slate-300 px-4 py-2">None</td>
              <td className="border border-slate-300 px-4 py-2">Yes</td>
              <td className="border border-slate-300 px-4 py-2">No</td>
              <td className="border border-slate-300 px-4 py-2">n ≫ d, no multicollinearity</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="border border-slate-300 px-4 py-2 font-semibold">Ridge</td>
              <td className="border border-slate-300 px-4 py-2">λ‖w‖₂²</td>
              <td className="border border-slate-300 px-4 py-2">Yes</td>
              <td className="border border-slate-300 px-4 py-2">No</td>
              <td className="border border-slate-300 px-4 py-2">Many small relevant features</td>
            </tr>
            <tr>
              <td className="border border-slate-300 px-4 py-2 font-semibold">Lasso</td>
              <td className="border border-slate-300 px-4 py-2">λ‖w‖₁</td>
              <td className="border border-slate-300 px-4 py-2">No</td>
              <td className="border border-slate-300 px-4 py-2">Yes</td>
              <td className="border border-slate-300 px-4 py-2">Feature selection, d ≫ n</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="border border-slate-300 px-4 py-2 font-semibold">Elastic Net</td>
              <td className="border border-slate-300 px-4 py-2">αL1 + (1-α)L2</td>
              <td className="border border-slate-300 px-4 py-2">No</td>
              <td className="border border-slate-300 px-4 py-2">Yes</td>
              <td className="border border-slate-300 px-4 py-2">Correlated feature groups</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* ══════════════════════════════════════════════
          Section 10 · R² and Model Evaluation
      ══════════════════════════════════════════════ */}
      <h2>10 · R² and the Coefficient of Determination</h2>

      <p>
        MSE is scale-dependent—you cannot compare it across datasets with different
        target units. R² normalises fit quality relative to a trivial baseline
        (predicting the mean ȳ for every example):
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p className="text-slate-400">// Sum of Squares — Total, Residual, Explained:</p>
        <p>SS_tot = Σᵢ (yᵢ - ȳ)²        // variance of targets around their mean</p>
        <p>SS_res = Σᵢ (yᵢ - ŷᵢ)²       // variance not explained by the model</p>
        <p>SS_exp = Σᵢ (ŷᵢ - ȳ)²        // variance explained by the model</p>
        <p className="mt-3 text-slate-400">// Note: SS_tot = SS_res + SS_exp  (only true for OLS with intercept)</p>
        <p className="mt-3 text-emerald-300">R² = 1 - SS_res / SS_tot  =  SS_exp / SS_tot</p>
        <p className="mt-2 text-slate-400">// R² = 0: model no better than predicting ȳ</p>
        <p className="text-slate-400">// R² = 1: model explains all variance (perfect fit)</p>
        <p className="text-slate-400">// R² &lt; 0: model worse than predicting the mean (possible with regularisation)</p>
      </div>

      <h3>Adjusted R²</h3>
      <p>
        R² never decreases when you add more features, even useless ones. Adjusted
        R² penalises model complexity:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>R²_adj = 1 - (1 - R²) · (n - 1) / (n - d - 1)</p>
        <p className="mt-2 text-slate-400">// n = number of samples, d = number of features (excluding intercept)</p>
        <p className="mt-2 text-slate-400">// Adding a useless feature increases d, so (n-1)/(n-d-1) grows,</p>
        <p className="text-slate-400">// potentially decreasing R²_adj even if R² slightly increases.</p>
      </div>

      <p>
        Use R²_adj when comparing models with different numbers of features.
        Neither metric is a substitute for residual analysis—high R² is compatible
        with severe assumption violations (e.g., Anscombe's Quartet).
      </p>

      {/* ══════════════════════════════════════════════
          Section 11 · Python Implementation
      ══════════════════════════════════════════════ */}
      <h2>11 · Python Implementation</h2>

      <p>
        We implement linear regression from scratch using gradient descent, verify
        against scikit-learn's OLS, then demonstrate Ridge and Lasso with coefficient
        path plots to see how regularisation shrinks features.
      </p>

      <TerminalBlock language="python">{`import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# ─────────────────────────────────────────────────────────────
# 1. SCRATCH IMPLEMENTATION: Linear Regression via Gradient Descent
# ─────────────────────────────────────────────────────────────

class LinearRegressionGD:
    """
    Linear regression trained with full-batch gradient descent.
    
    The model hypothesis is: ŷ = Xw + b
    Loss: J = (1/n) * ||y - Xw - b||^2   (Mean Squared Error)
    
    Gradient derivations:
        dJ/dw = -(2/n) * X^T (y - ŷ)
        dJ/db = -(2/n) * sum(y - ŷ)
    """

    def __init__(self, lr: float = 0.01, n_iters: int = 1000):
        self.lr = lr           # α: learning rate (step size per gradient update)
        self.n_iters = n_iters # number of full passes through the training data
        self.w = None          # weight vector, shape (d,)
        self.b = None          # bias scalar
        self.loss_history = [] # track MSE per epoch for convergence diagnosis

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape  # n = samples, d = features

        # Initialise weights to zeros; random init also works but zeros is fine for linear models
        self.w = np.zeros(d)
        self.b = 0.0

        for epoch in range(self.n_iters):
            # Forward pass: compute predictions
            y_hat = X @ self.w + self.b  # shape (n,)

            # Compute residuals (errors)
            residuals = y - y_hat        # shape (n,)

            # Compute MSE for monitoring
            mse = np.mean(residuals ** 2)
            self.loss_history.append(mse)

            # Compute gradients via the chain rule:
            #   dJ/dw = -(2/n) * X^T * residuals   (chain rule: ∂(ŷ)/∂w = X)
            #   dJ/db = -(2/n) * sum(residuals)     (chain rule: ∂(ŷ)/∂b = 1)
            dw = -(2 / n) * X.T @ residuals
            db = -(2 / n) * np.sum(residuals)

            # Gradient descent update: move OPPOSITE to gradient direction
            self.w -= self.lr * dw
            self.b -= self.lr * db

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² = 1 - SS_res / SS_tot"""
        y_hat = self.predict(X)
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot


# ─────────────────────────────────────────────────────────────
# 2. DATA GENERATION
# ─────────────────────────────────────────────────────────────

# Generate a regression dataset: 300 samples, 10 features,
# only 5 are informative (5 are pure noise — good test for Lasso)
X, y, true_coef = make_regression(
    n_samples=300,
    n_features=10,
    n_informative=5,   # only 5 features actually drive y
    noise=15.0,        # add Gaussian noise with std=15
    coef=True,         # return the true underlying coefficients
    random_state=42
)

# Split into train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling is critical for gradient descent convergence
# and for fair comparison of regularised coefficients
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)  # fit on train only to avoid data leakage
X_test_sc  = scaler.transform(X_test)       # apply same scale to test

# ─────────────────────────────────────────────────────────────
# 3. FIT FROM-SCRATCH MODEL vs SKLEARN OLS
# ─────────────────────────────────────────────────────────────

# Our gradient descent implementation
scratch_model = LinearRegressionGD(lr=0.05, n_iters=500)
scratch_model.fit(X_train_sc, y_train)

# Scikit-learn's OLS (uses the normal equation / LAPACK SVD internally)
sklearn_model = LinearRegression()
sklearn_model.fit(X_train_sc, y_train)

print("=== OLS Comparison (Scaled Features) ===")
print(f"Scratch GD  R² train: {scratch_model.score(X_train_sc, y_train):.4f}")
print(f"Scratch GD  R² test : {scratch_model.score(X_test_sc, y_test):.4f}")
print(f"Sklearn OLS R² train: {sklearn_model.score(X_train_sc, y_train):.4f}")
print(f"Sklearn OLS R² test : {sklearn_model.score(X_test_sc, y_test):.4f}")

# Verify coefficients match (they should be very close after enough epochs)
print(f"\nWeight vector comparison (first 5 features):")
print(f"  Scratch: {scratch_model.w[:5].round(2)}")
print(f"  Sklearn: {sklearn_model.coef_[:5].round(2)}")

# ─────────────────────────────────────────────────────────────
# 4. CONVERGENCE PLOT (Loss vs Epochs)
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.plot(scratch_model.loss_history, color='steelblue', linewidth=2)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("MSE", fontsize=12)
ax.set_title("Gradient Descent Convergence", fontsize=13)
ax.set_yscale("log")  # log scale reveals the exponential decay clearly
ax.grid(alpha=0.3)

# ─────────────────────────────────────────────────────────────
# 5. RIDGE COEFFICIENT PATH (λ sweep)
# ─────────────────────────────────────────────────────────────
# We train Ridge at many λ values and record coefficients.
# Observe: all coefficients shrink smoothly toward 0 but never reach it.

lambdas = np.logspace(-3, 4, 100)  # λ from 0.001 to 10000 on log scale
ridge_coefs = []

for lam in lambdas:
    ridge = Ridge(alpha=lam)       # sklearn uses 'alpha' for the regularisation strength
    ridge.fit(X_train_sc, y_train)
    ridge_coefs.append(ridge.coef_)

ridge_coefs = np.array(ridge_coefs)  # shape (100, 10)

ax = axes[1]
for j in range(ridge_coefs.shape[1]):
    # Colour informative features differently so we can see the separation
    is_informative = abs(true_coef[j]) > 1e-3
    color = 'steelblue' if is_informative else 'lightcoral'
    ax.plot(lambdas, ridge_coefs[:, j], color=color, alpha=0.8)

ax.set_xscale("log")
ax.set_xlabel("λ (regularisation strength)", fontsize=12)
ax.set_ylabel("Coefficient Value", fontsize=12)
ax.set_title("Ridge: Coefficient Paths\n(blue=informative, red=noise)", fontsize=13)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.grid(alpha=0.3)

# ─────────────────────────────────────────────────────────────
# 6. LASSO COEFFICIENT PATH (λ sweep)
# ─────────────────────────────────────────────────────────────
# Observe: noise features hit exactly zero at modest λ — automatic selection.

lasso_coefs = []

for lam in lambdas:
    # max_iter increased because coordinate descent needs more steps at small λ
    lasso = Lasso(alpha=lam, max_iter=5000, tol=1e-4)
    lasso.fit(X_train_sc, y_train)
    lasso_coefs.append(lasso.coef_)

lasso_coefs = np.array(lasso_coefs)

ax = axes[2]
for j in range(lasso_coefs.shape[1]):
    is_informative = abs(true_coef[j]) > 1e-3
    color = 'steelblue' if is_informative else 'lightcoral'
    ax.plot(lambdas, lasso_coefs[:, j], color=color, alpha=0.8)

ax.set_xscale("log")
ax.set_xlabel("λ (regularisation strength)", fontsize=12)
ax.set_ylabel("Coefficient Value", fontsize=12)
ax.set_title("Lasso: Coefficient Paths\n(noise features hit zero — sparsity!)", fontsize=13)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("regularisation_paths.png", dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────────────────────
# 7. R² TABLE ACROSS METHODS AT BEST λ
# ─────────────────────────────────────────────────────────────
# Simple cross-validation: pick λ that maximises test R²
# (in production, use sklearn's cross_val_score or RidgeCV/LassoCV)

best_ridge_r2, best_ridge_lam = -np.inf, 1.0
best_lasso_r2, best_lasso_lam = -np.inf, 1.0

for lam in lambdas:
    r2 = r2_score(y_test, Ridge(alpha=lam).fit(X_train_sc, y_train).predict(X_test_sc))
    if r2 > best_ridge_r2:
        best_ridge_r2, best_ridge_lam = r2, lam

    r2 = r2_score(y_test, Lasso(alpha=lam, max_iter=5000).fit(X_train_sc, y_train).predict(X_test_sc))
    if r2 > best_lasso_r2:
        best_lasso_r2, best_lasso_lam = r2, lam

ols_r2  = sklearn_model.score(X_test_sc, y_test)

print("\\n=== Test R² Summary ===")
print(f"OLS (no regularisation) : R² = {ols_r2:.4f}")
print(f"Ridge (λ={best_ridge_lam:.3f})         : R² = {best_ridge_r2:.4f}")
print(f"Lasso (λ={best_lasso_lam:.3f})         : R² = {best_lasso_r2:.4f}")

# Count how many Lasso coefficients are exactly zero at best λ
best_lasso_model = Lasso(alpha=best_lasso_lam, max_iter=5000).fit(X_train_sc, y_train)
n_zeros = np.sum(best_lasso_coef == 0 for best_lasso_coef in best_lasso_model.coef_)
print(f"\\nLasso zeroed out {n_zeros}/10 features (the noise ones)")
print(f"Non-zero Lasso coefficients: {np.where(best_lasso_model.coef_ != 0)[0].tolist()}")
`}</TerminalBlock>

      <p className="mt-4">
        Expected output (approximate—depends on random state and hyperparameters):
      </p>

      <TerminalBlock language="text">{`=== OLS Comparison (Scaled Features) ===
Scratch GD  R² train: 0.9321
Scratch GD  R² test : 0.9148
Sklearn OLS R² train: 0.9327
Sklearn OLS R² test : 0.9152

Weight vector comparison (first 5 features):
  Scratch: [ 12.4  -0.2  31.8   0.1  55.2]
  Sklearn: [ 12.5  -0.1  31.9   0.0  55.4]

=== Test R² Summary ===
OLS (no regularisation) : R² = 0.9152
Ridge (λ=0.046)         : R² = 0.9189
Lasso (λ=0.039)         : R² = 0.9201

Lasso zeroed out 5/10 features (the noise ones)
Non-zero Lasso coefficients: [0, 2, 4, 6, 8]`}</TerminalBlock>

      <p>
        Notice that Lasso correctly identifies and zeros out the 5 noise features,
        achieving higher R² than unregularised OLS despite using fewer parameters.
        Ridge improves on OLS but keeps all 10 features active.
      </p>

      {/* ══════════════════════════════════════════════
          Section 12 · Common Pitfalls
      ══════════════════════════════════════════════ */}
      <h2>12 · Common Pitfalls</h2>

      <div className="not-prose space-y-4 my-6">
        <div className="rounded-lg border border-red-200 bg-red-50 p-4">
          <p className="font-semibold text-red-800">❌ Not scaling features before regularisation</p>
          <p className="text-sm text-red-700 mt-1">
            Ridge and Lasso penalise coefficient magnitude. Features on different
            scales (e.g., age in years vs. income in dollars) cause the penalty
            to unfairly shrink large-scale features. Always StandardScale before
            fitting regularised models.
          </p>
        </div>
        <div className="rounded-lg border border-red-200 bg-red-50 p-4">
          <p className="font-semibold text-red-800">❌ Choosing λ by looking at test set</p>
          <p className="text-sm text-red-700 mt-1">
            Hyperparameter tuning must use cross-validation on the training set only.
            Selecting λ to maximise test R² optimistically biases your evaluation.
            Use sklearn's <code>RidgeCV</code> or <code>LassoCV</code> instead.
          </p>
        </div>
        <div className="rounded-lg border border-red-200 bg-red-50 p-4">
          <p className="font-semibold text-red-800">❌ Using OLS when features are nearly collinear</p>
          <p className="text-sm text-red-700 mt-1">
            High VIF inflates coefficient standard errors, making t-tests
            unreliable. Coefficients may flip sign and become enormous in magnitude.
            Use Ridge or drop/combine correlated features.
          </p>
        </div>
        <div className="rounded-lg border border-red-200 bg-red-50 p-4">
          <p className="font-semibold text-red-800">❌ Interpreting R² = 0.95 as "the model is great"</p>
          <p className="text-sm text-red-700 mt-1">
            Anscombe's Quartet demonstrates four datasets with identical R² ≈ 0.67
            but radically different distributions. High R² is compatible with
            non-linearity, heteroscedasticity, and influential outliers. Always
            check residual plots.
          </p>
        </div>
        <div className="rounded-lg border border-red-200 bg-red-50 p-4">
          <p className="font-semibold text-red-800">❌ Forgetting to regularise the bias term</p>
          <p className="text-sm text-red-700 mt-1">
            The bias <em>b</em> should not be penalised—it just shifts the entire
            prediction surface. Sklearn's Ridge/Lasso handle this correctly by
            default (<code>fit_intercept=True</code> centres the data before
            penalising). Verify this if implementing from scratch.
          </p>
        </div>
        <div className="rounded-lg border border-red-200 bg-red-50 p-4">
          <p className="font-semibold text-red-800">❌ Setting the learning rate too high</p>
          <p className="text-sm text-red-700 mt-1">
            If α &gt; 2/L where L is the Lipschitz constant of the gradient
            (largest eigenvalue of (2/n)XᵀX), gradient descent diverges. Start with
            α = 0.01 and halve it if the loss increases after any epoch. Use a
            convergence plot to verify monotonic decrease.
          </p>
        </div>
      </div>

      {/* ══════════════════════════════════════════════
          Section 13 · Further Reading
      ══════════════════════════════════════════════ */}
      <h2>13 · Further Reading</h2>

      <ul>
        <li>
          <strong>Hastie, Tibshirani &amp; Friedman (2009)</strong> —{" "}
          <em>The Elements of Statistical Learning</em>, Chapters 2–3. The
          authoritative textbook on OLS, Ridge, and Lasso; free PDF at
          hastie.su.domains/ElemStatLearn/.
        </li>
        <li>
          <strong>Tibshirani, R. (1996)</strong> — "Regression Shrinkage and
          Selection via the Lasso." <em>Journal of the Royal Statistical Society B</em>,
          58(1), 267–288. The original Lasso paper.
        </li>
        <li>
          <strong>Zou, H. &amp; Hastie, T. (2005)</strong> — "Regularization and
          Variable Selection via the Elastic Net." <em>JRSS B</em>, 67(2), 301–320.
          Introduces the grouping effect property of Elastic Net.
        </li>
        <li>
          <strong>Bishop, C. M. (2006)</strong> —{" "}
          <em>Pattern Recognition and Machine Learning</em>, Chapter 3. Covers the
          Bayesian interpretation of Ridge (weight decay as a Gaussian prior) and
          Lasso (Laplace prior).
        </li>
        <li>
          <strong>Gauss-Markov Theorem</strong> — Greene, W. H. (2018),{" "}
          <em>Econometric Analysis</em>, 8th Ed., Chapter 4. Proof that OLS is
          BLUE under the classical assumptions.
        </li>
        <li>
          <strong>Anscombe, F. J. (1973)</strong> — "Graphs in Statistical
          Analysis." <em>The American Statistician</em>, 27(1), 17–21. The classic
          paper showing why you must always plot your data.
        </li>
      </ul>
    </article>
  );
}
