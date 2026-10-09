"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import numpy as np

# ── Derivatives via Finite Differences ──────────────────
def f(x):
    return x**3 - 2*x**2 + x

def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)   # Central difference

x0 = 2.0
analytical  = 3*x0**2 - 4*x0 + 1              # df/dx = 3x² - 4x + 1
numerical   = numerical_derivative(f, x0)
print(f"Analytical derivative at x={x0}: {analytical}")
print(f"Numerical  derivative at x={x0}: {numerical:.6f}")

# ── Gradient of a Multi-Variable Function ────────────────
# f(x, y) = x² + 2xy + y³
# ∂f/∂x = 2x + 2y
# ∂f/∂y = 2x + 3y²
def grad_f(x, y):
    return np.array([2*x + 2*y, 2*x + 3*y**2])

print(f"\\nGradient at (1, 2): {grad_f(1, 2)}")   # [6, 14]

# ── Gradient Descent (Manual) ────────────────────────────
# Minimise f(x, y) = (x-3)² + (y-5)²
# True minimum at (3, 5) — gradient is exactly zero there
def loss(x, y):
    return (x - 3)**2 + (y - 5)**2

def grad_loss(x, y):
    return np.array([2*(x - 3), 2*(y - 5)])

x, y  = 0.0, 0.0     # start far from minimum
lr    = 0.1
steps = 50

for i in range(steps):
    g    = grad_loss(x, y)
    x   -= lr * g[0]
    y   -= lr * g[1]

print(f"\\nGradient Descent result after {steps} steps:")
print(f"  x={x:.6f}, y={y:.6f}")   # Should be ≈ 3.0, 5.0
print(f"  Loss={loss(x, y):.8f}")  # Should be ≈ 0`;

export default function CalculusPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">

      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-sky-100 text-sky-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-sky-500"></span>
          Track 0 · Mathematical Foundations
        </span>
      </div>

      <h1>Calculus & Optimization</h1>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Define the derivative geometrically and analytically.</li>
        <li>Compute partial derivatives and assemble the gradient vector ∇f.</li>
        <li>Apply the multivariate chain rule — the mathematical backbone of backpropagation.</li>
        <li>Understand convex vs non-convex loss landscapes and why it matters.</li>
        <li>Implement gradient descent from scratch and understand its convergence properties.</li>
      </ul>

      <h2>1 · The Derivative</h2>
      <p>
        The derivative of f at point x is the slope of the tangent line — the instantaneous rate
        of change:
      </p>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4">
        <p>f&apos;(x) = df/dx = lim_(h→0) [ f(x+h) − f(x) ] / h</p>
      </div>
      <p>
        In ML, we almost never compute derivatives of scalar functions of one variable. We compute
        derivatives of a scalar-valued <em>loss function</em> L with respect to millions of
        parameters θ (weights). This leads to the gradient.
      </p>

      <h2>2 · Partial Derivatives & The Gradient</h2>
      <p>
        For a function of multiple variables f(x₁, x₂, …, xₙ), the partial derivative ∂f/∂xᵢ
        measures how f changes as we vary xᵢ while holding all other variables fixed.
      </p>
      <p>
        The <strong>gradient</strong> ∇f assembles all partial derivatives into a vector:
      </p>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 space-y-1">
        <p>∇f(x) = [∂f/∂x₁, ∂f/∂x₂, …, ∂f/∂xₙ]ᵀ</p>
      </div>
      <p>
        The gradient points in the direction of steepest <em>ascent</em> of f. To minimise a loss
        function we move in the <em>opposite</em> direction — the negative gradient.
      </p>

      <h2>3 · The Chain Rule</h2>
      <p>
        When functions are composed — f(g(x)) — the chain rule computes the derivative of the
        composition:
      </p>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 space-y-1">
        <p>d/dx [ f(g(x)) ] = f&apos;(g(x)) · g&apos;(x)</p>
        <p></p>
        <p>Multivariate (Jacobian) form:</p>
        <p>∂L/∂x = (∂L/∂y) · (∂y/∂x)     where y = g(x)</p>
      </div>
      <p>
        <strong>This is backpropagation.</strong> A neural network is a deeply composed function.
        During backprop, we apply the chain rule repeatedly from the loss back through every layer,
        computing ∂L/∂W for each weight matrix W. The gradient then tells us exactly how to update
        each weight to reduce the loss.
      </p>

      <h2>4 · The Jacobian & Hessian</h2>
      <p>
        When both input and output are vectors, the derivative is a matrix called the
        <strong> Jacobian</strong>:
      </p>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 space-y-1">
        <p>J_ij = ∂fᵢ / ∂xⱼ     (m×n matrix for f: ℝⁿ → ℝᵐ)</p>
      </div>
      <p>
        The <strong>Hessian</strong> H is the matrix of second-order partial derivatives of a
        scalar function:
      </p>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 space-y-1">
        <p>H_ij = ∂²f / ∂xᵢ ∂xⱼ</p>
      </div>
      <p>
        If H is positive definite everywhere, the function is strictly convex and gradient descent
        will always find the global minimum. XGBoost uses second-order Taylor expansion with the
        Hessian to compute optimal leaf weights analytically.
      </p>

      <h2>5 · Convexity & Loss Landscapes</h2>
      <p>
        A function f is <strong>convex</strong> if the line segment between any two points on its
        graph lies above or on the graph:
      </p>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4">
        <p>f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)    for all λ ∈ [0,1]</p>
      </div>
      <p>Equivalently, H is positive semi-definite everywhere.</p>
      <ul>
        <li><strong>Linear regression MSE loss</strong> is convex — gradient descent finds the global minimum.</li>
        <li><strong>Logistic regression log-loss</strong> is convex — same guarantee.</li>
        <li><strong>Neural network loss surfaces</strong> are non-convex — many local minima and saddle points exist. This is why we need careful initialisation, momentum, and adaptive learning rates.</li>
      </ul>

      <h2>6 · Gradient Descent</h2>
      <p>
        Gradient descent iteratively moves parameters θ in the direction of the negative gradient
        by a step size α (the learning rate):
      </p>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 space-y-1">
        <p>θ_(t+1) = θ_t − α ∇_θ L(θ_t)</p>
      </div>
      <p>
        <strong>Learning rate α</strong> is the most critical hyperparameter:
      </p>
      <ul>
        <li>Too large: overshoots the minimum, loss oscillates or diverges.</li>
        <li>Too small: convergence is extremely slow; you may run out of compute budget.</li>
        <li>Just right: smooth, efficient descent toward a minimum.</li>
      </ul>

      <h3>Variants</h3>
      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Variant</th>
              <th className="px-4 py-3 text-left">Batch Size</th>
              <th className="px-4 py-3 text-left">Properties</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white"><td className="px-4 py-3">Batch GD</td><td className="px-4 py-3">Full dataset</td><td className="px-4 py-3">Exact gradient; slow per step; impractical for large N</td></tr>
            <tr className="bg-slate-50"><td className="px-4 py-3">Stochastic GD (SGD)</td><td className="px-4 py-3">1 sample</td><td className="px-4 py-3">Very noisy; fast per step; regularising effect</td></tr>
            <tr className="bg-white"><td className="px-4 py-3">Mini-batch GD</td><td className="px-4 py-3">32–512 samples</td><td className="px-4 py-3">Best tradeoff; GPU parallelism; industry standard</td></tr>
          </tbody>
        </table>
      </div>

      <h2>7 · Python Implementation</h2>
      <TerminalBlock language="python" filename="calculus_gradient_descent.py" code={code} />

      <h2>8 · Common Pitfalls</h2>
      <ul>
        <li><strong>Not normalising features before gradient descent.</strong> If features have very different scales (e.g., age in [0,100] vs income in [0,10⁶]), the loss surface is an elongated ellipse and gradient descent zigzags slowly. Always standardise (subtract mean, divide by σ) first.</li>
        <li><strong>Using too large a learning rate without a schedule.</strong> A fixed large α can work early in training when gradients are large, but overshoot near the minimum. Use learning rate decay or cosine annealing.</li>
        <li><strong>Saddle points in deep networks.</strong> In high dimensions, most critical points (∇L = 0) are saddle points, not local minima. Modern optimisers (Adam, RMSProp) escape saddle points faster than vanilla SGD.</li>
      </ul>

      <h2>9 · Further Reading</h2>
      <ul>
        <li>Goodfellow et al. (2016). <em>Deep Learning</em>, Chapter 4: Numerical Computation.</li>
        <li>Boyd, S. & Vandenberghe, L. (2004). <em>Convex Optimization</em>. Cambridge University Press. (Free PDF online)</li>
        <li>Ruder, S. (2016). "An overview of gradient descent optimisation algorithms." arXiv:1609.04747.</li>
      </ul>

    </article>
  );
}
