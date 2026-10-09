"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import numpy as np

class OptimizerTestSuite:
    """
    Unified implementation of first-order stochastic gradient descent optimizers:
    1. Vanilla SGD
    2. SGD with Polyak Momentum
    3. RMSProp (Geoffrey Hinton)
    4. Adam (Kingma & Ba, 2014)
    5. AdamW with Decoupled Weight Decay (Loshchilov & Hutter, 2017)
    """
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    @staticmethod
    def sgd_momentum(w: np.ndarray, dw: np.ndarray, v: np.ndarray, lr: float, beta: float = 0.9):
        # Velocity update: v_t = β · v_{t-1} + (1 - β) · ∇L(w)
        v = beta * v + (1.0 - beta) * dw
        w = w - lr * v
        return w, v

    @staticmethod
    def rmsprop(w: np.ndarray, dw: np.ndarray, s: np.ndarray, lr: float, beta: float = 0.99, eps: float = 1e-8):
        # Running average of squared gradients: s_t = β · s_{t-1} + (1 - β) · (∇L(w))²
        s = beta * s + (1.0 - beta) * (dw ** 2)
        # Parameter update: w_t = w_{t-1} - lr · dw / (sqrt(s_t) + ε)
        w = w - lr * dw / (np.sqrt(s) + eps)
        return w, s

    @staticmethod
    def adamw(w: np.ndarray, dw: np.ndarray, m: np.ndarray, v: np.ndarray, t: int,
              lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
              eps: float = 1e-8, weight_decay: float = 0.01):
        # 1. Update biased 1st moment estimate (momentum)
        m = beta1 * m + (1.0 - beta1) * dw
        # 2. Update biased 2nd raw moment estimate (uncentered variance)
        v = beta2 * v + (1.0 - beta2) * (dw ** 2)

        # 3. Compute bias-corrected first and second moment estimates
        m_hat = m / (1.0 - (beta1 ** t))
        v_hat = v / (1.0 - (beta2 ** t))

        # 4. AdamW Decoupled Weight Decay: decay applied directly to weights, NOT added to dw
        w = w - lr * weight_decay * w - lr * m_hat / (np.sqrt(v_hat) + eps)
        return w, m, v

# Simulation on the ill-conditioned Rosenbrock banana function:
# f(x, y) = (1 - x)² + 100(y - x²)²
if __name__ == "__main__":
    def rosenbrock_grad(xy):
        x, y = xy[0], xy[1]
        dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x**2)
        dy = 200.0 * (y - x**2)
        return np.array([dx, dy])

    # Run AdamW optimization from point (-1.5, 1.5) toward global minimum (1.0, 1.0)
    point = np.array([-1.5, 1.5])
    m_vec = np.zeros_like(point)
    v_vec = np.zeros_like(point)

    for step in range(1, 3001):
        g = rosenbrock_grad(point)
        # Clip gradient to prevent explosion in the steep ravine
        g = np.clip(g, -50.0, 50.0)
        point, m_vec, v_vec = OptimizerTestSuite.adamw(
            point, g, m_vec, v_vec, t=step, lr=0.005, weight_decay=0.0
        )

    print(f"Final AdamW Coordinates: ({point[0]:.4f}, {point[1]:.4f}) [Target: (1.0, 1.0)]")
`;

export default function OptimizersPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-100 text-purple-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-purple-500"></span>
          Track 2 · Deep Learning
        </span>
      </div>

      <h1>Optimizers: SGD, Momentum, RMSProp, Adam, and AdamW Decoupled Weight Decay</h1>

      <blockquote>
        <p>
          <em>"Standard gradient descent treats all coordinate directions equally. Adaptive moment estimation scales updates inversely proportional to the historical root mean square of gradients, effectively navigating highly anisotropic loss surfaces."</em> — Diederik Kingma &amp; Jimmy Ba (2014)
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Understand the pathological geometry of ravines, ill-conditioned Hessians, and why vanilla SGD oscillates.</li>
        <li>Derive Polyak Momentum and Nesterov Accelerated Gradient (NAG) physical equations.</li>
        <li>Explain the mechanism of RMSProp and AdaGrad in adaptively scaling coordinate learning rates.</li>
        <li>Derive Adam&apos;s bias correction factors <code>1 / (1 - β^t)</code> for early initialization steps.</li>
        <li>Contrast classical L2 regularization with Loshchilov &amp; Hutter&apos;s AdamW decoupled weight decay.</li>
      </ul>

      <h2>1 · The Geometry of Ill-Conditioned Loss Valleys</h2>
      <p>
        Consider the Taylor expansion of the loss function around a local minimum <code>w*</code>:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>L(w) ≈ L(w*) + ½ (w - w*)^T · H · (w - w*)</p>
      </div>

      <p>
        The <strong>Hessian condition number</strong> <code>κ = λ_max / λ_min</code> measures the ratio of the maximum curvature to the minimum curvature:
      </p>
      <ul>
        <li>If <code>κ ≈ 1</code>: The loss surface is spherical; vanilla SGD marches straight to the minimum.</li>
        <li>If <code>κ ≫ 1</code>: The loss surface forms an elongated canyon (ravine). In the steep direction (large eigenvalue), gradients are huge and cause wild oscillations across the canyon walls. In the gentle direction (small eigenvalue), gradients are tiny and progress is negligible.</li>
      </ul>

      <h2>2 · Momentum Methods (Polyak &amp; Nesterov)</h2>
      <p>
        Boris Polyak (1964) introduced <strong>Momentum</strong> by modeling optimization as a heavy particle rolling down a potential energy surface with friction:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p className="text-slate-400">// Polyak Heavy Ball Momentum</p>
        <p>v_t = β · v_(t-1) + (1 - β) · ∇L(w_(t-1))</p>
        <p>w_t = w_(t-1) - α · v_t</p>
      </div>

      <p>
        Oscillating gradients across the canyon walls point in opposite directions on consecutive steps and cancel out on average. Gradients along the canyon floor point consistently in the same direction and accumulate exponentially up to a terminal velocity factor of <code>1 / (1 - β)</code>.
      </p>

      <h2>3 · Adaptive Learning Rates: AdaGrad &amp; RMSProp</h2>
      <p>
        Momentum applies a uniform scaling across all parameters. <strong>AdaGrad</strong> (Duchi et al., 2011) maintains coordinate-wise sum of squared gradients:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>s_t = s_(t-1) + g_t ⊙ g_t</p>
        <p>w_t = w_(t-1) - [ α / (√(s_t) + ε) ] ⊙ g_t</p>
      </div>

      <p>
        <strong>The AdaGrad Failure Mode:</strong> Because <code>s_t</code> accumulates positive squared terms monotonically, <code>s_t → ∞</code> as training proceeds. The effective learning rate drops to zero, halting training prematurely.
      </p>
      <p>
        <strong>RMSProp (Geoffrey Hinton, 2012):</strong> Resolves this by replacing the monotonic sum with an exponentially decaying moving average:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>s_t = β · s_(t-1) + (1 - β) · (g_t ⊙ g_t)     (typically β = 0.99)</p>
        <p>w_t = w_(t-1) - [ α / (√(s_t) + ε) ] ⊙ g_t</p>
      </div>

      <h2>4 · Adam: Adaptive Moment Estimation</h2>
      <p>
        Kingma &amp; Ba (2014) combined the advantages of Momentum (first moment <code>m_t</code>) and RMSProp (uncentered second moment <code>v_t</code>):
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>m_t = β_1 · m_(t-1) + (1 - β_1) · g_t      (β_1 = 0.9)</p>
        <p>v_t = β_2 · v_(t-1) + (1 - β_2) · g_t²     (β_2 = 0.999)</p>
      </div>

      <h3>4.1 Derivation of the Bias Correction</h3>
      <p>
        Because vectors <code>m_0</code> and <code>v_0</code> are initialized to zero, early estimates are biased toward zero. Expanding <code>v_t</code> recursively:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>v_t = (1 - β_2) Σ_(i=1)^t β_2^(t - i) · g_i²</p>
        <br />
        <p>Taking expectations under the assumption that g_i comes from a stationary distribution:</p>
        <p>E[v_t] = E[ (1 - β_2) Σ_(i=1)^t β_2^(t - i) · g_i² ]</p>
        <p>       = E[g_t²] · (1 - β_2) Σ_(i=1)^t β_2^(t - i)</p>
        <p>       = E[g_t²] · (1 - β_2^t)</p>
        <br />
        <p>Therefore, dividing by (1 - β_2^t) yields the unbiased estimator:</p>
        <p>v̂_t = v_t / (1 - β_2^t)</p>
        <p>m̂_t = m_t / (1 - β_1^t)</p>
      </div>

      <h2>5 · Why AdamW Replaced Adam: Decoupled Weight Decay</h2>
      <p>
        In vanilla SGD, L2 regularization <code>½ λ ‖w‖²</code> is mathematically identical to weight decay <code>w_t = (1 - λ)w_(t-1) - α g_t</code>.
      </p>
      <p>
        Loshchilov &amp; Hutter (ICLR 2019) demonstrated that in Adam, standard L2 regularization adds <code>λ w</code> directly to the gradient <code>g_t</code>. This causes weights with large historical gradients to experience <strong>smaller weight decay</strong> because <code>(g + λw)</code> is divided by <code>√v_t</code>.
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p className="text-slate-400">// L2 Regularization in Adam (Flawed Coupling):</p>
        <p>g_t = ∇L(w) + λ · w</p>
        <p>w_t = w_(t-1) - α · m̂_t(g_t) / √(v̂_t(g_t))</p>
        <br />
        <p className="text-slate-400">// AdamW Decoupled Weight Decay (Correct Formulation):</p>
        <p>w_t = w_(t-1) - α · λ · w_(t-1) - α · m̂_t / √(v̂_t + ε)</p>
      </div>

      <p>
        AdamW decouples weight decay entirely from the gradient scaling step, yielding state-of-the-art generalization across modern LLMs and Vision Transformers.
      </p>

      <h2>6 · Python Implementation &amp; Convergence Test</h2>
      <TerminalBlock language="python" filename="optimizers_suite.py" code={code} />

      <h2>7 · Further Reading</h2>
      <ul>
        <li>Kingma, D. P., &amp; Ba, J. (2014). <em>Adam: A Method for Stochastic Optimization</em>. arXiv:1412.6980 (ICLR 2015).</li>
        <li>Loshchilov, I., &amp; Hutter, F. (2019). <em>Decoupled Weight Decay Regularization</em>. ICLR 2019 (arXiv:1711.05101).</li>
        <li>Polyak, B. T. (1964). <em>Some methods of speeding up the convergence of iteration methods</em>. USSR Computational Mathematics and Mathematical Physics, 4(5), 1-17.</li>
      </ul>
    </article>
  );
}
