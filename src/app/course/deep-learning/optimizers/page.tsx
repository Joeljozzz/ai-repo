"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";
import MathBlock from "@/components/MathBlock";
import ConceptCard from "@/components/ConceptCard";
import ComparisonTable from "@/components/ComparisonTable";

const optimizersSuiteCode = `import numpy as np
from typing import Dict, List, Tuple, Optional

# ==============================================================================
# UNIFIED DEEP LEARNING OPTIMIZER SUITE
# ==============================================================================

class Optimizer:
    """Abstract base class for all neural network optimizers."""
    def __init__(self, params: List[np.ndarray], lr: float = 1e-3, weight_decay: float = 0.0):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay
        self.state: Dict[int, Dict[str, np.ndarray]] = {}

    def step(self, grads: List[np.ndarray]):
        raise NotImplementedError

    def zero_grad(self):
        pass

class SGD(Optimizer):
    """
    Stochastic Gradient Descent with Classical Polyak Momentum and Nesterov Acceleration.
    """
    def __init__(self, params: List[np.ndarray], lr: float = 0.01, momentum: float = 0.9, 
                 nesterov: bool = False, weight_decay: float = 0.0):
        super().__init__(params, lr, weight_decay)
        self.momentum = momentum
        self.nesterov = nesterov
        for i, p in enumerate(self.params):
            self.state[i] = {"velocity": np.zeros_like(p)}

    def step(self, grads: List[np.ndarray]):
        for i, (p, g) in enumerate(zip(self.params, grads)):
            d_p = g + self.weight_decay * p
            v = self.state[i]["velocity"]
            v = self.momentum * v + d_p
            self.state[i]["velocity"] = v
            
            if self.nesterov:
                update = d_p + self.momentum * v
            else:
                update = v
                
            p -= self.lr * update

class RMSProp(Optimizer):
    """
    Root Mean Square Propagation (Geoffrey Hinton, 2012).
    Tracks exponential moving average of squared gradients to normalize coordinate steps.
    """
    def __init__(self, params: List[np.ndarray], lr: float = 0.001, alpha: float = 0.99, 
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params, lr, weight_decay)
        self.alpha = alpha
        self.eps = eps
        for i, p in enumerate(self.params):
            self.state[i] = {"square_avg": np.zeros_like(p)}

    def step(self, grads: List[np.ndarray]):
        for i, (p, g) in enumerate(zip(self.params, grads)):
            d_p = g + self.weight_decay * p
            s = self.state[i]["square_avg"]
            s = self.alpha * s + (1.0 - self.alpha) * (d_p ** 2)
            self.state[i]["square_avg"] = s
            p -= self.lr * (d_p / (np.sqrt(s) + self.eps))

class AdamW(Optimizer):
    """
    Adam with Decoupled Weight Decay (Loshchilov & Hutter, ICLR 2019).
    Decouples L2 regularization from moving average momentum and second moment scales.
    """
    def __init__(self, params: List[np.ndarray], lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999), 
                 eps: float = 1e-8, weight_decay: float = 0.01):
        super().__init__(params, lr, weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_count = 0
        for i, p in enumerate(self.params):
            self.state[i] = {
                "exp_avg": np.zeros_like(p),      # m_t: 1st moment
                "exp_avg_sq": np.zeros_like(p)    # v_t: 2nd moment
            }

    def step(self, grads: List[np.ndarray]):
        self.step_count += 1
        b1, b2 = self.beta1, self.beta2
        bias_correction1 = 1.0 - (b1 ** self.step_count)
        bias_correction2 = 1.0 - (b2 ** self.step_count)

        for i, (p, g) in enumerate(zip(self.params, grads)):
            m = self.state[i]["exp_avg"]
            v = self.state[i]["exp_avg_sq"]

            # 1. Update biased 1st moment estimate
            m = b1 * m + (1.0 - b1) * g
            # 2. Update biased 2nd raw moment estimate
            v = b2 * v + (1.0 - b2) * (g ** 2)
            self.state[i]["exp_avg"] = m
            self.state[i]["exp_avg_sq"] = v

            # 3. Compute bias-corrected moments
            m_hat = m / bias_correction1
            v_hat = v / bias_correction2

            # 4. Decoupled weight decay update: w = w - lr * lambda * w - lr * (m_hat / (sqrt(v_hat) + eps))
            p -= self.lr * self.weight_decay * p
            p -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps))

# ==============================================================================
# BENCHMARK: OPTIMIZING THE ROSENBROCK ILL-CONDITIONED VALLEY
# ==============================================================================
if __name__ == "__main__":
    def rosenbrock(x, y):
        return (1.0 - x)**2 + 100.0 * (y - x**2)**2

    def rosenbrock_grad(xy):
        x, y = xy[0], xy[1]
        dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x**2)
        dy = 200.0 * (y - x**2)
        return [np.array([dx, dy])]

    init_point = np.array([-1.5, 2.0])

    # Test AdamW vs SGD with Momentum
    adam_param = [init_point.copy()]
    opt_adam = AdamW(adam_param, lr=0.05, weight_decay=0.0)

    sgd_param = [init_point.copy()]
    opt_sgd = SGD(sgd_param, lr=0.001, momentum=0.9)

    print("Benchmarking Optimizers on Ill-conditioned Rosenbrock Ravine:")
    for step in range(1, 1001):
        g_adam = rosenbrock_grad(adam_param[0])
        opt_adam.step(g_adam)

        g_sgd = rosenbrock_grad(sgd_param[0])
        opt_sgd.step(g_sgd)

        if step % 200 == 0:
            loss_adam = rosenbrock(adam_param[0][0], adam_param[0][1])
            loss_sgd = rosenbrock(sgd_param[0][0], sgd_param[0][1])
            print(f"Step {step:4d} | AdamW Loss: {loss_adam:.6f} | SGD-M Loss: {loss_sgd:.6f}")
`;

export default function OptimizersPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      {/* ── Track Badge ── */}
      <div className="not-prose mb-4">
        <span className="inline-block rounded-full bg-purple-100 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-purple-700">
          Track 2 · Deep Learning
        </span>
      </div>

      <h1>Optimization Algorithms: Non-Convex Loss Landscapes, Momentum, Adam, and Decoupled Weight Decay (AdamW)</h1>

      <p className="lead text-lg text-slate-700">
        Training deep neural networks requires traversing non-convex, high-dimensional loss landscapes plagued by ill-conditioned
        ravines, saddle points, and stochastic gradient noise. This lecture covers the full theoretical evolution of first-order
        optimizers: from Stochastic Gradient Descent (SGD) and Polyak Momentum to adaptive methods (AdaGrad, RMSProp, Adam) and
        the mathematical proof behind Decoupled Weight Decay (AdamW).
      </p>

      {/* ── Learning Objectives ── */}
      <h2>Learning Objectives</h2>
      <ul>
        <li>
          Analyze high-dimensional loss landscape geometry: condition numbers of the Hessian matrix, ravines, and why saddle points outnumber local minima (Dauphin et al., 2014).
        </li>
        <li>
          Derive Polyak Classical Momentum and Nesterov Accelerated Gradient (NAG) using physical analogies of friction and velocity.
        </li>
        <li>
          Understand why AdaGrad fails in deep neural networks due to monotonic gradient accumulation and how RMSProp repairs it.
        </li>
        <li>
          Derive from first principles the analytical bias correction formulas for Adam&apos;s first (<span className="font-mono text-sm">1 - β₁ᵗ</span>) and second (<span className="font-mono text-sm">1 - β₂ᵗ</span>) moments.
        </li>
        <li>
          Prove why naive L2 regularization fails in Adam and derive the decoupled weight decay update rule of AdamW (Loshchilov &amp; Hutter, 2017).
        </li>
        <li>
          Implement a unified, object-oriented suite of modern optimizers from scratch in Python and benchmark them on ill-conditioned valleys.
        </li>
      </ul>

      {/* ══════════════════════════════════════════════
          Section 1 · Geometry of Deep Loss Landscapes
      ══════════════════════════════════════════════ */}
      <h2>1 · The Geometry of Deep Loss Landscapes</h2>
      <p>
        In convex optimization, the gradient always points toward the global minimum. In deep neural networks, the loss surface
        <span className="font-mono text-sm">L(θ)</span> exhibits severe pathological geometry:
      </p>

      <ConceptCard variant="theorem" title="Ill-Conditioned Ravines &amp; The Hessian Condition Number">
        <p>
          Let <span className="font-mono text-xs">H = ∇² L(θ)</span> be the Hessian matrix at the current parameter position.
          The condition number <span className="font-mono text-xs">κ(H) = λ_max / λ_min</span> measures curvature disparity across orthogonal directions.
        </p>
        <p className="font-mono text-xs bg-purple-100/60 p-2 rounded">
          In deep networks, κ(H) frequently exceeds 10⁴ - 10⁶ (forming steep ravines with nearly flat floors).
        </p>
        <p>
          Vanilla gradient descent oscillates violently between the steep ravine walls while making negligible progress along
          the flat floor where the minimum resides. Momentum and adaptive scaling algorithms exist specifically to resolve this disparity.
        </p>
      </ConceptCard>

      {/* ══════════════════════════════════════════════
          Section 2 · Momentum: Polyak & Nesterov
      ══════════════════════════════════════════════ */}
      <h2>2 · Momentum: Damping Oscillations via Inertia</h2>
      <p>
        Polyak (1964) introduced the &ldquo;heavy ball&rdquo; method, modeling optimization as a physical particle rolling
        down a potential well under gravity and friction.
      </p>

      <MathBlock
        title="Polyak Classical Momentum Derivation"
        steps={[
          {
            label: "Velocity Recurrence",
            formula: "v_t = β · v_(t-1) + (1 - β) · ∇L(θ_t),   where β ∈ [0.9, 0.99]",
            annotation: "Exponential moving average of past gradients. β acts as the momentum coefficient (friction).",
          },
          {
            label: "Parameter Step",
            formula: "θ_(t+1) = θ_t - η · v_t",
            annotation: "Updates parameters along the accumulated velocity vector.",
          },
          {
            label: "Terminal Velocity",
            formula: "If ∇L is constant  ⟹  v_∞ = (1 / (1 - β)) · ∇L",
            annotation: "For β = 0.9, step size accelerates up to 10x along consistent gradient trajectories.",
          },
        ]}
        result="Oscillatory cancellation: Opposing oscillations in ravines cancel to zero (+g - g = 0), while consistent signals amplify"
        resultNote="Nesterov Accelerated Gradient (NAG) improves on this by computing the gradient at the predicted lookahead point θ_t - β v_(t-1), providing an anticipatory braking mechanism."
      />

      {/* ══════════════════════════════════════════════
          Section 3 · Adaptive Methods: RMSProp & Adam
      ══════════════════════════════════════════════ */}
      <h2>3 · Adaptive Coordinate Scaling: RMSProp &amp; Adam</h2>
      <p>
        Rather than using a single scalar learning rate <span className="font-mono text-sm">η</span> for all parameters,
        adaptive optimizers scale updates inversely proportional to historical gradient magnitudes along each coordinate.
      </p>

      <h3>3.1 The Adam Optimizer (Kingma &amp; Ba, 2014)</h3>
      <p>
        Adam (Adaptive Moment Estimation) combines momentum (first raw moment) with RMSProp (second uncentered raw moment),
        incorporating analytical bias corrections for zero-initialized moving averages.
      </p>

      <MathBlock
        title="Analytical Bias Correction Proof in Adam"
        steps={[
          {
            label: "Moment Recurrence",
            formula: "m_t = β_1 · m_(t-1) + (1 - β_1) · g_t = (1 - β_1) ∑_(i=1)^t β_1^(t-i) · g_i",
            annotation: "Unrolling the recurrence relation assuming m_0 = 0.",
          },
          {
            label: "Expected Value",
            formula: "𝔼[m_t] = 𝔼[ (1 - β_1) ∑_(i=1)^t β_1^(t-i) · g_i ] ≈ 𝔼[g_t] · (1 - β_1) ∑_(i=1)^t β_1^(t-i)",
            annotation: "Assuming the true gradient distribution is stationary around the current point.",
          },
          {
            label: "Finite Geometric Series",
            formula: "∑_(i=1)^t β_1^(t-i) = (1 - β_1^t) / (1 - β_1)",
            annotation: "Summing the finite geometric series.",
          },
          {
            label: "Resulting Expectation",
            formula: "𝔼[m_t] = (1 - β_1^t) · 𝔼[g_t]",
            annotation: "Proves that m_t is biased towards 0 by factor (1 - β_1^t) when t is small.",
          },
        ]}
        result="Bias-Corrected Estimator: m̂_t = m_t / (1 - β_1^t)  and  v̂_t = v_t / (1 - β_2^t)"
        resultNote="Without this correction, early steps (t < 50) would suffer from massive learning rate distortion due to near-zero variance estimates."
      />

      {/* ══════════════════════════════════════════════
          Section 4 · The AdamW Breakthrough
      ══════════════════════════════════════════════ */}
      <h2>4 · The AdamW Breakthrough: Decoupled Weight Decay</h2>
      <p>
        For years, practitioners observed that SGD with momentum generalized better than Adam on computer vision tasks.
        Loshchilov &amp; Hutter (ICLR 2019) proved that this was caused by a fundamental mathematical bug in how deep learning
        libraries implemented L2 regularization with adaptive optimizers.
      </p>

      <MathBlock
        title="L2 Regularization vs. Decoupled Weight Decay in Adam"
        steps={[
          {
            label: "Standard L2 Regularization",
            formula: "L_reg(θ) = L(θ) + (λ / 2) ||θ||_2^2  ⟹  g_t = ∇L(θ_t) + λ θ_t",
            annotation: "The penalty is added directly into the gradient before computing moments.",
          },
          {
            label: "Coupled Update in Adam",
            formula: "θ_(t+1) = θ_t - η · [ m̂_t(∇L + λθ) / (√(v̂_t(∇L + λθ)) + ε) ]",
            annotation: "Weights with historically large gradients receive heavily dampened decay (decay is divided by √v_t!).",
          },
          {
            label: "Decoupled Weight Decay (AdamW)",
            formula: "θ_(t+1) = θ_t - η · λ · θ_t - η · [ m̂_t / (√(v̂_t) + ε) ]",
            annotation: "Decouples the regularization step from the adaptive gradient moment scaling.",
          },
        ]}
        result="AdamW: Restores true weight decay equivalence, matching or exceeding SGD generalization on Transformers and Vision models"
        resultNote="All modern LLMs (LLaMA, GPT-4, Mistral, Gemma) are trained with AdamW rather than vanilla Adam."
      />

      <ComparisonTable
        title="Comprehensive Deep Learning Optimizer Comparison"
        subtitle="Mathematical updates, memory overhead, hyperparameter defaults, and primary domains."
        columns={[
          { key: "optimizer", header: "Optimizer", width: "15%" },
          { key: "memory", header: "Memory / Param", width: "15%" },
          { key: "adaptive", header: "Adaptive LR?", width: "15%" },
          { key: "decay", header: "Weight Decay", width: "20%" },
          { key: "canonical", header: "Canonical Domain", width: "35%" },
        ]}
        data={[
          {
            optimizer: <strong className="text-slate-900">SGD + Momentum</strong>,
            memory: "1 state (velocity)",
            adaptive: <span className="text-rose-600 font-semibold">No</span>,
            decay: "Coupled L2 is mathematically exact",
            canonical: "ResNet, ConvNeXt, fine-tuning where generalization is paramount",
          },
          {
            optimizer: <strong className="text-slate-900">RMSProp</strong>,
            memory: "1 state (squared avg)",
            adaptive: <span className="text-emerald-600 font-semibold">Yes</span>,
            decay: "Coupled L2",
            canonical: "Recurrent networks (RNN/LSTM), early deep reinforcement learning (DQN)",
          },
          {
            optimizer: <strong className="text-slate-900">Adam</strong>,
            memory: "2 states (m, v)",
            adaptive: <span className="text-emerald-600 font-semibold">Yes</span>,
            decay: "Coupled (Flawed: penalizes rare features less)",
            canonical: "Historically widespread; superseded by AdamW",
          },
          {
            optimizer: <strong className="text-slate-900">AdamW</strong>,
            memory: "2 states (m, v)",
            adaptive: <span className="text-emerald-600 font-semibold">Yes</span>,
            decay: <span className="text-emerald-700 font-semibold">Decoupled (Correct)</span>,
            canonical: "Industry standard for LLMs, Transformers (BERT, GPT, ViT), modern CNNs",
          },
        ]}
      />

      {/* ══════════════════════════════════════════════
          Section 5 · Python Optimizer Suite
      ══════════════════════════════════════════════ */}
      <h2>5 · Full Python Optimizer Suite &amp; Rosenbrock Benchmark</h2>
      <p>
        The implementation below builds an object-oriented optimizer hierarchy supporting SGD with Nesterov momentum,
        RMSProp, and AdamW with decoupled weight decay, followed by a benchmark on the non-convex Rosenbrock banana function.
      </p>

      <TerminalBlock language="python" filename="deep_optimizers_suite.py" code={optimizersSuiteCode} />

      {/* ══════════════════════════════════════════════
          Section 6 · Learning Rate Schedulers
      ══════════════════════════════════════════════ */}
      <h2>6 · Learning Rate Schedules: Warmup and Cosine Annealing</h2>
      <p>
        Keeping the learning rate <span className="font-mono text-sm">η</span> constant throughout training leads to suboptimal convergence.
      </p>

      <MathBlock
        title="Cosine Annealing with Linear Warmup"
        steps={[
          {
            label: "Linear Warmup Phase (t ≤ T_warmup)",
            formula: "η_t = η_max · (t / T_warmup)",
            annotation: "Linearly ramps from 0 to η_max to prevent large destabilizing steps before Adam's moment statistics stabilize.",
          },
          {
            label: "Cosine Annealing Phase (t > T_warmup)",
            formula: "η_t = η_min + (1/2) · (η_max - η_min) · [ 1 + cos( π · (t - T_warmup) / (T_total - T_warmup) ) ]",
            annotation: "Smoothly decays learning rate to η_min (usually η_max / 10 or 0).",
          },
        ]}
        result="Prevents early training divergence while guaranteeing asymptotic convergence to flat minima"
        resultNote="Flat minima generalize significantly better to out-of-distribution test sets than sharp minima (Keskar et al., 2016)."
      />

      {/* ── References ── */}
      <h2>7 · Academic References</h2>
      <ul className="text-sm space-y-2">
        <li>
          <strong>Kingma, D. P., &amp; Ba, J. (2014)</strong>. <em>Adam: A Method for Stochastic Optimization</em>. ICLR 2015.
        </li>
        <li>
          <strong>Loshchilov, I., &amp; Hutter, F. (2019)</strong>. <em>Decoupled Weight Decay Regularization (AdamW)</em>. ICLR 2019.
        </li>
        <li>
          <strong>Dauphin, Y. N., et al. (2014)</strong>. <em>Identifying and attacking the saddle point problem in high-dimensional non-convex optimization</em>. NeurIPS 2014.
        </li>
        <li>
          <strong>Polyak, B. T. (1964)</strong>. <em>Some methods of speeding up the convergence of iteration methods</em>. USSR Computational Mathematics and Mathematical Physics, 4(5), 1–17.
        </li>
      </ul>
    </article>
  );
}
