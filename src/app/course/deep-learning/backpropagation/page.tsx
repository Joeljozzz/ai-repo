"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import numpy as np

class ComputationalGraphDense:
    """
    Two-layer computational graph demonstrating exact reverse-mode automatic
    differentiation (backpropagation) with intermediate Jacobian tensors.
    
    Forward equations:
      Z1 = X · W1 + b1
      A1 = ReLU(Z1)
      Z2 = A1 · W2 + b2
      Loss = 1/2 ||Z2 - Y||_F²
    """
    def __init__(self, d_in: int, d_hid: int, d_out: int):
        self.W1 = np.random.randn(d_in, d_hid) * np.sqrt(2.0 / d_in)
        self.b1 = np.zeros((1, d_hid))
        self.W2 = np.random.randn(d_hid, d_out) * np.sqrt(2.0 / d_hid)
        self.b2 = np.zeros((1, d_out))

    def forward_and_backward(self, X: np.ndarray, Y: np.ndarray):
        N = X.shape[0]

        # ── 1. FORWARD PASS ───────────────────────────────────────────
        Z1 = np.dot(X, self.W1) + self.b1           # (N, d_hid)
        A1 = np.maximum(0, Z1)                     # (N, d_hid)
        Z2 = np.dot(A1, self.W2) + self.b2          # (N, d_out)
        loss = 0.5 * np.mean(np.sum((Z2 - Y)**2, axis=1))

        # ── 2. BACKWARD PASS (REVERSE-MODE AUTODIFF) ──────────────────
        # Step A: Upstream gradient from Loss to Z2
        # ∂L/∂Z2 = (1/N) * (Z2 - Y)
        dZ2 = (1.0 / N) * (Z2 - Y)                 # (N, d_out)

        # Step B: Parameter gradients for Layer 2
        # ∂L/∂W2 = A1^T · ∂L/∂Z2
        dW2 = np.dot(A1.T, dZ2)                    # (d_hid, d_out)
        # ∂L/∂b2 = Σ rows(∂L/∂Z2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)   # (1, d_out)

        # Step C: Backpropagate to activation A1
        # ∂L/∂A1 = ∂L/∂Z2 · W2^T
        dA1 = np.dot(dZ2, self.W2.T)               # (N, d_hid)

        # Step D: Backpropagate through non-linear activation (ReLU)
        # ∂L/∂Z1 = ∂L/∂A1 ⊙ 1(Z1 > 0)
        dZ1 = dA1 * (Z1 > 0).astype(float)         # (N, d_hid)

        # Step E: Parameter gradients for Layer 1
        # ∂L/∂W1 = X^T · ∂L/∂Z1
        dW1 = np.dot(X.T, dZ1)                     # (d_in, d_hid)
        # ∂L/∂b1 = Σ rows(∂L/∂Z1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)   # (1, d_hid)

        return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def numerical_gradient_check(self, X: np.ndarray, Y: np.ndarray, epsilon: float = 1e-6):
        """
        Validates analytical backpropagation gradients against two-sided finite difference approximations:
        ∂L/∂θ ≈ ( L(θ + ε) - L(θ - ε) ) / (2ε)
        """
        _, analytical_grads = self.forward_and_backward(X, Y)

        # Check gradient for W2 as a representative tensor
        W2_num = np.zeros_like(self.W2)
        for i in range(self.W2.shape[0]):
            for j in range(self.W2.shape[1]):
                orig = self.W2[i, j]
                
                self.W2[i, j] = orig + epsilon
                Z1_p = np.maximum(0, np.dot(X, self.W1) + self.b1)
                loss_p = 0.5 * np.mean(np.sum((np.dot(Z1_p, self.W2) + self.b2 - Y)**2, axis=1))

                self.W2[i, j] = orig - epsilon
                Z1_m = np.maximum(0, np.dot(X, self.W1) + self.b1)
                loss_m = 0.5 * np.mean(np.sum((np.dot(Z1_m, self.W2) + self.b2 - Y)**2, axis=1))

                self.W2[i, j] = orig
                W2_num[i, j] = (loss_p - loss_m) / (2.0 * epsilon)

        rel_error = np.linalg.norm(analytical_grads["W2"] - W2_num) / (
            np.linalg.norm(analytical_grads["W2"]) + np.linalg.norm(W2_num) + 1e-15
        )
        print(f"Gradient Check Relative Error for W2: {rel_error:.2e}")
        assert rel_error < 1e-5, "Gradient check failed!"

if __name__ == "__main__":
    X_sample = np.random.randn(32, 10)
    Y_sample = np.random.randn(32, 2)
    model = ComputationalGraphDense(d_in=10, d_hid=16, d_out=2)
    model.numerical_gradient_check(X_sample, Y_sample)
`;

export default function BackpropagationPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-100 text-purple-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-purple-500"></span>
          Track 2 · Deep Learning
        </span>
      </div>

      <h1>Backpropagation: The Multivariate Chain Rule, Computational Graphs, and Gradient Flow</h1>

      <blockquote>
        <p>
          <em>"Backpropagation is not a biological learning model, nor is it a mysterious heuristic; it is simply reverse-mode automatic differentiation applied systematically to a directed acyclic computational graph."</em> — Yann LeCun (1988)
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Represent arbitrary deep models as Directed Acyclic Graphs (DAGs) of primitive operators.</li>
        <li>Derive the multivariate chain rule across tensor dimensions and compute exact Jacobian-vector products.</li>
        <li>Contrast forward-mode automatic differentiation with reverse-mode automatic differentiation in high parameter spaces.</li>
        <li>Diagnose the mathematical origins of vanishing and exploding gradients in deep networks.</li>
        <li>Implement two-sided finite difference gradient checking to verify analytical backpropagation calculations.</li>
      </ul>

      <h2>1 · Computational Graphs as Directed Acyclic Graphs (DAGs)</h2>
      <p>
        In modern deep learning frameworks (PyTorch, JAX, TensorFlow), neural architectures are constructed as computational graphs where nodes correspond to variables (tensors) and edges correspond to mathematical operations:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p className="text-slate-400">// Forward Pass: Evaluation from inputs to scalar objective L</p>
        <p>x ──▶ [Linear: W, b] ──▶ z ──▶ [Activation: σ] ──▶ a ──▶ [Loss: y] ──▶ L</p>
        <br />
        <p className="text-slate-400">// Backward Pass: Reverse-mode propagation of upstream adjoints</p>
        <p>∂L/∂x ◀── [∂z/∂x · ∂L/∂z] ◀── ∂L/∂z ◀── [∂a/∂z · ∂L/∂a] ◀── ∂L/∂a ◀── ∂L/∂L (= 1.0)</p>
      </div>

      <p>
        Every operation implements two methods:
      </p>
      <ol>
        <li><strong>Forward:</strong> Computes output tensor <code>y = f(x)</code> and caches variables required for gradients.</li>
        <li><strong>Backward:</strong> Receives upstream gradient <code>∂L/∂y</code> and computes downstream gradient <code>∂L/∂x = (∂L/∂y) · (∂y/∂x)</code>.</li>
      </ol>

      <h2>2 · The Multivariate Chain Rule and Tensor Derivations</h2>
      <p>
        Consider a dense linear layer mapping <code>z = W · a + b</code> with a scalar loss function <code>L</code>. We seek the exact gradient tensors <code>∂L/∂W</code> and <code>∂L/∂b</code>.
      </p>

      <h3>2.1 Upstream Adjoint Definition</h3>
      <p>
        Let <code>δ = ∂L/∂z</code> be the upstream error gradient. By the tensor chain rule:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>∂L/∂W_ij = Σ_k (∂L/∂z_k) · (∂z_k/∂W_ij)</p>
        <br />
        <p>Since z_k = Σ_m W_km · a_m + b_k, the partial derivative simplifies to:</p>
        <p>∂z_k / ∂W_ij = a_j   if k = i else 0</p>
        <br />
        <p>Therefore:</p>
        <p>∂L/∂W_ij = (∂L/∂z_i) · a_j = δ_i · a_j</p>
        <br />
        <p>In vector outer-product form:</p>
        <p>∂L/∂W = δ · a^T</p>
        <br />
        <p>For mini-batch tensor processing with N examples (A ∈ ℝ^(N×d_in), δ ∈ ℝ^(N×d_out)):</p>
        <p>∂L/∂W = (1/N) · A^T · δ</p>
        <p>∂L/∂b = (1/N) · Σ_(rows) δ</p>
      </div>

      <h2>3 · Forward-Mode vs. Reverse-Mode Automatic Differentiation</h2>
      <p>
        Consider a function mapping <code>f: ℝⁿ → ℝᵐ</code>:
      </p>

      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Autodiff Mode</th>
              <th className="px-4 py-3 text-left">Direction</th>
              <th className="px-4 py-3 text-left">Computational Complexity</th>
              <th className="px-4 py-3 text-left">Ideal Use Case</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Forward-Mode</td>
              <td className="px-4 py-3">With input flow (Jacobian-Vector Products)</td>
              <td className="px-4 py-3"><code>O(n)</code> passes to compute full gradient</td>
              <td className="px-4 py-3">Functions with few inputs and many outputs (<code>n ≪ m</code>)</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Reverse-Mode (Backprop)</td>
              <td className="px-4 py-3">Against input flow (Vector-Jacobian Products)</td>
              <td className="px-4 py-3"><code>O(m)</code> passes. For scalar loss (<code>m = 1</code>), requires <strong>1 single pass</strong>!</td>
              <td className="px-4 py-3">Machine Learning (millions of inputs/parameters, 1 scalar loss: <code>n ≫ 1</code>)</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>4 · Vanishing and Exploding Gradient Analysis</h2>
      <p>
        In an <code>L</code>-layer network, the gradient of the loss with respect to early weights <code>W^[1]</code> expands by the chain rule into a product of <code>L</code> Jacobians:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>∂L / ∂W^[1] = [ ∏_(l=2)^L ( ∂a^[l] / ∂z^[l] · W^[l] ) ] · (∂a^[1] / ∂z^[1]) · x</p>
      </div>

      <p>
        Let <code>λ_max</code> denote the largest singular value of the weight matrices:
      </p>
      <ul>
        <li><strong>Vanishing Gradients (<code>λ_max &lt; 1</code> or saturating activations):</strong> The product scales as <code>(λ_max)^L → 0</code> as <code>L → ∞</code>. Early layers stop updating, preventing deep feature representations.</li>
        <li><strong>Exploding Gradients (<code>λ_max &gt; 1</code>):</strong> The product scales as <code>(λ_max)^L → ∞</code>, resulting in weight overflows and numerical collapse (NaNs).</li>
      </ul>

      <h3>Remedies:</h3>
      <ul>
        <li><strong>Residual Skip Connections (He et al., ResNet):</strong> Formulates <code>a^[l] = a^[l-1] + F(a^[l-1])</code>, allowing gradients to propagate directly through the identity addition without multiplying through matrix chains.</li>
        <li><strong>Normalization Layers (Batch / Layer Norm):</strong> Constrains internal activations to zero mean and unit variance.</li>
        <li><strong>Gradient Clipping:</strong> Caps <code>‖g‖</code> at a maximum threshold <code>c</code> if <code>‖g‖ &gt; c</code>.</li>
      </ul>

      <h2>5 · Python Implementation &amp; Numerical Gradient Check</h2>
      <TerminalBlock language="python" filename="computational_graph_backprop.py" code={code} />

      <h2>6 · Common Pitfalls</h2>
      <ul>
        <li>
          <strong>Forgetting to clear cached gradients between mini-batches:</strong> In PyTorch, calling <code>loss.backward()</code> accumulates gradients into <code>tensor.grad</code> by default (using addition). Failing to run <code>optimizer.zero_grad()</code> causes gradients to sum across steps.
        </li>
        <li>
          <strong>In-place tensor modifications during forward execution:</strong> Mutating tensors in-place (e.g., <code>x += 1</code>) invalidates cached values required by autograd to compute reverse-mode derivatives, triggering runtime runtime errors.
        </li>
      </ul>

      <h2>7 · Further Reading</h2>
      <ul>
        <li>Rumelhart, D. E., Hinton, G. E., &amp; Williams, R. J. (1986). <em>Learning representations by back-propagating errors</em>. Nature, 323(6088), 533-536.</li>
        <li>Baydin, A. G., et al. (2018). <em>Automatic differentiation in machine learning: a survey</em>. Journal of Machine Learning Research, 18(153), 1-43.</li>
        <li>Griewank, A., &amp; Walther, A. (2008). <em>Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation</em> (2nd ed.). SIAM.</li>
      </ul>
    </article>
  );
}
