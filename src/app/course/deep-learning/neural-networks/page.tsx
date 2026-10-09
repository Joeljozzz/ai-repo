"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import numpy as np

class MultilayerPerceptron:
    """
    Fully-connected Feedforward Neural Network (Multilayer Perceptron)
    implements:
    - He (Kaiming) weight initialization: W ~ N(0, sqrt(2 / d_in))
    - ReLU activation for hidden layers: f(z) = max(0, z)
    - Softmax activation for output layer: σ(z)_i = e^(z_i) / Σ e^(z_j)
    - Categorical Cross-Entropy Loss: L = - 1/N Σ Σ y_ij log(ŷ_ij)
    - Full Analytical Backpropagation via the Multivariable Chain Rule
    """
    def __init__(self, layer_dims: list, learning_rate: float = 0.05):
        self.lr = learning_rate
        self.weights = []
        self.biases = []

        # He / Kaiming Initialization for ReLU networks
        for i in range(len(layer_dims) - 1):
            d_in = layer_dims[i]
            d_out = layer_dims[i + 1]
            limit = np.sqrt(2.0 / d_in)
            W = np.random.randn(d_in, d_out) * limit
            b = np.zeros((1, d_out))
            self.weights.append(W)
            self.biases.append(b)

    @staticmethod
    def relu(Z: np.ndarray) -> np.ndarray:
        return np.maximum(0, Z)

    @staticmethod
    def relu_derivative(Z: np.ndarray) -> np.ndarray:
        return (Z > 0).astype(float)

    @staticmethod
    def softmax(Z: np.ndarray) -> np.ndarray:
        # Subtract max(Z) along row for numerical overflow stability
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X: np.ndarray):
        """
        Computes forward activations and caches pre-activations Z and activations A
        for use in the backward pass.
        """
        A = X
        A_cache = [A]
        Z_cache = []

        for i in range(len(self.weights) - 1):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = self.relu(Z)
            Z_cache.append(Z)
            A_cache.append(A)

        # Output layer with Softmax
        Z_out = np.dot(A, self.weights[-1]) + self.biases[-1]
        A_out = self.softmax(Z_out)
        Z_cache.append(Z_out)
        A_cache.append(A_out)

        return A_out, Z_cache, A_cache

    def backward(self, X: np.ndarray, y_one_hot: np.ndarray, Z_cache: list, A_cache: list):
        N = X.shape[0]
        L = len(self.weights)
        dW_list = [None] * L
        db_list = [None] * L

        # 1. Output error: derivative of Cross-Entropy with Softmax simplifies directly to:
        # dZ_L = A_out - y_one_hot
        dZ = A_cache[-1] - y_one_hot

        for l in reversed(range(L)):
            # Gradients for layer l
            dW_list[l] = (1.0 / N) * np.dot(A_cache[l].T, dZ)
            db_list[l] = (1.0 / N) * np.sum(dZ, axis=0, keepdims=True)

            if l > 0:
                # Backpropagate error to previous layer:
                # dA_{l-1} = dZ_l · W_l^T
                dA_prev = np.dot(dZ, self.weights[l].T)
                # dZ_{l-1} = dA_{l-1} ⊙ ReLU'(Z_{l-1})
                dZ = dA_prev * self.relu_derivative(Z_cache[l - 1])

        # Parameter update (Vanilla SGD)
        for l in range(L):
            self.weights[l] -= self.lr * dW_list[l]
            self.biases[l] -= self.lr * db_list[l]

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 500, n_classes: int = 3):
        # One-hot encode targets
        N = X.shape[0]
        y_one_hot = np.zeros((N, n_classes))
        y_one_hot[np.arange(N), y] = 1.0

        for epoch in range(epochs):
            A_out, Z_cache, A_cache = self.forward(X)
            self.backward(X, y_one_hot, Z_cache, A_cache)

    def predict(self, X: np.ndarray) -> np.ndarray:
        A_out, _, _ = self.forward(X)
        return np.argmax(A_out, axis=1)

# Verification on Iris dataset
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    data = load_iris()
    X = StandardScaler().fit_transform(data.data)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    mlp = MultilayerPerceptron(layer_dims=[4, 16, 8, 3], learning_rate=0.1)
    mlp.fit(X_train, y_train, epochs=400, n_classes=3)
    preds = mlp.predict(X_test)
    print(f"Scratch MLP Accuracy: {accuracy_score(y_test, preds):.4f}")
`;

export default function NeuralNetworksPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-100 text-purple-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-purple-500"></span>
          Track 2 · Deep Learning
        </span>
      </div>

      <h1>Neural Networks (MLP): Architecture, Calculus, and the Universal Approximation Theorem</h1>

      <blockquote>
        <p>
          <em>"A feedforward neural network with a single hidden layer and non-linear activation function can approximate any continuous function on a compact subset of ℝⁿ to arbitrary accuracy."</em> — George Cybenko (1989) &amp; Kurt Hornik (1991)
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>State the Universal Approximation Theorem and explain why depth enables exponential parameter efficiency.</li>
        <li>Derive the matrix equations for the forward pass across arbitrary layer dimensions.</li>
        <li>Analyze non-linear activation functions (Sigmoid, Tanh, ReLU, Leaky ReLU, GELU) and their gradient saturation profiles.</li>
        <li>Understand weight initialization schemes: He (Kaiming) vs. Xavier (Glorot) initialization.</li>
        <li>Implement a multi-layer perceptron from scratch in NumPy with forward caching and vectorized backpropagation.</li>
      </ul>

      <h2>1 · The Universal Approximation Theorem</h2>
      <p>
        In 1989, George Cybenko established that standard multi-layer feedforward networks with continuous, sigmoidal activation functions are <strong>dense in the space of continuous functions</strong> on <code>I_n = [0, 1]^n</code>. Hornik (1991) extended this to any non-constant, bounded continuous activation function.
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p className="text-slate-400">// Mathematical Formulation</p>
        <p>Let σ(·) be a non-constant, bounded, continuous activation function.</p>
        <p>For any continuous function f ∈ C(K), compact K ⊂ ℝⁿ, and ε &gt; 0, there exist m, α_i, w_i, b_i such that:</p>
        <p>| F(x) - f(x) | &lt; ε,   ∀ x ∈ K</p>
        <p>where F(x) = Σ_(i=1)^m α_i · σ(w_i^T x + b_i)</p>
      </div>

      <p>
        <strong>Why Depth Matters:</strong> While a single hidden layer <em>can</em> approximate any function, the required width <code>m</code> scales exponentially with input dimension <code>n</code> (Curse of Dimensionality). Stacking layers hierarchically (depth) allows the network to learn compositional abstractions (edges → shapes → parts → objects) with exponentially fewer total parameters.
      </p>

      <h2>2 · The Matrix Equations of the Forward Pass</h2>
      <p>
        Let <code>L</code> denote the number of layers, with <code>a^[0] = x</code> as the input feature vector. For any layer <code>l ∈ &#123;1, ..., L&#125;</code>:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p className="text-slate-400">// Pre-activation linear mapping:</p>
        <p>z^[l] = W^[l] · a^[l-1] + b^[l]</p>
        <br />
        <p className="text-slate-400">// Post-activation non-linear transformation:</p>
        <p>a^[l] = g^[l]( z^[l] )</p>
        <br />
        <p className="text-slate-400">// Dimensionality in batch processing (batch size N):</p>
        <p>A^[l-1] ∈ ℝ^(N × n_[l-1])</p>
        <p>W^[l]   ∈ ℝ^(n_[l-1] × n_[l])</p>
        <p>b^[l]   ∈ ℝ^(1 × n_[l])</p>
        <p>Z^[l]   ∈ ℝ^(N × n_[l])</p>
        <p>A^[l]   ∈ ℝ^(N × n_[l])</p>
      </div>

      <h2>3 · Activation Functions and the Vanishing Gradient Dilemma</h2>
      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Activation</th>
              <th className="px-4 py-3 text-left">Formula g(z)</th>
              <th className="px-4 py-3 text-left">Derivative g&apos;(z)</th>
              <th className="px-4 py-3 text-left">Failure Modes &amp; Characteristics</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Sigmoid</td>
              <td className="px-4 py-3 font-mono">1 / (1 + e^-z)</td>
              <td className="px-4 py-3 font-mono">g(z)(1 - g(z))</td>
              <td className="px-4 py-3">Max derivative is 0.25. Causes vanishing gradients in deep networks; not zero-centered.</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Tanh</td>
              <td className="px-4 py-3 font-mono">(e^z - e^-z) / (e^z + e^-z)</td>
              <td className="px-4 py-3 font-mono">1 - g(z)²</td>
              <td className="px-4 py-3">Zero-centered, but saturates for |z| &gt; 3, vanishing gradients persist.</td>
            </tr>
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">ReLU</td>
              <td className="px-4 py-3 font-mono">max(0, z)</td>
              <td className="px-4 py-3 font-mono">1 if z &gt; 0 else 0</td>
              <td className="px-4 py-3">Constant gradient of 1 for z &gt; 0. Eliminates saturation; susceptible to "Dying ReLU".</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">GELU</td>
              <td className="px-4 py-3 font-mono">z · Φ(z)</td>
              <td className="px-4 py-3 font-mono">Φ(z) + z · ϕ(z)</td>
              <td className="px-4 py-3">Smooth, probabilistic gating. Default in modern Transformers (BERT, GPT-3/4, LLaMA).</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>4 · Weight Initialization Theory</h2>
      <p>
        If initial weights are too small, activations shrink exponentially across layers until gradients disappear. If too large, activations explode, leading to numerical overflow (NaNs) and saturated gradients.
      </p>

      <h3>4.1 Xavier / Glorot Initialization (for Sigmoid / Tanh)</h3>
      <p>
        Glorot &amp; Bengio (2010) set the variance of activations to be equal to the variance of inputs across layers:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>Var(W) = 2 / (n_in + n_out)</p>
        <p>W ~ N( 0,  √(2 / (n_in + n_out)) )</p>
      </div>

      <h3>4.2 He / Kaiming Initialization (for ReLU)</h3>
      <p>
        He et al. (2015) accounted for the fact that ReLU zeroes out approximately half the neurons at each step (variance halved). To compensate, the variance factor is doubled:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>Var(W) = 2 / n_in</p>
        <p>W ~ N( 0,  √(2 / n_in) )</p>
      </div>

      <h2>5 · Python Implementation from First Principles</h2>
      <TerminalBlock language="python" filename="mlp_from_scratch.py" code={code} />

      <h2>6 · Common Architectural Pitfalls</h2>
      <ul>
        <li>
          <strong>Softmax with Raw Cross-Entropy without Log-Sum-Exp trick:</strong> Calculating <code>e^z_i</code> on outputs with values exceeding 80 overflows float64. Always compute cross-entropy using the stable <code>LogSoftmax</code> formulation.
        </li>
        <li>
          <strong>Initializing all weights to zero:</strong> If all weights are identical, every neuron computes the exact same activation and gradient, causing complete symmetry collapse where neurons cannot differentiate.
        </li>
      </ul>

      <h2>7 · Further Reading</h2>
      <ul>
        <li>Cybenko, G. (1989). <em>Approximation by superpositions of a sigmoidal function</em>. Mathematics of Control, Signals, and Systems, 2(4), 303-314.</li>
        <li>Glorot, X., &amp; Bengio, Y. (2010). <em>Understanding the difficulty of training deep feedforward neural networks</em>. AISTATS, 249-256.</li>
        <li>He, K., et al. (2015). <em>Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification</em>. ICCV, 1026-1034.</li>
      </ul>
    </article>
  );
}
