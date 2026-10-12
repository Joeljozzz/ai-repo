"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";
import MathBlock from "@/components/MathBlock";
import ConceptCard from "@/components/ConceptCard";
import ComparisonTable from "@/components/ComparisonTable";
import AlgorithmBlock from "@/components/AlgorithmBlock";

const mlpModularCode = `import numpy as np
from typing import List, Dict, Tuple, Optional, Callable

# ==============================================================================
# 1. CORE ACTIVATION FUNCTIONS & DERIVATIVES
# ==============================================================================

class Activation:
    """Base class for all activation functions."""
    @staticmethod
    def forward(Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    @staticmethod
    def backward(Z: np.ndarray, dA: np.ndarray) -> np.ndarray:
        """Computes dZ given upstream gradient dA and pre-activation Z."""
        raise NotImplementedError

class Sigmoid(Activation):
    @staticmethod
    def forward(Z: np.ndarray) -> np.ndarray:
        # Clip Z to avoid numerical underflow/overflow in exp(-Z)
        Z_clipped = np.clip(Z, -500, 500)
        return 1.0 / (1.0 + np.exp(-Z_clipped))
    
    @staticmethod
    def backward(Z: np.ndarray, dA: np.ndarray) -> np.ndarray:
        s = Sigmoid.forward(Z)
        return dA * s * (1.0 - s)

class Tanh(Activation):
    @staticmethod
    def forward(Z: np.ndarray) -> np.ndarray:
        return np.tanh(Z)
    
    @staticmethod
    def backward(Z: np.ndarray, dA: np.ndarray) -> np.ndarray:
        t = np.tanh(Z)
        return dA * (1.0 - t ** 2)

class ReLU(Activation):
    @staticmethod
    def forward(Z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, Z)
    
    @staticmethod
    def backward(Z: np.ndarray, dA: np.ndarray) -> np.ndarray:
        return dA * (Z > 0).astype(np.float64)

class LeakyReLU(Activation):
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        
    def forward(self, Z: np.ndarray) -> np.ndarray:
        return np.where(Z > 0, Z, self.alpha * Z)
        
    def backward(self, Z: np.ndarray, dA: np.ndarray) -> np.ndarray:
        dZ = np.ones_like(Z)
        dZ[Z <= 0] = self.alpha
        return dA * dZ

class GELU(Activation):
    """Gaussian Error Linear Unit (Hendrycks & Gimpel, 2016) - used in BERT, GPT, LLaMA."""
    @staticmethod
    def forward(Z: np.ndarray) -> np.ndarray:
        # Fast tanh approximation: 0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3)))
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        inner = sqrt_2_over_pi * (Z + 0.044715 * (Z ** 3))
        return 0.5 * Z * (1.0 + np.tanh(inner))
        
    @staticmethod
    def backward(Z: np.ndarray, dA: np.ndarray) -> np.ndarray:
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        inner = sqrt_2_over_pi * (Z + 0.044715 * (Z ** 3))
        tanh_inner = np.tanh(inner)
        sech2_inner = 1.0 - tanh_inner ** 2
        d_inner = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * (Z ** 2))
        dZ = 0.5 * (1.0 + tanh_inner) + 0.5 * Z * sech2_inner * d_inner
        return dA * dZ

class Softmax:
    """Stable multi-class Softmax layer."""
    @staticmethod
    def forward(Z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        Z_scaled = Z / temperature
        # Subtract max per sample for numerical stability against float overflow
        shift_Z = Z_scaled - np.max(Z_scaled, axis=-1, keepdims=True)
        exp_Z = np.exp(shift_Z)
        return exp_Z / np.sum(exp_Z, axis=-1, keepdims=True)

# ==============================================================================
# 2. WEIGHT INITIALIZERS (HE & XAVIER)
# ==============================================================================

class Initializer:
    @staticmethod
    def xavier_uniform(fan_in: int, fan_out: int) -> np.ndarray:
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=(fan_in, fan_out))
        
    @staticmethod
    def xavier_normal(fan_in: int, fan_out: int) -> np.ndarray:
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.randn(fan_in, fan_out) * std

    @staticmethod
    def he_normal(fan_in: int, fan_out: int) -> np.ndarray:
        std = np.sqrt(2.0 / fan_in)
        return np.random.randn(fan_in, fan_out) * std

# ==============================================================================
# 3. MODULAR DEEP FEEDFORWARD NEURAL NETWORK
# ==============================================================================

class DeepNeuralNetwork:
    """
    Arbitrary-depth fully connected multilayer network supporting:
    - Layer-specific activation functions (ReLU, GELU, Sigmoid, Tanh)
    - Kaiming He / Xavier weight initialization
    - Vectorized forward evaluation with forward cache
    - Analytical multivariable backpropagation
    - Combined Softmax-CrossEntropy gradient simplification (dZ = S - Y)
    - L2 Weight Decay Regularization
    """
    def __init__(self, layer_dims: List[int], activations: List[Activation], 
                 init_method: str = "he", l2_lambda: float = 1e-4):
        self.layer_dims = layer_dims
        self.activations = activations
        self.l2_lambda = l2_lambda
        self.num_layers = len(layer_dims) - 1
        
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        
        for i in range(self.num_layers):
            din = layer_dims[i]
            dout = layer_dims[i + 1]
            if init_method == "he":
                W = Initializer.he_normal(din, dout)
            else:
                W = Initializer.xavier_normal(din, dout)
            b = np.zeros((1, dout))
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Forward propagation through L layers.
        Returns:
            A_final: Final activation output
            Z_cache: List of pre-activation matrices [Z_1, Z_2, ..., Z_L]
            A_cache: List of post-activation matrices [A_0, A_1, ..., A_L] where A_0 = X
        """
        A = X
        A_cache = [A]
        Z_cache = []
        
        for l in range(self.num_layers):
            Z = np.dot(A, self.weights[l]) + self.biases[l]
            Z_cache.append(Z)
            
            # If final layer and problem is multi-class classification
            if l == self.num_layers - 1:
                A = Softmax.forward(Z)
            else:
                A = self.activations[l].forward(Z)
            A_cache.append(A)
            
        return A, Z_cache, A_cache

    def compute_loss(self, y_pred: np.ndarray, y_true_one_hot: np.ndarray) -> float:
        """Categorical Cross-Entropy with L2 regularization."""
        m = y_true_one_hot.shape[0]
        eps = 1e-15
        clipped_pred = np.clip(y_pred, eps, 1.0 - eps)
        ce_loss = -np.sum(y_true_one_hot * np.log(clipped_pred)) / m
        
        # Add L2 regularization penalty
        l2_cost = 0.5 * self.l2_lambda * sum(np.sum(W ** 2) for W in self.weights)
        return float(ce_loss + l2_cost)

    def backward(self, y_true_one_hot: np.ndarray, 
                 Z_cache: List[np.ndarray], A_cache: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Vectorized reverse-mode backpropagation through L layers.
        """
        m = y_true_one_hot.shape[0]
        L = self.num_layers
        dW_list = [None] * L
        db_list = [None] * L
        
        # Output layer gradient: Softmax + Cross Entropy simplifies to: dZ = S - Y
        dZ = A_cache[-1] - y_true_one_hot
        
        for l in reversed(range(L)):
            A_prev = A_cache[l]
            # Gradient w.r.t weights and bias with L2 penalty
            dW = (np.dot(A_prev.T, dZ) / m) + (self.l2_lambda * self.weights[l])
            db = np.sum(dZ, axis=0, keepdims=True) / m
            
            dW_list[l] = dW
            db_list[l] = db
            
            if l > 0:
                # Upstream gradient to previous layer's activations
                dA_prev = np.dot(dZ, self.weights[l].T)
                # Chain rule through activation function
                Z_prev = Z_cache[l - 1]
                dZ = self.activations[l - 1].backward(Z_prev, dA_prev)
                
        return dW_list, db_list

    def step(self, dW_list: List[np.ndarray], db_list: List[np.ndarray], lr: float = 0.01):
        """Gradient descent parameter update."""
        for l in range(self.num_layers):
            self.weights[l] -= lr * dW_list[l]
            self.biases[l] -= lr * db_list[l]

# ==============================================================================
# 4. VERIFICATION PIPELINE
# ==============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    # Synthetic 3-class classification problem
    N = 300
    D_in = 8
    C = 3
    X_syn = np.random.randn(N, D_in)
    labels = np.random.randint(0, C, size=N)
    Y_one_hot = np.eye(C)[labels]

    # Build 4-layer architecture: 8 -> 32 (GELU) -> 16 (ReLU) -> 3 (Softmax)
    model = DeepNeuralNetwork(
        layer_dims=[D_in, 32, 16, C],
        activations=[GELU(), ReLU()],
        init_method="he",
        l2_lambda=1e-3
    )

    print("Architecture Initialized:")
    for i, w in enumerate(model.weights):
        print(f" Layer {i+1} Weights: {w.shape} | Biases: {model.biases[i].shape}")

    # Training loop demonstration
    for epoch in range(1, 101):
        y_hat, z_c, a_c = model.forward(X_syn)
        loss = model.compute_loss(y_hat, Y_one_hot)
        dW, db = model.backward(Y_one_hot, z_c, a_c)
        model.step(dW, db, lr=0.1)
        if epoch % 20 == 0:
            preds = np.argmax(y_hat, axis=1)
            acc = np.mean(preds == labels) * 100
            print(f" Epoch {epoch:3d} | Loss: {loss:.4f} | Batch Accuracy: {acc:.1f}%")
`;

export default function NeuralNetworksPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      {/* ── Track Badge ── */}
      <div className="not-prose mb-4">
        <span className="inline-block rounded-full bg-purple-100 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-purple-700">
          Track 2 · Deep Learning Foundations
        </span>
      </div>

      <h1>Neural Networks &amp; Multilayer Perceptrons: Foundations, Activations, Initializations, and Layer Mechanics</h1>

      <p className="lead text-lg text-slate-700">
        A deep neural network is a parameterized composition of non-linear affine transformations capable of learning
        hierarchical geometric representations from raw features. This lecture covers the complete mathematical foundations
        of Multilayer Perceptrons (MLPs), an exhaustive catalog of non-linear activation functions (including their derivation,
        vanishing gradient profiles, and saturation boundaries), weight initialization theory (Glorot and He proofs),
        universal approximation guarantees, and complete backward-pass vector calculus.
      </p>

      {/* ── Learning Objectives ── */}
      <h2>Learning Objectives</h2>
      <ul>
        <li>
          Mathematically formalize the forward pass of an <em>L</em>-layer feedforward neural network as recursive affine mappings and point-wise non-linearities.
        </li>
        <li>
          Derive from first principles the activation functions: Sigmoid, Softmax, Tanh, ReLU, Leaky ReLU, ELU, SELU, and GELU, including their derivatives and Jacobians.
        </li>
        <li>
          Understand the mathematical justification for coupling Softmax with Cross-Entropy loss, proving why the gradient collapses to <span className="font-mono text-sm">S - y</span> and prevents output saturation.
        </li>
        <li>
          Analyze the Universal Approximation Theorem (Cybenko 1989, Hornik 1991) and distinguish between representational power and polynomial sample learnability.
        </li>
        <li>
          Prove why naive random initialization causes signal collapse or explosion, and derive the variance preservation constraints of Xavier/Glorot and He/Kaiming initializations.
        </li>
        <li>
          Construct a modular, vectorized multi-layer network from scratch in NumPy with forward caching and analytic backpropagation.
        </li>
      </ul>

      {/* ══════════════════════════════════════════════
          Section 1 · The Mathematical Model of an MLP
      ══════════════════════════════════════════════ */}
      <h2>1 · The Formal Architecture of an L-Layer Network</h2>
      <p>
        Consider an input vector <span className="font-mono text-sm">x ∈ ℝ^(d_0)</span>. An <em>L</em>-layer neural network
        recursively transforms this input across a series of hidden representations <span className="font-mono text-sm">a^[1], a^[2], …, a^[L-1]</span>{" "}
        until generating an output <span className="font-mono text-sm">ŷ = a^[L] ∈ ℝ^(d_L)</span>.
      </p>

      <MathBlock
        title="Forward Propagation Recurrence Relations"
        steps={[
          {
            label: "Layer l Pre-activation",
            formula: "z^[l] = W^[l] · a^[l-1] + b^[l],   where W^[l] ∈ ℝ^(d_l × d_(l-1)), b^[l] ∈ ℝ^(d_l)",
            annotation: "Affine transformation projecting the previous layer's representation into d_l dimensional space.",
          },
          {
            label: "Layer l Activation",
            formula: "a^[l] = g^[l](z^[l]),   where a^[0] = x",
            annotation: "Element-wise application of non-linear scalar activation function g^[l] : ℝ → ℝ.",
          },
          {
            label: "Full Composition",
            formula: "f_θ(x) = (g^[L] ∘ W^[L] ∘ ⋯ ∘ g^[1] ∘ W^[1])(x)",
            annotation: "Parameterized by collection θ = {W^[1], b^[1], ..., W^[L], b^[L]}.",
          },
        ]}
        result="Without non-linear activations g^[l], the entire network collapses to a single affine transform: W_net = W^[L] ⋯ W^[1]"
        resultNote="Linear layers alone cannot compute non-linear decision boundaries or XOR-like parity functions regardless of network depth."
      />

      <ConceptCard variant="theorem" title="Why Non-Linearity Is Mandatory">
        <p>
          Let each layer transformation be strictly linear: <span className="font-mono text-xs">z^[1] = W^[1]x + b^[1]</span> and{" "}
          <span className="font-mono text-xs">z^[2] = W^[2]z^[1] + b^[2]</span>. Substituting layer 1 into layer 2 yields:
        </p>
        <p className="font-mono text-xs bg-purple-100/60 p-2 rounded">
          z^[2] = W^[2](W^[1]x + b^[1]) + b^[2] = (W^[2]W^[1])x + (W^[2]b^[1] + b^[2]) = W_eff x + b_eff
        </p>
        <p>
          Because matrix multiplication is closed under composition, a network of 100 linear layers has the exact representational
          capacity of a single linear layer. Non-linear activation functions provide the curvature necessary to partition
          complex, high-dimensional Riemannian manifolds.
        </p>
      </ConceptCard>

      {/* ══════════════════════════════════════════════
          Section 2 · Exhaustive Activation Functions Catalog
      ══════════════════════════════════════════════ */}
      <h2>2 · The Non-Linear Activation Functions Catalog</h2>
      <p>
        The choice of activation function directly dictates gradient flow dynamics, optimization speed, saturation risk,
        and expressivity. Below is the rigorous mathematical derivation and practical analysis of every major activation function.
      </p>

      <h3>2.1 The Logistic Sigmoid Function</h3>
      <p>
        Historically inspired by biological neuron firing thresholds, the sigmoid function squashes any real value into the
        open interval <span className="font-mono text-sm">(0, 1)</span>.
      </p>

      <MathBlock
        title="Sigmoid Definition & Derivative Derivation"
        steps={[
          {
            label: "Definition",
            formula: "σ(z) = 1 / (1 + e^(-z)) = e^z / (e^z + 1)",
            annotation: "Maps ℝ → (0, 1). Interpretable as a Bernoulli class probability.",
          },
          {
            label: "Quotient Rule",
            formula: "d/dz [σ(z)] = -1 / (1 + e^(-z))^2 · d/dz [1 + e^(-z)] = (e^(-z)) / (1 + e^(-z))^2",
            annotation: "Differentiating with respect to pre-activation input z.",
          },
          {
            label: "Algebraic Factoring",
            formula: "= (1 / (1 + e^(-z))) · (e^(-z) / (1 + e^(-z))) = σ(z) · (1 - σ(z))",
            annotation: "The derivative can be computed directly using the cached forward activation.",
          },
        ]}
        result="Maximum derivative: σ'(0) = 0.5 · (1 - 0.5) = 0.25"
        resultNote="Because σ'(z) ≤ 0.25 everywhere, backpropagating through L sigmoid layers attenuates the gradient by at least (0.25)^L, causing catastrophic vanishing gradients."
      />

      <ConceptCard variant="warning" title="The Two Fatal Flaws of Sigmoid in Hidden Layers">
        <ul className="list-disc pl-5 space-y-2">
          <li>
            <strong>Vanishing Gradient in Tails:</strong> When <span className="font-mono text-xs">|z| &gt; 4</span>,{" "}
            <span className="font-mono text-xs">σ'(z) ≈ 0</span>. Once a neuron saturates at 0 or 1, the local gradient is virtually zero,
            permanently halting learning for that neuron and all downstream layers.
          </li>
          <li>
            <strong>Non-Zero-Centered Outputs:</strong> Since <span className="font-mono text-xs">a = σ(z) &gt; 0</span> strictly,
            all incoming activations into the subsequent layer are strictly positive. In the backward pass,{" "}
            <span className="font-mono text-xs">∂L/∂W = a · ∂L/∂z</span>. This forces all weight gradients in a neuron to share the
            same sign, causing destructive <em>zigzagging</em> optimization trajectories during gradient descent.
          </li>
        </ul>
      </ConceptCard>

      <h3>2.2 Hyperbolic Tangent (Tanh)</h3>
      <p>
        The hyperbolic tangent function addresses the non-zero-centered problem of the sigmoid function by rescaling the output
        to the interval <span className="font-mono text-sm">(-1, 1)</span>.
      </p>

      <MathBlock
        title="Tanh Definition & Derivative"
        steps={[
          {
            label: "Definition",
            formula: "tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z)) = 2·σ(2z) - 1",
            annotation: "Zero-centered: tanh(0) = 0, with range (-1, 1).",
          },
          {
            label: "Derivative",
            formula: "d/dz [tanh(z)] = 1 - tanh^2(z)",
            annotation: "Peak derivative occurs at z = 0, where tanh'(0) = 1.0.",
          },
        ]}
        result="Peak gradient: tanh'(0) = 1.0 (4x higher than Sigmoid)"
        resultNote="Tanh is strictly preferred over Sigmoid for hidden layers in shallow networks, but still suffers from saturation and vanishing gradients when |z| > 3."
      />

      <h3>2.3 Rectified Linear Unit (ReLU)</h3>
      <p>
        Introduced in deep networks by Nair &amp; Hinton (2010) and popularized by AlexNet (Krizhevsky et al., 2012), ReLU completely
        revolutionized deep learning by solving the positive-side vanishing gradient problem.
      </p>

      <MathBlock
        title="ReLU Definition & Subgradient"
        steps={[
          {
            label: "Definition",
            formula: "f(z) = max(0, z) = { z if z > 0; 0 if z ≤ 0 }",
            annotation: "Piecewise linear activation function.",
          },
          {
            label: "Subgradient",
            formula: "f'(z) = { 1 if z > 0; 0 if z < 0; [0, 1] if z = 0 }",
            annotation: "In practice, implementations conventionally assign f'(0) = 0.",
          },
        ]}
        result="Non-saturating in the positive domain: f'(z) = 1 for all z > 0"
        resultNote="Allows training networks with dozens or hundreds of layers without gradient attenuation along active paths."
      />

      <ConceptCard variant="warning" title="The Dying ReLU Problem &amp; Remedies">
        <p>
          If a large gradient update causes a neuron's weights to shift such that <span className="font-mono text-xs">z = Wx + b &lt; 0</span>{" "}
          for all samples in the dataset, the neuron will output 0 permanently. Its gradient will be 0 forever:
        </p>
        <p className="font-mono text-xs bg-amber-100/60 p-2 rounded">
          ∂L/∂W = (∂L/∂z) · x = 0 · x = 0  (Neuron never updates again)
        </p>
        <p>
          In large networks, up to 10%–40% of neurons can become permanently inactive (&ldquo;dead&rdquo;). This motivated variants:
        </p>
        <ul className="list-disc pl-5 mt-2 space-y-1 text-xs">
          <li><strong>Leaky ReLU (Maas et al., 2013):</strong> <span className="font-mono">f(z) = max(αz, z)</span> with fixed small slope <span className="font-mono">α ≈ 0.01</span>.</li>
          <li><strong>Parametric ReLU (He et al., 2015):</strong> <span className="font-mono">f(z) = max(αz, z)</span> where <span className="font-mono">α</span> is a learnable parameter trained via backprop.</li>
          <li><strong>ELU (Clevert et al., 2015):</strong> <span className="font-mono">f(z) = z</span> if <span className="font-mono">z &gt; 0</span>, else <span className="font-mono">α(e^z - 1)</span>. Smooth curve pushes mean activations closer to zero.</li>
        </ul>
      </ConceptCard>

      <h3>2.4 Gaussian Error Linear Unit (GELU)</h3>
      <p>
        Used in state-of-the-art Transformers including BERT, GPT-2, GPT-3, GPT-4, and LLaMA, GELU (Hendrycks &amp; Gimpel, 2016)
        probabilistically weights inputs by their value rather than hard-thresholding by sign.
      </p>

      <MathBlock
        title="GELU Formulation & Intuition"
        steps={[
          {
            label: "Probabilistic Form",
            formula: "GELU(z) = z · P(X ≤ z),   where X ~ 𝒩(0, 1)",
            annotation: "Multiplying input z by the cumulative distribution function (CDF) of standard Gaussian Φ(z).",
          },
          {
            label: "Analytical Formula",
            formula: "GELU(z) = z · Φ(z) = z · (1/2) · [1 + erf(z / √2)]",
            annotation: "Smooth everywhere; curvature exists in the negative regime.",
          },
          {
            label: "Fast Approximation",
            formula: "GELU(z) ≈ 0.5z · (1 + tanh(√(2/π) · (z + 0.044715 · z³)))",
            annotation: "Highly accurate approximation used in hardware inference kernels to avoid computing erf().",
          },
        ]}
        result="Smooth, non-monotonic curvature around z = 0"
        resultNote="For negative z near zero, GELU drops below zero slightly (minimum around z ≈ -0.75, GELU ≈ -0.17) before asymptoting to 0, providing smooth gradient dynamics."
      />

      <h3>2.5 The Softmax Activation: Theory, Temperature, and Cross-Entropy Coupling</h3>
      <p>
        The Softmax function normalizes a vector of unconstrained real-valued logits <span className="font-mono text-sm">z ∈ ℝ^C</span> into
        a valid discrete probability distribution <span className="font-mono text-sm">s ∈ ℝ^C</span> satisfying{" "}
        <span className="font-mono text-sm">s_i &gt; 0</span> and <span className="font-mono text-sm">∑ s_i = 1</span>.
      </p>

      <MathBlock
        title="Softmax & Temperature Formulation"
        steps={[
          {
            label: "Softmax Equation",
            formula: "Softmax(z)_i = e^(z_i / T) / (∑_(j=1)^C e^(z_j / T))",
            annotation: "T > 0 is the temperature hyperparameter. Default T = 1.0.",
          },
          {
            label: "High Temperature (T → ∞)",
            formula: "lim_(T → ∞) Softmax(z)_i = 1 / C",
            annotation: "Yields a uniform probability distribution, maximizing entropy.",
          },
          {
            label: "Low Temperature (T → 0)",
            formula: "lim_(T → 0) Softmax(z)_i = { 1 if z_i = max(z); 0 otherwise }",
            annotation: "Collapses to the hard argmax indicator function.",
          },
        ]}
        result="Numerical Stability Fix: Softmax(z) = Softmax(z - max(z))"
        resultNote="Because e^z overflows double precision float at z > 709, subtracting the row maximum prevents float overflow without altering the mathematical probability distribution."
      />

      <h4>Softmax Jacobian &amp; The Elegant Cross-Entropy Cancellation</h4>
      <p>
        Because each output coordinate <span className="font-mono text-sm">s_i</span> depends on all logits <span className="font-mono text-sm">z_j</span>,
        its derivative is a full <span className="font-mono text-sm">C × C</span> Jacobian matrix:
      </p>

      <MathBlock
        title="Softmax Jacobian Derivation"
        steps={[
          {
            label: "Case 1: i = j",
            formula: "∂s_i / ∂z_i = (e^(z_i) · ∑ e^(z_k) - e^(z_i) · e^(z_i)) / (∑ e^(z_k))^2 = s_i · (1 - s_i)",
            annotation: "Diagonal entries of the Jacobian.",
          },
          {
            label: "Case 2: i ≠ j",
            formula: "∂s_i / ∂z_j = (0 - e^(z_i) · e^(z_j)) / (∑ e^(z_k))^2 = -s_i · s_j",
            annotation: "Off-diagonal entries of the Jacobian.",
          },
          {
            label: "Unified Form",
            formula: "∂s_i / ∂z_j = s_i · (δ_ij - s_j),   where δ_ij is the Kronecker delta",
            annotation: "Compact matrix representation of the Softmax Jacobian.",
          },
        ]}
        result="Coupling with Cross-Entropy Loss: L = - ∑ y_k log(s_k)"
        resultNote="Computing the gradient of Cross-Entropy w.r.t logits yields: ∂L/∂z_i = s_i - y_i. The division by s_k cancels perfectly with the Softmax derivative, preventing vanishing gradients!"
      />

      <ComparisonTable
        title="Comprehensive Activation Function Comparison Matrix"
        subtitle="Tradeoffs across range, zero-centeredness, vanishing gradient susceptibility, and canonical use cases."
        columns={[
          { key: "name", header: "Activation", width: "15%" },
          { key: "formula", header: "Formula", width: "20%" },
          { key: "range", header: "Range", width: "12%" },
          { key: "zeroCenter", header: "Zero-Centered?", width: "12%" },
          { key: "saturation", header: "Saturation / Vanishing Grads", width: "20%" },
          { key: "primaryUse", header: "Primary Application", width: "21%" },
        ]}
        data={[
          {
            name: <strong className="text-slate-900">Sigmoid</strong>,
            formula: <code className="text-xs">1 / (1 + e^-z)</code>,
            range: "(0, 1)",
            zeroCenter: <span className="text-rose-600 font-semibold">No</span>,
            saturation: "Severe on both tails (|z| > 4)",
            primaryUse: "Binary classification output, gating in LSTM/GRU",
          },
          {
            name: <strong className="text-slate-900">Tanh</strong>,
            formula: <code className="text-xs">(e^z - e^-z)/(e^z + e^-z)</code>,
            range: "(-1, 1)",
            zeroCenter: <span className="text-emerald-600 font-semibold">Yes</span>,
            saturation: "Severe on both tails (|z| > 3)",
            primaryUse: "Recurrent neural network state transitions",
          },
          {
            name: <strong className="text-slate-900">ReLU</strong>,
            formula: <code className="text-xs">max(0, z)</code>,
            range: "[0, ∞)",
            zeroCenter: <span className="text-rose-600 font-semibold">No</span>,
            saturation: "Zero saturation for z > 0; Dying ReLU for z < 0",
            primaryUse: "Default standard for CNNs and deep MLPs",
          },
          {
            name: <strong className="text-slate-900">Leaky ReLU</strong>,
            formula: <code className="text-xs">max(αz, z), α≈0.01</code>,
            range: "(-∞, ∞)",
            zeroCenter: <span className="text-amber-600 font-semibold">Partial</span>,
            saturation: "None; constant slope in both regimes",
            primaryUse: "GAN discriminators, networks prone to dying units",
          },
          {
            name: <strong className="text-slate-900">GELU</strong>,
            formula: <code className="text-xs">z · Φ(z)</code>,
            range: "[-0.17, ∞)",
            zeroCenter: <span className="text-amber-600 font-semibold">Partial</span>,
            saturation: "Smooth non-monotonic curvature",
            primaryUse: "Transformers (BERT, GPT, LLaMA, ViT)",
          },
          {
            name: <strong className="text-slate-900">Softmax</strong>,
            formula: <code className="text-xs">e^z_i / ∑ e^z_j</code>,
            range: "(0, 1), ∑=1",
            zeroCenter: <span className="text-rose-600 font-semibold">No</span>,
            saturation: "Saturates when one logit dominates; mitigated by Cross-Entropy",
            primaryUse: "Multi-class classification output, Attention weights",
          },
        ]}
      />

      {/* ══════════════════════════════════════════════
          Section 3 · Universal Approximation Theorem
      ══════════════════════════════════════════════ */}
      <h2>3 · The Universal Approximation Theorem (UAT)</h2>
      <p>
        The theoretical foundation for neural networks is rooted in the Universal Approximation Theorem.
      </p>

      <ConceptCard variant="theorem" title="Cybenko (1989) &amp; Hornik (1991) Theorem Statement">
        <p>
          Let <span className="font-mono text-xs">σ</span> be any continuous non-constant, bounded, and sigmoidal activation function.
          Then, for any continuous function <span className="font-mono text-xs">f : [0, 1]^d → ℝ</span> and any target tolerance{" "}
          <span className="font-mono text-xs">ε &gt; 0</span>, there exists a single hidden layer neural network with finite width{" "}
          <span className="font-mono text-xs">N</span>:
        </p>
        <p className="font-mono text-xs bg-purple-100/60 p-2 rounded">
          F(x) = ∑_(i=1)^N α_i · σ(w_i^T x + b_i)
        </p>
        <p>
          such that <span className="font-mono text-xs">|F(x) - f(x)| &lt; ε</span> for all <span className="font-mono text-xs">x ∈ [0, 1]^d</span>.
        </p>
      </ConceptCard>

      <ConceptCard variant="insight" title="The Critical Distinction: Expressivity vs. Learnability">
        <ul className="list-disc pl-5 space-y-2 text-sm">
          <li>
            <strong>UAT guarantees existence, not discoverability:</strong> The theorem proves that a set of weights exists that solves
            the problem. It does <em>not</em> prove that gradient descent or backpropagation can ever find those weights in polynomial time.
          </li>
          <li>
            <strong>Exponential Width in Shallow Networks:</strong> To approximate high-frequency oscillations or complex topologies with
            a single hidden layer, the required width <span className="font-mono text-xs">N</span> scales exponentially with dimension{" "}
            <span className="font-mono text-xs">O(2^d)</span>.
          </li>
          <li>
            <strong>Why Deep is Exponentially Superior:</strong> Deep networks (depth <span className="font-mono text-xs">L</span>) can
            compose simpler sub-functions hierarchically, enabling them to represent functions requiring exponential width with only
            polynomial parameters (Telgarsky 2016, Liang &amp; Srikant 2016).
          </li>
        </ul>
      </ConceptCard>

      {/* ══════════════════════════════════════════════
          Section 4 · Weight Initialization Theory
      ══════════════════════════════════════════════ */}
      <h2>4 · Weight Initialization: Preserving Signal Variance Across Layers</h2>
      <p>
        Improper weight initialization will cause training to fail before it begins. If weights are initialized too small,
        signal variance exponentially collapses towards zero. If weights are too large, activations explode to infinity or saturate.
      </p>

      <h3>4.1 The Symmetry Breaking Problem</h3>
      <p>
        If all weights are initialized to zero (<span className="font-mono text-sm">W = 0</span>), every neuron in layer <span className="font-mono text-sm">l</span> receives
        the identical pre-activation: <span className="font-mono text-sm">z_i = b_i</span>. Consequently, all neurons compute identical activations,
        receive identical backpropagation gradients, and update to identical values. The network functions as a single neuron,
        rendering width completely useless.
      </p>

      <h3>4.2 Xavier / Glorot Initialization (for Sigmoid and Tanh)</h3>
      <p>
        Glorot &amp; Bengio (2010) analyzed variance flow under linear and tanh activations.
        Let <span className="font-mono text-sm">z_i = ∑_(j=1)^(d_(in)) W_ij x_j</span>. Assuming weights and inputs are independent with zero mean:
      </p>

      <MathBlock
        title="Glorot Variance Derivation"
        steps={[
          {
            label: "Layer Variance",
            formula: "Var(z_i) = ∑_(j=1)^(d_(in)) Var(W_ij) · Var(x_j) = d_(in) · Var(W) · Var(x)",
            annotation: "Assuming i.i.d zero-mean weights and inputs.",
          },
          {
            label: "Forward Constraint",
            formula: "Var(z) = Var(x)  ⟹  d_(in) · Var(W) = 1  ⟹  Var(W) = 1 / d_(in)",
            annotation: "Preserves activation signal variance during the forward pass.",
          },
          {
            label: "Backward Constraint",
            formula: "Var(∂L/∂x) = Var(∂L/∂z)  ⟹  d_(out) · Var(W) = 1  ⟹  Var(W) = 1 / d_(out)",
            annotation: "Preserves gradient signal variance during the backward pass.",
          },
          {
            label: "Harmonic Mean Compromise",
            formula: "Var(W) = 2 / (d_(in) + d_(out))",
            annotation: "Xavier Normal: W ~ 𝒩(0, 2 / (d_(in) + d_(out))). Xavier Uniform: W ~ 𝒰(-√6 / √(d_(in)+d_(out)), +√6 / √(d_(in)+d_(out))).",
          },
        ]}
        result="Preserves signal magnitude across 20+ layers of Tanh activations"
        resultNote="Xavier initialization fails on ReLU because ReLU zeros out 50% of the distribution, causing signal variance to halve at every layer."
      />

      <h3>4.3 He / Kaiming Initialization (for ReLU Networks)</h3>
      <p>
        He et al. (2015) accounted for the rectification step: because ReLU sets negative inputs to zero, the variance of
        the activation is half the variance of the pre-activation: <span className="font-mono text-sm">Var(f(z)) = (1/2) · Var(z)</span>.
      </p>

      <MathBlock
        title="He / Kaiming Variance Derivation"
        steps={[
          {
            label: "ReLU Variance Drop",
            formula: "Var(z^[l]) = d_(in) · Var(W^[l]) · Var(a^[l-1]) = d_(in) · Var(W^[l]) · (1/2) · Var(z^[l-1])",
            annotation: "Factoring in the 50% sparsity induced by max(0, z).",
          },
          {
            label: "Equating Variances",
            formula: "Var(z^[l]) = Var(z^[l-1])  ⟹  (1/2) · d_(in) · Var(W) = 1",
            annotation: "Condition for stable forward signal propagation.",
          },
          {
            label: "He Normal Formula",
            formula: "Var(W) = 2 / d_(in)  ⟹  W ~ 𝒩(0, √(2 / d_(in)))",
            annotation: "He Uniform: W ~ 𝒰(-√(6 / d_(in)), +√(6 / d_(in))).",
          },
        ]}
        result="The universal industry default for all ReLU, Leaky ReLU, and GELU deep models"
        resultNote="Enabled training of 30+ layer deep networks without Batch Normalization."
      />

      {/* ══════════════════════════════════════════════
          Section 5 · Full Modular Architecture Code
      ══════════════════════════════════════════════ */}
      <h2>5 · Production-Grade Modular Neural Network Implementation</h2>
      <p>
        Below is an end-to-end vectorized Python implementation of an arbitrary-depth Multilayer Perceptron. It incorporates
        He and Xavier initializers, modular activation classes (ReLU, GELU, Sigmoid, Tanh), stable Softmax with temperature,
        vectorized forward caching, analytical backpropagation, and L2 weight decay.
      </p>

      <TerminalBlock language="python" filename="deep_neural_network_core.py" code={mlpModularCode} />

      {/* ══════════════════════════════════════════════
          Section 6 · Algorithmic Execution Steps
      ══════════════════════════════════════════════ */}
      <h2>6 · Complete Training Step Algorithm</h2>
      <AlgorithmBlock
        name="Deep MLP Forward-Backward Mini-Batch Step"
        input="Batch features X ∈ ℝ^(m × d_0), One-hot targets Y ∈ ℝ^(m × C), Weights {W^[l]}, Biases {b^[l]}"
        output="Updated parameters {W^[l], b^[l]}, Scalar loss J"
        timeComplexity="O(m · ∑ d_(l-1) · d_l)"
        spaceComplexity="O(m · ∑ d_l) for forward activation caches"
        steps={[
          {
            step: 1,
            title: "Forward Feedthrough & Cache Creation",
            detail: "Iterate l = 1 ... L: compute Z^[l] = A^[l-1] W^[l] + b^[l], apply g^[l](Z^[l]) to get A^[l]. Store (Z^[l], A^[l]) in execution cache.",
            complexity: "O(m · d_(l-1) · d_l)",
          },
          {
            step: 2,
            title: "Loss Computation & Softmax Output Error",
            detail: "Compute cross-entropy loss with L2 regularization. Compute output layer error directly: dZ^[L] = A^[L] - Y.",
            complexity: "O(m · C)",
          },
          {
            step: 3,
            title: "Reverse Mode Gradient Backpropagation",
            detail: "Iterate l = L ... 1: compute dW^[l] = (1/m)(A^[l-1])^T dZ^[l] + λ W^[l], db^[l] = (1/m) ∑ dZ^[l]. Propagate dA^[l-1] = dZ^[l] (W^[l])^T and dZ^[l-1] = dA^[l-1] ⊙ g'^[l-1](Z^[l-1]).",
            complexity: "O(m · d_(l-1) · d_l)",
          },
          {
            step: 4,
            title: "Parameter Update via Optimizer Step",
            detail: "Apply gradient descent update: W^[l] ← W^[l] - η dW^[l], b^[l] ← b^[l] - η db^[l].",
            complexity: "O(∑ d_(l-1) · d_l)",
          },
        ]}
      />

      {/* ══════════════════════════════════════════════
          Section 7 · Common Pitfalls & Diagnostics
      ══════════════════════════════════════════════ */}
      <h2>7 · Practical Diagnostics &amp; Engineering Pitfalls</h2>
      <div className="not-prose grid grid-cols-1 md:grid-cols-2 gap-4 my-6">
        <div className="rounded-xl border border-red-200 bg-red-50/70 p-5">
          <h4 className="font-bold text-red-900 text-sm flex items-center gap-2">
            <span>⚠️</span> Saturated Softmax Float Overflow
          </h4>
          <p className="text-xs text-red-800 mt-2 leading-relaxed">
            Directly evaluating <code>np.exp(Z) / np.sum(np.exp(Z))</code> fails with NaN when <code>Z &gt; 709</code> due to 64-bit float limit.
            Always subtract the row maximum: <code>Z_stable = Z - np.max(Z, axis=-1, keepdims=True)</code> before computing the exponential.
          </p>
        </div>

        <div className="rounded-xl border border-amber-200 bg-amber-50/70 p-5">
          <h4 className="font-bold text-amber-900 text-sm flex items-center gap-2">
            <span>⚠️</span> Using Sigmoid in Deep Hidden Layers
          </h4>
          <p className="text-xs text-amber-800 mt-2 leading-relaxed">
            Stacking 5+ layers of Sigmoids guarantees that the earliest layers receive near-zero gradients (<span className="font-mono">≤ (0.25)^5 ≈ 0.00097</span>).
            Use GELU or ReLU in all hidden layers and restrict Sigmoid exclusively to binary output layers or gating gates.
          </p>
        </div>

        <div className="rounded-xl border border-blue-200 bg-blue-50/70 p-5">
          <h4 className="font-bold text-blue-900 text-sm flex items-center gap-2">
            <span>💡</span> Zero-Initializing Biases vs Weights
          </h4>
          <p className="text-xs text-blue-800 mt-2 leading-relaxed">
            While initializing weights to zero destroys symmetry, initializing biases to zero (<code>b = 0</code>) is completely safe
            and standard practice, as asymmetric weight initialization breaks symmetry across hidden units immediately.
          </p>
        </div>

        <div className="rounded-xl border border-purple-200 bg-purple-50/70 p-5">
          <h4 className="font-bold text-purple-900 text-sm flex items-center gap-2">
            <span>💡</span> Gradient Norm Monitoring
          </h4>
          <p className="text-xs text-purple-800 mt-2 leading-relaxed">
            Track the L2 norm of gradients per layer: <code>||dW^[l]||_2</code>. If early layers have gradient norms 1,000x smaller than late layers,
            the network is vanishing. If norms explode (&gt;100), implement gradient clipping: <code>g = g · min(1, max_norm / ||g||_2)</code>.
          </p>
        </div>
      </div>

      {/* ── References ── */}
      <h2>8 · Academic References &amp; Primary Literature</h2>
      <ul className="text-sm space-y-2">
        <li>
          <strong>Cybenko, G. (1989)</strong>. <em>Approximation by superpositions of a sigmoidal function</em>. Mathematics of Control, Signals and Systems, 2(4), 303–314.
        </li>
        <li>
          <strong>Hornik, K. (1991)</strong>. <em>Approximation capabilities of multilayer feedforward networks</em>. Neural Networks, 4(2), 251–257.
        </li>
        <li>
          <strong>Glorot, X., &amp; Bengio, Y. (2010)</strong>. <em>Understanding the difficulty of training deep feedforward neural networks</em>. AISTATS 2010.
        </li>
        <li>
          <strong>He, K., Zhang, X., Ren, S., &amp; Sun, J. (2015)</strong>. <em>Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification</em>. ICCV 2015.
        </li>
        <li>
          <strong>Hendrycks, D., &amp; Gimpel, K. (2016)</strong>. <em>Gaussian Error Linear Units (GELUs)</em>. arXiv:1606.08415.
        </li>
        <li>
          <strong>Nair, V., &amp; Hinton, G. E. (2010)</strong>. <em>Rectified linear units improve restricted boltzmann machines</em>. ICML 2010.
        </li>
      </ul>
    </article>
  );
}
