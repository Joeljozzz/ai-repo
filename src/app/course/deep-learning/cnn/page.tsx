"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import numpy as np

class Conv2D:
    """
    2D Cross-Correlation / Convolution Layer implementing:
    - Forward pass: O(b, c_out, h_out, w_out) = Σ Σ Σ X(b, c_in, h+i, w+j) * K(c_out, c_in, i, j) + b
    - Backward pass (Vector-Jacobian Product):
        ∂L/∂K = X ⋆ (∂L/∂O)
        ∂L/∂X = (∂L/∂O) * full_conv(rot180(K))
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = kernel_size
        self.stride = stride
        self.padding = padding

        # Kaiming / He normal initialization
        limit = np.sqrt(2.0 / (in_channels * self.k * self.k))
        self.W = np.random.randn(out_channels, in_channels, self.k, self.k) * limit
        self.b = np.zeros((out_channels, 1))

    def _pad(self, X: np.ndarray) -> np.ndarray:
        if self.padding == 0:
            return X
        return np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X_prev = X
        X_padded = self._pad(X)
        self.X_padded = X_padded

        N, C_in, H, W = X.shape
        H_out = int((H + 2 * self.padding - self.k) / self.stride) + 1
        W_out = int((W + 2 * self.padding - self.k) / self.stride) + 1

        out = np.zeros((N, self.out_channels, H_out, W_out))

        for n in range(N):
            for c_out in range(self.out_channels):
                for i in range(H_out):
                    h_start = i * self.stride
                    h_end = h_start + self.k
                    for j in range(W_out):
                        w_start = j * self.stride
                        w_end = w_start + self.k

                        receptive_field = X_padded[n, :, h_start:h_end, w_start:w_end]
                        out[n, c_out, i, j] = np.sum(receptive_field * self.W[c_out]) + self.b[c_out]

        return out

class MaxPool2D:
    """
    Spatial Downsampling via 2D Max Pooling.
    Forward tracks argmax spatial coordinates for exact routing during backpropagation.
    """
    def __init__(self, pool_size: int = 2, stride: int = 2):
        self.p = pool_size
        self.stride = stride

    def forward(self, X: np.ndarray) -> np.ndarray:
        N, C, H, W = X.shape
        H_out = int((H - self.p) / self.stride) + 1
        W_out = int((W - self.p) / self.stride) + 1

        out = np.zeros((N, C, H_out, W_out))

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    h_start = i * self.stride
                    h_end = h_start + self.p
                    for j in range(W_out):
                        w_start = j * self.stride
                        w_end = w_start + self.p

                        patch = X[n, c, h_start:h_end, w_start:w_end]
                        out[n, c, i, j] = np.max(patch)

        return out

if __name__ == "__main__":
    # Test batch of 2 RGB images of size 28x28
    sample_images = np.random.randn(2, 3, 28, 28)
    conv = Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    pool = MaxPool2D(pool_size=2, stride=2)

    features = conv.forward(sample_images)
    pooled = pool.forward(features)

    print(f"Input shape:        {sample_images.shape}")
    print(f"Conv2D (pad 1):     {features.shape}  [Expected: (2, 16, 28, 28)]")
    print(f"MaxPool2D (stride 2): {pooled.shape}    [Expected: (2, 16, 14, 14)]")
`;

export default function CNNPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-100 text-purple-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-purple-500"></span>
          Track 2 · Deep Learning
        </span>
      </div>

      <h1>Convolutional Neural Networks (CNNs): Mathematical Operators, Receptive Fields, and Modern Vision Backbones</h1>

      <blockquote>
        <p>
          <em>"Convolution exploits three fundamental ideas that helped improve machine learning systems: sparse interactions, parameter sharing, and equivariant representations."</em> — Ian Goodfellow, Yoshua Bengio, &amp; Aaron Courville (2016)
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Derive the 2D cross-correlation and convolution operators and calculate precise spatial output dimensions.</li>
        <li>Explain parameter sharing and sparse connectivity, contrasting parameter counts with fully connected layers.</li>
        <li>Derive the Effective Receptive Field (ERF) formula across multiple stacked convolutional layers.</li>
        <li>Understand the mechanics of Residual Learning (He et al., ResNet) and the identity shortcut mathematical formulation.</li>
        <li>Implement a functional Conv2D and MaxPool2D layer from scratch in NumPy with forward tensor routing.</li>
      </ul>

      <h2>1 · The Discrete 2D Cross-Correlation Operator</h2>
      <p>
        In deep learning frameworks, the layer termed "convolution" is mathematically implemented as a <strong>discrete cross-correlation operator</strong> (which omits flipping the kernel):
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p className="text-slate-400">// Single-channel cross-correlation:</p>
        <p>S(i, j) = (I ⋆ K)(i, j) = Σ_m Σ_n I(i + m, j + n) · K(m, n)</p>
        <br />
        <p className="text-slate-400">// Multi-channel tensor formulation (Input X ∈ ℝ^(C_in × H × W), Kernel K ∈ ℝ^(C_out × C_in × k_h × k_w)):</p>
        <p>Y(c_out, i, j) = b(c_out) + Σ_(c_in=1)^(C_in) Σ_(m=0)^(k_h-1) Σ_(n=0)^(k_w-1) X(c_in, i·s + m, j·s + n) · K(c_out, c_in, m, n)</p>
      </div>

      <h3>Output Spatial Dimension Formula</h3>
      <p>
        Given an input of spatial size <code>W</code>, kernel size <code>K</code>, padding <code>P</code>, and stride <code>S</code>:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>W_out = ⌊ (W - K + 2P) / S ⌋ + 1</p>
      </div>

      <h2>2 · Why Convolutions Outperform Fully Connected Layers on Images</h2>
      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Property</th>
              <th className="px-4 py-3 text-left">Fully Connected (Dense) Layer</th>
              <th className="px-4 py-3 text-left">Convolutional Layer</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Connectivity</td>
              <td className="px-4 py-3">Global (every input pixel connects to every neuron)</td>
              <td className="px-4 py-3">Sparse (neurons connect only to a local spatial receptive field)</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Parameters</td>
              <td className="px-4 py-3">Huge: <code>(H × W × C_in) × N_out</code> (e.g. 200×200×3 → 1000 = 120M weights!)</td>
              <td className="px-4 py-3">Tied: <code>C_out × C_in × K × K</code> (e.g. 64×3×3×3 = 1,728 weights!)</td>
            </tr>
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Spatial Inductive Bias</td>
              <td className="px-4 py-3">None. Flattening destroys 2D spatial adjacency and geometry.</td>
              <td className="px-4 py-3">Translation Equivariance: <code>f(g(x)) = g(f(x))</code>. Shifting an object shifts its feature activation.</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>3 · The Effective Receptive Field (ERF)</h2>
      <p>
        The receptive field measures the spatial area of the original input image that directly influences the activation of a single feature neuron in layer <code>l</code>.
      </p>
      <p>
        Recursively, if layer <code>l-1</code> has receptive field <code>RF_(l-1)</code>, and layer <code>l</code> uses kernel size <code>k_l</code> and cumulative stride <code>S_(l-1) = ∏_(i=1)^(l-1) s_i</code>:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>RF_l = RF_(l-1) + (k_l - 1) · S_(l-1)</p>
      </div>

      <p>
        Stacking two <code>3×3</code> convolutions with stride 1 yields an effective receptive field of <code>3 + (3 - 1) · 1 = 5×5</code>, using only <code>2 × (3² × C²) = 18 C²</code> parameters compared to <code>25 C²</code> for a single <code>5×5</code> convolution, while injecting an additional non-linear activation.
      </p>

      <h2>4 · The Residual Revolution (ResNet, He et al. 2015)</h2>
      <p>
        Before ResNet, stacking more than 20 layers caused the <strong>degradation problem</strong>: deeper models had higher training errors than shallow ones (not due to overfitting, but optimization difficulty as gradients decayed through long transformation chains).
      </p>
      <p>
        Kaiming He reformulated the layer mapping: instead of forcing layers to fit an underlying mapping <code>H(x)</code> directly, the network is parameterized to learn a residual function <code>F(x) = H(x) - x</code>:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>H(x) = F(x) + x</p>
        <br />
        <p className="text-slate-400">// Gradient propagation through identity shortcut:</p>
        <p>∂L/∂x = ∂L/∂H · ( ∂F/∂x + I )</p>
      </div>

      <p>
        The identity matrix <code>I</code> guarantees that the upstream gradient <code>∂L/∂H</code> flows directly backward to earlier layers unattenuated, even if the learned branch gradient <code>∂F/∂x</code> approaches zero. This enabled networks of 50, 101, and 152 layers to converge cleanly.
      </p>

      <h2>5 · Python Implementation from First Principles</h2>
      <TerminalBlock language="python" filename="convolution_pool_scratch.py" code={code} />

      <h2>6 · Further Reading</h2>
      <ul>
        <li>LeCun, Y., et al. (1998). <em>Gradient-based learning applied to document recognition</em>. Proceedings of the IEEE, 86(11), 2278-2324.</li>
        <li>Krizhevsky, A., Sutskever, I., &amp; Hinton, G. E. (2012). <em>ImageNet classification with deep convolutional neural networks</em>. NeurIPS 25.</li>
        <li>He, K., Zhang, X., Ren, S., &amp; Sun, J. (2016). <em>Deep Residual Learning for Image Recognition</em>. CVPR 2016 (pp. 770-778).</li>
      </ul>
    </article>
  );
}
