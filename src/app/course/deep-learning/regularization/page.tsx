"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import numpy as np

class RegularizationDemo:
    """
    Demonstrating primary deep learning regularization mechanisms:
    1. Inverted Dropout (Srivastava et al., 2014)
    2. Batch Normalization (Ioffe & Szegedy, 2015)
    3. Layer Normalization (Ba, Kiros, & Hinton, 2016)
    4. Weight Decay (L2 penalty)
    """
    @staticmethod
    def inverted_dropout(X: np.ndarray, drop_prob: float = 0.5, training: bool = True):
        """
        Inverted Dropout divides active activations by keep_prob = (1 - p) during training.
        This preserves the expected value E[a] across both train and test phases without
        requiring any arithmetic scaling at inference time.
        """
        if not training or drop_prob == 0.0:
            return X, None

        keep_prob = 1.0 - drop_prob
        # Sample Bernoulli mask
        mask = (np.random.rand(*X.shape) < keep_prob) / keep_prob
        return X * mask, mask

    @staticmethod
    def batch_norm_forward(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5):
        """
        Batch Normalization normalizes across the BATCH dimension (axis=0) for each feature.
        μ_B = (1/m) Σ x_i
        σ_B² = (1/m) Σ (x_i - μ_B)²
        x̂_i = (x_i - μ_B) / sqrt(σ_B² + ε)
        y_i = γ · x̂_i + β
        """
        N, D = X.shape
        mean = np.mean(X, axis=0, keepdims=True)
        var = np.var(X, axis=0, keepdims=True)

        X_hat = (X - mean) / np.sqrt(var + eps)
        out = gamma * X_hat + beta
        return out, (X, X_hat, mean, var, gamma, eps)

    @staticmethod
    def layer_norm_forward(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5):
        """
        Layer Normalization normalizes across the FEATURE dimension (axis=1) for each sample.
        Independent of batch size, making it ideal for RNNs, Transformers, and batch size = 1.
        """
        mean = np.mean(X, axis=1, keepdims=True)
        var = np.var(X, axis=1, keepdims=True)

        X_hat = (X - mean) / np.sqrt(var + eps)
        out = gamma * X_hat + beta
        return out

if __name__ == "__main__":
    X_batch = np.array([
        [1.0, 50.0, 100.0],
        [2.0, 55.0, 120.0],
        [1.5, 48.0, 110.0],
        [3.0, 60.0, 130.0]
    ])

    # 1. Test Inverted Dropout
    dropped, mask = RegularizationDemo.inverted_dropout(X_batch, drop_prob=0.5, training=True)
    print("Inverted Dropout Output (scaled by 1/keep_prob):\\n", dropped)

    # 2. Test Batch Norm
    gamma_bn = np.ones((1, 3))
    beta_bn = np.zeros((1, 3))
    bn_out, _ = RegularizationDemo.batch_norm_forward(X_batch, gamma_bn, beta_bn)
    print("\\nBatchNorm Output (zero mean across rows):\\n", np.round(bn_out, 3))

    # 3. Test Layer Norm
    gamma_ln = np.ones((1, 3))
    beta_ln = np.zeros((1, 3))
    ln_out = RegularizationDemo.layer_norm_forward(X_batch, gamma_ln, beta_ln)
    print("\\nLayerNorm Output (zero mean across columns):\\n", np.round(ln_out, 3))
`;

export default function RegularizationPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-100 text-purple-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-purple-500"></span>
          Track 2 · Deep Learning
        </span>
      </div>

      <h1>Regularization: Weight Decay, Inverted Dropout, Batch Normalization, and Layer Normalization</h1>

      <blockquote>
        <p>
          <em>"Regularization is any modification we make to a learning algorithm that is intended to reduce its generalization error but not its training error."</em> — Ian Goodfellow, Yoshua Bengio, &amp; Aaron Courville (2016)
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Derive Weight Decay (L2 penalty) from the Gaussian prior MAP Bayesian perspective.</li>
        <li>Explain Inverted Dropout and prove how scaling by <code>1 / (1 - p)</code> during training preserves expected values at test time.</li>
        <li>Derive Batch Normalization equations and analyze how it mitigates internal covariate shift.</li>
        <li>Contrast Batch Normalization with Layer Normalization, Instance Normalization, and Group Normalization.</li>
        <li>Implement Inverted Dropout, BatchNorm, and LayerNorm from scratch in NumPy.</li>
      </ul>

      <h2>1 · L2 Regularization &amp; Weight Decay: The Bayesian Connection</h2>
      <p>
        Adding an L2 penalty to the objective function:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>J̃(w) = J(w) + (λ / 2) ‖w‖₂²</p>
        <br />
        <p>The gradient becomes:</p>
        <p>∇_w J̃(w) = ∇_w J(w) + λ w</p>
        <br />
        <p>In standard SGD with step size α:</p>
        <p>w_(t) = w_(t-1) - α ( ∇ J(w) + λ w ) = (1 - α λ) w_(t-1) - α ∇ J(w)</p>
      </div>

      <p>
        At every update step, the weights are shrunk by a constant factor <code>(1 - α λ)</code> before taking the gradient step.
      </p>
      <p>
        <strong>The Bayesian Interpretation:</strong> Minimizing an L2-regularized loss is mathematically equivalent to computing the Maximum A Posteriori (MAP) estimate of <code>w</code> under a zero-mean <strong>Gaussian prior</strong> <code>w ~ N(0, σ_w² I)</code> where <code>λ = σ² / σ_w²</code>.
      </p>

      <h2>2 · Inverted Dropout (Srivastava et al., 2014)</h2>
      <p>
        Dropout randomly zeroes out a fraction <code>p</code> of neuron activations during each forward pass. This prevents <strong>co-adaptation</strong> of features (neurons relying on specific other neurons to compensate for errors), approximating an ensemble of <code>2^N</code> thinned subnetworks.
      </p>

      <h3>Why Inverted Dropout is Standard</h3>
      <p>
        In standard dropout, activations are left untouched during training and multiplied by <code>(1 - p)</code> at test time.
      </p>
      <p>
        <strong>Inverted Dropout</strong> inverts this scaling by dividing by <code>keep_prob = (1 - p)</code> during training:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>r_j ~ Bernoulli( 1 - p )</p>
        <p>ã_j = ( r_j · a_j ) / ( 1 - p )</p>
      </div>

      <p>
        Since <code>E[r_j] = 1 - p</code>, the expected value is preserved:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>E[ ã_j ] = E[ ( r_j · a_j ) / (1 - p) ] = [ (1 - p) / (1 - p) ] · a_j = a_j</p>
      </div>

      <p>
        This completely eliminates the need for any modification during inference: the test-time model simply executes the unmasked network with zero overhead.
      </p>

      <h2>3 · Batch Normalization (Ioffe &amp; Szegedy, 2015)</h2>
      <p>
        As parameters in early layers update during training, the distribution of activations feeding into subsequent layers shifts continuously (termed <em>Internal Covariate Shift</em>).
      </p>
      <p>
        <strong>Batch Normalization (BatchNorm)</strong> forces activations across the mini-batch to have zero mean and unit variance, followed by a learned affine transformation to preserve representational capacity:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>1. Mini-batch Mean:      μ_B = (1 / m) Σ_(i=1)^m x_i</p>
        <p>2. Mini-batch Variance:  σ_B² = (1 / m) Σ_(i=1)^m (x_i - μ_B)²</p>
        <p>3. Normalize:            x̂_i = (x_i - μ_B) / √(σ_B² + ε)</p>
        <p>4. Scale &amp; Shift:        y_i = γ · x̂_i + β</p>
      </div>

      <p>
        Parameters <code>γ</code> and <code>β</code> are learnable. If the network determines the unnormalized activations were optimal, it can learn <code>γ = √(σ_B²)</code> and <code>β = μ_B</code> to recover the identity mapping.
      </p>

      <h2>4 · Normalization Taxonomy: Batch vs. Layer vs. Group</h2>
      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Method</th>
              <th className="px-4 py-3 text-left">Normalization Axis</th>
              <th className="px-4 py-3 text-left">Batch Size Dependency</th>
              <th className="px-4 py-3 text-left">Standard Domain</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">BatchNorm</td>
              <td className="px-4 py-3">Across Batch <code>(N)</code> for each feature channel.</td>
              <td className="px-4 py-3">High. Fails for small batch sizes (<code>N &lt; 8</code>).</td>
              <td className="px-4 py-3">Convolutional Neural Networks (ResNet, ConvNeXt)</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">LayerNorm</td>
              <td className="px-4 py-3">Across Features <code>(C, H, W)</code> independently for each sample.</td>
              <td className="px-4 py-3">None. Identical computation whether batch size is 1 or 1,000.</td>
              <td className="px-4 py-3">Transformers, LLMs (BERT, GPT, LLaMA), RNNs</td>
            </tr>
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">GroupNorm</td>
              <td className="px-4 py-3">Partitions channels into <code>G</code> groups; normalizes within each group.</td>
              <td className="px-4 py-3">None. Stable for batch size = 1.</td>
              <td className="px-4 py-3">Computer Vision with high-resolution inputs / small batches</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>5 · Python Implementation from First Principles</h2>
      <TerminalBlock language="python" filename="regularization_techniques.py" code={code} />

      <h2>6 · Further Reading</h2>
      <ul>
        <li>Srivastava, N., et al. (2014). <em>Dropout: A simple way to prevent neural networks from overfitting</em>. JMLR, 15(1), 1929–1958.</li>
        <li>Ioffe, S., &amp; Szegedy, C. (2015). <em>Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift</em>. ICML 2015.</li>
        <li>Ba, J. L., Kiros, J. R., &amp; Hinton, G. E. (2016). <em>Layer Normalization</em>. arXiv:1607.06450.</li>
      </ul>
    </article>
  );
}
