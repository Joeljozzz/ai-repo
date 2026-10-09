"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import numpy as np

class MultiHeadAttention:
    """
    Scaled Dot-Product and Multi-Head Attention as formulated in
    'Attention Is All You Need' (Vaswani et al., 2017).
    
    Attention(Q, K, V) = softmax( (Q · K^T) / sqrt(d_k) + M ) · V
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
    """
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # typically 64

        # Projection weight matrices
        scale = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale

    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None):
        """
        Q, K, V shapes: (batch_size, num_heads, seq_len, d_k)
        """
        # 1. Compute raw affinity energy scores: (Q · K^T) / sqrt(d_k)
        scores = np.matmul(Q, K.swapaxes(-1, -2)) / np.sqrt(self.d_k)

        # 2. Apply causal autoregressive mask if provided (e.g. Decoder masking)
        if mask is not None:
            # Mask out future positions with negative infinity
            scores = np.where(mask == 0, -1e9, scores)

        # 3. Apply Softmax along the last dimension (over keys)
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # 4. Weighted combination of values
        context = np.matmul(attention_weights, V)
        return context, attention_weights

    def forward(self, X: np.ndarray, mask: np.ndarray = None):
        """
        Input X shape: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = X.shape

        # Linear projections
        Q = np.dot(X, self.W_q)  # (batch, seq, d_model)
        K = np.dot(X, self.W_k)
        V = np.dot(X, self.W_v)

        # Reshape and transpose into multi-head format: (batch, num_heads, seq_len, d_k)
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).swapaxes(1, 2)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).swapaxes(1, 2)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k).swapaxes(1, 2)

        # Scaled dot-product attention per head
        context, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads back: (batch, seq_len, d_model)
        context = context.swapaxes(1, 2).reshape(batch_size, seq_len, self.d_model)

        # Final linear output projection
        output = np.dot(context, self.W_o)
        return output, attn_weights

if __name__ == "__main__":
    # Test on a sequence of 6 tokens with d_model=64 and 4 heads
    batch_size, seq_len, d_model = 2, 6, 64
    x_tokens = np.random.randn(batch_size, seq_len, d_model)

    mha = MultiHeadAttention(d_model=d_model, num_heads=4)
    out, weights = mha.forward(x_tokens)

    print(f"Input sequence tensor: {x_tokens.shape}")
    print(f"Output contextualized: {out.shape}   [Expected: (2, 6, 64)]")
    print(f"Attention map weights: {weights.shape} [Expected: (2, 4, 6, 6)]")
`;

export default function AttentionPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-100 text-purple-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-purple-500"></span>
          Track 2 · Deep Learning
        </span>
      </div>

      <h1>Attention Mechanisms: Bahdanau Alignment, Scaled Dot-Product, and Multi-Head Attention</h1>

      <blockquote>
        <p>
          <em>"The fundamental bottleneck of sequence-to-sequence networks was compressing an arbitrary length input sequence into a single static context vector. Attention replaces this bottleneck with dynamic, data-dependent memory retrieval."</em> — Dzmitry Bahdanau, Kyunghyun Cho, &amp; Yoshua Bengio (2014)
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Understand the theoretical progression from additive attention (Bahdanau) to multiplicative attention (Luong).</li>
        <li>Derive the Scaled Dot-Product Attention equation and prove why the <code>1 / √d_k</code> normalization factor is necessary.</li>
        <li>Explain the Query, Key, and Value database retrieval metaphor in high-dimensional vector spaces.</li>
        <li>Analyze Multi-Head Attention as parallel subspace projections and derive total computational complexity.</li>
        <li>Implement vectorized Multi-Head Attention from scratch in NumPy with causal autoregressive masking.</li>
      </ul>

      <h2>1 · The Information Bottleneck of Seq2Seq Models</h2>
      <p>
        In classical Recurrent Neural Network encoder-decoder architectures (Sutskever et al., Cho et al.), the encoder compressed a source sequence <code>(x_1, ..., x_T)</code> into a single fixed-length vector <code>h_T</code>.
      </p>
      <p>
        For sentences longer than 20 words, performance deteriorated rapidly due to the <strong>information bottleneck</strong> and gradient decay over long temporal chains.
      </p>
      <p>
        <strong>Bahdanau Additive Attention (2014):</strong> Enabled the decoder at step <code>t</code> to selectively look back across <em>all</em> encoder hidden states <code>(h_1, ..., h_T)</code> by computing learned alignment scores:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p className="text-slate-400">// Bahdanau Additive Energy Score:</p>
        <p>e_ti = v_a^T · tanh( W_a · s_(t-1) + U_a · h_i )</p>
        <br />
        <p className="text-slate-400">// Luong Multiplicative Dot-Product Score (1915):</p>
        <p>e_ti = s_t^T · W_a · h_i</p>
      </div>

      <h2>2 · Query, Key, and Value Vector Formulation</h2>
      <p>
        Vaswani et al. (2017) formalized attention as a differentiable database lookup:
      </p>
      <ul>
        <li><strong>Query (Q):</strong> What the current token is seeking (search prompt).</li>
        <li><strong>Key (K):</strong> What each candidate token offers (database indexing tags).</li>
        <li><strong>Value (V):</strong> The actual information payload transferred if a match occurs.</li>
      </ul>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>Attention(Q, K, V) = Softmax( (Q · K^T) / √d_k ) · V</p>
      </div>

      <h3>Why Divide by √d_k? (Variance Preservation)</h3>
      <p>
        Assume the components of <code>q</code> and <code>k</code> are independent random variables with zero mean and unit variance <code>Var(q_i) = Var(k_i) = 1</code>. The dot product is:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>q · k = Σ_(i=1)^(d_k) q_i · k_i</p>
        <p>E[ q · k ] = Σ E[q_i] · E[k_i] = 0</p>
        <p>Var( q · k ) = Σ Var(q_i · k_i) = Σ [ Var(q_i) · Var(k_i) ] = d_k</p>
      </div>

      <p>
        For large projection dimensions (e.g., <code>d_k = 64</code> or <code>128</code>), the variance of <code>Q · K^T</code> grows to <code>d_k</code>, producing large dot products. Passing large magnitudes into the Softmax function pushes activations into regions with vanishingly small derivatives (saturation), choking gradient flow during backpropagation.
      </p>
      <p>
        Dividing by <strong><code>√d_k</code></strong> scales the variance back down to exactly 1.0, preserving gradient stability.
      </p>

      <h2>3 · Multi-Head Attention Theory</h2>
      <p>
        A single attention head tends to average information across tokens. <strong>Multi-Head Attention</strong> projects queries, keys, and values <code>h</code> times with independently learned linear projection matrices:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>MultiHead(Q, K, V) = Concat( head_1, ..., head_h ) · W_O</p>
        <p>where head_i = Attention( Q · W_i^Q,  K · W_i^K,  V · W_i^V )</p>
        <br />
        <p>W_i^Q ∈ ℝ^(d_model × d_k),   W_i^K ∈ ℝ^(d_model × d_k),   W_i^V ∈ ℝ^(d_model × d_v),   W_O ∈ ℝ^(h·d_v × d_model)</p>
      </div>

      <p>
        This enables the model to simultaneously attend to information from different representation subspaces at different positions (e.g., Head 1 tracks grammatical subject-verb agreement, Head 2 resolves pronoun coreference, Head 3 tracks local punctuation boundaries).
      </p>

      <h2>4 · Computational Complexity: Attention vs. Convolution vs. Recurrence</h2>
      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Layer Type</th>
              <th className="px-4 py-3 text-left">Complexity Per Layer</th>
              <th className="px-4 py-3 text-left">Sequential Operations</th>
              <th className="px-4 py-3 text-left">Maximum Path Length</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Self-Attention</td>
              <td className="px-4 py-3 font-mono">O(n² · d)</td>
              <td className="px-4 py-3 font-mono">O(1) [Fully Parallel]</td>
              <td className="px-4 py-3 font-mono">O(1)</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Recurrent (RNN)</td>
              <td className="px-4 py-3 font-mono">O(n · d²)</td>
              <td className="px-4 py-3 font-mono">O(n) [Non-parallelizable]</td>
              <td className="px-4 py-3 font-mono">O(n)</td>
            </tr>
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Convolutional</td>
              <td className="px-4 py-3 font-mono">O(k · n · d²)</td>
              <td className="px-4 py-3 font-mono">O(1)</td>
              <td className="px-4 py-3 font-mono">O(log_k(n))</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>5 · Python Implementation from First Principles</h2>
      <TerminalBlock language="python" filename="multihead_attention.py" code={code} />

      <h2>6 · Further Reading</h2>
      <ul>
        <li>Vaswani, A., et al. (2017). <em>Attention Is All You Need</em>. Advances in Neural Information Processing Systems (NeurIPS 30).</li>
        <li>Bahdanau, D., Cho, K., &amp; Bengio, Y. (2015). <em>Neural Machine Translation by Jointly Learning to Align and Translate</em>. ICLR 2015.</li>
        <li>Luong, M. T., Pham, H., &amp; Manning, C. D. (2015). <em>Effective Approaches to Attention-based Neural Machine Translation</em>. EMNLP 2015.</li>
      </ul>
    </article>
  );
}
