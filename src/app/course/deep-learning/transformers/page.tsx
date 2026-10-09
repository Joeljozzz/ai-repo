"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import numpy as np

class TransformerBlock:
    """
    Standard Transformer Encoder Block (Vaswani et al., 2017):
    - Multi-Head Self-Attention (MHA)
    - Pre-Layer Normalization (Pre-LN): Ba et al. (2016)
    - Position-Wise Feed-Forward Network (FFN): FFN(x) = max(0, x · W1 + b1) · W2 + b2
    - Residual Connections: x + Sublayer(LN(x))
    """
    def __init__(self, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, eps: float = 1e-5):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_ff = d_ff
        self.eps = eps

        # Layer Norm 1 parameters
        self.gamma1 = np.ones((1, 1, d_model))
        self.beta1 = np.zeros((1, 1, d_model))

        # Layer Norm 2 parameters
        self.gamma2 = np.ones((1, 1, d_model))
        self.beta2 = np.zeros((1, 1, d_model))

        # MHA projection matrices
        scale = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale

        # Position-wise Feed-Forward Network matrices
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros((1, 1, d_ff))
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        self.b2 = np.zeros((1, 1, d_model))

    def layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        # Normalize across feature dimension (last axis)
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + self.eps)
        return gamma * x_hat + beta

    def multi_head_attention(self, x: np.ndarray) -> np.ndarray:
        B, T, D = x.shape
        Q = np.dot(x, self.W_q).reshape(B, T, self.num_heads, self.d_k).swapaxes(1, 2)
        K = np.dot(x, self.W_k).reshape(B, T, self.num_heads, self.d_k).swapaxes(1, 2)
        V = np.dot(x, self.W_v).reshape(B, T, self.num_heads, self.d_k).swapaxes(1, 2)

        scores = np.matmul(Q, K.swapaxes(-1, -2)) / np.sqrt(self.d_k)
        attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = attn / np.sum(attn, axis=-1, keepdims=True)

        context = np.matmul(attn, V).swapaxes(1, 2).reshape(B, T, D)
        return np.dot(context, self.W_o)

    def feed_forward(self, x: np.ndarray) -> np.ndarray:
        # FFN(x) = max(0, x·W1 + b1) · W2 + b2
        hidden = np.maximum(0, np.dot(x, self.W1) + self.b1)
        return np.dot(hidden, self.W2) + self.b2

    def forward(self, x: np.ndarray) -> np.ndarray:
        # 1. Pre-LN Self-Attention with Residual Addition
        norm1 = self.layer_norm(x, self.gamma1, self.beta1)
        attn_out = self.multi_head_attention(norm1)
        x = x + attn_out

        # 2. Pre-LN Feed-Forward with Residual Addition
        norm2 = self.layer_norm(x, self.gamma2, self.beta2)
        ffn_out = self.feed_forward(norm2)
        x = x + ffn_out

        return x

def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
    PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )
    """
    pe = np.zeros((seq_len, d_model))
    pos = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(pos * div_term)
    pe[:, 1::2] = np.cos(pos * div_term)
    return pe

if __name__ == "__main__":
    B, T, D = 2, 8, 64  # Batch size 2, Sequence length 8, Hidden dim 64
    x = np.random.randn(B, T, D)
    
    # Add positional encoding
    pe = sinusoidal_positional_encoding(T, D)
    x = x + pe

    transformer = TransformerBlock(d_model=D, num_heads=4, d_ff=128)
    out = transformer.forward(x)
    print(f"Input with PE shape:       {x.shape}")
    print(f"Transformer output shape:  {out.shape} [Expected: (2, 8, 64)]")
`;

export default function TransformersPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-100 text-purple-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-purple-500"></span>
          Track 2 · Deep Learning
        </span>
      </div>

      <h1>The Transformer Architecture: Encoder-Decoder, Pre-LN, and Positional Encodings</h1>

      <blockquote>
        <p>
          <em>"The Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution."</em> — Vaswani et al. (2017)
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Understand the complete architectural taxonomy: Encoder-only (BERT), Decoder-only (GPT), and Encoder-Decoder (T5).</li>
        <li>Derive Sinusoidal Positional Encodings and prove why they enable relative positional linear projections via trigonometric angle-addition identities.</li>
        <li>Analyze the stability difference between Post-LN (Vaswani et al.) and Pre-LN (Radford et al., Baevski et al.).</li>
        <li>Derive parameter budgets across projection matrices, FFN layers, and LayerNorm components.</li>
        <li>Implement a full Transformer Encoder block from scratch in NumPy with LayerNorm and Feed-Forward Networks.</li>
      </ul>

      <h2>1 · The Architectural Taxonomy: BERT vs. GPT vs. T5</h2>
      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Architecture</th>
              <th className="px-4 py-3 text-left">Attention Type</th>
              <th className="px-4 py-3 text-left">Training Objective</th>
              <th className="px-4 py-3 text-left">Canonical Models</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Encoder-Only</td>
              <td className="px-4 py-3">Bidirectional (all tokens attend to all tokens)</td>
              <td className="px-4 py-3">Masked Language Modeling (MLM: predict 15% hidden tokens)</td>
              <td className="px-4 py-3">BERT, RoBERTa, DeBERTa (classification, embeddings)</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Decoder-Only</td>
              <td className="px-4 py-3">Causal / Autoregressive (token <code>i</code> only attends to <code>≤ i</code>)</td>
              <td className="px-4 py-3">Next-Token Prediction (Causal Language Modeling)</td>
              <td className="px-4 py-3">GPT-3/4, LLaMA, Mistral, Claude, Gemini (generative AI)</td>
            </tr>
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Encoder-Decoder</td>
              <td className="px-4 py-3">Bidirectional Encoder + Causal Decoder with Cross-Attention</td>
              <td className="px-4 py-3">Sequence-to-Sequence (span corruption, translation)</td>
              <td className="px-4 py-3">Original Vaswani 2017 Transformer, T5, BART</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>2 · Sinusoidal Positional Encodings: The Trigonometric Identity Proof</h2>
      <p>
        Because self-attention is <strong>permutation equivariant</strong> (reordering input tokens produces the exact same permutation of output representations), the network has zero inherent awareness of word order.
      </p>
      <p>
        Vaswani et al. injected positional information by adding static sinusoidal vectors <code>PE ∈ ℝ^(L × d_model)</code> directly to input embeddings:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )</p>
        <p>PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )</p>
      </div>

      <h3>Why This Allows the Model to Learn Relative Positions</h3>
      <p>
        By the classical trigonometric angle-addition identity:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>sin(α + β) = sin(α) cos(β) + cos(α) sin(β)</p>
        <p>cos(α + β) = cos(α) cos(β) - sin(α) sin(β)</p>
      </div>

      <p>
        For any fixed position offset <code>k</code>, the positional encoding at position <code>pos + k</code> can be expressed as a <strong>linear transformation</strong> of the encoding at position <code>pos</code>:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>[ PE(pos+k, 2i)   ]   =   [ cos(k·ω_i)   sin(k·ω_i) ]   [ PE(pos, 2i)   ]</p>
        <p>[ PE(pos+k, 2i+1) ]       [ -sin(k·ω_i)  cos(k·ω_i) ]   [ PE(pos, 2i+1) ]</p>
        <br />
        <p>where ω_i = 1 / 10000^(2i / d_model)</p>
      </div>

      <p>
        This allows the self-attention projection weights <code>W_Q</code> and <code>W_K</code> to easily learn attention affinities that depend strictly on relative distance <code>k</code>.
      </p>

      <h2>3 · Pre-LN vs. Post-LN: The Training Stability Revolution</h2>
      <p>
        The original Vaswani (2017) architecture used <strong>Post-LN</strong>:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p className="text-slate-400">// Post-LN (Original Vaswani 2017):</p>
        <p>x_(l+1) = LayerNorm( x_l + Sublayer(x_l) )</p>
        <br />
        <p className="text-slate-400">// Pre-LN (Radford et al. GPT-2, Modern Standard):</p>
        <p>x_(l+1) = x_l + Sublayer( LayerNorm(x_l) )</p>
      </div>

      <p>
        <strong>Why Pre-LN Won:</strong> In Post-LN, gradients passing through the residual stream are scaled by the derivative of LayerNorm at every block. In deep networks (&gt;12 layers), the expected gradient norm for early layers decays exponentially, requiring delicate learning rate warmups to prevent training divergence.
      </p>
      <p>
        In Pre-LN, the identity shortcut <code>x_l</code> remains an unobstructed highway from the final layer back to the input embeddings (<code>∂L/∂x_0 ≈ ∂L/∂x_L</code>), allowing deep models to train stably from epoch 1.
      </p>

      <h2>4 · The Position-Wise Feed-Forward Network (FFN)</h2>
      <p>
        While self-attention models relationships <em>between</em> tokens, the Feed-Forward Network (FFN) operates on each token representation independently and identically:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>FFN(x) = max(0, x · W_1 + b_1) · W_2 + b_2</p>
        <p>where W_1 ∈ ℝ^(d_model × d_ff),   W_2 ∈ ℝ^(d_ff × d_model),   typically d_ff = 4 · d_model</p>
      </div>

      <p>
        Modern research (Geva et al., 2021) showed that the FFN layers function as <strong>key-value associative memories</strong>: the first layer acts as keys detecting factual patterns, while the second layer outputs values corresponding to retrieved factual knowledge.
      </p>

      <h2>5 · Python Implementation from First Principles</h2>
      <TerminalBlock language="python" filename="transformer_block_scratch.py" code={code} />

      <h2>6 · Further Reading</h2>
      <ul>
        <li>Vaswani, A., et al. (2017). <em>Attention Is All You Need</em>. NeurIPS 2017.</li>
        <li>Radford, A., et al. (2019). <em>Language Models are Unsupervised Multitask Learners</em> (GPT-2). OpenAI Technical Report.</li>
        <li>Ba, J. L., Kiros, J. R., &amp; Hinton, G. E. (2016). <em>Layer Normalization</em>. arXiv:1607.06450.</li>
        <li>Geva, M., et al. (2021). <em>Transformer Feed-Forward Layers Are Key-Value Memories</em>. EMNLP 2021.</li>
      </ul>
    </article>
  );
}
