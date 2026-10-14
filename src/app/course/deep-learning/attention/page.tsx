"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";
import MathBlock from "@/components/MathBlock";
import ConceptCard from "@/components/ConceptCard";
import ComparisonTable from "@/components/ComparisonTable";

const attentionSuiteCode = `import numpy as np
from typing import Tuple, Optional

# ==============================================================================
# 1. SCALED DOT-PRODUCT ATTENTION WITH CAUSAL MASKING
# ==============================================================================

def scaled_dot_product_attention(
    Q: np.ndarray, 
    K: np.ndarray, 
    V: np.ndarray, 
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Scaled Dot-Product Attention:
        Attention(Q, K, V) = Softmax( (Q · K^T) / sqrt(d_k) + M ) · V
    
    Shapes:
        Q: (batch_size, num_heads, seq_len_q, d_k)
        K: (batch_size, num_heads, seq_len_k, d_k)
        V: (batch_size, num_heads, seq_len_k, d_v)
        mask: (seq_len_q, seq_len_k) or broadcastable
    """
    d_k = Q.shape[-1]
    
    # 1. Compute raw pairwise compatibility scores: (Q · K^T) / sqrt(d_k)
    scores = np.matmul(Q, K.swapaxes(-1, -2)) / np.sqrt(d_k)

    # 2. Inject additive causal / padding mask
    if mask is not None:
        # Fill masked positions with -1e9 (approximating -infinity before exp)
        scores = np.where(mask == 0, -1e9, scores)

    # 3. Stable Softmax over the key dimension
    shift_scores = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(shift_scores)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 4. Multiply attention weights by values
    context = np.matmul(attention_weights, V)
    return context, attention_weights

# ==============================================================================
# 2. MULTI-HEAD ATTENTION (MHA) & GROUPED-QUERY ATTENTION (GQA)
# ==============================================================================

class MultiHeadAttention:
    """
    Production Multi-Head and Grouped-Query Attention (Vaswani 2017, Ainslie 2023).
    Supports:
    - Standard Multi-Head Attention (num_q_heads == num_kv_heads)
    - Grouped-Query Attention (num_q_heads > num_kv_heads, e.g. LLaMA-3)
    - Multi-Query Attention (num_kv_heads == 1)
    """
    def __init__(self, d_model: int = 512, num_q_heads: int = 8, num_kv_heads: int = 8):
        assert d_model % num_q_heads == 0, "d_model must be divisible by num_q_heads"
        assert num_q_heads % num_kv_heads == 0, "num_q_heads must be a multiple of num_kv_heads"
        
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_q_heads // num_kv_heads
        self.d_k = d_model // num_q_heads
        
        # Projection weight matrices initialized with Xavier Normal
        scale_q = np.sqrt(2.0 / (d_model + d_model))
        scale_kv = np.sqrt(2.0 / (d_model + (self.num_kv_heads * self.d_k)))
        
        self.W_q = np.random.randn(d_model, d_model) * scale_q
        self.W_k = np.random.randn(d_model, self.num_kv_heads * self.d_k) * scale_kv
        self.W_v = np.random.randn(d_model, self.num_kv_heads * self.d_k) * scale_kv
        self.W_o = np.random.randn(d_model, d_model) * scale_q

    def forward(self, X: np.ndarray, is_causal: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        batch_size, seq_len, _ = X.shape
        
        # 1. Project inputs
        Q = np.dot(X, self.W_q)  # (B, S, d_model)
        K = np.dot(X, self.W_k)  # (B, S, num_kv_heads * d_k)
        V = np.dot(X, self.W_v)  # (B, S, num_kv_heads * d_k)
        
        # 2. Reshape into head dimensions
        Q = Q.reshape(batch_size, seq_len, self.num_q_heads, self.d_k).swapaxes(1, 2)
        K = K.reshape(batch_size, seq_len, self.num_kv_heads, self.d_k).swapaxes(1, 2)
        V = V.reshape(batch_size, seq_len, self.num_kv_heads, self.d_k).swapaxes(1, 2)
        
        # 3. For GQA / MQA: repeat KV heads to match Q heads
        if self.num_groups > 1:
            K = np.repeat(K, self.num_groups, axis=1)
            V = np.repeat(V, self.num_groups, axis=1)

        # 4. Construct causal autoregressive mask (lower triangular)
        mask = None
        if is_causal:
            mask = np.tril(np.ones((seq_len, seq_len)))

        # 5. Execute Scaled Dot-Product Attention across all heads in parallel
        context, attn_weights = scaled_dot_product_attention(Q, K, V, mask=mask)

        # 6. Concatenate heads and project output: (B, num_heads, S, d_k) -> (B, S, d_model)
        context = context.swapaxes(1, 2).reshape(batch_size, seq_len, self.d_model)
        output = np.dot(context, self.W_o)
        
        return output, attn_weights

# ==============================================================================
# VERIFICATION & NUMERICAL RUN
# ==============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    B, S, D = 2, 4, 128
    dummy_input = np.random.randn(B, S, D)

    # 1. Standard Multi-Head Attention (MHA: 8 Q-heads, 8 KV-heads)
    mha = MultiHeadAttention(d_model=D, num_q_heads=8, num_kv_heads=8)
    out_mha, weights_mha = mha.forward(dummy_input, is_causal=True)
    print("MHA Output Shape:", out_mha.shape)
    print("MHA Causal Attention Upper Triangle (Tokens cannot look ahead):\\n", 
          np.round(weights_mha[0, 0], 3))

    # 2. Grouped-Query Attention (GQA: 8 Q-heads, 2 KV-heads - LLaMA-3 Style)
    gqa = MultiHeadAttention(d_model=D, num_q_heads=8, num_kv_heads=2)
    out_gqa, _ = gqa.forward(dummy_input, is_causal=True)
    print("\\nGQA Output Shape:", out_gqa.shape)
    print("KV Cache Memory Reduction Factor:", 8 // 2, "x bandwidth savings")
`;

export default function AttentionPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      {/* ── Track Badge ── */}
      <div className="not-prose mb-4">
        <span className="inline-block rounded-full bg-purple-100 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-purple-700">
          Track 2 · Deep Learning
        </span>
      </div>

      <h1>Attention Mechanisms: Bahdanau, Scaled Dot-Product, Multi-Head, and Grouped-Query Attention (GQA)</h1>

      <p className="lead text-lg text-slate-700">
        Before attention mechanisms, sequence transduction relied on Recurrent Neural Networks (RNNs and LSTMs) that compressed
        arbitrary-length sequential contexts into a single fixed-dimensional hidden state vector. This information bottleneck
        prevented models from capturing long-range dependencies. Attention replaced this static compression with dynamic,
        content-based routing. This lecture covers the mathematical evolution of attention: from Bahdanau additive alignment to
        Vaswani scaled dot-product attention, the analytical proof for variance scaling by <span className="font-mono text-sm">1/√d_k</span>,
        Multi-Head Attention (MHA), and modern memory-efficient Grouped-Query Attention (GQA).
      </p>

      {/* ── Learning Objectives ── */}
      <h2>Learning Objectives</h2>
      <ul>
        <li>
          Understand the Seq2Seq information bottleneck problem and how Bahdanau (2014) additive attention solved it.
        </li>
        <li>
          Derive the Scaled Dot-Product Attention equation and prove mathematically why dividing by <span className="font-mono text-sm">√d_k</span> is necessary to prevent Softmax gradient saturation.
        </li>
        <li>
          Analyze Multi-Head Attention (MHA) and explain why multiple attention heads capture distinct linguistic subspaces (syntactic, positional, coreference).
        </li>
        <li>
          Evaluate the memory-bandwidth bottleneck of autoregressive inference and contrast MHA, Multi-Query Attention (MQA), and Grouped-Query Attention (GQA).
        </li>
        <li>
          Formulate causal autoregressive masking and implement an end-to-end vectorized MHA/GQA module from scratch in NumPy.
        </li>
      </ul>

      {/* ══════════════════════════════════════════════
          Section 1 · The Seq2Seq Bottleneck & Bahdanau
      ══════════════════════════════════════════════ */}
      <h2>1 · The Seq2Seq Bottleneck &amp; The Origin of Attention</h2>
      <p>
        In traditional Encoder-Decoder architectures (Sutskever et al., 2014), the encoder compresses an entire sequence
        <span className="font-mono text-sm">x = (x_1, x_2, ..., x_T)</span> into a single context vector <span className="font-mono text-sm">c = h_T</span>.
        For sentences exceeding 20–30 tokens, BLEU scores degraded drastically as the vector ran out of bits to encode nuanced semantic syntax.
      </p>

      <MathBlock
        title="Bahdanau Additive Attention Formulation (Bahdanau et al., 2014)"
        steps={[
          {
            label: "Alignment Score",
            formula: "e_ij = v_a^T · tanh(W_a · s_(i-1) + U_a · h_j)",
            annotation: "Single hidden-layer feedforward network scoring compatibility between decoder state s_(i-1) and encoder annotation h_j.",
          },
          {
            label: "Attention Weights",
            formula: "α_ij = exp(e_ij) / (∑_(k=1)^T exp(e_ik))",
            annotation: "Normalized distribution representing the degree to which output token i attends to input token j.",
          },
          {
            label: "Dynamic Context Vector",
            formula: "c_i = ∑_(j=1)^T α_ij · h_j",
            annotation: "Time-varying weighted sum of all encoder hidden states.",
          },
        ]}
        result="Information Bottleneck Destroyed: The decoder directly queries the exact encoder states relevant to the current prediction"
        resultNote="Luong et al. (2015) subsequently introduced multiplicative dot-product attention: e_ij = s_i^T h_j, paving the way for matrix multiplication acceleration on GPUs."
      />

      {/* ══════════════════════════════════════════════
          Section 2 · Scaled Dot-Product Attention & Proof
      ══════════════════════════════════════════════ */}
      <h2>2 · Scaled Dot-Product Attention &amp; The Variance Scaling Proof</h2>
      <p>
        In &ldquo;Attention Is All You Need&rdquo; (Vaswani et al., 2017), the authors replaced recurrence entirely with
        <strong>Scaled Dot-Product Attention</strong> operating across three matrix projections: Queries (<span className="font-mono text-sm">Q</span>),
        Keys (<span className="font-mono text-sm">K</span>), and Values (<span className="font-mono text-sm">V</span>).
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p className="text-amber-400 font-bold mb-2">// The Canonical Equation</p>
        <p className="text-base text-emerald-300">Attention(Q, K, V) = Softmax( (Q · K^T) / √d_k + M ) · V</p>
        <p className="mt-3 text-slate-400">// Where:</p>
        <p>  Q ∈ ℝ^(n × d_k)    — Queries representing what tokens are searching for</p>
        <p>  K ∈ ℝ^(m × d_k)    — Keys representing what tokens have to offer</p>
        <p>  V ∈ ℝ^(m × d_v)    — Values containing the actual contextual content</p>
        <p>  M ∈ ℝ^(n × m)      — Additive attention mask (e.g. -∞ for causal future positions)</p>
      </div>

      <ConceptCard variant="theorem" title="Proof: Why Must We Divide by √d_k?">
        <p>
          Assume the components of query vector <span className="font-mono text-xs">q</span> and key vector <span className="font-mono text-xs">k</span>{" "}
          are independent random variables with zero mean and unit variance:
        </p>
        <p className="font-mono text-xs bg-purple-100/60 p-2 rounded">
          𝔼[q_i] = 0,   Var(q_i) = 1,   𝔼[k_i] = 0,   Var(k_i) = 1
        </p>
        <p>
          Their dot product is the sum of <span className="font-mono text-xs">d_k</span> independent products:{" "}
          <span className="font-mono text-xs">q · k = ∑_(i=1)^(d_k) q_i k_i</span>.
        </p>
        <ul className="list-disc pl-5 space-y-1 text-xs">
          <li><strong>Expected Value:</strong> <span className="font-mono">𝔼[q · k] = ∑ 𝔼[q_i] · 𝔼[k_i] = 0</span>.</li>
          <li><strong>Variance:</strong> <span className="font-mono">Var(q · k) = ∑ Var(q_i k_i) = ∑ [Var(q_i)Var(k_i) + Var(q_i)(𝔼[k_i])² + Var(k_i)(𝔼[q_i])²] = ∑ (1 · 1) = d_k</span>.</li>
        </ul>
        <p className="mt-2">
          Therefore, the standard deviation of the raw dot product is <span className="font-mono text-xs">σ = √d_k</span>.
          For large <span className="font-mono text-xs">d_k</span> (e.g. <span className="font-mono text-xs">d_k = 64</span> or <span className="font-mono text-xs">128</span>),
          dot products frequently reach values &gt; 30.
        </p>
        <p className="font-mono text-xs bg-red-100/60 p-2 rounded text-red-900 mt-2">
          When logits z into Softmax are large, Softmax saturates: S_i ≈ 1 and S_j ≈ 0.
          As proven in Section 2 of Neural Networks, the Softmax gradient is S_i(1 - S_i) ≈ 0,
          causing catastrophic vanishing gradients during backpropagation!
        </p>
        <p className="mt-2">
          Dividing by <span className="font-mono text-xs">√d_k</span> scales the variance back to strictly <span className="font-mono text-xs">Var = 1</span>,
          ensuring smooth gradients across all sequence lengths.
        </p>
      </ConceptCard>

      {/* ══════════════════════════════════════════════
          Section 3 · Multi-Head Attention (MHA)
      ══════════════════════════════════════════════ */}
      <h2>3 · Multi-Head Attention: Attending to Multiple Subspaces</h2>
      <p>
        Rather than performing a single attention function with <span className="font-mono text-sm">d_model</span>-dimensional keys,
        queries, and values, Multi-Head Attention projects them <span className="font-mono text-sm">h</span> times with distinct,
        learnable linear projections.
      </p>

      <MathBlock
        title="Multi-Head Attention Formulation"
        steps={[
          {
            label: "Head Projection",
            formula: "head_i = Attention( Q · W_i^Q,   K · W_i^K,   V · W_i^V )",
            annotation: "Where W_i^Q ∈ ℝ^(d_model × d_k), W_i^K ∈ ℝ^(d_model × d_k), W_i^V ∈ ℝ^(d_model × d_v).",
          },
          {
            label: "Concatenation & Output",
            formula: "MultiHead(Q, K, V) = Concat(head_1, head_2, ..., head_h) · W^O",
            annotation: "Output projection W^O ∈ ℝ^(h·d_v × d_model). Typically d_k = d_v = d_model / h.",
          },
        ]}
        result="Dimensional Equivalence: Total computational cost is identical to single-head attention with full dimensionality"
        resultNote="Multiple heads permit the network to jointly attend to information from different representation subspaces (e.g., Head 1: syntactic structure; Head 2: pronoun coreference; Head 3: positional proximity)."
      />

      {/* ══════════════════════════════════════════════
          Section 4 · GQA, MQA, and Inference Bottlenecks
      ══════════════════════════════════════════════ */}
      <h2>4 · The Inference Memory Wall: MHA vs MQA vs GQA</h2>
      <p>
        During autoregressive LLM inference, previously generated Key and Value vectors must be cached in GPU High Bandwidth
        Memory (HBM) to avoid recomputing past tokens (&ldquo;KV Cache&rdquo;).
        For long context windows (e.g., 32k - 128k tokens), the KV Cache memory footprint dominates GPU VRAM, saturating memory bandwidth.
      </p>

      <ComparisonTable
        title="Attention Architectures: Memory Bandwidth & Quality Tradeoffs"
        subtitle="Comparing Multi-Head, Multi-Query, and Grouped-Query Attention."
        columns={[
          { key: "type", header: "Architecture", width: "20%" },
          { key: "kvHeads", header: "KV Heads relative to Q", width: "25%" },
          { key: "kvCacheSize", header: "KV-Cache Memory Footprint", width: "25%" },
          { key: "models", header: "Prominent Models", width: "30%" },
        ]}
        data={[
          {
            type: <strong className="text-slate-900">Multi-Head (MHA)</strong>,
            kvHeads: "num_kv_heads == num_q_heads (1:1 ratio)",
            kvCacheSize: "Baseline (100% memory)",
            models: "Original Transformer, GPT-3, LLaMA-1",
          },
          {
            type: <strong className="text-slate-900">Multi-Query (MQA)</strong>,
            kvHeads: "num_kv_heads == 1 (all Q share 1 KV)",
            kvCacheSize: <span className="text-emerald-700 font-semibold">Reduced by h (e.g. 1/32 ≈ 3.1%)</span>,
            models: "PaLM, StarCoder, Falcon",
          },
          {
            type: <strong className="text-slate-900">Grouped-Query (GQA)</strong>,
            kvHeads: "1 < num_kv_heads < num_q_heads (e.g. 8:1)",
            kvCacheSize: <span className="text-emerald-700 font-semibold">Reduced by 4x - 8x with zero quality loss</span>,
            models: "LLaMA-2-70B, LLaMA-3, Mistral 7B, Gemma",
          },
        ]}
      />

      {/* ══════════════════════════════════════════════
          Section 5 · Python Implementation
      ══════════════════════════════════════════════ */}
      <h2>5 · Complete Vectorized Multi-Head &amp; GQA Implementation</h2>
      <p>
        Below is a complete implementation supporting both standard Multi-Head Attention and Grouped-Query Attention with
        causal masking and parallel head computation.
      </p>

      <TerminalBlock language="python" filename="attention_mechanisms.py" code={attentionSuiteCode} />

      {/* ── References ── */}
      <h2>6 · Academic References</h2>
      <ul className="text-sm space-y-2">
        <li>
          <strong>Vaswani, A., et al. (2017)</strong>. <em>Attention Is All You Need</em>. NeurIPS 2017.
        </li>
        <li>
          <strong>Bahdanau, D., Cho, K., &amp; Bengio, Y. (2014)</strong>. <em>Neural machine translation by jointly learning to align and translate</em>. ICLR 2015.
        </li>
        <li>
          <strong>Ainslie, J., et al. (2023)</strong>. <em>GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints</em>. EMNLP 2023.
        </li>
        <li>
          <strong>Dao, T., et al. (2022)</strong>. <em>FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness</em>. NeurIPS 2022.
        </li>
      </ul>
    </article>
  );
}
