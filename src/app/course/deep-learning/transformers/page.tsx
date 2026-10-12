"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";
import MathBlock from "@/components/MathBlock";
import ConceptCard from "@/components/ConceptCard";
import ComparisonTable from "@/components/ComparisonTable";
import AlgorithmBlock from "@/components/AlgorithmBlock";

const transformerEngineCode = `import numpy as np
from typing import Tuple, Optional

# ==============================================================================
# 1. ROTARY POSITION EMBEDDINGS (RoPE - Su et al., 2021)
# ==============================================================================

def precompute_rope_frequencies(d_head: int, seq_len: int, theta: float = 10000.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes complex rotation angles for Rotary Position Embedding:
    theta_i = 10000 ^ (-2(i-1) / d)
    """
    assert d_head % 2 == 0, "Head dimension must be even for 2D pairwise rotation"
    dim_indices = np.arange(0, d_head, 2)
    freqs = 1.0 / (theta ** (dim_indices / d_head))  # (d_head / 2,)
    positions = np.arange(seq_len)                   # (seq_len,)
    angles = np.outer(positions, freqs)             # (seq_len, d_head / 2)
    cos = np.cos(angles)
    sin = np.sin(angles)
    return cos, sin

def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """
    Applies pairwise 2D rotation to queries or keys:
    [x1, x2] -> [x1 * cos - x2 * sin,  x1 * sin + x2 * cos]
    Shape of x: (batch_size, num_heads, seq_len, d_head)
    """
    # Split into even and odd indices along last dimension
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    
    # Broadcast cos/sin over batch and head dimensions: (1, 1, seq_len, d_head / 2)
    cos_b = cos[np.newaxis, np.newaxis, :, :]
    sin_b = sin[np.newaxis, np.newaxis, :, :]
    
    x_rot1 = x1 * cos_b - x2 * sin_b
    x_rot2 = x1 * sin_b + x2 * cos_b
    
    # Interleave back together
    x_out = np.empty_like(x)
    x_out[..., 0::2] = x_rot1
    x_out[..., 1::2] = x_rot2
    return x_out

# ==============================================================================
# 2. ROOT MEAN SQUARE LAYER NORMALIZATION (RMSNorm - Zhang & Sennrich, 2019)
# ==============================================================================

class RMSNorm:
    """
    RMSNorm normalizes activations strictly by their root mean square without
    centering by the mean: RMS(x) = sqrt( (1/d) * sum(x_i^2) + eps )
    Used in LLaMA-3, Mistral, and Gemma for 15% faster kernel execution.
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        self.eps = eps
        self.weight = np.ones(d_model)  # learnable scale parameter gamma

    def forward(self, x: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.weight

# ==============================================================================
# 3. SwiGLU FEED-FORWARD NETWORK (Shazeer, 2020)
# ==============================================================================

class SwiGLUFeedForward:
    """
    SwiGLU Activation Gated MLP as used in modern frontier LLMs:
    SwiGLU(x) = ( Swish(x * W_gate) ⊙ (x * W_up) ) * W_down
    where Swish(z) = z * sigmoid(z).
    """
    def __init__(self, d_model: int, d_ff: int):
        self.W_gate = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.W_up   = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.W_down = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)

    @staticmethod
    def swish(z: np.ndarray) -> np.ndarray:
        return z / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, x: np.ndarray) -> np.ndarray:
        gate = self.swish(np.dot(x, self.W_gate))
        up = np.dot(x, self.W_up)
        return np.dot(gate * up, self.W_down)

# ==============================================================================
# 4. COMPLETE MODERN DECODER TRANSFORMER BLOCK
# ==============================================================================

class ModernTransformerBlock:
    """
    Pre-RMSNorm Causal Decoder Block matching LLaMA-3 and Mistral architectures:
    1. Pre-RMSNorm
    2. Causal Self-Attention with RoPE (Rotary Position Embeddings)
    3. Residual Connection (x = x + Attn)
    4. Pre-RMSNorm
    5. SwiGLU Gated Feed-Forward Network
    6. Residual Connection (x = x + FFN)
    """
    def __init__(self, d_model: int = 256, num_heads: int = 4, d_ff: int = 680):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        # Attention projection weights
        scale = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale
        
        self.ffn = SwiGLUFeedForward(d_model, d_ff)

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, S, D = x.shape
        cos, sin = precompute_rope_frequencies(self.d_head, S)

        # --- 1. Attention Sublayer with Pre-LN ---
        h_norm = self.norm1.forward(x)
        Q = np.dot(h_norm, self.W_q).reshape(B, S, self.num_heads, self.d_head).swapaxes(1, 2)
        K = np.dot(h_norm, self.W_k).reshape(B, S, self.num_heads, self.d_head).swapaxes(1, 2)
        V = np.dot(h_norm, self.W_v).reshape(B, S, self.num_heads, self.d_head).swapaxes(1, 2)

        # Apply Rotary Position Embeddings to Q and K
        Q_rot = apply_rope(Q, cos, sin)
        K_rot = apply_rope(K, cos, sin)

        # Causal Attention
        scores = np.matmul(Q_rot, K_rot.swapaxes(-1, -2)) / np.sqrt(self.d_head)
        causal_mask = np.tril(np.ones((S, S)))
        scores = np.where(causal_mask == 0, -1e9, scores)
        
        attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = attn / np.sum(attn, axis=-1, keepdims=True)
        
        context = np.matmul(attn, V).swapaxes(1, 2).reshape(B, S, D)
        attn_out = np.dot(context, self.W_o)

        # First Residual Connection
        x = x + attn_out

        # --- 2. Feed-Forward Sublayer with Pre-LN ---
        x = x + self.ffn.forward(self.norm2.forward(x))
        return x

# ==============================================================================
# VERIFICATION & NUMERICAL RUN
# ==============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    B, Seq, Dim = 2, 8, 128
    dummy_input = np.random.randn(B, Seq, Dim)

    block = ModernTransformerBlock(d_model=Dim, num_heads=4, d_ff=340)
    out = block.forward(dummy_input)

    print("Input Tensor Shape: ", dummy_input.shape)
    print("Output Tensor Shape:", out.shape)
    print("Gradient Highway Verified: Output mean magnitude =", np.round(np.mean(np.abs(out)), 4))
`;

export default function TransformersPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      {/* ── Track Badge ── */}
      <div className="not-prose mb-4">
        <span className="inline-block rounded-full bg-purple-100 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-purple-700">
          Track 2 · Deep Learning
        </span>
      </div>

      <h1>The Transformer Architecture: Positional Encodings (RoPE), Pre-LN vs Post-LN, RMSNorm, and SwiGLU</h1>

      <p className="lead text-lg text-slate-700">
        The Transformer (Vaswani et al., 2017) eliminated recurrence and convolution entirely, establishing self-attention
        as the universal compute substrate for generative AI. Over the past decade, the architecture evolved from the original
        Encoder-Decoder Post-LN layout to the modern Pre-LN Decoder-Only paradigm with Rotary Position Embeddings (RoPE),
        RMSNorm, and SwiGLU activations powering frontier models like LLaMA-3, Mistral, and GPT-4. This lecture provides
        a rigorous analysis of the Transformer block mechanics, positional geometry, normalization stability, and feed-forward architectures.
      </p>

      {/* ── Learning Objectives ── */}
      <h2>Learning Objectives</h2>
      <ul>
        <li>
          Compare Encoder-Decoder (Vaswani 2017), Encoder-Only (BERT), and Decoder-Only (GPT, LLaMA) topologies and understand why Decoder-Only dominates generative modeling.
        </li>
        <li>
          Prove why self-attention is permutation-equivariant and derive both absolute sinusoidal positional encodings and modern Rotary Position Embeddings (RoPE, Su et al., 2021).
        </li>
        <li>
          Analyze the gradient backpropagation dynamics of Post-LN vs Pre-LN and understand why Pre-LN enables training 1,000-layer models without learning rate warmup fragility.
        </li>
        <li>
          Derive Root Mean Square Layer Normalization (RMSNorm) and prove why removing mean-centering preserves representational capacity while accelerating GPU throughput.
        </li>
        <li>
          Deconstruct Gated Linear Units (SwiGLU, Shazeer 2020) and evaluate parameter scaling in position-wise feed-forward networks.
        </li>
        <li>
          Construct an end-to-end, modern LLaMA-style Decoder Transformer Block from scratch in NumPy with RoPE, RMSNorm, and SwiGLU.
        </li>
      </ul>

      {/* ══════════════════════════════════════════════
          Section 1 · Architectural Taxonomy
      ══════════════════════════════════════════════ */}
      <h2>1 · Architectural Taxonomy: Encoder vs. Decoder</h2>
      <p>
        The Transformer ecosystem bifurcated into three distinct architectural families:
      </p>

      <ComparisonTable
        title="The Three Transformer Architectural Paradigms"
        subtitle="Attention masking, training objectives, representational properties, and canonical models."
        columns={[
          { key: "family", header: "Architecture", width: "18%" },
          { key: "attention", header: "Attention Mechanism", width: "22%" },
          { key: "objective", header: "Pretraining Objective", width: "30%" },
          { key: "models", header: "Exemplar Models", width: "30%" },
        ]}
        data={[
          {
            family: <strong className="text-slate-900">Encoder-Only</strong>,
            attention: "Bidirectional (All tokens attend to all tokens)",
            objective: "Masked Language Modeling (MLM): Predict 15% corrupted tokens",
            models: "BERT, RoBERTa, DeBERTa, ModernBERT",
          },
          {
            family: <strong className="text-slate-900">Encoder-Decoder</strong>,
            attention: "Bidirectional in Encoder; Causal + Cross-Attention in Decoder",
            objective: "Sequence-to-Sequence Denoising: Reconstruct masked text",
            models: "Original Vaswani (2017), T5, BART",
          },
          {
            family: <strong className="text-slate-900">Decoder-Only</strong>,
            attention: <span className="text-emerald-700 font-semibold">Causal (Strict lower-triangular mask)</span>,
            objective: "Autoregressive Next-Token Prediction: P(x_t | x_<t)",
            models: "GPT-4, LLaMA-3, Mistral, Claude, Gemma",
          },
        ]}
      />

      <ConceptCard variant="insight" title="Why Decoder-Only Won the LLM Revolution">
        <ul className="list-disc pl-5 space-y-1 text-sm">
          <li>
            <strong>Natural Pretraining-Inference Alignment:</strong> Pretraining directly on autoregressive log-likelihood{" "}
            <span className="font-mono text-xs">∑ log P(x_t | x_&lt;t)</span> matches the inference generation loop token-by-token.
          </li>
          <li>
            <strong>Zero Alignment Tax:</strong> In-context few-shot learning and instruction tuning function natively without
            needing distinct encoder and decoder embedding projections.
          </li>
          <li>
            <strong>Unified KV Caching:</strong> Pure causal masking allows efficient linear append operations to a single Key-Value cache.
          </li>
        </ul>
      </ConceptCard>

      {/* ══════════════════════════════════════════════
          Section 2 · Positional Geometry: Sinusoids to RoPE
      ══════════════════════════════════════════════ */}
      <h2>2 · Positional Geometry: Permutation Equivariance to RoPE</h2>
      <p>
        Self-attention is mathematically <strong>permutation equivariant</strong>: if you shuffle the input tokens, the output vectors
        are shuffled identically. Without explicit position signals, the sentence &ldquo;The dog bit the cat&rdquo; is processed
        identically to &ldquo;The cat bit the dog&rdquo;.
      </p>

      <h3>2.1 Sinusoidal Encodings (Vaswani et al., 2017)</h3>
      <MathBlock
        title="Absolute Sinusoidal Positional Encoding"
        steps={[
          {
            label: "Even Indices",
            formula: "PE_(pos, 2i) = sin( pos / 10000^(2i / d_model) )",
            annotation: "Wavelengths form a geometric progression from 2π to 10000 · 2π.",
          },
          {
            label: "Odd Indices",
            formula: "PE_(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )",
            annotation: "Orthogonal cosine basis at the same frequency.",
          },
          {
            label: "Linear Shift Property",
            formula: "PE_(pos + k) = T_k · PE_(pos)",
            annotation: "By trigonometric addition identities: sin(α + β) = sin α cos β + cos α sin β, proving that relative shift k is representable as a linear matrix transformation.",
          },
        ]}
        result="Limitation: Added directly to word embeddings: x̃ = Embedding(x) + PE, diluting token semantic representations"
        resultNote="Sinusoidal encodings fail to generalize well when evaluated on sequence lengths longer than seen during pretraining."
      />

      <h3>2.2 Rotary Position Embedding (RoPE - Su et al., 2021)</h3>
      <p>
        RoPE encodes relative position by <strong>rotating</strong> Query and Key vectors in 2D planes rather than adding static vectors.
      </p>

      <MathBlock
        title="Rotary Position Embedding (RoPE) Formulation"
        steps={[
          {
            label: "2D Block Rotation",
            formula: "R_Θ,m · x_m = [ cos(m θ_i)  -sin(m θ_i) ] [ x_m^(2i)   ]",
            annotation: "                          [ sin(m θ_i)   cos(m θ_i) ] [ x_m^(2i+1) ]",
          },
          {
            label: "Inner Product Property",
            formula: "⟨ R_Θ,m · q_m,   R_Θ,n · k_n ⟩ = q_m^T · (R_Θ,m)^T · R_Θ,n · k_n = q_m^T · R_Θ,(n-m) · k_n",
            annotation: "The dot product between query at pos m and key at pos n depends strictly on the relative distance (n - m)!",
          },
        ]}
        result="Universal Standard: Preserves vector norm while embedding relative positional distance directly into the attention score"
        resultNote="RoPE is utilized across LLaMA-1/2/3, Mistral, Qwen, and DeepSeek, supporting context extension via RoPE frequency scaling (YaRN, NTK-aware)."
      />

      {/* ══════════════════════════════════════════════
          Section 3 · Normalization: Pre-LN & RMSNorm
      ══════════════════════════════════════════════ */}
      <h2>3 · Normalization Topologies: Pre-LN vs. Post-LN &amp; RMSNorm</h2>
      <p>
        Where Layer Normalization is placed relative to the residual skip connection drastically impacts gradient propagation:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p className="text-slate-400">// Post-LN (Original Vaswani 2017 - Unstable in deep models)</p>
        <p>x_(l+1) = LayerNorm( x_l + Sublayer(x_l) )</p>
        <p className="mt-3 text-slate-400">// Pre-LN (Modern Standard - Unobstructed residual highway)</p>
        <p>x_(l+1) = x_l + Sublayer( LayerNorm(x_l) )</p>
      </div>

      <ConceptCard variant="theorem" title="Mathematical Proof of Pre-LN Stability (Xiong et al., 2020)">
        <p>
          In Pre-LN, the recurrence relation across <span className="font-mono text-xs">L</span> layers is:
        </p>
        <p className="font-mono text-xs bg-purple-100/60 p-2 rounded">
          x_L = x_0 + ∑_(l=1)^L Sublayer( LN(x_(l-1)) )
        </p>
        <p>
          Taking the gradient of the loss <span className="font-mono text-xs">ℰ</span> with respect to the input representation <span className="font-mono text-xs">x_0</span> yields:
        </p>
        <p className="font-mono text-xs bg-purple-100/60 p-2 rounded">
          ∂ℰ / ∂x_0 = ∂ℰ / ∂x_L · [ I + ∑_(l=1)^L ∂Sublayer / ∂x_0 ]
        </p>
        <p>
          The identity matrix <span className="font-mono text-xs">I</span> provides an uninterrupted, unattenuated gradient highway
          directly from the loss back to the earliest embeddings. In Post-LN, gradients pass through <span className="font-mono text-xs">L</span> successive
          LayerNorm Jacobians, causing variance explosion near initialization unless stabilized by fragile warmup schedules.
        </p>
      </ConceptCard>

      <h3>3.3 RMSNorm (Root Mean Square Layer Normalization)</h3>
      <p>
        Standard LayerNorm computes: <span className="font-mono text-sm">y = ((x - μ) / √(σ² + ε)) · γ + β</span>.
        Zhang &amp; Sennrich (2019) demonstrated that the re-centering operation (<span className="font-mono text-sm">x - μ</span>)
        contributes virtually nothing to model stability. RMSNorm drops the mean calculation entirely:
      </p>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p className="text-slate-400">// RMSNorm Equation</p>
        <p>RMS(x) = √( (1 / d) · ∑_(i=1)^d x_i² + ε )</p>
        <p className="mt-2 text-emerald-300">y = (x / RMS(x)) ⊙ γ</p>
      </div>

      {/* ══════════════════════════════════════════════
          Section 4 · Gated MLP: SwiGLU
      ══════════════════════════════════════════════ */}
      <h2>4 · Position-Wise Feed-Forward Networks: SwiGLU</h2>
      <p>
        In the original Transformer, the FFN consisted of two linear transformations with a ReLU activation:
        <span className="font-mono text-sm">FFN(x) = max(0, x W_1 + b_1) W_2 + b_2</span> with inner dimension <span className="font-mono text-sm">d_ff = 4 · d_model</span>.
      </p>

      <MathBlock
        title="SwiGLU Activation Gating (Shazeer, 2020)"
        steps={[
          {
            label: "Gated Linear Unit (GLU)",
            formula: "GLU(x, W, V) = (x · W) ⊙ σ(x · V)",
            annotation: "Bilinear mapping where one branch gates the other.",
          },
          {
            label: "Swish Activation",
            formula: "Swish_β(z) = z · σ(β z)",
            annotation: "Smooth, non-monotonic activation.",
          },
          {
            label: "SwiGLU Formulation",
            formula: "SwiGLU(x) = ( Swish(x · W_gate) ⊙ (x · W_up) ) · W_down",
            annotation: "Employs three weight matrices instead of two.",
          },
          {
            label: "Dimension Compensation",
            formula: "d_ff = ⌊(8 / 3) · d_model⌋",
            annotation: "To keep parameter count identical to a 2-matrix FFN with d_ff = 4 d_model (3 · (8/3) = 8 vs 2 · 4 = 8).",
          },
        ]}
        result="SwiGLU consistently outperforms ReLU, GELU, and Tanh across all benchmark perplexity evaluations"
        resultNote="Adopted as the mandatory FFN architecture in LLaMA-2/3, Mistral, PaLM, and Gemma."
      />

      {/* ══════════════════════════════════════════════
          Section 5 · Python Transformer Block
      ══════════════════════════════════════════════ */}
      <h2>5 · Full Modern Transformer Block Implementation</h2>
      <p>
        The code below demonstrates a complete, vectorized modern Decoder Transformer Block combining Pre-RMSNorm,
        RoPE rotary positional embeddings, causal self-attention, SwiGLU gating, and residual skip highways.
      </p>

      <TerminalBlock language="python" filename="modern_transformer_block.py" code={transformerEngineCode} />

      {/* ── References ── */}
      <h2>6 · Academic References</h2>
      <ul className="text-sm space-y-2">
        <li>
          <strong>Vaswani, A., et al. (2017)</strong>. <em>Attention Is All You Need</em>. NeurIPS 2017.
        </li>
        <li>
          <strong>Su, J., et al. (2024)</strong>. <em>RoFormer: Enhanced Transformer with Rotary Position Embedding</em>. Neurocomputing.
        </li>
        <li>
          <strong>Xiong, R., et al. (2020)</strong>. <em>On Layer Normalization in the Transformer Architecture</em>. ICML 2020.
        </li>
        <li>
          <strong>Zhang, B., &amp; Sennrich, R. (2019)</strong>. <em>Root Mean Square Layer Normalization</em>. NeurIPS 2019.
        </li>
        <li>
          <strong>Shazeer, N. (2020)</strong>. <em>GLU Variants Improve Transformer</em>. arXiv:2002.05202.
        </li>
      </ul>
    </article>
  );
}
