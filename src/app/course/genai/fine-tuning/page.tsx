"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import numpy as np

class LoRALinear:
    """
    Low-Rank Adaptation (LoRA) for Parameter-Efficient Fine-Tuning (PEFT)
    (Hu et al., Microsoft 2021).
    
    Forward Equation:
      h = W_0 · x + ΔW · x = W_0 · x + (α / r) · (B · A) · x
      where W_0 ∈ ℝ^(d_out × d_in) is frozen,
            A   ∈ ℝ^(r × d_in)      initialized ~ N(0, σ²),
            B   ∈ ℝ^(d_out × r)     initialized to 0.0,
            r ≪ min(d_in, d_out)    is the intrinsic rank.
    """
    def __init__(self, d_in: int, d_out: int, r: int = 4, lora_alpha: float = 16.0):
        self.d_in = d_in
        self.d_out = d_out
        self.r = r
        self.scaling = lora_alpha / r

        # Frozen pre-trained base weight
        self.W0 = np.random.randn(d_out, d_in) * 0.02
        self.W0_frozen = True

        # Trainable low-rank decomposition matrices
        # Matrix A initialized from Gaussian
        self.A = np.random.randn(r, d_in) * np.sqrt(2.0 / d_in)
        # Matrix B initialized to EXACTLY ZERO: ensures ΔW = 0 at training start!
        self.B = np.zeros((d_out, r))

    def forward(self, x: np.ndarray) -> np.ndarray:
        # 1. Base model path (frozen)
        base_out = np.dot(x, self.W0.T)

        # 2. Low-rank adapter path: (x · A^T) · B^T · (α / r)
        lora_intermediate = np.dot(x, self.A.T)  # (batch, r)
        lora_out = np.dot(lora_intermediate, self.B.T) * self.scaling

        return base_out + lora_out

    def merge_weights(self):
        """
        At deployment / inference time: merge low-rank matrices back into base weights.
        W_deployed = W_0 + (α / r) · (B · A)
        Zero additional inference latency or memory overhead!
        """
        delta_W = self.scaling * np.dot(self.B, self.A)
        self.W0 += delta_W
        print("LoRA weights merged into W0. Deployable as a standard linear layer.")

if __name__ == "__main__":
    d_in, d_out, rank = 4096, 4096, 8
    layer = LoRALinear(d_in, d_out, r=rank, lora_alpha=16.0)

    # Base parameters vs. LoRA trainable parameters
    base_params = d_in * d_out
    lora_params = (d_in * rank) + (rank * d_out)
    reduction = 100.0 * (1.0 - lora_params / base_params)

    print(f"Base Linear Layer Parameters: {base_params:,}")
    print(f"LoRA Trainable Parameters:    {lora_params:,} (Rank {rank})")
    print(f"Parameter Reduction:          {reduction:.2f}% fewer trainable weights!")

    x_input = np.random.randn(2, d_in)
    out_initial = layer.forward(x_input)
    # Verify that initial low-rank contribution is exactly zero (B=0)
    base_only = np.dot(x_input, layer.W0.T)
    print(f"Initial ΔW zero-deviation:    {np.allclose(out_initial, base_only)}")
`;

export default function FineTuningPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-100 text-indigo-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-indigo-500"></span>
          Track 3 · Generative AI &amp; Large Language Models
        </span>
      </div>

      <h1>Fine-Tuning &amp; Parameter-Efficient Fine-Tuning (PEFT): LoRA, QLoRA, and Prefix Tuning</h1>

      <blockquote>
        <p>
          <em>"Pre-trained language models have a very low intrinsic dimension: the weight updates during task-specific adaptation occupy a tiny subspace compared to the full parameter space."</em> — Edward Hu et al. (Microsoft Research, 2021)
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Contrast Full Fine-Tuning with Parameter-Efficient Fine-Tuning (PEFT) across GPU VRAM requirements.</li>
        <li>Derive Low-Rank Adaptation (LoRA) weight decomposition: <code>ΔW = (α / r) · B · A</code>.</li>
        <li>Explain why initializing matrix <code>B = 0</code> ensures <code>ΔW = 0</code> at initialization, preventing degradation.</li>
        <li>Understand QLoRA innovations (Dettmers et al.): NormalFloat4 (NF4) data type, Double Quantization, and Paged Optimizers.</li>
        <li>Implement a functional LoRA Linear layer from scratch in NumPy with weight merging capabilities.</li>
      </ul>

      <h2>1 · The VRAM Bottleneck of Full Fine-Tuning</h2>
      <p>
        In Full Fine-Tuning (FFT), every single weight in the network is updated. In 16-bit precision (FP16/BF16), storing a 70B parameter model requires 140 GB of VRAM. However, during training with AdamW, memory consumption explodes:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p className="text-slate-400">// Memory Breakdown for AdamW Training per parameter (bytes):</p>
        <p>Model Weights (FP16):           2 bytes</p>
        <p>Gradients (FP16):               2 bytes</p>
        <p>AdamW 1st Moment m (FP32):      4 bytes</p>
        <p>AdamW 2nd Moment v (FP32):      4 bytes</p>
        <p>Master Copy of Weights (FP32):  4 bytes</p>
        <p>Total per parameter:            16 bytes</p>
        <br />
        <p>For a 70B parameter model: 70 × 10⁹ × 16 bytes = 1,120 GB (over 14 × A100 80GB GPUs)!</p>
      </div>

      <p>
        Additionally, full fine-tuning suffers from <strong>Catastrophic Forgetting</strong>: specializing the model on a narrow domain causes it to lose its general conversational and reasoning capabilities.
      </p>

      <h2>2 · Low-Rank Adaptation (LoRA): Hu et al. (2021)</h2>
      <p>
        Aghajanyan et al. (2020) demonstrated that the <em>intrinsic rank</em> of parameter updates <code>ΔW</code> is extraordinarily small.
      </p>
      <p>
        For a pre-trained weight matrix <code>W_0 ∈ ℝ^(d × k)</code>, LoRA freezes <code>W_0</code> and decomposes the update matrix <code>ΔW</code> into two low-rank matrices <code>B ∈ ℝ^(d × r)</code> and <code>A ∈ ℝ^(r × k)</code> where <code>r ≪ min(d, k)</code>:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>h = W_0 · x + ΔW · x = W_0 · x + (α / r) · B · A · x</p>
      </div>

      <h3>Why Initialize B = 0 and A ~ N(0, σ²)?</h3>
      <ul>
        <li>If both <code>A</code> and <code>B</code> were initialized from Gaussian noise, <code>ΔW = B · A ≠ 0</code> at step 0, introducing massive random perturbation into the pre-trained model and destroying initial performance.</li>
        <li>By setting <code>B = 0</code>, the initial low-rank product is identically zero: <code>ΔW = 0 · A = 0</code>. The model begins training producing the exact pre-trained outputs, gradually learning task offsets smoothly.</li>
      </ul>

      <h3>Zero Inference Latency (Weight Merging)</h3>
      <p>
        At deployment time, the adapter weights can be permanently added into the base matrix:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>W_deployed = W_0 + (α / r) · (B · A)</p>
      </div>

      <p>
        The merged model is functionally a standard linear layer with zero added latency or extra memory.
      </p>

      <h2>3 · QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)</h2>
      <p>
        QLoRA enables fine-tuning a 65B model on a single 48GB GPU by introducing three core architectural breakthroughs:
      </p>

      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Innovation</th>
              <th className="px-4 py-3 text-left">Mechanism</th>
              <th className="px-4 py-3 text-left">Engineering Benefit</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">NormalFloat4 (NF4)</td>
              <td className="px-4 py-3">Quantiles matched to normal distribution of neural weights; equal information theoretical content per bin.</td>
              <td className="px-4 py-3">Better precision than standard 4-bit integer (INT4) or floating point (FP4).</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Double Quantization (DQ)</td>
              <td className="px-4 py-3">Quantizes the quantization constants (scales) themselves from 32-bit floats down to 8-bit integers.</td>
              <td className="px-4 py-3">Saves ~0.37 bits per parameter (3 GB savings on a 65B model).</td>
            </tr>
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Paged Optimizers</td>
              <td className="px-4 py-3">Utilizes CUDA Unified Memory to automatically page optimizer states to CPU RAM during activation memory spikes.</td>
              <td className="px-4 py-3">Completely prevents CUDA Out-of-Memory (OOM) crashes on long sequences.</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>4 · Python Implementation of LoRA Layer</h2>
      <TerminalBlock language="python" filename="lora_peft_scratch.py" code={code} />

      <h2>5 · Further Reading</h2>
      <ul>
        <li>Hu, E. J., et al. (2021). <em>LoRA: Low-Rank Adaptation of Large Language Models</em>. ICLR 2022.</li>
        <li>Dettmers, T., et al. (2023). <em>QLoRA: Efficient Finetuning of Quantized LLMs</em>. NeurIPS 2023.</li>
        <li>Li, X. L., &amp; Liang, P. (2021). <em>Prefix-Tuning: Optimizing Continuous Prompts for Generation</em>. ACL 2021.</li>
      </ul>
    </article>
  );
}
