"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import numpy as np

class BPEandScaling:
    """
    Byte-Pair Encoding (BPE) subword tokenization and
    Chinchilla Compute-Optimal Scaling Laws (Hoffmann et al., DeepMind 2022).
    """
    @staticmethod
    def chinchilla_optimal_tokens(compute_flops: float):
        """
        Hoffmann et al. (2022) established that for compute-optimal training:
          C ≈ 6 · N · D   (where N = parameters, D = tokens)
          N_opt ∝ C^0.5
          D_opt ∝ C^0.5
        Token count and parameter count should scale in equal proportion:
        Rule of thumb: ~20 tokens per model parameter.
        """
        # Given target compute budget in FLOPs:
        # C = 6 * N * D, with N = G * C^0.5, D = H * C^0.5
        N_opt = 0.6 * np.sqrt(compute_flops / 6.0)
        D_opt = (compute_flops / 6.0) / N_opt
        return N_opt, D_opt

    @staticmethod
    def bpe_merge_step(corpus_words: dict):
        """
        Single iteration of Byte-Pair Encoding:
        Identifies most frequent contiguous symbol pair and merges them into a new vocabulary token.
        """
        pair_counts = {}
        for word, freq in corpus_words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + freq

        if not pair_counts:
            return None, corpus_words

        best_pair = max(pair_counts, key=pair_counts.get)
        
        # Merge best pair in corpus
        merged_corpus = {}
        target = " ".join(best_pair)
        replacement = "".join(best_pair)
        for word, freq in corpus_words.items():
            new_word = word.replace(target, replacement)
            merged_corpus[new_word] = freq

        return best_pair, merged_corpus

if __name__ == "__main__":
    # 1. Chinchilla Scaling Calculation
    # Compute budget equivalent to 10^24 FLOPs
    flops = 1e24
    N_params, D_tokens = BPEandScaling.chinchilla_optimal_tokens(flops)
    print(f"For Compute Budget C = 10^24 FLOPs:")
    print(f"  Optimal Parameters (N): {N_params / 1e9:.2f} Billion")
    print(f"  Optimal Tokens     (D): {D_tokens / 1e9:.2f} Billion (Ratio D/N ≈ {D_tokens/N_params:.1f})")

    # 2. BPE Toy Corpus
    corpus = {
        "l o w </w>": 5,
        "l o w e r </w>": 2,
        "n e w e s t </w>": 6,
        "w i d e s t </w>": 3
    }
    pair, updated_corpus = BPEandScaling.bpe_merge_step(corpus)
    print(f"\\nMost frequent contiguous pair: {pair}")
    print("Corpus after merge:", updated_corpus)
`;

export default function LLMFoundationsPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-100 text-indigo-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-indigo-500"></span>
          Track 3 · Generative AI &amp; Large Language Models
        </span>
      </div>

      <h1>Large Language Model Foundations: Tokenization (BPE), Scaling Laws, and Alignment (RLHF)</h1>

      <blockquote>
        <p>
          <em>"Kaplan et al. initially suggested that scaling parameters was more important than scaling dataset size. Chinchilla proved that for compute-optimal training, dataset size and model size must be scaled in equal proportion: approximately 20 tokens per model parameter."</em> — Jordan Hoffmann et al. (DeepMind, 2022)
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Understand Byte-Pair Encoding (BPE), SentencePiece, and the Byte-level fallback representation.</li>
        <li>Derive Chinchilla compute-optimal scaling laws and compute FLOP requirements: <code>C ≈ 6ND</code>.</li>
        <li>Explain the mechanics of Reinforcement Learning from Human Feedback (RLHF): SFT, Reward Modeling, and PPO / DPO.</li>
        <li>Analyze emergent capabilities, in-context learning, and autoregressive cross-entropy perplexity.</li>
        <li>Implement BPE vocabulary merging and Chinchilla compute budget allocation in Python.</li>
      </ul>

      <h2>1 · Subword Tokenization: Byte-Pair Encoding (BPE)</h2>
      <p>
        Natural language cannot be fed as raw characters (inefficient sequence lengths) or entire words (infinite vocabulary size, out-of-vocabulary failure on unseen conjugations).
      </p>
      <p>
        <strong>Byte-Pair Encoding (Sennrich et al., 2016; Radford et al., 2019):</strong>
      </p>
      <ol>
        <li>Initialize vocabulary <code>V</code> with individual unicode characters / raw bytes (256 base byte tokens).</li>
        <li>Tokenize text into sequences of individual characters.</li>
        <li>Count occurrences of all contiguous token pairs <code>(t_1, t_2)</code> across the corpus.</li>
        <li>Merge the most frequent pair into a single new token <code>t_new = t_1 ∘ t_2</code> and add it to <code>V</code>.</li>
        <li>Repeat until vocabulary reaches a target threshold (e.g. <code>32,000</code> in LLaMA, <code>100,000</code> in GPT-4).</li>
      </ol>

      <p>
        <strong>Byte-Level Fallback:</strong> By initializing with the 256 raw bytes of UTF-8, modern tokenizers (GPT-4, LLaMA) eliminate out-of-vocabulary (OOV) tokens entirely. Any arbitrary binary data or foreign script can be represented.
      </p>

      <h2>2 · Scaling Laws: Kaplan vs. Chinchilla (Hoffmann et al., 2022)</h2>
      <p>
        Training large autoregressive models requires billions of dollars in GPU compute. Determining the optimal allocation of parameters <code>N</code> and training tokens <code>D</code> given a floating-point operation (FLOP) budget <code>C</code> is critical.
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p className="text-slate-400">// Forward + Backward Pass Compute per token:</p>
        <p>Forward pass:  ~ 2 N  FLOPs per token</p>
        <p>Backward pass: ~ 4 N  FLOPs per token</p>
        <p>Total Compute: C ≈ 6 · N · D  FLOPs</p>
      </div>

      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Regime</th>
              <th className="px-4 py-3 text-left">Scaling Rule</th>
              <th className="px-4 py-3 text-left">Token-to-Parameter Ratio</th>
              <th className="px-4 py-3 text-left">Implications &amp; Historical Models</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Kaplan et al. (OpenAI, 2020)</td>
              <td className="px-4 py-3 font-mono">N ∝ C^0.73,  D ∝ C^0.27</td>
              <td className="px-4 py-3">~ 1.5 to 4 tokens per parameter</td>
              <td className="px-4 py-3">Led to severely undertrained massive models: GPT-3 (175B on 300B tokens), Gopher (280B on 300B tokens).</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Chinchilla (DeepMind, 2022)</td>
              <td className="px-4 py-3 font-mono">N ∝ C^0.50,  D ∝ C^0.50</td>
              <td className="px-4 py-3"><strong>~ 20 tokens per parameter</strong></td>
              <td className="px-4 py-3">Proved that a 70B model trained on 1.4T tokens beats a 280B model while using 4x less inference compute!</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>3 · Alignment: SFT, Reward Modeling, and RLHF / DPO</h2>
      <p>
        A raw pre-trained language model simply completes text statistically (e.g. prompt: "Write an essay on Rome" → completion: "Write an essay on Greece and Carthage"). It is not an assistant.
      </p>

      <h3>3.1 Supervised Fine-Tuning (SFT)</h3>
      <p>
        Fine-tunes the base model on tens of thousands of high-quality human demonstrations <code>(prompt, response)</code> using standard cross-entropy loss to teach instruction-following format.
      </p>

      <h3>3.2 Reinforcement Learning from Human Feedback (RLHF / PPO)</h3>
      <p>
        Ouyang et al. (InstructGPT, 2022) align model values with human preferences:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p className="text-slate-400">// 1. Reward Model (RM): Predicts scalar human preference score r_θ(x, y)</p>
        <p>Loss_RM = - E_(y_w, y_l) [ ln σ( r_θ(x, y_w) - r_θ(x, y_l) ) ]</p>
        <br />
        <p className="text-slate-400">// 2. Proximal Policy Optimization (PPO) with KL-Penalty against Base Reference Model:</p>
        <p>obj(ϕ) = E_(x, y ~ π_ϕ) [ r_θ(x, y) - β · D_KL( π_ϕ(y|x) || π_ref(y|x) ) ]</p>
      </div>

      <p>
        The <strong>KL penalty <code>β · D_KL</code></strong> prevents the policy model <code>π_ϕ</code> from deviating too far from the reference model <code>π_ref</code>, stopping "reward hacking" (generating gibberish that tricks the reward model).
      </p>

      <h3>3.3 Direct Preference Optimization (DPO, Rafailov et al. 2023)</h3>
      <p>
        DPO mathematically derives the optimal policy analytically, proving that the reward model can be expressed directly in terms of the policy itself:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>L_DPO(π_θ; π_ref) = - E [ ln σ( β · ln( π_θ(y_w|x) / π_ref(y_w|x) ) - β · ln( π_θ(y_l|x) / π_ref(y_l|x) ) ) ]</p>
      </div>

      <p>
        DPO completely eliminates the unstable reinforcement learning loop, training directly on preference pairs via a stable binary cross-entropy objective.
      </p>

      <h2>4 · Python Implementation of BPE &amp; Scaling Calculations</h2>
      <TerminalBlock language="python" filename="llm_foundations.py" code={code} />

      <h2>5 · Further Reading</h2>
      <ul>
        <li>Hoffmann, J., et al. (2022). <em>Training Compute-Optimal Large Language Models</em> (Chinchilla). arXiv:2203.15556.</li>
        <li>Ouyang, L., et al. (2022). <em>Training language models to follow instructions with human feedback</em> (InstructGPT). NeurIPS 2022.</li>
        <li>Rafailov, R., et al. (2023). <em>Direct Preference Optimization: Your Language Model is Secretly a Reward Model</em>. NeurIPS 2023.</li>
      </ul>
    </article>
  );
}
