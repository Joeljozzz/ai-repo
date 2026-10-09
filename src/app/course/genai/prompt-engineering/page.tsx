"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import json

class PromptEngineeringPatterns:
    """
    Demonstrating formal prompt engineering patterns:
    1. Zero-Shot vs Few-Shot (Brown et al., 2020)
    2. Chain-of-Thought (CoT) with intermediate rationale (Wei et al., 2022)
    3. Self-Consistency majority voting (Wang et al., 2022)
    4. Tree-of-Thoughts (ToT) search exploration (Yao et al., 2023)
    5. Constitutional AI Self-Critique & Revision (Anthropic, 2022)
    """
    @staticmethod
    def format_cot_prompt(question: str, exemplars: list = None) -> str:
        """
        Formats a Chain-of-Thought prompt enforcing explicit step-by-step rationales.
        """
        system = "You are an analytical AI reasoning engine. Solve problems step-by-step before concluding with the final answer.\\n\\n"
        prompt = system
        if exemplars:
            for ex in exemplars:
                prompt += f"Q: {ex['q']}\\nA: Let's think step by step.\\n{ex['rationale']}\\nTherefore, the answer is {ex['answer']}.\\n\\n"
        prompt += f"Q: {question}\\nA: Let's think step by step."
        return prompt

    @staticmethod
    def self_consistency_voting(simulated_responses: list) -> str:
        """
        Samples multiple reasoning paths from the LLM at temperature T = 0.7,
        extracts the final answer from each path, and applies majority voting (Wang et al., 2022).
        """
        from collections import Counter
        extracted_answers = []
        for resp in simulated_responses:
            if "Therefore, the answer is" in resp:
                ans = resp.split("Therefore, the answer is")[-1].strip().rstrip(".")
                extracted_answers.append(ans)
        vote_counts = Counter(extracted_answers)
        winner, count = vote_counts.most_common(1)[0]
        return f"Majority Voted Answer: '{winner}' with {count}/{len(simulated_responses)} consensus."

if __name__ == "__main__":
    math_q = "A store sells apples for $2 each. If Alice buys 4 apples and pays with a $20 bill, how much change does she receive?"
    cot_prompt = PromptEngineeringPatterns.format_cot_prompt(math_q)
    print("Formatted Chain-of-Thought Prompt:\\n" + "─" * 50)
    print(cot_prompt)

    # Simulated sampling over 3 diverse reasoning paths (T=0.7)
    sample_paths = [
        "4 apples cost 4 * 2 = $8. Change = 20 - 8 = $12. Therefore, the answer is $12.",
        "Cost = $8. Alice pays $20. 20 - 8 = 12 dollars. Therefore, the answer is $12.",
        "Total is 4 * 2 = 8. 20 minus 8 is 10. Therefore, the answer is $10."  # Noise path
    ]
    consensus = PromptEngineeringPatterns.self_consistency_voting(sample_paths)
    print("\\nSelf-Consistency Aggregation:\\n" + "─" * 50)
    print(consensus)
`;

export default function PromptEngineeringPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-100 text-indigo-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-indigo-500"></span>
          Track 3 · Generative AI &amp; Large Language Models
        </span>
      </div>

      <h1>Prompt Engineering: In-Context Learning, Chain-of-Thought, and Search-Based Reasoning (Tree-of-Thoughts)</h1>

      <blockquote>
        <p>
          <em>"Prompt engineering is not mere phrasing; it is the programmatic specification of conditioning distributions over the pre-trained model's latent representations to steer inference without gradient updates."</em>
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Understand the theoretical mechanics of In-Context Learning (ICL) as implicit meta-optimization (von Oswald et al., 2023).</li>
        <li>Derive the Chain-of-Thought (CoT) reasoning paradigm and explain why it expands the model&apos;s computational budget per problem.</li>
        <li>Explain Self-Consistency decoding (Wang et al.) as marginalizing over latent reasoning paths.</li>
        <li>Analyze Tree-of-Thoughts (ToT) search exploration combining BFS/DFS tree search with deliberate valuation.</li>
        <li>Implement Chain-of-Thought templating and Self-Consistency majority voting from scratch in Python.</li>
      </ul>

      <h2>1 · In-Context Learning as Implicit Gradient Descent</h2>
      <p>
        In Brown et al. (GPT-3, 2020), foundation models demonstrated the ability to learn new tasks purely from a few conditioning demonstrations <code>(x_1, y_1), ..., (x_k, y_k)</code> in the prompt without parameter updates.
      </p>
      <p>
        Johannes von Oswald et al. (ICML 2023) and Dai et al. (2022) proved mathematically that <strong>in-context learning implements an implicit form of gradient descent</strong>:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p className="text-slate-400">// Theoretical Equivalence:</p>
        <p>Attention updates over prompt tokens act as an internal forward pass of a meta-optimizer,</p>
        <p>producing hidden state shifts ΔW_prompt that are mathematically equivalent</p>
        <p>to taking standard gradient steps on the demonstration loss.</p>
      </div>

      <h2>2 · Chain-of-Thought (CoT): Expanding Test-Time Compute</h2>
      <p>
        In standard autoregressive generation, generating a direct answer <code>y</code> immediately after question <code>x</code> constrains the model to <code>O(1)</code> sequential forward passes of compute.
      </p>
      <p>
        <strong>Chain-of-Thought (Wei et al., 2022)</strong> prompts the model to generate intermediate reasoning tokens <code>r_1, r_2, ..., r_k</code> before outputting the final answer:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>P(Answer | Question) = Σ_(r) P(Rationale | Question) · P(Answer | Question, Rationale)</p>
      </div>

      <p>
        Each generated rationale token acts as an external memory scratchpad that the Transformer can attend back to via self-attention, allocating <strong><code>O(k)</code> sequential layers of computational depth</strong> to complex arithmetic and logical deductions.
      </p>

      <h2>3 · Prompting Methodology Taxonomy</h2>
      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Technique</th>
              <th className="px-4 py-3 text-left">Mechanism</th>
              <th className="px-4 py-3 text-left">Benchmark Gains</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Few-Shot Prompting</td>
              <td className="px-4 py-3">Provides 3-5 demonstrations of input/output pairs to establish format and schema.</td>
              <td className="px-4 py-3">+15-20% on classification and JSON extraction.</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Zero-Shot CoT (Kojima 2022)</td>
              <td className="px-4 py-3">Appends trigger phrase <em>"Let's think step by step"</em> to the query prompt.</td>
              <td className="px-4 py-3">Jumps GSM8K arithmetic accuracy from 17.7% to 78.7%!</td>
            </tr>
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Self-Consistency (Wang 2022)</td>
              <td className="px-4 py-3">Samples 10 diverse reasoning paths at temperature 0.7; takes majority vote across answers.</td>
              <td className="px-4 py-3">+10-15% over greedy single-path Chain-of-Thought.</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Tree-of-Thoughts (Yao 2023)</td>
              <td className="px-4 py-3">Maintains a tree of thoughts; uses BFS/DFS with lookahead valuation and backtracking.</td>
              <td className="px-4 py-3">Solves Game of 24 (accuracy jumps from 4% in CoT to 74% in ToT).</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>4 · Python Implementation of CoT &amp; Self-Consistency</h2>
      <TerminalBlock language="python" filename="prompt_engineering_suite.py" code={code} />

      <h2>5 · Further Reading</h2>
      <ul>
        <li>Wei, J., et al. (2022). <em>Chain-of-Thought Prompting Elicits Reasoning in Large Language Models</em>. NeurIPS 2022.</li>
        <li>Wang, X., et al. (2022). <em>Self-Consistency Improves Chain of Thought Reasoning in Language Models</em>. ICLR 2023.</li>
        <li>Yao, S., et al. (2023). <em>Tree of Thoughts: Deliberate Problem Solving with Large Language Models</em>. NeurIPS 2023.</li>
        <li>von Oswald, J., et al. (2023). <em>Transformers learn in-context by gradient descent</em>. ICML 2023.</li>
      </ul>
    </article>
  );
}
