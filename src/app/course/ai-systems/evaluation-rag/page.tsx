"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import numpy as np

class RAGASEvaluationEngine:
    """
    RAG Triad and RAGAS Framework Implementation (Es et al., 2023):
    1. Faithfulness: Is the answer grounded exclusively in retrieved context?
    2. Answer Relevance: Does the generated answer address the user query?
    3. Context Precision: Are the ground-truth relevant chunks ranked at the top?
    4. Context Recall: Were all necessary facts successfully retrieved?
    """
    @staticmethod
    def compute_context_precision(retrieved_chunk_relevance: list) -> float:
        """
        Mean Average Precision (mAP) over retrieved context ranking.
        retrieved_chunk_relevance: binary list indicating if chunk k is relevant [1, 0, 1, ...]
        Context Precision@K = (1 / Total Relevant) * Σ (Precision@k * rel(k))
        """
        total_relevant = sum(retrieved_chunk_relevance)
        if total_relevant == 0:
            return 0.0

        precisions = []
        running_relevant = 0
        for k, rel in enumerate(retrieved_chunk_relevance, start=1):
            if rel == 1:
                running_relevant += 1
                precisions.append(running_relevant / k)

        return float(np.sum(precisions) / total_relevant)

    @staticmethod
    def compute_context_recall(ground_truth_claims: list, retrieved_context_text: str) -> float:
        """
        Fraction of ground-truth statements supported by retrieved context:
        Context Recall = |Supported Claims| / |Total Ground Truth Claims|
        """
        supported = 0
        for claim in ground_truth_claims:
            # Simulated semantic verification (in production: LLM prompt check)
            if any(term in retrieved_context_text.lower() for term in claim.lower().split()):
                supported += 1
        return supported / len(ground_truth_claims) if ground_truth_claims else 0.0

    @staticmethod
    def compute_faithfulness(generated_claims: list, retrieved_context_text: str) -> float:
        """
        Groundedness: |Claims Inferable from Context| / |Total Generated Claims|
        Detects ungrounded hallucinations!
        """
        inferable = 0
        for claim in generated_claims:
            # Check if claim tokens reside inside retrieved context
            overlap = len(set(claim.lower().split()) & set(retrieved_context_text.lower().split()))
            if overlap >= 3:
                inferable += 1
        return inferable / len(generated_claims) if generated_claims else 0.0

if __name__ == "__main__":
    # Test Context Precision@5 where 1st and 3rd chunks are relevant
    # Ideal ranking places 1s as early as possible
    retrieval_binary_feedback = [1, 0, 1, 0, 0]
    precision_score = RAGASEvaluationEngine.compute_context_precision(retrieval_binary_feedback)
    print(f"Context Precision@5: {precision_score:.4f} (Precision@1=1.0, Precision@3=0.67 -> mAP = 0.833)")

    ground_truth = [
        "AdamW decouples weight decay from gradient updates",
        "Learning rate is multiplied by weight decay directly"
    ]
    retrieved_text = "AdamW optimizer decouples weight decay directly from gradient updates."
    recall_score = RAGASEvaluationEngine.compute_context_recall(ground_truth, retrieved_text)
    print(f"Context Recall:       {recall_score:.2f}")

    generated_statements = [
        "AdamW decouples weight decay",
        "It was created by Albert Einstein in 1905"  # Hallucinated statement!
    ]
    faithfulness = RAGASEvaluationEngine.compute_faithfulness(generated_statements, retrieved_text)
    print(f"Faithfulness Score:   {faithfulness:.2f} (50% ungrounded hallucination detected!)")
`;

export default function EvaluationRAGPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-teal-100 text-teal-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-teal-500"></span>
          Track 4 · AI Systems &amp; RAG Engineering
        </span>
      </div>

      <h1>Evaluating RAG Systems: The RAG Triad, RAGAS Metrics, and LLM-as-a-Judge</h1>

      <blockquote>
        <p>
          <em>"Traditional NLP metrics like BLEU and ROUGE evaluate surface n-gram overlap, completely failing to measure factual groundedness or retrieval precision. Modern RAG evaluation requires reference-free, component-wise decomposition into Faithfulness, Relevance, and Context Precision."</em> — Shahul Es et al. (2023)
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Understand why classical n-gram overlap metrics (BLEU, ROUGE) fail for generative AI pipelines.</li>
        <li>Derive the RAG Triad: Context Relevance, Groundedness (Faithfulness), and Answer Relevance.</li>
        <li>Formulate the RAGAS framework mathematical metrics: Context Precision (mAP), Context Recall, and Faithfulness.</li>
        <li>Explain the mechanics and bias mitigation strategies of the <strong>LLM-as-a-Judge</strong> paradigm (Zheng et al., 2023).</li>
        <li>Implement a full programmatic RAG evaluation harness from scratch in Python.</li>
      </ul>

      <h2>1 · The Failure of BLEU and ROUGE in RAG</h2>
      <p>
        In traditional machine translation and summarization, BLEU and ROUGE measure lexical surface overlap. In production RAG systems, they exhibit catastrophic vulnerabilities:
      </p>
      <ul>
        <li><strong>Paraphrasing Penalty:</strong> An answer can be 100% factually correct but share zero vocabulary with the reference answer, receiving a BLEU score of 0.0.</li>
        <li><strong>Hallucination Blindness:</strong> Adding a single word ("Alice did <em>not</em> murder Bob") flips the factual validity entirely while preserving 95% n-gram overlap.</li>
      </ul>

      <h2>2 · The RAG Triad Architecture</h2>
      <p>
        The <strong>RAG Triad (TruLens)</strong> isolates evaluation into three distinct sub-components:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>                  [ USER QUERY ]</p>
        <p>                   /          \</p>
        <p>      Context Relevance      Answer Relevance</p>
        <p>                 /              \</p>
        <p>[ RETRIEVED CONTEXT ] ──Faithfulness──▶ [ GENERATED ANSWER ]</p>
      </div>

      <ol>
        <li><strong>Context Relevance:</strong> Does the retrieved context chunk contain the information necessary to address the query? (Evaluates the retrieval index).</li>
        <li><strong>Faithfulness (Groundedness):</strong> Is every claim in the generated answer directly inferable from the retrieved context? (Detects hallucinations).</li>
        <li><strong>Answer Relevance:</strong> Does the final answer directly address the original question without irrelevant digressions? (Evaluates generation instruction-following).</li>
      </ol>

      <h2>3 · The RAGAS Mathematical Metrics (Es et al., 2023)</h2>
      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Metric</th>
              <th className="px-4 py-3 text-left">Mathematical Formulation</th>
              <th className="px-4 py-3 text-left">Target Measurement</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Faithfulness</td>
              <td className="px-4 py-3 font-mono">|V_claims(Answer, Context)| / |Total_claims(Answer)|</td>
              <td className="px-4 py-3">Measures hallucination rate; penalizes claims unbacked by context.</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Context Precision</td>
              <td className="px-4 py-3 font-mono">Σ_(k=1)^K [ (Precision@k · v_k) ] / Total_Relevant_Chunks</td>
              <td className="px-4 py-3">Mean Average Precision (mAP); rewards placing relevant chunks at top ranks.</td>
            </tr>
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Context Recall</td>
              <td className="px-4 py-3 font-mono">|Attributed_claims(Ground_Truth, Context)| / |Total_claims(Ground_Truth)|</td>
              <td className="px-4 py-3">Ensures retriever caught all necessary facts to answer comprehensively.</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>4 · LLM-as-a-Judge: Protocols and Bias Mitigations</h2>
      <p>
        Using a frontier model (GPT-4, Claude 3.5 Sonnet) as an automated evaluator correlates with human expert judgment at &gt;85% agreement (Zheng et al., 2023). However, systems must mitigate four systematic judge biases:
      </p>
      <ul>
        <li><strong>Position Bias:</strong> Models favor candidate answers placed in Position A over Position B. <em>Remedy:</em> Run evaluation twice swapping position ordering, averaging scores.</li>
        <li><strong>Verbosity Bias:</strong> Judges favor longer, verbose responses regardless of information density. <em>Remedy:</em> Instruct judge explicitly to penalize fluff and wordiness.</li>
        <li><strong>Self-Enhancement Bias:</strong> Models favor answers generated by themselves. <em>Remedy:</em> Anonymize generator identities in judge prompts.</li>
      </ul>

      <h2>5 · Python Implementation of RAGAS Evaluation Engine</h2>
      <TerminalBlock language="python" filename="ragas_evaluation_engine.py" code={code} />

      <h2>6 · Further Reading</h2>
      <ul>
        <li>Es, S., et al. (2023). <em>RAGAS: Automated Evaluation of Retrieval Augmented Generation</em>. EMNLP 2023.</li>
        <li>Zheng, L., et al. (2023). <em>Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena</em>. NeurIPS 2023.</li>
        <li>Min, S., et al. (2023). <em>FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation</em>. EMNLP 2023.</li>
      </ul>
    </article>
  );
}
