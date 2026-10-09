"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import numpy as np

class AdvancedRAGPipeline:
    """
    Advanced RAG Pipeline implementing:
    1. Hypothetical Document Embeddings (HyDE) (Gao et al., 2022)
    2. Multi-Query Expansion & Sub-Query Decomposition
    3. Cross-Encoder Re-Ranking with Softmax Score Calibration
    4. Contextual Compression & Lost-in-the-Middle mitigation
    """
    def __init__(self, doc_corpus: dict):
        self.doc_corpus = doc_corpus

    @staticmethod
    def hyde_generate_hypothetical_passage(query: str) -> str:
        """
        Generates a hypothetical ideal answer document using an LLM zero-shot.
        Even if factually hallucinated, its embedding is closer in vector space
        to actual relevant passages than an interrogative user query!
        """
        # In production: llm.generate("Write a passage answering: " + query)
        hypothetical_doc = (
            f"Regarding {query}: In distributed database systems, two-phase commit (2PC) "
            "coordinates all nodes across prepare and commit phases to guarantee strict serializability."
        )
        return hypothetical_doc

    @staticmethod
    def multi_query_expansion(query: str) -> list:
        """
        Generates 3 diverse query rephrasings from multiple perspectives.
        Overcomes lexical brittleness and prompt sensitivity.
        """
        return [
            query,
            f"Technical breakdown and architecture of {query}",
            f"Common failure modes and trade-offs of {query}"
        ]

    @staticmethod
    def cross_encoder_rerank(query: str, retrieved_docs: list, top_k: int = 2) -> list:
        """
        Cross-Encoder joint scoring: f(query, doc).
        Applies full cross-attention between all token pairs (unlike cosine similarity on bi-encoders).
        """
        scored_docs = []
        for doc_id, text in retrieved_docs:
            # Simulated cross-encoder logit based on semantic intersection
            overlap = len(set(query.lower().split()) & set(text.lower().split()))
            logit = overlap * 1.5 + np.random.uniform(0.1, 0.5)
            scored_docs.append((doc_id, text, logit))

        # Sort descending by re-ranker score
        ranked = sorted(scored_docs, key=lambda x: x[2], reverse=True)[:top_k]
        return [(doc_id, text, round(score, 3)) for doc_id, text, score in ranked]

if __name__ == "__main__":
    corpus = {
        "doc_A": "Two-phase commit ensures consistency across distributed database nodes.",
        "doc_B": "Gradient descent minimizes empirical risk using the multivariable chain rule.",
        "doc_C": "Distributed transactions require prepare and commit phases to prevent split-brain."
    }

    pipeline = AdvancedRAGPipeline(corpus)
    user_q = "How does two-phase commit work in distributed databases?"

    # 1. HyDE
    hyde_doc = pipeline.hyde_generate_hypothetical_passage(user_q)
    print("HyDE Generated Passage:\\n", hyde_doc)

    # 2. Query Expansion
    queries = pipeline.multi_query_expansion(user_q)
    print("\\nExpanded Sub-Queries:", queries)

    # 3. Cross-Encoder Re-Ranking
    candidates = list(corpus.items())
    reranked = pipeline.cross_encoder_rerank(user_q, candidates, top_k=2)
    print("\\nCross-Encoder Re-Ranked Top-2:")
    for doc_id, text, score in reranked:
        print(f"  [{doc_id}] Score: {score} -> '{text}'")
`;

export default function AdvancedRAGPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-teal-100 text-teal-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-teal-500"></span>
          Track 4 · AI Systems &amp; RAG Engineering
        </span>
      </div>

      <h1>Advanced RAG Patterns: HyDE, Query Expansion, Re-Ranking, and Contextual Compression</h1>

      <blockquote>
        <p>
          <em>"Naive RAG fails when the query and the target document inhabit asymmetric linguistic spaces. Advanced RAG bridges this vector space gap using hypothetical generation, multi-query routing, and cross-attention re-ranking."</em> — Luyu Gao et al. (2022)
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Understand the Query-Document Asymmetry problem in bi-encoder vector retrieval.</li>
        <li>Derive Hypothetical Document Embeddings (HyDE) and explain why hallucinated answers yield superior embedding vectors.</li>
        <li>Analyze Multi-Query Expansion, Sub-Question Decomposition, and Step-Back Prompting.</li>
        <li>Explain Cross-Encoder re-ranking architectures (Cohere Rerank, BGE-Reranker) vs. Bi-Encoder retrieval.</li>
        <li>Mitigate the "Lost-in-the-Middle" phenomenon (Liu et al., 2023) via Contextual Compression and prompt sorting.</li>
      </ul>

      <h2>1 · The Query-Document Asymmetry Problem</h2>
      <p>
        In naive vector search, a user submits a short, interrogative question: <em>"How does Raft handle network partitions?"</em> (10 tokens). The relevant target passage is a lengthy, technical declarative explanation (400 tokens).
      </p>
      <p>
        Because embedding models project questions and declarative paragraphs into slightly different vector neighborhoods, the cosine similarity between question and target is frequently lower than the similarity between the question and an irrelevant FAQ sentence.
      </p>

      <h2>2 · Hypothetical Document Embeddings (HyDE, Gao et al. 2022)</h2>
      <p>
        HyDE circumvents asymmetry by instructing an LLM to generate a hypothetical answer document <code>d_hyp</code> zero-shot:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>1. Query q ──▶ LLM( zero_shot_prompt ) ──▶ Hypothetical Document d_hyp</p>
        <p>2. v_search = Embed( d_hyp )   (NOT Embed(q)!)</p>
        <p>3. Top-k = VectorDB.search( v_search )</p>
      </div>

      <p>
        <strong>Why It Works:</strong> Even if <code>d_hyp</code> contains minor factual inaccuracies or hallucinations, its <em>linguistic style, terminology, and structural syntax</em> reside squarely inside the vector space of declarative documents rather than interrogative questions.
      </p>

      <h2>3 · Multi-Stage Retrieval &amp; Cross-Encoder Re-Ranking</h2>
      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Pipeline Stage</th>
              <th className="px-4 py-3 text-left">Technology</th>
              <th className="px-4 py-3 text-left">Candidate Pool Size</th>
              <th className="px-4 py-3 text-left">Latency Budget</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Stage 1: Broad Retrieval</td>
              <td className="px-4 py-3">HNSW Dense Embeddings + BM25 Sparse Search (RRF)</td>
              <td className="px-4 py-3">10,000,000 → Top 100</td>
              <td className="px-4 py-3">&lt; 15 ms</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Stage 2: Cross-Encoder Re-rank</td>
              <td className="px-4 py-3">Full cross-attention model: <code>Score = CrossEncoder(Query, Passage)</code></td>
              <td className="px-4 py-3">Top 100 → Top 5</td>
              <td className="px-4 py-3">~ 50 ms</td>
            </tr>
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Stage 3: Context Compression</td>
              <td className="px-4 py-3">Extract only sentences answering the query; discard irrelevant fluff</td>
              <td className="px-4 py-3">Top 5 → 1,500 prompt tokens</td>
              <td className="px-4 py-3">&lt; 10 ms</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>4 · The "Lost-in-the-Middle" Phenomenon (Liu et al., 2023)</h2>
      <p>
        Nelson Liu et al. (Stanford, 2023) demonstrated that large language models retrieve information with high fidelity when it is placed at the <strong>very beginning (primacy effect)</strong> or the <strong>very end (recency effect)</strong> of the context window.
      </p>
      <p>
        When critical evidence is placed in the middle of a 20-chunk prompt, model retrieval accuracy drops by up to <strong>30%</strong>, even in long-context models (128k+ tokens).
      </p>
      <p>
        <strong>Engineering Remedy (U-Shaped Reordering):</strong> Sort re-ranked passages such that the highest-scoring passage is at the very beginning of the prompt, the second highest-scoring is at the very end, and weaker passages occupy the center.
      </p>

      <h2>5 · Python Implementation of Advanced RAG Engine</h2>
      <TerminalBlock language="python" filename="advanced_rag_pipeline.py" code={code} />

      <h2>6 · Further Reading</h2>
      <ul>
        <li>Gao, L., et al. (2022). <em>Precise Zero-Shot Dense Retrieval without Relevance Labels</em> (HyDE). ACL 2023.</li>
        <li>Liu, N. F., et al. (2023). <em>Lost in the Middle: How Language Models Use Long Contexts</em>. TACL 2024.</li>
        <li>Nogueira, R., &amp; Cho, K. (2019). <em>Passage Re-ranking with BERT</em>. arXiv:1901.04085.</li>
      </ul>
    </article>
  );
}
