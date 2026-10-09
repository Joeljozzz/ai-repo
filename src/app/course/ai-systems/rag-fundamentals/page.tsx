"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import numpy as np

class VectorIndexHNSWSim:
    """
    Simulated Vector Database Index demonstrating:
    - Cosine Similarity & Inner Product metric spaces.
    - Dense Embedding Retrieval with Top-K Max-Heap ranking.
    - Reciprocal Rank Fusion (RRF) for Hybrid Dense + Sparse Search:
        RRF_Score(d) = Σ [ 1 / (k + rank_dense(d)) ] + Σ [ 1 / (k + rank_sparse(d)) ]
    """
    def __init__(self, d_dim: int = 128, rrf_k: int = 60):
        self.d_dim = d_dim
        self.rrf_k = rrf_k
        self.doc_ids = []
        self.embeddings = []
        self.doc_texts = []

    def insert(self, doc_id: str, text: str, embedding: np.ndarray):
        norm = np.linalg.norm(embedding)
        norm_emb = embedding / norm if norm > 0 else embedding
        self.doc_ids.append(doc_id)
        self.doc_texts.append(text)
        self.embeddings.append(norm_emb)

    def dense_search(self, query_emb: np.ndarray, top_k: int = 5):
        # Normalize query vector for cosine similarity via dot product
        q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-12)
        X = np.array(self.embeddings)  # (N, D)
        # Cosine similarity: S = X · q
        similarities = np.dot(X, q_norm)
        # Retrieve top indices sorted descending
        ranked_indices = np.argsort(-similarities)[:top_k]
        return [(self.doc_ids[idx], similarities[idx], idx) for idx in ranked_indices]

    def hybrid_search_rrf(self, query_emb: np.ndarray, sparse_matches: list, top_k: int = 3):
        """
        sparse_matches: list of doc_ids returned by lexical keyword search (e.g. BM25).
        Combines dense vector rank and sparse keyword rank using Reciprocal Rank Fusion.
        """
        dense_results = self.dense_search(query_emb, top_k=len(self.doc_ids))
        dense_ranks = {doc_id: rank + 1 for rank, (doc_id, _, _) in enumerate(dense_results)}
        sparse_ranks = {doc_id: rank + 1 for rank, doc_id in enumerate(sparse_matches)}

        all_docs = set(dense_ranks.keys()).union(set(sparse_ranks.keys()))
        rrf_scores = {}

        for doc_id in all_docs:
            score = 0.0
            if doc_id in dense_ranks:
                score += 1.0 / (self.rrf_k + dense_ranks[doc_id])
            if doc_id in sparse_ranks:
                score += 1.0 / (self.rrf_k + sparse_ranks[doc_id])
            rrf_scores[doc_id] = score

        sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return sorted_docs

if __name__ == "__main__":
    np.random.seed(42)
    db = VectorIndexHNSWSim(d_dim=64)

    # Insert sample documentation chunks
    corpus = [
        ("doc_1", "Gradient descent updates weights along the negative gradient."),
        ("doc_2", "AdamW decouples weight decay from adaptive moment estimation."),
        ("doc_3", "Transformers compute attention as softmax(Q K^T / sqrt(d)) V."),
        ("doc_4", "RAG retrieves external chunks to ground LLM generation in truth."),
    ]
    for doc_id, text in corpus:
        dummy_emb = np.random.randn(64)
        db.insert(doc_id, text, dummy_emb)

    q_vec = np.random.randn(64)
    # Simulate a sparse keyword match identifying doc_4 and doc_1
    sparse_hits = ["doc_4", "doc_1"]
    hybrid_top = db.hybrid_search_rrf(q_vec, sparse_hits, top_k=2)

    print("Hybrid Search RRF Top Results:")
    for doc_id, rrf_score in hybrid_top:
        print(f"  {doc_id} -> RRF Score: {rrf_score:.6f}")
`;

export default function RAGFundamentalsPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-teal-100 text-teal-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-teal-500"></span>
          Track 4 · AI Systems &amp; RAG Engineering
        </span>
      </div>

      <h1>Retrieval-Augmented Generation (RAG): Architecture, Chunking, and Hybrid Fusion</h1>

      <blockquote>
        <p>
          <em>"Parametric memory in neural weights is static, expensive to update, and prone to hallucination. Non-parametric retrieval grounds large language models in dynamic, verifiable external truth."</em> — Patrick Lewis et al. (Meta AI, 2020)
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Understand the tri-part RAG lifecycle: Ingestion (Indexing), Retrieval, and Augmented Generation.</li>
        <li>Analyze chunking trade-offs: Fixed-size with overlap, Semantic chunking, and Recursive Character Splitting.</li>
        <li>Derive Reciprocal Rank Fusion (RRF) for combining sparse lexical search (BM25) with dense semantic vector search.</li>
        <li>Explain Cross-Encoder re-ranking vs. Bi-Encoder retrieval architectures.</li>
        <li>Implement a functional hybrid vector retrieval engine with RRF from first principles in NumPy.</li>
      </ul>

      <h2>1 · Parametric vs. Non-Parametric Memory</h2>
      <p>
        In standard generative LLMs, knowledge is stored exclusively in <strong>parametric memory</strong> (the neural network weights <code>θ</code> learned during pre-training). This exhibits fundamental engineering liabilities:
      </p>
      <ul>
        <li><strong>Knowledge Cutoffs:</strong> The model has no knowledge of events after its training date.</li>
        <li><strong>Hallucination:</strong> When parameters lack certainty, the model fabricates statistically plausible but factually incorrect assertions.</li>
        <li><strong>Enterprise Privacy:</strong> Proprietary internal databases cannot be safely baked into shared foundation model weights.</li>
      </ul>
      <p>
        <strong>RAG (Lewis et al., 2020)</strong> couples the parametric model with <strong>non-parametric memory</strong> (a searchable vector database), retrieving top-<code>k</code> relevant document chunks at inference time and injecting them into the LLM context prompt.
      </p>

      <h2>2 · The Data Ingestion &amp; Chunking Science</h2>
      <p>
        Chunking transforms arbitrary-length source documents into coherent passages suitable for embedding models.
      </p>

      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Chunking Strategy</th>
              <th className="px-4 py-3 text-left">Mechanism</th>
              <th className="px-4 py-3 text-left">Trade-Offs</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Fixed-Size with Overlap</td>
              <td className="px-4 py-3">Split every <code>C</code> tokens with <code>O</code> overlap (e.g. 512 tokens with 10% overlap).</td>
              <td className="px-4 py-3">Simple and fast. Arbitrarily severs sentences, paragraphs, or tables mid-thought.</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Recursive Character</td>
              <td className="px-4 py-3">Attempts splitting on double newlines <code>\n\n</code>, then single newlines, spaces, and finally characters.</td>
              <td className="px-4 py-3">Preserves structural paragraph integrity. The recommended general baseline.</td>
            </tr>
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Semantic Chunking</td>
              <td className="px-4 py-3">Embeds consecutive sentences; splits where cosine distance between adjacent sentences exceeds a threshold.</td>
              <td className="px-4 py-3">Produces semantically coherent passages. Computationally expensive (requires N embedding calls).</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>3 · Bi-Encoders vs. Cross-Encoders</h2>
      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Architecture</th>
              <th className="px-4 py-3 text-left">Forward Pass Computation</th>
              <th className="px-4 py-3 text-left">Latency &amp; Scalability</th>
              <th className="px-4 py-3 text-left">Role in Production Pipeline</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Bi-Encoder</td>
              <td className="px-4 py-3"><code>Score = E(query) · E(document)^T</code>. Documents embedded offline; dot product at query time.</td>
              <td className="px-4 py-3">Sub-millisecond via HNSW vector index across millions of documents.</td>
              <td className="px-4 py-3">First-stage retrieval: filter 10,000,000 documents down to Top-100 candidates.</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Cross-Encoder (Re-ranker)</td>
              <td className="px-4 py-3"><code>Score = Model( [CLS] query [SEP] document [SEP] )</code>. Full cross-attention between every query token and document token.</td>
              <td className="px-4 py-3">High latency (tens of ms per pair); cannot be pre-indexed offline.</td>
              <td className="px-4 py-3">Second-stage re-ranking: score the Top-100 candidates down to Top-5 passages for LLM prompt.</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>4 · Hybrid Search &amp; Reciprocal Rank Fusion (RRF)</h2>
      <p>
        Dense vector embeddings understand conceptual synonyms (e.g., matching "automobile repair" with "car maintenance"), but fail on exact alphanumeric keyword lookups (e.g. error codes, product serial numbers <code>ERR_403_X9</code>).
      </p>
      <p>
        Production RAG combines <strong>BM25 lexical search</strong> with <strong>dense vector search</strong> using <strong>Reciprocal Rank Fusion (RRF)</strong> (Cormack et al., 2009):
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>RRF_Score(d ∈ D) = Σ_(m ∈ M) [ 1 / (k + rank_m(d)) ]</p>
        <br />
        <p>where:</p>
        <p>  M      = Set of retrieval systems (Dense Vector, Sparse BM25)</p>
        <p>  rank_m = Rank of document d in system m (1, 2, 3, ...)</p>
        <p>  k      = Smoothing constant (empirically set to 60 to prevent top ranks from dominating)</p>
      </div>

      <h2>5 · Python Implementation: Vector Retrieval &amp; Hybrid RRF</h2>
      <TerminalBlock language="python" filename="hybrid_rag_rrf.py" code={code} />

      <h2>6 · Further Reading</h2>
      <ul>
        <li>Lewis, P., et al. (2020). <em>Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks</em>. NeurIPS 2020.</li>
        <li>Cormack, G. V., Clarke, C. L., &amp; Buettcher, S. (2009). <em>Reciprocal rank fusion outperformscondorcet and individual rank learning methods</em>. SIGIR &apos;09, 758–759.</li>
        <li>Karpukhin, V., et al. (2020). <em>Dense Passage Retrieval for Open-Domain Question Answering</em>. EMNLP 2020.</li>
      </ul>
    </article>
  );
}
