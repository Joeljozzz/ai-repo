"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import numpy as np

class MatryoshkaEmbeddings:
    """
    Matryoshka Representation Learning (MRL) (Kusupati et al., NeurIPS 2022).
    Enables vector truncation to arbitrary lower dimensions d_m ∈ {64, 128, 256, 512, 1024}
    while preserving up to 99% of downstream cosine similarity retrieval accuracy.
    
    Multi-Scale Contrastive Loss:
      L_MRL = Σ_(m=1)^M w_m · L_InfoNCE( z_{1:d_m}, z_{1:d_m}^+ )
    """
    def __init__(self, full_dim: int = 1024, nested_dims: list = [64, 128, 256, 512, 1024]):
        self.full_dim = full_dim
        self.nested_dims = nested_dims

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def truncate_and_normalize(self, full_embedding: np.ndarray, target_dim: int) -> np.ndarray:
        """
        Extract the first target_dim coordinates and re-normalize to unit length.
        """
        assert target_dim <= self.full_dim, "Target dimension exceeds full dimension."
        truncated = full_embedding[:target_dim]
        norm = np.linalg.norm(truncated)
        return truncated / norm if norm > 0 else truncated

    def evaluate_retrieval_fidelity(self, query_full: np.ndarray, doc_full: np.ndarray):
        """
        Compares cosine similarity across all nested Matryoshka sub-dimensions.
        """
        results = {}
        for dim in self.nested_dims:
            q_trunc = self.truncate_and_normalize(query_full, dim)
            d_trunc = self.truncate_and_normalize(doc_full, dim)
            sim = self.cosine_similarity(q_trunc, d_trunc)
            results[dim] = sim
        return results

if __name__ == "__main__":
    np.random.seed(42)
    # Simulate a correlated high-dimensional embedding pair (similar concepts)
    base_signal = np.random.randn(1024)
    # Target document has identical core semantic signal plus minor noise
    doc_embedding = base_signal + np.random.randn(1024) * 0.3
    query_embedding = base_signal + np.random.randn(1024) * 0.3

    mrl = MatryoshkaEmbeddings(full_dim=1024, nested_dims=[64, 128, 256, 512, 1024])
    similarities = mrl.evaluate_retrieval_fidelity(query_embedding, doc_embedding)

    print("Matryoshka Truncation Cosine Similarity Preservation:")
    full_sim = similarities[1024]
    for dim, sim in similarities.items():
        retention = (sim / full_sim) * 100.0
        bytes_per_vec = dim * 4  # FP32
        print(f"  Dimension {dim:4d} | Vector Size: {bytes_per_vec:4d} bytes | Cosine Sim: {sim:.4f} ({retention:.1f}% fidelity)")
`;

export default function EmbeddingsPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-100 text-indigo-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-indigo-500"></span>
          Track 3 · Generative AI &amp; Large Language Models
        </span>
      </div>

      <h1>Dense Embeddings &amp; Vector Spaces: Contrastive Learning (InfoNCE) and Matryoshka Representation Learning</h1>

      <blockquote>
        <p>
          <em>"Dense embeddings project discrete linguistic symbols into a continuous semantic manifold where geometric distance directly mirrors semantic similarity. Matryoshka embeddings structure this manifold hierarchically, allowing vectors to be sliced to 1/16th their size with negligible loss in retrieval fidelity."</em> — Aditya Kusupati et al. (NeurIPS 2022)
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Understand the theoretical evolution from sparse bag-of-words (TF-IDF) to dense neural embeddings.</li>
        <li>Derive the InfoNCE Contrastive Loss objective used in dual-encoder embedding models (SimCLR, CLIP, BGE, OpenAI).</li>
        <li>Explain Matryoshka Representation Learning (MRL) and the nested information theory behind adaptive vector truncation.</li>
        <li>Analyze the trade-offs of embedding dimensions: search speed, memory footprint, and retrieval recall.</li>
        <li>Implement Matryoshka vector truncation and cosine similarity preservation analysis in Python.</li>
      </ul>

      <h2>1 · From Sparse Lexical Matching to Dense Manifolds</h2>
      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Representation</th>
              <th className="px-4 py-3 text-left">Vector Dimensionality</th>
              <th className="px-4 py-3 text-left">Semantic Relationship</th>
              <th className="px-4 py-3 text-left">Primary Limitation</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Sparse (TF-IDF / BM25)</td>
              <td className="px-4 py-3"><code>|V|</code> (Vocabulary size, ~50,000 to 500,000)</td>
              <td className="px-4 py-3">Orthogonal: <code>x_i · x_j = 0</code> if no exact token overlap.</td>
              <td className="px-4 py-3">Synonym blindness: "cardiac arrest" and "heart attack" have zero overlap.</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Dense Neural Embeddings</td>
              <td className="px-4 py-3">Dense continuous <code>ℝ^d</code> (typically <code>d ∈ [384, 1536]</code>)</td>
              <td className="px-4 py-3">Geometric alignment: <code>cos θ ≈ 1.0</code> for semantically related passages.</td>
              <td className="px-4 py-3">High memory consumption; struggles with exact alphanumeric product ID lookups.</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>2 · The InfoNCE Contrastive Learning Objective</h2>
      <p>
        Modern dense embedding models (e.g., E5, BGE, text-embedding-3) are trained using a dual-encoder architecture with the <strong>InfoNCE (Information Noise-Contrastive Estimation) loss</strong> (van den Oord et al., 2018):
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>L_InfoNCE = - ln [ exp( sim(q, d^+) / τ ) / ( exp( sim(q, d^+) / τ ) + Σ_(j=1)^K exp( sim(q, d_j^-) / τ ) ) ]</p>
        <br />
        <p>where:</p>
        <p>  sim(u, v) = (u · v) / (‖u‖ ‖v‖)  (Cosine similarity)</p>
        <p>  q         = Query representation vector</p>
        <p>  d^+       = True positive passage (relevant document)</p>
        <p>  d_j^-     = In-batch negative passages (unrelated documents)</p>
        <p>  τ         = Temperature hyperparameter (controls sharpness of softmax distribution)</p>
      </div>

      <p>
        <strong>Theoretical Connection:</strong> Minimizing InfoNCE maximizes a lower bound on the <strong>Mutual Information</strong> <code>I(Q; D^+)</code> between query and relevant passage, pulling matching pairs together in vector space while pushing unrelated pairs apart.
      </p>

      <h2>3 · Matryoshka Representation Learning (MRL, Kusupati et al. 2022)</h2>
      <p>
        Standard embedding models produce rigid, fixed-dimensional vectors (e.g. 1536 dimensions). If an engineer needs lower memory consumption or faster vector database latency, they must re-train an entirely new embedding model from scratch.
      </p>
      <p>
        Inspired by Russian nesting dolls (Matryoshka dolls), <strong>MRL</strong> trains the model such that the <strong>first <code>k</code> dimensions</strong> of the embedding form a fully functional, self-sufficient representation for any subset of dimensions <code>k ∈ &#123;64, 128, 256, 512, 1024&#125;</code>:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>L_MRL = Σ_(m=1)^M c_m · L_InfoNCE( q_(1:d_m), d^+_(1:d_m) )</p>
      </div>

      <h3>Production Engineering Architecture: Adaptive Search</h3>
      <ol>
        <li><strong>Coarse First-Stage Retrieval:</strong> Truncate embeddings to <code>d = 64</code> or <code>128</code>. Perform search across 10 million vectors in memory at 12x higher speed and 1/16th memory cost to identify the top 500 candidates.</li>
        <li><strong>Fine Second-Stage Re-ranking:</strong> Re-score the 500 candidates using the full <code>1024</code>-dimensional embeddings to guarantee maximum semantic precision.</li>
      </ol>

      <h2>4 · Python Implementation of Matryoshka Truncation Engine</h2>
      <TerminalBlock language="python" filename="matryoshka_embeddings.py" code={code} />

      <h2>5 · Further Reading</h2>
      <ul>
        <li>Kusupati, A., et al. (2022). <em>Matryoshka Representation Learning</em>. Advances in Neural Information Processing Systems (NeurIPS 35).</li>
        <li>van den Oord, A., Li, Y., &amp; Vinyals, O. (2018). <em>Representation Learning with Contrastive Predictive Coding</em>. arXiv:1807.03748.</li>
        <li>Reimers, N., &amp; Gurevych, I. (2019). <em>Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks</em>. EMNLP 2019.</li>
      </ul>
    </article>
  );
}
