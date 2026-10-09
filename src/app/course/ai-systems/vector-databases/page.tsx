"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import heapq
import numpy as np
from typing import List, Tuple, Dict, Set

class HNSWNode:
    def __init__(self, doc_id: str, vec: np.ndarray, level: int):
        self.doc_id = doc_id
        self.vec = vec
        self.level = level
        # Adjacency list: level -> list of neighbor node references
        self.neighbors: Dict[int, List['HNSWNode']] = {l: [] for l in range(level + 1)}

class HierarchicalNSW:
    """
    Hierarchical Navigable Small World (HNSW) Index (Malkov & Yashunin, 2018).
    - Multi-layer graph representation with logarithmic search complexity O(log N).
    - Top layers contain long-range highway links (skip-list concept).
    - Bottom layer (Level 0) contains dense local connectivity.
    - Metric: Cosine Similarity / Inner Product.
    """
    def __init__(self, d_dim: int, M: int = 16, ef_construction: int = 64, mL: float = 1.0 / np.log(16)):
        self.d_dim = d_dim
        self.M = M                          # Max connections per node
        self.ef_construction = ef_construction  # Beam search exploration width
        self.mL = mL                        # Normalization factor for level generation
        self.entry_point: HNSWNode = None
        self.max_level = -1

    def _dist(self, v1: np.ndarray, v2: np.ndarray) -> float:
        # Cosine distance: 1.0 - (v1 · v2) / (||v1|| * ||v2||)
        denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12
        return 1.0 - np.dot(v1, v2) / denom

    def _sample_level(self) -> int:
        # Exponential distribution: floor(-ln(uniform(0, 1)) * mL)
        return int(-np.log(np.random.uniform(0.0001, 1.0)) * self.mL)

    def search_layer(self, query: np.ndarray, entry_points: List[HNSWNode], ef: int, level: int) -> List[Tuple[float, HNSWNode]]:
        """
        Greedy beam search at a specific graph layer.
        Maintains candidate min-heap and visited set.
        """
        v = entry_points[0]
        visited: Set[HNSWNode] = {v}
        # Candidates min-heap: (dist, node)
        candidates = [(self._dist(query, v.vec), v)]
        # Results max-heap (negated dist): (-dist, node)
        w = [(-self._dist(query, v.vec), v)]

        while len(candidates) > 0:
            c_dist, c_node = heapq.heappop(candidates)
            furthest_result_dist = -w[0][0]

            if c_dist > furthest_result_dist:
                break

            for neighbor in c_node.neighbors.get(level, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    n_dist = self._dist(query, neighbor.vec)
                    furthest_result_dist = -w[0][0]

                    if n_dist < furthest_result_dist or len(w) < ef:
                        heapq.heappush(candidates, (n_dist, neighbor))
                        heapq.heappush(w, (-n_dist, neighbor))
                        if len(w) > ef:
                            heapq.heappop(w)

        return sorted([(-neg_d, node) for neg_d, node in w], key=lambda x: x[0])

    def insert(self, doc_id: str, vec: np.ndarray):
        level = self._sample_level()
        new_node = HNSWNode(doc_id, vec, level)

        if self.entry_point is None:
            self.entry_point = new_node
            self.max_level = level
            return

        curr_obj = self.entry_point
        curr_dist = self._dist(vec, curr_obj.vec)

        # Phase 1: Fast greedy traversal down to node's level
        for l in range(self.max_level, level, -1):
            changed = True
            while changed:
                changed = False
                for neighbor in curr_obj.neighbors.get(l, []):
                    d = self._dist(vec, neighbor.vec)
                    if d < curr_dist:
                        curr_dist = d
                        curr_obj = neighbor
                        changed = True

        # Phase 2: Beam search and bi-directional edge insertion
        entry_nodes = [curr_obj]
        for l in range(min(self.max_level, level), -1, -1):
            neighbors_found = self.search_layer(vec, entry_nodes, ef=self.ef_construction, level=l)
            # Connect up to M nearest neighbors
            selected = neighbors_found[:self.M]
            for d, n in selected:
                new_node.neighbors[l].append(n)
                n.neighbors[l].append(new_node)
            entry_nodes = [node for _, node in neighbors_found]

        if level > self.max_level:
            self.max_level = level
            self.entry_point = new_node

if __name__ == "__main__":
    np.random.seed(42)
    index = HierarchicalNSW(d_dim=16, M=8, ef_construction=32)

    # Insert 100 high-dimensional document vectors
    for i in range(100):
        vec = np.random.randn(16)
        index.insert(f"doc_{i}", vec)

    q = np.random.randn(16)
    results = index.search_layer(q, [index.entry_point], ef=5, level=0)
    print(f"HNSW Top Match: {results[0][1].doc_id} with Cosine Distance {results[0][1].vec[:3]}...")
`;

export default function VectorDatabasesPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-teal-100 text-teal-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-teal-500"></span>
          Track 4 · AI Systems &amp; RAG Engineering
        </span>
      </div>

      <h1>Vector Databases: HNSW Graph Indexing, Product Quantization (IVF-PQ), and Metric Spaces</h1>

      <blockquote>
        <p>
          <em>"Exact nearest neighbor search scales as O(N · d), becoming computationally impossible across billions of vectors. HNSW solves this by organizing high-dimensional space into multi-layer skip graphs, achieving logarithmic query times with over 99% recall."</em> — Yury Malkov &amp; Dmitry Yashunin (2018)
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Derive distance metric spaces: Euclidean (L2), Inner Product (Dot Product), and Cosine Distance.</li>
        <li>Explain the Hierarchical Navigable Small World (HNSW) multi-layer skip-graph architecture.</li>
        <li>Understand Inverted File Indexing with Product Quantization (IVF-PQ) for sub-byte memory compression.</li>
        <li>Contrast vector database architectures: Pinecone, Qdrant, Weaviate, Milvus, and pgvector.</li>
        <li>Implement an HNSW index with beam-search layer routing from scratch in Python.</li>
      </ul>

      <h2>1 · Metric Spaces in High-Dimensional Vector Search</h2>
      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Metric</th>
              <th className="px-4 py-3 text-left">Mathematical Formula</th>
              <th className="px-4 py-3 text-left">Characteristics &amp; Properties</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Euclidean Distance (L2)</td>
              <td className="px-4 py-3 font-mono">‖u - v‖₂ = √( Σ (u_i - v_i)² )</td>
              <td className="px-4 py-3">True metric. Sensitive to vector magnitude; computationally slower (square root).</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Inner Product (Dot Product)</td>
              <td className="px-4 py-3 font-mono">⟨u, v⟩ = Σ u_i · v_i</td>
              <td className="px-4 py-3">Extremely fast (AVX/SIMD vectorized instructions). Not a metric unless normalized.</td>
            </tr>
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Cosine Similarity</td>
              <td className="px-4 py-3 font-mono">cos θ = (u · v) / (‖u‖ ‖v‖)</td>
              <td className="px-4 py-3">Measures angular orientation, invariant to scale. When normalized (<code>‖u‖=1</code>), matches Inner Product.</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>2 · The HNSW Algorithm: Multi-Layer Skip-List Graph</h2>
      <p>
        The <strong>Hierarchical Navigable Small World (HNSW)</strong> index (Malkov &amp; Yashunin, 2018) is the gold standard algorithm powering modern vector databases.
      </p>

      <h3>2.1 Structural Intuition</h3>
      <p>
        HNSW extends William Pugh&apos;s 1D <strong>Skip-List</strong> data structure to multi-dimensional graphs:
      </p>
      <ul>
        <li><strong>Top Layers (High Level):</strong> Sparse graphs containing only a few long-range highway links. Traversal takes massive steps across vector space in <code>O(1)</code> hops.</li>
        <li><strong>Bottom Layer (Level 0):</strong> Dense small-world graph containing all <code>N</code> data points, guaranteeing the clustering coefficient necessary for high recall.</li>
      </ul>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p className="text-slate-400">// Level Generation Probabilistic Rule:</p>
        <p>l = ⌊ - ln( uniform(0, 1) ) · m_L ⌋,   where m_L = 1 / ln(M)</p>
        <br />
        <p>P(node reaches level l) = e^(-l / m_L)</p>
      </div>

      <h2>3 · Product Quantization (IVF-PQ): Extreme Memory Compression</h2>
      <p>
        Storing 1 billion 1536-dimensional vectors (OpenAI text-embedding-3-large) in FP32 requires <strong>6.14 Terabytes of raw RAM</strong>.
      </p>
      <p>
        <strong>Product Quantization (Jégou et al., 2011)</strong> compresses vectors by slicing the <code>d</code>-dimensional vector into <code>m</code> smaller sub-vectors of dimension <code>d/m</code>:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>v = [ v_1, v_2, ..., v_m ],   v_i ∈ ℝ^(d/m)</p>
      </div>

      <ol>
        <li>Run K-Means clustering independently on each of the <code>m</code> sub-vector subspaces with <code>k* = 256</code> centroids (codebook).</li>
        <li>Replace each continuous sub-vector with an <strong>8-bit integer index (1 byte)</strong> pointing to its nearest centroid.</li>
        <li>A 1536-dimensional float vector (6,144 bytes) is compressed into <code>m = 96</code> bytes—a <strong>64x memory reduction</strong>!</li>
        <li>At query time, distances are computed via precomputed lookup tables (Asymmetric Distance Computation [ADC]) in cache memory without decompression.</li>
      </ol>

      <h2>4 · Industry Vector Database Architecture Comparison</h2>
      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Database</th>
              <th className="px-4 py-3 text-left">Core Index Engine</th>
              <th className="px-4 py-3 text-left">Storage Architecture</th>
              <th className="px-4 py-3 text-left">Filtering Capabilities</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Pinecone</td>
              <td className="px-4 py-3">Proprietary HNSW + Graph Quantization</td>
              <td className="px-4 py-3">Fully managed cloud, decoupled serverless storage</td>
              <td className="px-4 py-3">Single-stage metadata filtering</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Qdrant</td>
              <td className="px-4 py-3">Custom Rust HNSW with scalar quantization</td>
              <td className="px-4 py-3">On-disk mmap payload storage, memory-mapped vectors</td>
              <td className="px-4 py-3">Payload index filtering built directly into graph traversal</td>
            </tr>
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">pgvector</td>
              <td className="px-4 py-3">IVFFlat / HNSW extension in PostgreSQL</td>
              <td className="px-4 py-3">Native relational PostgreSQL engine (WAL, ACID)</td>
              <td className="px-4 py-3">Direct SQL relational joins with ACID guarantees</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>5 · Python Implementation of HNSW Graph Index</h2>
      <TerminalBlock language="python" filename="hnsw_vector_index.py" code={code} />

      <h2>6 · Further Reading</h2>
      <ul>
        <li>Malkov, Y. A., &amp; Yashunin, D. A. (2018). <em>Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs</em>. IEEE TPAMI, 42(4), 824-836.</li>
        <li>Jégou, H., Douze, M., &amp; Schmid, C. (2011). <em>Product quantization for nearest neighbor search</em>. IEEE TPAMI, 33(1), 117-128.</li>
      </ul>
    </article>
  );
}
