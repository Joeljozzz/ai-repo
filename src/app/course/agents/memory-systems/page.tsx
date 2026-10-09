"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import time
from typing import List, Dict

class MemorySystemArchitecture:
    """
    Cognitive Memory Architecture for Autonomous Agents:
    1. Working Memory (Short-Term FIFO Context Buffer)
    2. Episodic Memory (Experience Traces with Recency & Importance decay)
    3. Semantic Memory (Declarative knowledge base / Vector index)
    4. Procedural Memory (Prompt rules, system instructions, tools)
    """
    def __init__(self, decay_rate: float = 0.995):
        self.decay_rate = decay_rate
        self.episodic_store: List[Dict] = []
        self.semantic_knowledge: Dict[str, str] = {}

    def add_episodic_memory(self, memory_text: str, importance_score: float):
        """
        Stores an episodic event with an importance score [1.0 to 10.0] and creation timestamp.
        (Generative Agents: Park et al., Stanford 2023)
        """
        entry = {
            "text": memory_text,
            "importance": importance_score,
            "timestamp": time.time(),
            "access_count": 1
        }
        self.episodic_store.append(entry)

    def retrieve_memories(self, query: str, top_k: int = 2) -> List[Dict]:
        """
        Retrieval Score = α_recency · S_recency + α_importance · S_importance + α_relevance · S_relevance
        where S_recency = decay_rate^(hours_since_access)
        """
        now = time.time()
        scored_memories = []

        for mem in self.episodic_store:
            # Recency score exponentially decaying over time
            hours_elapsed = (now - mem["timestamp"]) / 3600.0
            recency = self.decay_rate ** hours_elapsed

            # Normalized importance [0.1 to 1.0]
            importance = mem["importance"] / 10.0

            # Lexical semantic relevance proxy
            overlap = len(set(query.lower().split()) & set(mem["text"].lower().split()))
            relevance = min(1.0, overlap * 0.4)

            # Combined tripartite retrieval function
            total_score = 0.3 * recency + 0.3 * importance + 0.4 * relevance
            scored_memories.append((total_score, mem))

        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [mem for score, mem in scored_memories[:top_k]]

if __name__ == "__main__":
    mem_sys = MemorySystemArchitecture()
    mem_sys.add_episodic_memory("User prefers responses formatted in clean bullet points without pleasantries.", importance_score=9.0)
    mem_sys.add_episodic_memory("User asked about weather in Seattle 3 weeks ago.", importance_score=2.0)
    mem_sys.add_episodic_memory("User is an expert AI research scientist; prefers mathematical rigor.", importance_score=9.5)

    retrieved = mem_sys.retrieve_memories("How should I format this technical report for the user?")
    print("Top Retrieved Episodic Memories:")
    for m in retrieved:
        print(f"  [Importance {m['importance']}] {m['text']}")
`;

export default function MemorySystemsPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-rose-100 text-rose-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-rose-500"></span>
          Track 5 · Agentic AI &amp; Orchestration
        </span>
      </div>

      <h1>Agent Memory Systems: Working, Episodic, Semantic, and Reflection Architectures</h1>

      <blockquote>
        <p>
          <em>"Memory in artificial agents is not just a vector database lookup; it is a tripartite cognitive structure that synthesizes raw observational streams into abstract episodic memories and hierarchical reflections."</em> — Joon Sung Park et al. (Stanford University, 2023)
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Understand the cognitive psychology taxonomy applied to AI: Working, Episodic, Semantic, and Procedural memory.</li>
        <li>Derive the Stanford Generative Agents memory retrieval function: Recency, Importance, and Relevance.</li>
        <li>Explain Memory Reflection (Park et al.) as hierarchical tree abstraction of atomic experiences.</li>
        <li>Analyze context compaction strategies: Sliding windows, Summarization trees, and MemGPT virtual paging.</li>
        <li>Implement a full episodic memory retrieval engine with exponential decay from scratch in Python.</li>
      </ul>

      <h2>1 · The Cognitive Memory Taxonomy for Agents</h2>
      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Memory Type</th>
              <th className="px-4 py-3 text-left">Cognitive Analog</th>
              <th className="px-4 py-3 text-left">Implementation in AI Systems</th>
              <th className="px-4 py-3 text-left">Persistence</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Working Memory</td>
              <td className="px-4 py-3">Short-term scratchpad</td>
              <td className="px-4 py-3">In-context prompt tokens (sliding conversation window, tool call outputs).</td>
              <td className="px-4 py-3">Transient (wiped between runs).</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Episodic Memory</td>
              <td className="px-4 py-3">Autobiographical experiences</td>
              <td className="px-4 py-3">Time-stamped log of user interactions, past errors, and successful tool trajectories.</td>
              <td className="px-4 py-3">Persistent vector index with decay functions.</td>
            </tr>
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Semantic Memory</td>
              <td className="px-4 py-3">Generalized facts &amp; concepts</td>
              <td className="px-4 py-3">Knowledge graphs, documentation embeddings, domain-specific databases.</td>
              <td className="px-4 py-3">Static or continuously updated vector store.</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Procedural Memory</td>
              <td className="px-4 py-3">Implicit skills &amp; habits</td>
              <td className="px-4 py-3">System prompt guidelines, few-shot tool examples, fine-tuned agent weights.</td>
              <td className="px-4 py-3">Permanent architectural configuration.</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>2 · The Stanford Memory Retrieval Function (Park et al., 2023)</h2>
      <p>
        In the seminal <em>Generative Agents</em> simulation (Stanford &amp; Google, 2023), 25 autonomous agents inhabited an interactive virtual town. At each step, agents retrieved relevant memories using a tripartite scoring objective:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>Score(m) = α_recency · S_recency(m) + α_importance · S_importance(m) + α_relevance · S_relevance(m, q)</p>
        <br />
        <p>1. Recency:    S_recency(m) = γ^(Δt)</p>
        <p>   where γ ∈ [0.99, 0.995] is the decay factor and Δt is hours since last memory access.</p>
        <br />
        <p>2. Importance: S_importance(m) ∈ [1, 10]</p>
        <p>   Rated by an LLM prompt: "On a scale of 1-10, how poignant/significant is this event?"</p>
        <br />
        <p>3. Relevance:  S_relevance(m, q) = cos( E(m), E(q) )</p>
        <p>   Cosine similarity between query embedding and memory embedding.</p>
      </div>

      <h2>3 · Memory Reflection: Synthesizing Abstractions</h2>
      <p>
        Storing hundreds of raw observations (e.g., "John drank coffee", "John sat at his desk", "John opened email") floods working memory with trivial noise.
      </p>
      <p>
        <strong>Reflection</strong> is periodic hierarchical synthesis: when the cumulative importance score of recent memories exceeds a threshold (e.g. 100), the agent prompts itself:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>Prompt: "Given the 100 recent observations above, what are the 3 most salient higher-level insights</p>
        <p>         you can infer regarding the subject's goals and psychological state?"</p>
        <br />
        <p>Output: "Insight 1: John is intensely focused on finishing his chemistry paper before Friday."</p>
      </div>

      <p>
        This high-level insight is written back into episodic memory as a first-class memory node, forming a hierarchical tree of abstractions.
      </p>

      <h2>4 · MemGPT: Virtual Context Paging (Packer et al., 2023)</h2>
      <p>
        Inspired by operating system virtual memory (RAM vs Disk paging), <strong>MemGPT</strong> divides memory into:
      </p>
      <ul>
        <li><strong>Core Memory (Main Context Window):</strong> The fixed-size LLM prompt holding immediate persona instructions and current conversation buffer.</li>
        <li><strong>Archival Memory (External Disk):</strong> Unlimited vector database storage.</li>
        <li><strong>Recall Mechanism:</strong> The agent issues programmatic tool calls <code>archival_memory_search(query)</code> and <code>core_memory_append(insight)</code> to page relevant blocks into and out of its own working context dynamically.</li>
      </ul>

      <h2>5 · Python Implementation of Agent Memory Retrieval Engine</h2>
      <TerminalBlock language="python" filename="agent_memory_system.py" code={code} />

      <h2>6 · Further Reading</h2>
      <ul>
        <li>Park, J. S., et al. (2023). <em>Generative Agents: Interactive Simulacra of Human Behavior</em>. UIST 2023 (arXiv:2304.03442).</li>
        <li>Packer, C., et al. (2023). <em>MemGPT: Towards LLMs as Operating Systems</em>. arXiv:2310.08560.</li>
        <li>Sumers, T. R., et al. (2023). <em>Cognitive Architectures for Language Agents</em>. TMLR 2024.</li>
      </ul>
    </article>
  );
}
