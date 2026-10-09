"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `class CognitiveAgentLoop:
    """
    Formal Cognitive Architecture of an Autonomous AI Agent:
    Perception -> Memory Retrieval -> Planning & Decomposition -> Action Execution -> Reflection
    (Wang et al., 2023; Weng, 2023)
    """
    def __init__(self, agent_id: str, system_goal: str):
        self.agent_id = agent_id
        self.system_goal = system_goal
        self.short_term_buffer = []  # In-context conversational history
        self.long_term_memory = []   # External episodic vector memories
        self.tool_inventory = {
            "calculator": lambda expr: eval(expr),
            "search_docs": lambda q: f"Retrieved facts for query: {q}"
        }

    def perceive(self, user_observation: str):
        """
        Ingests multimodal sensory input and environmental telemetry.
        """
        event = {"role": "environment", "content": user_observation}
        self.short_term_buffer.append(event)
        return event

    def plan_and_decompose(self, complex_task: str) -> list:
        """
        Decomposes high-level objectives into sequential, actionable sub-tasks
        (Subgoal Decomposition via Task-Tree Search).
        """
        # Simulated hierarchical task network (HTN) breakdown
        subtasks = [
            f"Step 1: Retrieve background knowledge regarding {complex_task}",
            f"Step 2: Compute quantitative metrics using calculator tool",
            f"Step 3: Synthesize output adhering to constraints"
        ]
        return subtasks

    def execute_action(self, tool_name: str, args: str):
        """
        Dispatches discrete API tool action to external environment.
        """
        if tool_name in self.tool_inventory:
            return self.tool_inventory[tool_name](args)
        return "Tool not available."

    def reflect_and_learn(self, task_outcome: dict):
        """
        Reflexion: Writes successful execution paths into long-term episodic memory.
        """
        memory_entry = {
            "goal": task_outcome["task"],
            "successful_strategy": task_outcome["plan"],
            "timestamp": "2026-10-09"
        }
        self.long_term_memory.append(memory_entry)
        print(f"[{self.agent_id}] Strategy consolidated into Long-Term Memory.")

if __name__ == "__main__":
    agent = CognitiveAgentLoop("ResearchAgent-01", "Automate financial analysis")
    obs = agent.perceive("Calculate annual profit margin from Q1-Q4 revenue sheets.")
    plan = agent.plan_and_decompose(obs["content"])
    print("Hierarchical Subtask Plan:")
    for step in plan:
        print(" ", step)

    calc_res = agent.execute_action("calculator", "(120000 - 85000) / 120000")
    print(f"\\nTool Execution Result (Profit Margin): {float(calc_res):.2%}")
    agent.reflect_and_learn({"task": obs["content"], "plan": plan})
`;

export default function AgentFundamentalsPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-rose-100 text-rose-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-rose-500"></span>
          Track 5 · Agentic AI &amp; Orchestration
        </span>
      </div>

      <h1>Agent Fundamentals: Cognitive Architectures, Perception, Memory, and Action Loops</h1>

      <blockquote>
        <p>
          <em>"An LLM is a stateless next-token predictor. An AI Agent wraps that language engine inside a continuous cognitive loop comprising perception, planning, working and long-term memory, and grounded tool execution."</em> — Lilian Weng (OpenAI, 2023)
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Define the formal boundary between standard LLM prompting and autonomous AI agents.</li>
        <li>Derive the five pillars of cognitive agent architecture: Perception, Memory, Planning, Tool Use, and Self-Reflection.</li>
        <li>Understand Task Decomposition strategies: Chain-of-Thought, Subgoal Decomposition, and Hierarchical Task Networks (HTN).</li>
        <li>Analyze the transition from passive question-answering to closed-loop autonomous environments.</li>
        <li>Implement a functional autonomous cognitive agent loop from scratch in Python.</li>
      </ul>

      <h2>1 · The Evolution: From Chatbots to Autonomous Agents</h2>
      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Level</th>
              <th className="px-4 py-3 text-left">Architecture</th>
              <th className="px-4 py-3 text-left">Autonomy &amp; Capabilities</th>
              <th className="px-4 py-3 text-left">Control Flow</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Level 1: Prompted LLM</td>
              <td className="px-4 py-3">Stateless forward pass</td>
              <td className="px-4 py-3">Passive text completion; no external side effects.</td>
              <td className="px-4 py-3">Linear (Single Request → Single Response)</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Level 2: RAG Pipeline</td>
              <td className="px-4 py-3">Bi-Encoder retrieval + LLM</td>
              <td className="px-4 py-3">Grounds answers in static external documents.</td>
              <td className="px-4 py-3">Deterministic Directed Acyclic Graph (DAG)</td>
            </tr>
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Level 3: ReAct Tool Agent</td>
              <td className="px-4 py-3">LLM + Tool Dispatcher</td>
              <td className="px-4 py-3">Decides when and how to call external APIs (calculators, web search).</td>
              <td className="px-4 py-3">Conditional Loop (While not finished, call tool)</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Level 4: Multi-Agent System</td>
              <td className="px-4 py-3">Swarm / Hierarchical Graph</td>
              <td className="px-4 py-3">Autonomous division of labor, peer code review, self-correction.</td>
              <td className="px-4 py-3">Dynamic Cyclic State Machine with Checkpointing</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>2 · The Cognitive Agent Architecture</h2>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>                ┌───────────────────────────────┐</p>
        <p>                │       AUTONOMOUS AGENT        │</p>
        <p>                │                               │</p>
        <p>PERCEPTION ────▶│  ┌─────────┐   ┌──────────┐  │────▶ ACTIONS (APIs, Code, Tools)</p>
        <p>(User, Env)     │  │ PLANNING│   │  MEMORY  │  │</p>
        <p>                │  │ (HTN,CoT│   │(Episodic,│  │</p>
        <p>                │  │  Trees) │   │ Working) │  │</p>
        <p>                │  └─────────┘   └──────────┘  │</p>
        <p>                │        ▲             ▲       │</p>
        <p>                │        └──────┬──────┘       │</p>
        <p>                │       [ REASONING CORE ]     │</p>
        <p>                │        (Foundation LLM)      │</p>
        <p>                └───────────────────────────────┘</p>
      </div>

      <ol>
        <li><strong>Perception:</strong> Ingests external inputs, tool execution stdout/stderr, and environmental state changes.</li>
        <li><strong>Working Memory:</strong> The in-context sliding window tracking recent conversation turns and immediate sub-goals.</li>
        <li><strong>Long-Term Memory:</strong> External vector stores storing episodic experiences, user preferences, and past successful execution traces.</li>
        <li><strong>Planning &amp; Decomposition:</strong> Splits complex goals into discrete sub-tasks using Tree Search or Hierarchical Task Networks.</li>
        <li><strong>Action Execution:</strong> Invokes external tools via structured JSON schemas, modifying environment state.</li>
      </ol>

      <h2>3 · Python Implementation of Cognitive Loop</h2>
      <TerminalBlock language="python" filename="cognitive_agent_loop.py" code={code} />

      <h2>4 · Further Reading</h2>
      <ul>
        <li>Weng, L. (2023). <em>LLM Powered Autonomous Agents</em>. Lil&apos;Log (lilianweng.github.io/posts/2023-06-23-agent/).</li>
        <li>Wang, L., et al. (2023). <em>A Survey on Large Language Model based Autonomous Agents</em>. arXiv:2308.11432.</li>
        <li>Park, J. S., et al. (2023). <em>Generative Agents: Interactive Simulacra of Human Behavior</em>. UIST 2023.</li>
      </ul>
    </article>
  );
}
