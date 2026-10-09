"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import json
from typing import List, Dict

class AgentMessage:
    def __init__(self, sender: str, recipient: str, content: str):
        self.sender = sender
        self.recipient = recipient
        self.content = content

class SupervisorAgent:
    """
    Supervisor Pattern for Multi-Agent Orchestration (Wu et al., AutoGen 2023).
    The Supervisor acts as a centralized orchestrator, breaking goals into tasks,
    delegating to specialized worker agents, and reviewing outputs before synthesis.
    """
    def __init__(self, name: str = "Supervisor"):
        self.name = name
        self.workers = {}

    def register_worker(self, role: str, handler):
        self.workers[role] = handler

    def coordinate(self, user_objective: str) -> str:
        print(f"[{self.name}] Ingested Objective: '{user_objective}'")

        # 1. Delegate research task to Research Specialist
        print(f"[{self.name}] Routing subtask to 'Researcher'...")
        research_findings = self.workers["Researcher"]("Analyze technical specs for quantum key distribution.")

        # 2. Delegate synthesis/code task to Engineer Specialist
        print(f"[{self.name}] Routing subtask to 'Engineer' with research context...")
        engineer_code = self.workers["Engineer"](f"Implement prototype based on: {research_findings}")

        # 3. Delegate quality review to Critic / Auditor
        print(f"[{self.name}] Routing review to 'Critic'...")
        audit_verdict = self.workers["Critic"](engineer_code)

        return f"Final Audited Deliverable:\\n{engineer_code}\\n\\nAuditor Verdict: {audit_verdict}"

if __name__ == "__main__":
    supervisor = SupervisorAgent()

    # Specialized Worker Agents
    supervisor.register_worker("Researcher", lambda task: f"Verified: BB84 protocol uses 4 photon polarization states.")
    supervisor.register_worker("Engineer", lambda task: f"class BB84Simulation:\\n    def generate_key(self): return [1, 0, 1, 1]")
    supervisor.register_worker("Critic", lambda code: "PASSED: Zero cryptographic side-channel leaks detected.")

    deliverable = supervisor.coordinate("Design a secure quantum key distribution module.")
    print("\\n" + "─" * 60)
    print(deliverable)
`;

export default function MultiAgentPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-rose-100 text-rose-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-rose-500"></span>
          Track 5 · Agentic AI &amp; Orchestration
        </span>
      </div>

      <h1>Multi-Agent Orchestration: Supervisor Patterns, AutoGen, and Swarm Topologies</h1>

      <blockquote>
        <p>
          <em>"A single agent attempting to solve an entire complex software project collapses under cognitive overload and context dilution. Multi-agent systems apply corporate division of labor: specialized personas communicate via structured protocols to achieve collective intelligence."</em> — Qingyun Wu et al. (Microsoft Research, 2023)
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Understand why single-agent architectures fail on long-horizon, multi-disciplinary tasks.</li>
        <li>Analyze Multi-Agent communication topologies: Hierarchical (Supervisor-Worker), Peer-to-Peer (Swarm), and Round-Robin.</li>
        <li>Explain AutoGen conversational patterns (ConversableAgent, GroupChatManager).</li>
        <li>Mitigate infinite conversational loops and context bloat in agent-to-agent chatter.</li>
        <li>Implement a full Supervisor-Worker orchestration pattern from scratch in Python.</li>
      </ul>

      <h2>1 · Why Multi-Agent Systems Outperform Single Agents</h2>
      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Failure Mode in Single Agents</th>
              <th className="px-4 py-3 text-left">Multi-Agent Solution</th>
              <th className="px-4 py-3 text-left">Empirical Advantage</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Persona Dilution</td>
              <td className="px-4 py-3">Strict separation of concerns (e.g. Architect, Coder, Tester, Security Auditor).</td>
              <td className="px-4 py-3">Narrow prompts maintain high fidelity and rigorous persona focus.</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Context Window Bloat</td>
              <td className="px-4 py-3">Message filtering: sub-agents only receive summaries relevant to their task.</td>
              <td className="px-4 py-3">Avoids filling the 128k context window with intermediate scratchpads.</td>
            </tr>
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Confirmation Bias / Blind Spots</td>
              <td className="px-4 py-3">Adversarial Review: a dedicated Critic agent is incentivized to find bugs.</td>
              <td className="px-4 py-3">Dramatically decreases production bugs and logical oversights.</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>2 · Multi-Agent Topology Taxonomy</h2>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>1. HIERARCHICAL (SUPERVISOR)       2. PEER-TO-PEER (SWARM)       3. SEQUENTIAL PIPELINE</p>
        <p>         [ SUPERVISOR ]                     [ AGENT A ]                 [ Plan ]</p>
        <p>          /    │    \                          ▲     ▲                        │</p>
        <p>         /     │     \                         │     │                        ▼</p>
        <p>    [Worker] [Worker] [Worker]             [ AGENT B ]◀───▶[ AGENT C ]    [ Code ]</p>
        <p>    (Search) (Code)   (Review)                                                 │</p>
        <p>                                                                               ▼</p>
        <p>                                                                           [ Test ]</p>
      </div>

      <h2>3 · Python Implementation of Supervisor Multi-Agent Architecture</h2>
      <TerminalBlock language="python" filename="multi_agent_supervisor.py" code={code} />

      <h2>4 · Mitigating Conversational Drift &amp; Infinite Loops</h2>
      <p>
        In multi-agent systems, agents frequently enter unproductive loops (Agent A: "Thanks!", Agent B: "You're welcome! Let me know if you need anything.", Agent A: "Will do!").
      </p>
      <ul>
        <li><strong>Termination Predicates:</strong> Enforce strict stopping triggers (e.g. <code>__END__</code>, maximum step counts, or validator approval tokens).</li>
        <li><strong>Structured JSON Bus:</strong> Ban free-form conversational chatter between agents; mandate rigid schemas: <code>{`{ "status": "APPROVED", "diff": "...", "notes": "..." }`}</code>.</li>
      </ul>

      <h2>5 · Further Reading</h2>
      <ul>
        <li>Wu, Q., et al. (2023). <em>AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation</em>. arXiv:2308.08155.</li>
        <li>Hong, S., et al. (2023). <em>MetaGPT: Meta Programming for Multi-Agent Collaborative Framework</em>. ICLR 2024.</li>
        <li>Li, G., et al. (2023). <em>Camel: Communicative agents for 'mind' exploration of large scale language model society</em>. NeurIPS 2023.</li>
      </ul>
    </article>
  );
}
