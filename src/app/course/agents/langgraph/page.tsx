"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `from typing import Dict, Any, Callable

class StateGraph:
    """
    StateGraph engine implementing cyclic state machines for agentic workflows:
    - Centralized mutable State schema
    - Nodes as pure state transformation functions: Node(state) -> partial_state_update
    - Conditional edges with routing logic: router(state) -> next_node_name
    - Checkpointing & Time-Travel state replay
    """
    def __init__(self, state_schema: type):
        self.state_schema = state_schema
        self.nodes: Dict[str, Callable[[dict], dict]] = {}
        self.edges: Dict[str, str] = {}
        self.conditional_edges: Dict[str, Callable[[dict], str]] = {}
        self.entry_point: str = None
        self.checkpoints: list = []

    def add_node(self, name: str, action: Callable[[dict], dict]):
        self.nodes[name] = action

    def set_entry_point(self, name: str):
        self.entry_point = name

    def add_edge(self, source: str, target: str):
        self.edges[source] = target

    def add_conditional_edges(self, source: str, router: Callable[[dict], str]):
        self.conditional_edges[source] = router

    def run(self, initial_state: dict, max_steps: int = 10) -> dict:
        state = initial_state.copy()
        current_node = self.entry_point

        for step in range(1, max_steps + 1):
            if current_node == "__END__" or current_node is None:
                print(f"Workflow terminated cleanly at step {step-1}.")
                break

            print(f"Executing Node: '{current_node}' | Current State: {state}")
            # Execute node and merge partial update
            update = self.nodes[current_node](state)
            state.update(update)

            # Record checkpoint for time-travel debugging
            self.checkpoints.append({"step": step, "node": current_node, "state": state.copy()})

            # Determine next node via conditional router or static edge
            if current_node in self.conditional_edges:
                current_node = self.conditional_edges[current_node](state)
            elif current_node in self.edges:
                current_node = self.edges[current_node]
            else:
                current_node = "__END__"

        return state

if __name__ == "__main__":
    # Example: Code Generation with Autonomous Unit-Test Evaluation Loop
    graph = StateGraph(state_schema=dict)

    def generator_node(state: dict) -> dict:
        code_draft = "def add(a, b): return a + b"
        return {"code": code_draft, "attempts": state.get("attempts", 0) + 1}

    def evaluator_node(state: dict) -> dict:
        # Simulate test passing after attempt 2
        tests_passed = (state["attempts"] >= 2)
        return {"tests_passed": tests_passed}

    def review_router(state: dict) -> str:
        if state["tests_passed"]:
            return "__END__"
        return "generator"  # Cyclic loop back to generator!

    graph.add_node("generator", generator_node)
    graph.add_node("evaluator", evaluator_node)
    graph.set_entry_point("generator")
    graph.add_edge("generator", "evaluator")
    graph.add_conditional_edges("evaluator", review_router)

    final_state = graph.run({"attempts": 0, "tests_passed": False})
    print("\\nFinal State after Cyclic Graph Execution:", final_state)
`;

export default function LangGraphPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-rose-100 text-rose-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-rose-500"></span>
          Track 5 · Agentic AI &amp; Orchestration
        </span>
      </div>

      <h1>LangGraph &amp; Cyclic Architectures: State Machines, Human-in-the-Loop, and Time-Travel</h1>

      <blockquote>
        <p>
          <em>"Standard DAG (Directed Acyclic Graph) pipelines are fundamentally inadequate for agentic behavior: agents require recursive feedback loops, conditional branching, human-in-the-loop approvals, and durable state checkpointing."</em> — LangChain Team (2024)
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Explain why cyclic graphs are necessary for production agent workflows (feedback loops, lint-repair cycles).</li>
        <li>Derive the core abstractions of LangGraph: <code>StateGraph</code>, Nodes, Static Edges, and Conditional Edges.</li>
        <li>Understand durable persistence and Checkpointing for fault-tolerant state recovery.</li>
        <li>Implement Human-in-the-Loop approval breakpoints with state modification.</li>
        <li>Implement a functional StateGraph engine with cyclic execution loops from scratch in Python.</li>
      </ul>

      <h2>1 · DAGs vs. Cyclic State Machines</h2>
      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Property</th>
              <th className="px-4 py-3 text-left">Directed Acyclic Graph (DAG)</th>
              <th className="px-4 py-3 text-left">Cyclic State Machine (LangGraph)</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Looping &amp; Recursion</td>
              <td className="px-4 py-3">Impossible. Execution flows strictly forward in one direction.</td>
              <td className="px-4 py-3">First-class. Nodes can route back to themselves or earlier nodes based on conditions.</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Error Recovery</td>
              <td className="px-4 py-3">Fails entire pipeline on exception.</td>
              <td className="px-4 py-3">Routes error state to a debugger / reflexer node for iterative self-correction.</td>
            </tr>
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">State Model</td>
              <td className="px-4 py-3">Ad-hoc parameter passing between sequential functions.</td>
              <td className="px-4 py-3">Centralized, typed schema with explicit reduction operations (e.g. <code>operator.add</code>).</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>2 · The Core LangGraph Architecture</h2>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>                ┌───────────────┐</p>
        <p>                │ [GENERATOR]   │◀────────────┐</p>
        <p>                └───────┬───────┘             │</p>
        <p>                        │ Static Edge         │ (Tests Failed:</p>
        <p>                        ▼                     │  Route Back)</p>
        <p>                ┌───────────────┐             │</p>
        <p>                │ [EVALUATOR]   │─────────────┘</p>
        <p>                └───────┬───────┘</p>
        <p>                        │ Conditional Edge (Tests Passed)</p>
        <p>                        ▼</p>
        <p>                   [ __END__ ]</p>
      </div>

      <h2>3 · Durable Checkpointing &amp; Time-Travel Debugging</h2>
      <p>
        In production systems, an agent workflow may run across days (e.g. autonomous market research, multi-day data pipeline).
      </p>
      <p>
        <strong>Checkpointers (e.g. PostgresSaver):</strong> Automatically serialize and write the complete state dictionary to durable storage after every node execution.
      </p>
      <ul>
        <li><strong>Crash Recovery:</strong> If the worker process restarts mid-execution, the graph resumes seamlessly from the last completed node checkpoint.</li>
        <li><strong>Human-in-the-Loop Breakpoints:</strong> The graph pauses before critical actions (e.g., executing a financial transaction or pushing to git). A human inspects the state, modifies parameters if needed, and clicks "Approve" to continue.</li>
        <li><strong>Time-Travel Replay:</strong> Engineers can inspect historical state at Step 4, fork a new execution branch with modified inputs, and debug alternate trajectories without re-running Steps 1-3.</li>
      </ul>

      <h2>4 · Python Implementation of StateGraph Engine</h2>
      <TerminalBlock language="python" filename="state_graph_scratch.py" code={code} />

      <h2>5 · Further Reading</h2>
      <ul>
        <li>LangChain Team (2024). <em>LangGraph: Building Language Agents as Graphs</em>. blog.langchain.dev/langgraph/.</li>
        <li>Harel, D. (1987). <em>Statecharts: A visual formalism for complex systems</em>. Science of Computer Programming, 8(3), 231-274.</li>
      </ul>
    </article>
  );
}
