"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import json
from typing import Dict, Any, Callable

class ToolRegistry:
    """
    Registry managing schema validation and invocation of agentic tools.
    """
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, description: str, parameters: Dict[str, Any], func: Callable):
        self._tools[name] = func
        self._schemas[name] = {
            "name": name,
            "description": description,
            "parameters": parameters
        }

    def get_tool_definitions(self) -> str:
        return json.dumps(list(self._schemas.values()), indent=2)

    def execute(self, name: str, args: Dict[str, Any]) -> str:
        if name not in self._tools:
            return f"Error: Tool '{name}' not found."
        try:
            result = self._tools[name](**args)
            return str(result)
        except Exception as e:
            return f"Error executing '{name}': {str(e)}"

class ReActAgent:
    """
    Autonomous ReAct (Reason + Act) Agent implementing the cognitive loop:
    Thought -> Action [Tool + Args] -> Observation -> Thought -> Finish
    (Yao et al., ICLR 2023)
    """
    def __init__(self, tools: ToolRegistry, max_iterations: int = 5):
        self.tools = tools
        self.max_iterations = max_iterations
        self.history = []

    def _mock_llm_step(self, question: str, iteration: int, last_observation: str) -> str:
        """
        Simulated LLM response returning structured ReAct thought and action tags.
        """
        if iteration == 1:
            return (
                "Thought: The user is asking for the current temperature in San Francisco and the humidity.\\n"
                "Action: get_weather\\n"
                "Action Input: {\\"city\\": \\"San Francisco\\"}"
            )
        elif iteration == 2:
            return (
                f"Thought: I received the weather observation: '{last_observation}'. Now I will synthesize the final answer.\\n"
                "Final Answer: The current temperature in San Francisco is 62°F with 78% humidity."
            )
        return "Final Answer: Done."

    def run(self, query: str) -> str:
        print(f"User Query: {query}\\n" + "─" * 50)
        observation = ""

        for i in range(1, self.max_iterations + 1):
            llm_output = self._mock_llm_step(query, iteration=i, last_observation=observation)
            print(f"Step {i}:\\n{llm_output}\\n")

            if "Final Answer:" in llm_output:
                final_ans = llm_output.split("Final Answer:")[1].strip()
                return final_ans

            # Parse Action and Action Input
            lines = llm_output.split("\\n")
            action_name = None
            action_input = {}
            for line in lines:
                if line.startswith("Action:"):
                    action_name = line.split("Action:")[1].strip()
                elif line.startswith("Action Input:"):
                    raw_args = line.split("Action Input:")[1].strip()
                    action_input = json.loads(raw_args)

            # Tool Execution
            if action_name:
                observation = self.tools.execute(action_name, action_input)
                print(f"Observation: {observation}\\n" + "─" * 50)

        return "Reached maximum iteration limit."

if __name__ == "__main__":
    registry = ToolRegistry()
    registry.register(
        name="get_weather",
        description="Returns weather metrics for a specified city.",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        func=lambda city: f"Temperature: 62°F, Humidity: 78% in {city}"
    )

    agent = ReActAgent(tools=registry)
    result = agent.run("What is the weather like in San Francisco right now?")
    print(f"Final Resolved Output: {result}")
`;

export default function ReactPatternPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-rose-100 text-rose-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-rose-500"></span>
          Track 5 · Agentic AI &amp; Orchestration
        </span>
      </div>

      <h1>ReAct: Synergizing Reasoning and Acting in Language Models</h1>

      <blockquote>
        <p>
          <em>"While Chain-of-Thought reasoning is ungrounded in external state, and pure action calling lacks dynamic planning, ReAct tightly interleaves verbal reasoning traces with discrete tool executions."</em> — Shunyu Yao et al. (Princeton &amp; Google Brain, 2022)
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Understand the theoretical limitations of pure Chain-of-Thought (CoT) and pure Action Generation.</li>
        <li>Derive the formal ReAct cognitive loop: <code>Thought_t → Action_t → Observation_t → Thought_(t+1)</code>.</li>
        <li>Design JSON schema specifications for LLM function/tool calling protocols.</li>
        <li>Analyze error recovery mechanisms: Self-Correction, Reflexion (Shinn et al.), and Tool Hallucination safeguards.</li>
        <li>Implement an autonomous ReAct agent execution engine with tool dispatching from scratch in Python.</li>
      </ul>

      <h2>1 · The Dilemma of Ungrounded Reasoning vs. Blind Action</h2>
      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Paradigm</th>
              <th className="px-4 py-3 text-left">Execution Mechanism</th>
              <th className="px-4 py-3 text-left">Core Vulnerability</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Pure Chain-of-Thought (CoT)</td>
              <td className="px-4 py-3">Internal token reasoning steps without external tool calls (Wei et al., 2022).</td>
              <td className="px-4 py-3">Hallucination. The model reasons on incorrect internal parametric memories without fact-checking.</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Pure Tool Calling (WebGPT)</td>
              <td className="px-4 py-3">Directly maps query to tool calls without intermediate reasoning scratchpads.</td>
              <td className="px-4 py-3">Lack of synthesis. Cannot handle multi-step composition or synthesize conflicting evidence.</td>
            </tr>
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">ReAct (Yao et al., 2022)</td>
              <td className="px-4 py-3"><strong>Interleaves</strong> natural language thoughts with discrete tool invocations.</td>
              <td className="px-4 py-3">Overcomes both: thoughts dynamically plan and adjust; actions retrieve verified ground truth.</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>2 · The Formal ReAct Cognitive Loop</h2>
      <p>
        At time step <code>t</code>, the agent receives context history <code>c_t = (q, o_1, a_1, ..., o_(t-1))</code>. Instead of directly predicting action <code>a_t</code>, the policy samples from two interleaved action spaces:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p className="text-slate-400">// Unified Agent Action Space: Â = A ∪ L</p>
        <p>A = External Tool Executions: &#123; search(q), sql_query(s), python_eval(c) &#125;</p>
        <p>L = Natural Language Reasoning Thoughts (Internal Scratchpad)</p>
        <br />
        <p className="text-slate-400">// Execution Sequence:</p>
        <p>1. Thought_t      ~ p_θ( thought | context_t )</p>
        <p>2. Action_t       ~ p_θ( action  | context_t, Thought_t )</p>
        <p>3. Observation_t  = Environment.execute( Action_t )</p>
        <p>4. context_(t+1)  = (context_t, Thought_t, Action_t, Observation_t)</p>
      </div>

      <h2>3 · Tool Schema Protocols: Structured Outputs</h2>
      <p>
        Modern foundation models (OpenAI function calling, Anthropic tool use, Gemini tool declarations) enforce JSON schema specifications. The model is constrained to generate syntactically valid JSON objects matching declared signatures:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>&#123;</p>
        <p>  &quot;name&quot;: &quot;execute_sql&quot;,</p>
        <p>  &quot;description&quot;: &quot;Executes a read-only SQL query against the enterprise database.&quot;,</p>
        <p>  &quot;parameters&quot;: &#123;</p>
        <p>    &quot;type&quot;: &quot;object&quot;,</p>
        <p>    &quot;properties&quot;: &#123;</p>
        <p>      &quot;query&quot;: &#123; &quot;type&quot;: &quot;string&quot;, &quot;description&quot;: &quot;The SQL statement&quot; &#125;</p>
        <p>    &#125;,</p>
        <p>    &quot;required&quot;: [&quot;query&quot;]</p>
        <p>  &#125;</p>
        <p>&#125;</p>
      </div>

      <h2>4 · Error Handling and Reflexion</h2>
      <p>
        In production systems, tools often fail (e.g. API timeouts, SQL syntax errors, empty search hits).
      </p>
      <p>
        <strong>Reflexion (Shinn et al., 2023):</strong> When a tool returns an error observation, the agent is prompted with an episodic reflection buffer:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p className="text-slate-400">// Self-Correction Reflection Trace:</p>
        <p>Observation: &quot;OperationalError: no such column &apos;revenue_2023&apos;&quot;</p>
        <p>Thought: &quot;My previous query failed because the column name was incorrect. I should inspect the schema with PRAGMA table_info before re-querying.&quot;</p>
        <p>Action: execute_sql(&quot;PRAGMA table_info(financials);&quot;)</p>
      </div>

      <h2>5 · Python Implementation of ReAct Engine</h2>
      <TerminalBlock language="python" filename="react_agent_engine.py" code={code} />

      <h2>6 · Further Reading</h2>
      <ul>
        <li>Yao, S., et al. (2023). <em>ReAct: Synergizing Reasoning and Acting in Language Models</em>. ICLR 2023.</li>
        <li>Shinn, N., et al. (2023). <em>Reflexion: Language Agents with Verbal Reinforcement Learning</em>. NeurIPS 2023.</li>
        <li>Schick, T., et al. (2023). <em>Toolformer: Language Models Can Teach Themselves to Use Tools</em>. NeurIPS 2023.</li>
      </ul>
    </article>
  );
}
