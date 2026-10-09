"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `from typing import Callable, Any

class Runnable:
    """
    Core primitive of LangChain Expression Language (LCEL).
    Implements functional UNIX-pipe composition:
      chain = prompt | model | output_parser
    All Runnables implement invoke(), batch(), and stream().
    """
    def __init__(self, func: Callable[[Any], Any]):
        self.func = func

    def invoke(self, input_data: Any) -> Any:
        return self.func(input_data)

    def __or__(self, other: 'Runnable') -> 'RunnableSequence':
        return RunnableSequence(self, other)

class RunnableSequence(Runnable):
    def __init__(self, first: Runnable, second: Runnable):
        self.first = first
        self.second = second

    def invoke(self, input_data: Any) -> Any:
        # Pipelined data flow: second.invoke(first.invoke(x))
        intermediate = self.first.invoke(input_data)
        return self.second.invoke(intermediate)

    def __or__(self, other: Runnable) -> 'RunnableSequence':
        return RunnableSequence(self, other)

class PromptTemplate(Runnable):
    def __init__(self, template: str):
        self.template = template
        super().__init__(self._format)

    def _format(self, variables: dict) -> str:
        return self.template.format(**variables)

class MockChatModel(Runnable):
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        super().__init__(self._generate)

    def _generate(self, prompt_text: str) -> str:
        return f'{{"thought": "Analyzed {prompt_text[:20]}...", "status": "200_OK"}}'

class StrOutputParser(Runnable):
    def __init__(self):
        super().__init__(self._parse)

    def _parse(self, model_output: str) -> str:
        import json
        try:
            parsed = json.loads(model_output)
            return parsed.get("status", model_output)
        except Exception:
            return model_output

if __name__ == "__main__":
    # LCEL Composition using Python bitwise OR operator overloading (__or__)
    prompt = PromptTemplate("Summarize the mathematical foundations of {topic} in one line.")
    model = MockChatModel()
    parser = StrOutputParser()

    # Functional Pipe Composition: chain = prompt | model | parser
    lcel_chain = prompt | model | parser

    result = lcel_chain.invoke({"topic": "Stochastic Gradient Descent"})
    print("LCEL Chain Result:", result)
`;

export default function LangChainPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-rose-100 text-rose-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-rose-500"></span>
          Track 5 · Agentic AI &amp; Orchestration
        </span>
      </div>

      <h1>LangChain Deep Dive: LCEL, Runnables, Streaming, and Tracing Architecture</h1>

      <blockquote>
        <p>
          <em>"LangChain Expression Language (LCEL) replaced fragile class inheritance with a unified functional composition interface: every component is a Runnable adhering to a standard protocol supporting synchronous, asynchronous, batch, and streaming execution."</em> — Harrison Chase (LangChain, 2023)
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Understand the structural evolution from legacy <code>LLMChain</code> to LangChain Expression Language (LCEL).</li>
        <li>Derive the <code>Runnable</code> interface specification: <code>invoke()</code>, <code>batch()</code>, <code>stream()</code>, and <code>astream_events()</code>.</li>
        <li>Implement Python pipe operator composition <code>chain = prompt | model | parser</code> via <code>__or__</code> overloading.</li>
        <li>Analyze parallel execution with <code>RunnableParallel</code> and fallback recovery with <code>with_fallbacks()</code>.</li>
        <li>Understand observability, latency profiling, and token tracing using LangSmith.</li>
      </ul>

      <h2>1 · The Architectural Evolution: Why LCEL Replaced Legacy Chains</h2>
      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Dimension</th>
              <th className="px-4 py-3 text-left">Legacy Chains (2022-2023)</th>
              <th className="px-4 py-3 text-left">LCEL Architecture (2024-Present)</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Design Pattern</td>
              <td className="px-4 py-3">Deep OOP inheritance hierarchies (<code>Chain</code>, <code>SequentialChain</code>).</td>
              <td className="px-4 py-3">Functional composition primitives (<code>RunnableSequence</code>).</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Composition Syntax</td>
              <td className="px-4 py-3">Imperative nested boilerplate instantiation.</td>
              <td className="px-4 py-3">Declarative UNIX pipe syntax: <code>chain = a | b | c</code>.</td>
            </tr>
            <tr className="bg-white">
              <td className="px-4 py-3 font-semibold">Streaming Support</td>
              <td className="px-4 py-3">Ad-hoc custom callback handlers; difficult to route token chunks.</td>
              <td className="px-4 py-3">First-class: every step yields token generators out-of-the-box.</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-semibold">Async &amp; Batching</td>
              <td className="px-4 py-3">Synchronous execution by default; manual threading.</td>
              <td className="px-4 py-3">Automatic async event loops (<code>ainvoke</code>) and concurrency batching.</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>2 · The Runnable Protocol Specification</h2>
      <p>
        Every LCEL primitive implements the standard <code>Runnable</code> interface:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p className="text-slate-400">// Synchronous &amp; Asynchronous Invocation Methods:</p>
        <p>invoke(input, config)   / ainvoke(input, config)    # Single sample</p>
        <p>batch(inputs, config)   / abatch(inputs, config)    # Concurrent mini-batches</p>
        <p>stream(input, config)   / astream(input, config)    # Generator yielding token chunks</p>
        <p>astream_events(input)                               # Real-time event stream for UI</p>
      </div>

      <h2>3 · Python Implementation of LCEL from First Principles</h2>
      <TerminalBlock language="python" filename="lcel_scratch.py" code={code} />

      <h2>4 · Observability &amp; Tracing with LangSmith</h2>
      <p>
        Because LLM calls are non-deterministic, production architectures require full trace observability. <strong>LangSmith</strong> hooks into the LCEL execution graph via runtime callbacks, recording:
      </p>
      <ul>
        <li><strong>Exact Latency Breakdown:</strong> Time spent in network serialization vs. model time-to-first-token (TTFT) vs. tool execution.</li>
        <li><strong>Token Accounting:</strong> Prompt tokens vs. completion tokens mapped to exact dollar costs.</li>
        <li><strong>Debugging Intermediates:</strong> Inspecting raw prompt interpolations before they hit the model to catch schema formatting bugs.</li>
      </ul>

      <h2>5 · Further Reading</h2>
      <ul>
        <li>Chase, H. (2023). <em>LangChain Expression Language (LCEL)</em>. LangChain Official Documentation.</li>
        <li>LangSmith Documentation (2024). <em>Observability and Evaluation for LLM Applications</em>. docs.smith.langchain.com.</li>
      </ul>
    </article>
  );
}
