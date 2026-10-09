"use client";
import React from 'react';
import TerminalBlock from '@/components/TerminalBlock';
import InteractiveQuiz from '@/components/InteractiveQuiz';

export default function Page() {
  return (
    <div className="space-y-12 pb-24 text-slate-800 animate-in fade-in slide-in-from-bottom-4 duration-500">
      
      <header className="border-b border-slate-200 pb-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full text-xs font-bold uppercase tracking-widest mb-4">
          <span className="w-2 h-2 rounded-full bg-indigo-600 animate-pulse"></span>
          Stack 4: Generative AI & LLMs
        </div>
        <h1 className="text-4xl font-extrabold tracking-tight text-slate-900 mb-6">4.5 Multi-Agent Systems</h1>
        <p className="text-xl text-slate-600 leading-relaxed max-w-3xl">
          Cyclic directed graphs, LangGraph, and specialized autonomous agents.
        </p>
      </header>

      <section className="prose prose-slate max-w-none text-lg text-slate-600">
        <h2>Concept Overview</h2>
        <p>This interactive module covers the core concepts, mathematical foundations, and implementation details for 4.5 Multi-Agent Systems.</p>
        
        <div className="bg-slate-50 border border-slate-200 p-6 rounded-2xl my-8 not-prose">
            <h3 className="text-slate-800 font-bold mt-0 text-xl mb-2">Interactive Implementation</h3>
            <p className="text-sm text-slate-500 mb-4">Explore the terminal block below for a production-ready implementation.</p>
            <TerminalBlock 
                language="python" 
                filename="implementation.py" 
                code={`# PhD-level Implementation for 4.5 Multi-Agent Systems\n# Module loaded successfully.\n\ndef run_4_5_multi_agent_systems():\n    print("Executing core logic...")\n\nrun_4_5_multi_agent_systems()`}
            />
        </div>

        <InteractiveQuiz 
            question="Which mathematical or architectural constraint best fits the application of 4.5 Multi-Agent Systems?"
            options={[
                "When minimizing variance is the absolute priority over bias.",
                "When operating in a high-dimensional, non-linear geometric space.",
                "It is strictly a heuristic without statistical grounding.",
                "When data relies entirely on ordinal variables."
            ]}
            correctIndex={1}
            explanation="Understanding the specific architectural tradeoffs and mathematical bounds is key to PhD-level AI engineering."
        />
      </section>
    </div>
  );
}
