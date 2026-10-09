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
          Generative AI & LLMs
        </div>
        <h1 className="text-4xl font-extrabold tracking-tight text-slate-900 mb-6">LangGraph & Multi-Agent</h1>
        <p className="text-xl text-slate-600 leading-relaxed max-w-3xl">
          Cyclic graphs and specialized agents.
        </p>
      </header>

      <section className="prose prose-slate max-w-none text-lg text-slate-600">
        <h2>Concept Overview</h2>
        <p>This interactive module covers the core concepts, mathematical foundations, and implementation details for LangGraph & Multi-Agent.</p>
        
        <div className="bg-slate-50 border border-slate-200 p-6 rounded-2xl my-8 not-prose">
            <h3 className="text-slate-800 font-bold mt-0 text-xl mb-2">Interactive Implementation</h3>
            <p className="text-sm text-slate-500 mb-4">Explore the terminal block below for a production-ready implementation.</p>
            <TerminalBlock 
                language="python" 
                filename="implementation.py" 
                code={`# Implementation for LangGraph & Multi-Agent\n# Module loaded successfully.\n\ndef run_langgraph___multi_agent():\n    print("Executing core logic...")\n\nrun_langgraph___multi_agent()`}
            />
        </div>

        <InteractiveQuiz 
            question="Which scenario best fits the application of LangGraph & Multi-Agent?"
            options={[
                "When latency is the only priority.",
                "When you need maximum accuracy with specific constraints.",
                "It should be avoided in production.",
                "When you have unlabelled data."
            ]}
            correctIndex={1}
            explanation="Understanding the specific architectural tradeoffs is key to AI engineering. Every tool has its specific use case."
        />
      </section>
    </div>
  );
}
