const fs = require('fs');
const path = require('path');

// Must mirror src/data/curriculum.ts
const curriculum = [
  { id: "math",          color: "sky",     label: "Track 0", title: "Mathematical Foundations",       modules: [
    { id: "descriptive-stats" },{ id: "probability" },{ id: "distributions" },
    { id: "linear-algebra" },  { id: "calculus" },   { id: "hypothesis-testing" },{ id: "statistical-errors" },
  ]},
  { id: "ml-core",       color: "emerald", label: "Track 1", title: "Classical Machine Learning",      modules: [
    { id: "learning-paradigms" },{ id: "data-preprocessing" },{ id: "model-evaluation" },
    { id: "linear-regression" },{ id: "logistic-regression" },{ id: "knn" },{ id: "naive-bayes" },
    { id: "svm" },{ id: "decision-trees" },{ id: "random-forests" },
    { id: "boosting" },{ id: "xgboost" },{ id: "clustering" },{ id: "dim-reduction" },
  ]},
  { id: "deep-learning", color: "purple",  label: "Track 2", title: "Deep Learning",                  modules: [
    { id: "neural-networks" },{ id: "backpropagation" },{ id: "optimizers" },{ id: "regularization" },
    { id: "cnn" },{ id: "rnn-lstm" },{ id: "attention" },{ id: "transformers" },
  ]},
  { id: "genai",         color: "indigo",  label: "Track 3", title: "Generative AI & LLMs",           modules: [
    { id: "llm-foundations" },{ id: "prompt-engineering" },{ id: "fine-tuning" },{ id: "embeddings" },
  ]},
  { id: "ai-systems",    color: "teal",    label: "Track 4", title: "AI Systems & RAG Engineering",   modules: [
    { id: "rag-fundamentals" },{ id: "vector-databases" },{ id: "advanced-rag" },{ id: "evaluation-rag" },
  ]},
  { id: "agents",        color: "rose",    label: "Track 5", title: "Agentic AI & Orchestration",     modules: [
    { id: "agent-fundamentals" },{ id: "react-pattern" },{ id: "memory-systems" },
    { id: "langchain" },{ id: "langgraph" },{ id: "multi-agent" },{ id: "production-agents" },
  ]},
];

const COMING_SOON_TEMPLATE = (stackTitle, color, modId) => `"use client";
import React from 'react';

export default function Page() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] text-center">
      <div className="w-16 h-16 bg-${color}-100 rounded-2xl flex items-center justify-center mb-6">
        <svg className="w-8 h-8 text-${color}-500" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" /></svg>
      </div>
      <h2 className="text-2xl font-bold text-slate-900 mb-2">Content In Progress</h2>
      <p className="text-slate-500 max-w-md">This module is being written to Stanford lecture quality. Check back soon.</p>
    </div>
  );
}
`;

const baseDir = path.join(__dirname, 'src/app/course');
let created = 0;
let skipped = 0;

for (const stack of curriculum) {
  for (const mod of stack.modules) {
    const dir = path.join(baseDir, stack.id, mod.id);
    const pagePath = path.join(dir, 'page.tsx');

    // Only create if the file does NOT already exist (never overwrite real content)
    if (!fs.existsSync(pagePath)) {
      fs.mkdirSync(dir, { recursive: true });
      fs.writeFileSync(pagePath, COMING_SOON_TEMPLATE(stack.title, stack.color, mod.id), 'utf8');
      console.log(`  Created: ${stack.id}/${mod.id}`);
      created++;
    } else {
      console.log(`  Skipped (exists): ${stack.id}/${mod.id}`);
      skipped++;
    }
  }
}
console.log(`\nDone. Created ${created} new pages, skipped ${skipped} existing pages.`);
