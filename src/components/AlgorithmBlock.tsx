"use client";
import React from "react";

interface Step {
  step: number | string;
  title: string;
  detail: string | React.ReactNode;
  complexity?: string;
}

interface AlgorithmBlockProps {
  name: string;
  input: string;
  output: string;
  timeComplexity: string;
  spaceComplexity: string;
  steps: Step[];
}

export default function AlgorithmBlock({
  name,
  input,
  output,
  timeComplexity,
  spaceComplexity,
  steps,
}: AlgorithmBlockProps) {
  return (
    <div className="not-prose my-8 overflow-hidden rounded-xl border border-indigo-100 bg-white shadow-sm ring-1 ring-slate-900/5">
      <div className="border-b border-indigo-50 bg-gradient-to-r from-indigo-50/70 via-white to-blue-50/40 px-6 py-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <span className="flex h-6 w-6 items-center justify-center rounded-md bg-indigo-600 text-xs font-bold text-white shadow-sm">
              λ
            </span>
            <h4 className="text-base font-bold text-slate-900">{name}</h4>
          </div>
          <div className="flex items-center gap-2 text-xs font-mono">
            <span className="rounded bg-slate-100 px-2.5 py-1 text-slate-700 border border-slate-200">
              Time: <strong className="text-indigo-600">{timeComplexity}</strong>
            </span>
            <span className="rounded bg-slate-100 px-2.5 py-1 text-slate-700 border border-slate-200">
              Space: <strong className="text-indigo-600">{spaceComplexity}</strong>
            </span>
          </div>
        </div>

        <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs font-mono text-slate-600 bg-white/80 p-2.5 rounded-lg border border-indigo-50">
          <div>
            <span className="font-semibold text-slate-500 uppercase tracking-wider">Input:</span> {input}
          </div>
          <div>
            <span className="font-semibold text-slate-500 uppercase tracking-wider">Output:</span> {output}
          </div>
        </div>
      </div>

      <div className="divide-y divide-slate-100 px-6 py-2">
        {steps.map((s, idx) => (
          <div key={idx} className="flex items-start gap-4 py-3.5">
            <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-indigo-50 text-xs font-bold text-indigo-700">
              {s.step}
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between gap-2">
                <span className="text-sm font-semibold text-slate-900">{s.title}</span>
                {s.complexity && (
                  <span className="text-[11px] font-mono text-slate-400">{s.complexity}</span>
                )}
              </div>
              <div className="mt-1 text-xs text-slate-600 leading-relaxed">{s.detail}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
