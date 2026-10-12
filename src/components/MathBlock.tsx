"use client";
import React from "react";

interface MathStep {
  label?: string;
  formula: string;
  annotation?: string;
}

interface MathBlockProps {
  title?: string;
  steps: MathStep[];
  result?: string;
  resultNote?: string;
}

export default function MathBlock({ title, steps, result, resultNote }: MathBlockProps) {
  return (
    <div className="not-prose bg-slate-900 rounded-xl overflow-hidden my-8 shadow-lg border border-slate-700/50">
      {title && (
        <div className="px-5 py-3 bg-slate-800/60 border-b border-slate-700/50 flex items-center gap-2">
          <span className="text-amber-400 text-xs">∑</span>
          <span className="text-slate-300 text-sm font-semibold tracking-wide">{title}</span>
        </div>
      )}
      <div className="px-5 py-4 space-y-3">
        {steps.map((step, i) => (
          <div key={i} className="flex items-start gap-3">
            {step.label && (
              <span className="text-slate-500 text-xs font-mono mt-1 w-24 shrink-0">{step.label}</span>
            )}
            <div className="flex-1">
              <p className="font-mono text-slate-100 text-sm">{step.formula}</p>
              {step.annotation && (
                <p className="text-slate-400 text-xs mt-1 font-sans">{step.annotation}</p>
              )}
            </div>
          </div>
        ))}
        {result && (
          <div className="mt-4 pt-3 border-t border-slate-700/50">
            <p className="font-mono text-emerald-300 text-sm font-bold">{result}</p>
            {resultNote && (
              <p className="text-slate-400 text-xs mt-1 font-sans">{resultNote}</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
