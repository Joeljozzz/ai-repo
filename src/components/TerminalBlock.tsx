"use client";

import React from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface TerminalBlockProps {
  language: string;
  filename?: string;
  code: string;
}

export default function TerminalBlock({ language, filename, code }: TerminalBlockProps) {
  return (
    <div className="rounded-xl overflow-hidden shadow-2xl border border-slate-700 bg-[#1e1e1e] my-8 font-mono">
      <div className="bg-[#2d2d2d] px-4 py-3 flex items-center gap-2 border-b border-black/40">
        <div className="flex gap-2">
          <div className="w-3 h-3 rounded-full bg-red-500 border border-red-600"></div>
          <div className="w-3 h-3 rounded-full bg-amber-500 border border-amber-600"></div>
          <div className="w-3 h-3 rounded-full bg-emerald-500 border border-emerald-600"></div>
        </div>
        {filename && (
          <span className="text-xs text-slate-400 ml-4">{filename}</span>
        )}
      </div>
      <SyntaxHighlighter 
        language={language} 
        style={vscDarkPlus} 
        showLineNumbers 
        customStyle={{ margin: 0, padding: '1.5rem', background: 'transparent' }}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
}
