"use client";

import React, { useState } from 'react';

interface InteractiveQuizProps {
  question: string;
  options: string[];
  correctIndex: number;
  explanation: string;
}

export default function InteractiveQuiz({ question, options, correctIndex, explanation }: InteractiveQuizProps) {
  const [selected, setSelected] = useState<number | null>(null);
  const [submitted, setSubmitted] = useState(false);

  const isCorrect = selected === correctIndex;

  return (
    <div className="bg-white border border-slate-200 rounded-2xl p-8 shadow-sm my-8 not-prose">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-8 h-8 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center font-bold text-sm">?</div>
        <h3 className="font-bold text-xl text-slate-800">{question}</h3>
      </div>
      
      <div className="space-y-3 mb-6">
        {options.map((opt, i) => (
          <button
            key={i}
            onClick={() => !submitted && setSelected(i)}
            disabled={submitted}
            className={`w-full text-left px-5 py-4 rounded-xl border transition-all ${
              submitted
                ? i === correctIndex
                  ? 'border-emerald-500 bg-emerald-50 text-emerald-900' // Correct answer highlighted
                  : selected === i
                    ? 'border-red-500 bg-red-50 text-red-900' // Wrong answer highlighted
                    : 'border-slate-200 bg-slate-50 text-slate-400 opacity-50' // Unselected
                : selected === i
                  ? 'border-blue-500 bg-blue-50 text-blue-900 ring-2 ring-blue-500/20' // Selected before submit
                  : 'border-slate-200 hover:border-blue-300 hover:bg-slate-50 text-slate-700'
            }`}
          >
            <div className="flex items-center gap-3">
              <div className={`w-5 h-5 rounded-full border flex items-center justify-center shrink-0 ${
                submitted 
                  ? i === correctIndex ? 'border-emerald-500 bg-emerald-500 text-white' : selected === i ? 'border-red-500 bg-red-500 text-white' : 'border-slate-300'
                  : selected === i ? 'border-blue-500 bg-blue-500' : 'border-slate-300'
              }`}>
                {submitted && i === correctIndex && <span className="text-xs">✓</span>}
                {submitted && selected === i && i !== correctIndex && <span className="text-xs">✕</span>}
              </div>
              <span className="font-medium text-sm sm:text-base">{opt}</span>
            </div>
          </button>
        ))}
      </div>

      {!submitted ? (
        <button
          onClick={() => setSubmitted(true)}
          disabled={selected === null}
          className="px-6 py-2.5 bg-slate-900 text-white rounded-lg font-medium hover:bg-slate-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Check Answer
        </button>
      ) : (
        <div className={`p-5 rounded-xl border ${isCorrect ? 'bg-emerald-50 border-emerald-200' : 'bg-rose-50 border-rose-200'}`}>
          <h4 className={`font-bold mb-2 ${isCorrect ? 'text-emerald-900' : 'text-rose-900'}`}>
            {isCorrect ? 'Correct! 🎉' : 'Not quite. 🤔'}
          </h4>
          <p className={isCorrect ? 'text-emerald-800' : 'text-rose-800'}>
            {explanation}
          </p>
          <button
            onClick={() => { setSelected(null); setSubmitted(false); }}
            className={`mt-4 px-4 py-2 text-sm font-medium rounded-lg border ${
              isCorrect ? 'border-emerald-300 text-emerald-800 hover:bg-emerald-100' : 'border-rose-300 text-rose-800 hover:bg-rose-100'
            }`}
          >
            Try Again
          </button>
        </div>
      )}
    </div>
  );
}
