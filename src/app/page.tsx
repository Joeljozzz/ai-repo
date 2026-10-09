"use client";

import React from 'react';
import Link from 'next/link';

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[70vh] text-center animate-in fade-in zoom-in duration-500">
      <div className="w-20 h-20 bg-blue-600 rounded-2xl shadow-xl flex items-center justify-center mb-8 transform -rotate-6 hover:rotate-0 transition-all cursor-pointer">
        <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/><path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4 4.5 4.5 0 0 1 3 4 4.5 4.5 0 0 1 3-4Z"/></svg>
      </div>
      
      <h1 className="text-5xl font-extrabold text-slate-900 tracking-tight mb-6">
        The Complete AI Engineering Curriculum
      </h1>
      
      <p className="text-xl text-slate-600 max-w-2xl mb-12">
        A fully interactive, multi-stack certification path covering Classical Machine Learning, Deep Learning, Generative AI, and MLOps.
      </p>

      <Link 
        href="/course/ml/linear-models" 
        className="px-8 py-4 bg-slate-900 text-white rounded-xl font-bold hover:bg-blue-600 transition-colors shadow-lg hover:shadow-blue-500/30 flex items-center gap-3"
      >
        Start Course
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>
      </Link>
    </div>
  );
}
