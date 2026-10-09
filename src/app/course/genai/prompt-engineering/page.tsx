"use client";
import React from 'react';

export default function Page() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] text-center">
      <div className="w-16 h-16 bg-indigo-100 rounded-2xl flex items-center justify-center mb-6">
        <svg className="w-8 h-8 text-indigo-500" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" /></svg>
      </div>
      <h2 className="text-2xl font-bold text-slate-900 mb-2">Content In Progress</h2>
      <p className="text-slate-500 max-w-md">This module is being written to Stanford lecture quality. Check back soon.</p>
    </div>
  );
}
