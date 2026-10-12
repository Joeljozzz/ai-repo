"use client";
import React from "react";

interface Column {
  key: string;
  header: string;
  width?: string;
}

interface ComparisonTableProps {
  title?: string;
  subtitle?: string;
  columns: Column[];
  data: Record<string, React.ReactNode>[];
}

export default function ComparisonTable({ title, subtitle, columns, data }: ComparisonTableProps) {
  return (
    <div className="not-prose my-8 overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
      {(title || subtitle) && (
        <div className="border-b border-slate-200 bg-slate-50/80 px-6 py-4">
          {title && <h4 className="text-base font-bold text-slate-900">{title}</h4>}
          {subtitle && <p className="mt-1 text-xs text-slate-500 font-sans">{subtitle}</p>}
        </div>
      )}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm border-collapse">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-100/75 text-xs font-semibold uppercase tracking-wider text-slate-700">
              {columns.map((col) => (
                <th key={col.key} className="px-5 py-3.5" style={{ width: col.width }}>
                  {col.header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {data.map((row, idx) => (
              <tr key={idx} className={idx % 2 === 0 ? "bg-white hover:bg-slate-50/50" : "bg-slate-50/30 hover:bg-slate-50/60"}>
                {columns.map((col) => (
                  <td key={col.key} className="px-5 py-3.5 text-slate-700 leading-relaxed align-top">
                    {row[col.key]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
