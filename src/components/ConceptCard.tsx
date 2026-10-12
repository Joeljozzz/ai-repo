"use client";
import React from "react";

type ConceptVariant = "definition" | "theorem" | "warning" | "insight" | "example";

const variantStyles: Record<ConceptVariant, { bg: string; border: string; icon: string; label: string; labelColor: string }> = {
  definition: { bg: "bg-blue-50", border: "border-blue-200", icon: "📖", label: "Definition", labelColor: "text-blue-700" },
  theorem:    { bg: "bg-purple-50", border: "border-purple-200", icon: "📐", label: "Theorem", labelColor: "text-purple-700" },
  warning:    { bg: "bg-amber-50", border: "border-amber-200", icon: "⚠️", label: "Common Pitfall", labelColor: "text-amber-700" },
  insight:    { bg: "bg-emerald-50", border: "border-emerald-200", icon: "💡", label: "Key Insight", labelColor: "text-emerald-700" },
  example:    { bg: "bg-slate-50", border: "border-slate-200", icon: "🔬", label: "Worked Example", labelColor: "text-slate-700" },
};

interface ConceptCardProps {
  variant?: ConceptVariant;
  title: string;
  children: React.ReactNode;
}

export default function ConceptCard({ variant = "definition", title, children }: ConceptCardProps) {
  const s = variantStyles[variant];
  return (
    <div className={`not-prose ${s.bg} ${s.border} border rounded-xl p-5 my-6`}>
      <div className="flex items-center gap-2 mb-3">
        <span className="text-base">{s.icon}</span>
        <span className={`text-xs font-bold uppercase tracking-widest ${s.labelColor}`}>{s.label}</span>
      </div>
      <h4 className="text-base font-bold text-slate-900 mb-2">{title}</h4>
      <div className="text-sm text-slate-700 leading-relaxed space-y-2">{children}</div>
    </div>
  );
}
