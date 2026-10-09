"use client";

import Link from "next/link";
import { ArrowRight, BookOpenText, BrainCircuit, Layers3, Sparkles, TrendingUp } from "lucide-react";
import { curriculum } from "@/data/curriculum";

const trackHighlights = [
  {
    title: "Foundations",
    value: "6+ modules",
    description: "Statistics, probability, calculus, and optimization that power modern ML systems.",
    icon: BrainCircuit,
  },
  {
    title: "Classical ML",
    value: "14 tracks",
    description: "Regression, trees, boosting, clustering, and evaluation grounded in first-principles math.",
    icon: Layers3,
  },
  {
    title: "Production AI",
    value: "RAG + agents",
    description: "Retrieval systems, evaluation loops, and autonomous tools for real-world deployment.",
    icon: TrendingUp,
  },
];

export default function Home() {
  return (
    <div className="space-y-10 py-6">
      <section className="overflow-hidden rounded-[28px] bg-slate-950 text-white shadow-[0_30px_80px_-30px_rgba(15,23,42,0.8)]">
        <div className="grid gap-8 px-6 py-8 md:px-10 md:py-10 lg:grid-cols-[1.3fr_0.7fr]">
          <div className="space-y-6">
            <div className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/5 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-200">
              <Sparkles className="h-3.5 w-3.5 text-cyan-300" />
              AI Engineering Handbook
            </div>

            <div className="space-y-4">
              <h1 className="max-w-2xl text-4xl font-black tracking-tight text-white md:text-5xl">
                Learn the full stack of modern AI engineering.
              </h1>
              <p className="max-w-xl text-lg text-slate-300">
                From mathematical foundations to production-grade RAG and agent systems, this curriculum helps you
                move from theory to shipping useful AI systems with confidence.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-4">
              <Link
                href="/course/math/descriptive-stats"
                className="inline-flex items-center gap-2 rounded-xl bg-cyan-400 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-300"
              >
                Start learning
                <ArrowRight className="h-4 w-4" />
              </Link>
              <Link
                href="#curriculum"
                className="inline-flex items-center gap-2 rounded-xl border border-white/15 bg-white/5 px-5 py-3 text-sm font-semibold text-white transition hover:bg-white/10"
              >
                Explore tracks
              </Link>
            </div>
          </div>

          <div className="rounded-2xl border border-white/10 bg-white/5 p-5 backdrop-blur-sm">
            <div className="space-y-4">
              <div className="flex items-center justify-between text-sm text-slate-300">
                <span>Learning roadmap</span>
                <span className="rounded-full bg-emerald-500/15 px-2 py-1 text-xs font-medium text-emerald-300">
                  Active
                </span>
              </div>

              <div className="space-y-3">
                {[
                  "Math foundations",
                  "Classical ML",
                  "Deep learning",
                  "LLMs and GenAI",
                  "RAG engineering",
                  "Agentic systems",
                ].map((item, index) => (
                  <div key={item} className="flex items-center gap-3">
                    <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-cyan-500/15 text-xs font-bold text-cyan-300">
                      {index + 1}
                    </div>
                    <span className="text-sm text-slate-200">{item}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        {trackHighlights.map(({ title, value, description, icon: Icon }) => (
          <div key={title} className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm ring-1 ring-slate-100">
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-slate-100 text-slate-700">
              <Icon className="h-5 w-5" />
            </div>
            <div className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">{value}</div>
            <h2 className="mb-2 text-xl font-bold text-slate-900">{title}</h2>
            <p className="text-sm leading-6 text-slate-600">{description}</p>
          </div>
        ))}
      </section>

      <section id="curriculum" className="space-y-5">
        <div className="flex items-end justify-between gap-4">
          <div>
            <div className="mb-2 inline-flex items-center gap-2 rounded-full bg-blue-100 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.2em] text-blue-700">
              <BookOpenText className="h-3.5 w-3.5" />
              Curriculum
            </div>
            <h2 className="text-3xl font-black tracking-tight text-slate-900">Explore the full learning path</h2>
          </div>
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          {curriculum.map((stack) => (
            <div key={stack.id} className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm ring-1 ring-slate-100">
              <div className="mb-2 flex items-center justify-between gap-3">
                <div>
                  <div className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">{stack.label}</div>
                  <h3 className="text-xl font-bold text-slate-900">{stack.title}</h3>
                </div>
                <div className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-semibold text-slate-600">
                  {stack.modules.length} modules
                </div>
              </div>

              <p className="mb-4 text-sm leading-6 text-slate-600">{stack.description}</p>

              <div className="space-y-2">
                {stack.modules.slice(0, 4).map((module) => (
                  <Link
                    key={module.id}
                    href={`/course/${stack.id}/${module.id}`}
                    className="flex items-center justify-between rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700 transition hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700"
                  >
                    <span className="font-medium">{module.title}</span>
                    <ArrowRight className="h-4 w-4 opacity-60" />
                  </Link>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="rounded-2xl border border-slate-200 bg-gradient-to-r from-blue-50 via-indigo-50 to-cyan-50 p-6">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <div className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Next milestone</div>
            <h3 className="text-2xl font-black tracking-tight text-slate-900">Go from intuition to deployable AI systems.</h3>
          </div>
          <Link
            href="/course/ml-core/linear-regression"
            className="inline-flex items-center gap-2 rounded-xl bg-slate-900 px-5 py-3 text-sm font-semibold text-white transition hover:bg-blue-600"
          >
            Begin with linear regression
            <ArrowRight className="h-4 w-4" />
          </Link>
        </div>
      </section>
    </div>
  );
}

