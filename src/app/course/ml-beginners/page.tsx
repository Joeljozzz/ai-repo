"use client";

import React, { useState } from "react";
import Link from "next/link";
import {
  Brain,
  Sparkles,
  ArrowRight,
  TrendingUp,
  Scale,
  LineChart,
  Globe,
  Boxes,
  MessageSquare,
  Smile,
  Clock,
  Rocket,
  Search,
  HelpCircle
} from "lucide-react";
import { mlBeginnersWeeks, quizzes, QuizItem } from "@/data/mlBeginnersData";
import InteractiveQuizSuite from "@/components/InteractiveQuizSuite";

const iconMap: Record<string, React.ComponentType<{ className?: string }>> = {
  Brain,
  Scale,
  LineChart,
  TrendingUp,
  Globe,
  Sparkles,
  Boxes,
  MessageSquare,
  Smile,
  Clock,
  Rocket
};

export default function MLBeginnersHub() {
  const [activeTab, setActiveTab] = useState<"roadmap" | "quizzes">("roadmap");
  const [selectedCategory, setSelectedCategory] = useState<string>("All");
  const [quizSearch, setQuizSearch] = useState("");
  const [activeQuizModal, setActiveQuizModal] = useState<QuizItem | null>(null);
  const [completedQuizzesCount, setCompletedQuizzesCount] = useState<number>(() => {
    if (typeof window === "undefined") return 0;
    let count = 0;
    for (const q of quizzes) {
      if (localStorage.getItem(`ml_quiz_${q.id}`) !== null) {
        count++;
      }
    }
    return count;
  });

  const categories = [
    "All",
    "Introduction",
    "Regression",
    "Web-App",
    "Classification",
    "Clustering",
    "NLP",
    "TimeSeries",
    "Reinforcement",
    "Real-World"
  ];

  const filteredWeeks = mlBeginnersWeeks
    .map((week) => {
      if (selectedCategory === "All") return week;
      const matchedLessons = week.lessons.filter((l) => l.category === selectedCategory);
      return { ...week, lessons: matchedLessons };
    })
    .filter((week) => week.lessons.length > 0);

  const filteredQuizzes = quizzes.filter((q) => {
    return q.title.toLowerCase().includes(quizSearch.toLowerCase());
  });

  return (
    <div className="space-y-10 py-6 pb-20">
      {/* ── Hero Section ── */}
      <section className="relative overflow-hidden rounded-[28px] bg-slate-950 p-8 text-white shadow-2xl md:p-12">
        <div className="relative z-10 max-w-3xl space-y-5">
          <div className="inline-flex items-center gap-2 rounded-full border border-emerald-400/30 bg-emerald-500/10 px-3.5 py-1.5 text-xs font-semibold uppercase tracking-wider text-emerald-300">
            <Sparkles className="h-3.5 w-3.5 text-emerald-400" />
            12 Weeks · 26 Lessons · 52 Interactive Quizzes
          </div>

          <h1 className="text-3xl font-black tracking-tight text-white sm:text-5xl">
            Machine Learning for Beginners
          </h1>

          <p className="text-base text-slate-300 sm:text-lg leading-relaxed">
            The world-renowned classical Machine Learning curriculum transformed into an interactive web experience.
            From statistical foundations and pumpkin regression to hotel review NLP, ARIMA time series, and Q-learning.
          </p>

          <div className="flex flex-wrap items-center gap-6 pt-2">
            <div className="flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-white/10 text-emerald-400 font-bold">
                26
              </div>
              <div className="text-xs">
                <div className="font-bold text-white">Full Lessons</div>
                <div className="text-slate-400">Step-by-step guides</div>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-white/10 text-cyan-400 font-bold">
                52
              </div>
              <div className="text-xs">
                <div className="font-bold text-white">Pre & Post Quizzes</div>
                <div className="text-slate-400">Instant score evaluation</div>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-emerald-500/20 text-emerald-300 font-bold">
                {completedQuizzesCount}/52
              </div>
              <div className="text-xs">
                <div className="font-bold text-white">Your Progress</div>
                <div className="text-slate-400">Saved to browser</div>
              </div>
            </div>
          </div>
        </div>

        {/* Ambient background glow */}
        <div className="pointer-events-none absolute -right-24 -top-24 h-96 w-96 rounded-full bg-emerald-500/15 blur-3xl" />
        <div className="pointer-events-none absolute -bottom-24 right-1/3 h-80 w-80 rounded-full bg-cyan-500/10 blur-3xl" />
      </section>

      {/* ── View Switcher & Filters ── */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between border-b border-slate-200 pb-4">
        {/* Tab Switcher */}
        <div className="inline-flex rounded-xl bg-slate-200/70 p-1">
          <button
            onClick={() => setActiveTab("roadmap")}
            className={`rounded-lg px-4 py-2 text-xs font-bold transition ${
              activeTab === "roadmap"
                ? "bg-white text-slate-900 shadow-sm"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            🗺️ 12-Week Roadmap
          </button>
          <button
            onClick={() => setActiveTab("quizzes")}
            className={`rounded-lg px-4 py-2 text-xs font-bold transition flex items-center gap-2 ${
              activeTab === "quizzes"
                ? "bg-white text-slate-900 shadow-sm"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            <HelpCircle className="h-3.5 w-3.5 text-blue-600" />
            52-Quiz Challenge Arena ({quizzes.length})
          </button>
        </div>

        {/* Category Filter Pills (Roadmap view) */}
        {activeTab === "roadmap" && (
          <div className="flex items-center gap-1.5 overflow-x-auto pb-1 text-xs">
            {categories.map((cat) => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`rounded-full px-3 py-1 font-medium transition shrink-0 ${
                  selectedCategory === cat
                    ? "bg-slate-900 text-white"
                    : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                }`}
              >
                {cat}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* ── Tab Content: 12-Week Roadmap ── */}
      {activeTab === "roadmap" && (
        <div className="space-y-8">
          {filteredWeeks.map((week) => {
            const IconComponent = iconMap[week.icon] || Brain;
            return (
              <div
                key={week.week}
                className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm ring-1 ring-slate-100"
              >
                {/* Week Header */}
                <div className="flex items-center justify-between border-b border-slate-100 bg-slate-50/70 px-6 py-4">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-100 text-emerald-700">
                      <IconComponent className="h-5 w-5" />
                    </div>
                    <div>
                      <div className="text-[11px] font-bold uppercase tracking-wider text-emerald-700">
                        Week {week.week}
                      </div>
                      <h2 className="text-lg font-bold text-slate-900">{week.title}</h2>
                    </div>
                  </div>
                  <span className="rounded-full bg-slate-200/60 px-3 py-1 text-xs font-semibold text-slate-700">
                    {week.lessons.length} {week.lessons.length === 1 ? "Lesson" : "Lessons"}
                  </span>
                </div>

                <div className="p-6">
                  <p className="mb-6 text-sm text-slate-600 leading-relaxed">
                    {week.description}
                  </p>

                  {/* Lessons Grid */}
                  <div className="grid gap-4 md:grid-cols-2">
                    {week.lessons.map((lesson) => (
                      <div
                        key={lesson.id}
                        className="flex flex-col justify-between rounded-xl border border-slate-200/80 bg-slate-50/40 p-5 transition hover:border-blue-300 hover:bg-blue-50/30"
                      >
                        <div className="space-y-2">
                          <div className="flex items-center justify-between gap-2">
                            <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                              Lesson {lesson.lessonNumber} · {lesson.readTime}
                            </span>
                            <span className="rounded-md bg-blue-100 px-2 py-0.5 text-[10px] font-semibold text-blue-700">
                              {lesson.category}
                            </span>
                          </div>

                          <h3 className="text-base font-bold text-slate-900 leading-snug">
                            {lesson.title}
                          </h3>

                          <p className="text-xs text-slate-600 line-clamp-2 leading-relaxed">
                            {lesson.shortDesc}
                          </p>
                        </div>

                        {/* Action buttons */}
                        <div className="mt-5 flex items-center justify-between gap-2 border-t border-slate-200/60 pt-3 text-xs">
                          <div className="flex items-center gap-2">
                            <button
                              onClick={() => {
                                const q = quizzes.find((item) => item.id === lesson.preQuizId);
                                if (q) setActiveQuizModal(q);
                              }}
                              className="rounded-lg bg-slate-200/80 px-2.5 py-1 font-semibold text-slate-700 hover:bg-blue-100 hover:text-blue-700 transition"
                              title="Take Pre-Lecture Quiz"
                            >
                              Pre-Quiz #{lesson.preQuizId}
                            </button>
                            <button
                              onClick={() => {
                                const q = quizzes.find((item) => item.id === lesson.postQuizId);
                                if (q) setActiveQuizModal(q);
                              }}
                              className="rounded-lg bg-slate-200/80 px-2.5 py-1 font-semibold text-slate-700 hover:bg-emerald-100 hover:text-emerald-700 transition"
                              title="Take Post-Lecture Quiz"
                            >
                              Post-Quiz #{lesson.postQuizId}
                            </button>
                          </div>

                          <Link
                            href={`/course/ml-beginners/${lesson.id}`}
                            className="inline-flex items-center gap-1 font-bold text-blue-600 hover:text-blue-800 transition"
                          >
                            Study Lesson
                            <ArrowRight className="h-3.5 w-3.5" />
                          </Link>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* ── Tab Content: 52-Quiz Challenge Arena ── */}
      {activeTab === "quizzes" && (
        <div className="space-y-6">
          {/* Search bar */}
          <div className="relative">
            <Search className="absolute left-3.5 top-3.5 h-4 w-4 text-slate-400" />
            <input
              type="text"
              value={quizSearch}
              onChange={(e) => setQuizSearch(e.target.value)}
              placeholder="Search by topic, algorithm, or lesson title (e.g., ARIMA, Regression, Q-Learning, Clustering)..."
              className="w-full rounded-xl border border-slate-200 bg-white pl-10 pr-4 py-3 text-sm text-slate-800 placeholder-slate-400 focus:border-blue-500 focus:outline-none shadow-sm"
            />
          </div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {filteredQuizzes.map((q) => {
              const isPre = q.title.toLowerCase().includes("pre-lecture");
              return (
                <div
                  key={q.id}
                  className="flex flex-col justify-between rounded-xl border border-slate-200 bg-white p-5 shadow-sm transition hover:border-blue-400 hover:shadow-md"
                >
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="flex h-6 w-6 items-center justify-center rounded-md bg-slate-100 text-xs font-bold text-slate-700">
                        #{q.id}
                      </span>
                      <span
                        className={`rounded-full px-2.5 py-0.5 text-[10px] font-bold uppercase ${
                          isPre
                            ? "bg-amber-100 text-amber-800"
                            : "bg-emerald-100 text-emerald-800"
                        }`}
                      >
                        {isPre ? "Pre-Quiz" : "Post-Quiz"}
                      </span>
                    </div>

                    <h3 className="font-bold text-slate-900 text-sm leading-snug">
                      {q.title}
                    </h3>

                    <p className="text-xs text-slate-500">
                      {q.quiz.length} multiple-choice challenge questions
                    </p>
                  </div>

                  <button
                    onClick={() => setActiveQuizModal(q)}
                    className="mt-4 w-full rounded-lg bg-slate-900 py-2 text-center text-xs font-bold text-white transition hover:bg-blue-600"
                  >
                    Start Quiz Now
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── Active Quiz Modal ── */}
      {activeQuizModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 p-4 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="relative max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-3xl bg-white p-6 shadow-2xl">
            <button
              onClick={() => setActiveQuizModal(null)}
              className="absolute right-5 top-5 flex h-8 w-8 items-center justify-center rounded-full bg-slate-100 text-slate-600 hover:bg-slate-200 transition"
            >
              ✕
            </button>
            <InteractiveQuizSuite
              quizData={activeQuizModal}
              badgeLabel="Arena Challenge"
              onComplete={() => {
                // refresh completed score tally
                let count = 0;
                for (const q of quizzes) {
                  if (localStorage.getItem(`ml_quiz_${q.id}`) !== null) count++;
                }
                setCompletedQuizzesCount(count);
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
