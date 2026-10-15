import React from "react";
import Link from "next/link";
import { notFound } from "next/navigation";
import {
  ArrowLeft,
  ArrowRight,
  Clock,
  Sparkles,
  BookOpen,
  CheckCircle,
  HelpCircle,
  Code2
} from "lucide-react";
import { getAllLessons, getLessonById, getQuizById } from "@/data/mlBeginnersData";
import InteractiveQuizSuite from "@/components/InteractiveQuizSuite";

interface PageProps {
  params: Promise<{ lessonId: string }>;
}

export async function generateStaticParams() {
  const lessons = getAllLessons();
  return lessons.map((lesson) => ({
    lessonId: lesson.id,
  }));
}

export default async function LessonPage({ params }: PageProps) {
  const { lessonId } = await params;
  const lesson = getLessonById(lessonId);

  if (!lesson) {
    notFound();
  }

  const allLessons = getAllLessons();
  const currentIndex = allLessons.findIndex((l) => l.id === lesson.id);
  const prevLesson = currentIndex > 0 ? allLessons[currentIndex - 1] : null;
  const nextLesson = currentIndex < allLessons.length - 1 ? allLessons[currentIndex + 1] : null;

  const preQuiz = getQuizById(lesson.preQuizId);
  const postQuiz = getQuizById(lesson.postQuizId);

  return (
    <article className="prose prose-slate max-w-none pb-24 not-prose space-y-10">
      {/* ── Breadcrumb & Navigation ── */}
      <div className="flex items-center justify-between border-b border-slate-200 pb-4 text-xs font-semibold">
        <Link
          href="/course/ml-beginners"
          className="inline-flex items-center gap-1.5 text-slate-500 hover:text-blue-600 transition"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to 12-Week Roadmap
        </Link>
        <span className="rounded-full bg-emerald-100 px-3 py-1 text-emerald-800">
          Week {lesson.week} · Lesson {lesson.lessonNumber} of 26
        </span>
      </div>

      {/* ── Lesson Hero Header ── */}
      <header className="space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          <span className="inline-flex items-center gap-1.5 rounded-md bg-blue-100 px-2.5 py-1 text-xs font-bold uppercase tracking-wider text-blue-700">
            <Sparkles className="h-3.5 w-3.5" />
            {lesson.category}
          </span>
          <span className="inline-flex items-center gap-1 text-xs text-slate-500 font-medium">
            <Clock className="h-3.5 w-3.5 text-slate-400" />
            {lesson.readTime} read
          </span>
        </div>

        <h1 className="text-3xl font-black tracking-tight text-slate-900 sm:text-4xl">
          {lesson.title}
        </h1>

        <p className="text-lg text-slate-600 leading-relaxed max-w-3xl">
          {lesson.shortDesc}
        </p>
      </header>

      {/* ── Learning Objectives Card ── */}
      <section className="rounded-2xl border border-blue-100 bg-blue-50/50 p-6 shadow-sm">
        <h2 className="mb-4 flex items-center gap-2 text-base font-bold text-blue-900">
          <BookOpen className="h-5 w-5 text-blue-600" />
          Learning Objectives
        </h2>
        <ul className="grid gap-2.5 sm:grid-cols-2 text-sm text-slate-700">
          {lesson.objectives.map((obj, i) => (
            <li key={i} className="flex items-start gap-2.5">
              <CheckCircle className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600" />
              <span>{obj}</span>
            </li>
          ))}
        </ul>
      </section>

      {/* ── Pre-Lecture Interactive Quiz ── */}
      {preQuiz && (
        <section className="space-y-3">
          <div className="flex items-center gap-2 text-sm font-bold text-slate-800 uppercase tracking-wider">
            <HelpCircle className="h-4 w-4 text-amber-500" />
            Step 1 · Baseline Self-Assessment
          </div>
          <p className="text-xs text-slate-500">
            Test your initial intuition before studying the lesson content.
          </p>
          <InteractiveQuizSuite
            quizData={preQuiz}
            badgeLabel="Pre-Lecture Quiz"
            titleOverride={`${lesson.title}: Pre-Lecture Quiz`}
          />
        </section>
      )}

      {/* ── Mathematical Formulations (if any) ── */}
      {lesson.mathFormulas && lesson.mathFormulas.length > 0 && (
        <section className="space-y-4">
          <h2 className="text-xl font-bold text-slate-900 border-b border-slate-200 pb-2">
            Mathematical Foundations
          </h2>
          <div className="grid gap-4 sm:grid-cols-2">
            {lesson.mathFormulas.map((mf, i) => (
              <div
                key={i}
                className="rounded-xl border border-slate-200 bg-slate-900 p-5 text-white shadow-sm"
              >
                <div className="text-xs font-bold uppercase tracking-wider text-cyan-400 mb-2">
                  {mf.name}
                </div>
                <div className="my-3 font-mono text-sm overflow-x-auto rounded-lg bg-white/5 p-3 text-cyan-200 border border-white/10">
                  {mf.formula}
                </div>
                <p className="text-xs text-slate-400 leading-relaxed">
                  {mf.description}
                </p>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ── Core Conceptual Sections ── */}
      <section className="space-y-8">
        <h2 className="text-xl font-bold text-slate-900 border-b border-slate-200 pb-2">
          Core Technical Breakdown
        </h2>

        {lesson.sections.map((sec, idx) => (
          <div key={idx} className="space-y-4">
            <h3 className="text-lg font-bold text-slate-900">
              {sec.title}
            </h3>
            <p className="text-slate-700 leading-relaxed text-sm sm:text-base">
              {sec.content}
            </p>

            {sec.codeSnippet && (
              <div className="overflow-hidden rounded-xl border border-slate-800 bg-slate-950 shadow-md">
                <div className="flex items-center justify-between border-b border-slate-800 bg-slate-900/90 px-4 py-2.5 text-xs text-slate-400 font-mono">
                  <span className="flex items-center gap-2">
                    <Code2 className="h-4 w-4 text-emerald-400" />
                    {sec.codeSnippet.title}
                  </span>
                  <span className="uppercase text-[10px] text-slate-500 font-bold">
                    {sec.codeSnippet.language}
                  </span>
                </div>
                <pre className="p-4 text-xs font-mono text-emerald-300 overflow-x-auto leading-relaxed">
                  <code>{sec.codeSnippet.code}</code>
                </pre>
              </div>
            )}
          </div>
        ))}
      </section>

      {/* ── Key Technical Terminology Glossary ── */}
      {lesson.keyTerms && lesson.keyTerms.length > 0 && (
        <section className="space-y-4">
          <h2 className="text-xl font-bold text-slate-900 border-b border-slate-200 pb-2">
            Essential Terminology & Definitions
          </h2>
          <div className="grid gap-3 sm:grid-cols-2">
            {lesson.keyTerms.map((item, i) => (
              <div
                key={i}
                className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm"
              >
                <div className="font-bold text-sm text-slate-900 mb-1">
                  {item.term}
                </div>
                <p className="text-xs text-slate-600 leading-relaxed">
                  {item.definition}
                </p>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ── Post-Lecture Interactive Quiz ── */}
      {postQuiz && (
        <section className="space-y-3 pt-6 border-t border-slate-200">
          <div className="flex items-center gap-2 text-sm font-bold text-slate-800 uppercase tracking-wider">
            <CheckCircle className="h-4 w-4 text-emerald-600" />
            Step 2 · Knowledge Retention & Mastery Check
          </div>
          <p className="text-xs text-slate-500">
            Verify what you learned with the post-lecture challenge quiz.
          </p>
          <InteractiveQuizSuite
            quizData={postQuiz}
            badgeLabel="Post-Lecture Quiz"
            titleOverride={`${lesson.title}: Post-Lecture Quiz`}
          />
        </section>
      )}

      {/* ── Hands-on Assignment / Challenge Card ── */}
      <section className="rounded-2xl border border-emerald-200 bg-gradient-to-br from-emerald-50 to-teal-50 p-6 shadow-sm">
        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-emerald-800 mb-2">
          <Sparkles className="h-4 w-4 text-emerald-600" />
          Hands-On Assignment & Coding Prompt
        </div>
        <h3 className="text-lg font-bold text-emerald-950 mb-2">
          {lesson.assignment.title}
        </h3>
        <p className="text-sm text-emerald-900 leading-relaxed">
          {lesson.assignment.instructions}
        </p>
      </section>

      {/* ── Next / Previous Navigation Footer ── */}
      <footer className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between border-t border-slate-200 pt-8">
        {prevLesson ? (
          <Link
            href={`/course/ml-beginners/${prevLesson.id}`}
            className="flex items-center gap-3 rounded-xl border border-slate-200 bg-white p-4 transition hover:border-blue-400 hover:shadow-sm"
          >
            <ArrowLeft className="h-5 w-5 text-slate-400" />
            <div className="text-left">
              <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                Previous · Lesson {prevLesson.lessonNumber}
              </div>
              <div className="text-sm font-bold text-slate-800">
                {prevLesson.title}
              </div>
            </div>
          </Link>
        ) : (
          <div />
        )}

        {nextLesson ? (
          <Link
            href={`/course/ml-beginners/${nextLesson.id}`}
            className="flex items-center gap-3 rounded-xl border border-blue-200 bg-blue-50/50 p-4 transition hover:border-blue-500 hover:bg-blue-50 hover:shadow-sm"
          >
            <div className="text-right">
              <div className="text-[11px] font-bold uppercase tracking-wider text-blue-600">
                Next · Lesson {nextLesson.lessonNumber}
              </div>
              <div className="text-sm font-bold text-slate-800">
                {nextLesson.title}
              </div>
            </div>
            <ArrowRight className="h-5 w-5 text-blue-600" />
          </Link>
        ) : (
          <Link
            href="/course/ml-beginners"
            className="flex items-center gap-2 rounded-xl bg-slate-900 px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-800"
          >
            Complete 12-Week Curriculum
            <Sparkles className="h-4 w-4 text-emerald-400" />
          </Link>
        )}
      </footer>
    </article>
  );
}
