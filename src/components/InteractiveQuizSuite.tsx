"use client";

import React, { useState } from "react";
import { CheckCircle2, XCircle, RotateCcw, Award, ArrowRight, HelpCircle } from "lucide-react";
import { QuizItem, getQuizById } from "@/data/mlBeginnersData";

interface InteractiveQuizSuiteProps {
  quizId?: number;
  quizData?: QuizItem;
  titleOverride?: string;
  badgeLabel?: string;
  onComplete?: (score: number, total: number) => void;
}

export default function InteractiveQuizSuite({
  quizId,
  quizData,
  titleOverride,
  badgeLabel = "Interactive Quiz",
  onComplete
}: InteractiveQuizSuiteProps) {
  const quiz = quizData || (quizId ? getQuizById(quizId) : undefined);

  const [currentIndex, setCurrentIndex] = useState(0);
  const [selectedOptionIndex, setSelectedOptionIndex] = useState<number | null>(null);
  const [isAnswerSubmitted, setIsAnswerSubmitted] = useState(false);
  const [userScore, setUserScore] = useState(0);
  const [isFinished, setIsFinished] = useState(false);
  const [savedScore, setSavedScore] = useState<number | null>(() => {
    if (typeof window === "undefined" || !quiz) return null;
    const stored = localStorage.getItem(`ml_quiz_${quiz.id}`);
    return stored !== null ? parseInt(stored, 10) : null;
  });

  if (!quiz || !quiz.quiz || quiz.quiz.length === 0) {
    return (
      <div className="bg-slate-50 border border-slate-200 rounded-2xl p-6 text-slate-500 text-sm">
        Quiz data not found for ID: {quizId}
      </div>
    );
  }

  const currentQ = quiz.quiz[currentIndex];
  const totalQuestions = quiz.quiz.length;

  const handleSelectOption = (index: number) => {
    if (isAnswerSubmitted) return;
    setSelectedOptionIndex(index);
  };

  const handleSubmitAnswer = () => {
    if (selectedOptionIndex === null) return;
    setIsAnswerSubmitted(true);

    const selectedOpt = currentQ.answerOptions[selectedOptionIndex];
    const isCorrect = selectedOpt.isCorrect === true || selectedOpt.isCorrect === "true";

    if (isCorrect) {
      setUserScore((prev) => prev + 1);
    }
  };

  const handleNextQuestion = () => {
    if (currentIndex + 1 < totalQuestions) {
      setCurrentIndex((prev) => prev + 1);
      setSelectedOptionIndex(null);
      setIsAnswerSubmitted(false);
    } else {
      setIsFinished(true);
      const finalScore = userScore + (
        currentQ.answerOptions[selectedOptionIndex!]?.isCorrect === true ||
        currentQ.answerOptions[selectedOptionIndex!]?.isCorrect === "true" ? 1 : 0
      );
      localStorage.setItem(`ml_quiz_${quiz.id}`, finalScore.toString());
      setSavedScore(finalScore);
      if (onComplete) {
        onComplete(finalScore, totalQuestions);
      }
    }
  };

  const handleRestart = () => {
    setCurrentIndex(0);
    setSelectedOptionIndex(null);
    setIsAnswerSubmitted(false);
    setUserScore(0);
    setIsFinished(false);
  };

  return (
    <div className="not-prose my-8 overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm ring-1 ring-slate-100 transition-all">
      {/* Header bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 bg-slate-50/80 px-6 py-4">
        <div className="flex items-center gap-2.5">
          <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-blue-600 text-xs font-bold text-white shadow-sm">
            <HelpCircle className="h-4 w-4" />
          </span>
          <div>
            <div className="text-[11px] font-bold uppercase tracking-wider text-blue-600">
              {badgeLabel} {quiz.id ? `· Quiz #${quiz.id}` : ""}
            </div>
            <h3 className="text-base font-bold text-slate-900">
              {titleOverride || quiz.title}
            </h3>
          </div>
        </div>

        {savedScore !== null && !isFinished && (
          <div className="flex items-center gap-1.5 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-700">
            <CheckCircle2 className="h-3.5 w-3.5" />
            Previous Best: {savedScore}/{totalQuestions}
          </div>
        )}
      </div>

      <div className="p-6 md:p-8">
        {!isFinished ? (
          <div>
            {/* Step indicators */}
            <div className="mb-6 flex items-center justify-between">
              <span className="text-xs font-semibold tracking-wide text-slate-400 uppercase">
                Question {currentIndex + 1} of {totalQuestions}
              </span>
              <div className="flex gap-1.5">
                {quiz.quiz.map((_, i) => (
                  <div
                    key={i}
                    className={`h-2 rounded-full transition-all ${
                      i === currentIndex
                        ? "w-8 bg-blue-600"
                        : i < currentIndex
                        ? "w-2.5 bg-emerald-500"
                        : "w-2.5 bg-slate-200"
                    }`}
                  />
                ))}
              </div>
            </div>

            {/* Question Text */}
            <h4 className="mb-6 text-lg font-bold text-slate-900 leading-snug">
              {currentQ.questionText}
            </h4>

            {/* Answer Options */}
            <div className="space-y-3 mb-6">
              {currentQ.answerOptions.map((opt, idx) => {
                const isThisSelected = selectedOptionIndex === idx;
                const isOptCorrect = opt.isCorrect === true || opt.isCorrect === "true";

                let borderBg = "border-slate-200 bg-white hover:border-blue-300 hover:bg-slate-50 text-slate-800";
                if (isAnswerSubmitted) {
                  if (isOptCorrect) {
                    borderBg = "border-emerald-500 bg-emerald-50/80 text-emerald-950 font-semibold";
                  } else if (isThisSelected) {
                    borderBg = "border-rose-500 bg-rose-50/80 text-rose-950 font-semibold";
                  } else {
                    borderBg = "border-slate-200 bg-slate-50/50 text-slate-400 opacity-60";
                  }
                } else if (isThisSelected) {
                  borderBg = "border-blue-600 bg-blue-50/70 text-blue-950 font-semibold ring-2 ring-blue-600/20";
                }

                return (
                  <button
                    key={idx}
                    onClick={() => handleSelectOption(idx)}
                    disabled={isAnswerSubmitted}
                    className={`w-full text-left p-4 rounded-xl border transition-all flex items-start gap-3.5 ${borderBg}`}
                  >
                    <div className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-inherit text-xs font-bold">
                      {isAnswerSubmitted && isOptCorrect && (
                        <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                      )}
                      {isAnswerSubmitted && isThisSelected && !isOptCorrect && (
                        <XCircle className="h-4 w-4 text-rose-600" />
                      )}
                      {(!isAnswerSubmitted || (!isOptCorrect && !isThisSelected)) && (
                        <span>{String.fromCharCode(65 + idx)}</span>
                      )}
                    </div>
                    <span className="text-sm sm:text-base leading-relaxed flex-1">
                      {opt.answerText}
                    </span>
                  </button>
                );
              })}
            </div>

            {/* Action Bar */}
            <div className="flex items-center justify-between pt-2">
              <div className="text-xs text-slate-400">
                {!isAnswerSubmitted
                  ? "Select the best option and click Submit"
                  : selectedOptionIndex !== null &&
                    (currentQ.answerOptions[selectedOptionIndex].isCorrect === true ||
                     currentQ.answerOptions[selectedOptionIndex].isCorrect === "true")
                  ? "🎉 Correct answer!"
                  : "💡 Keep learning! The correct answer is highlighted in green."}
              </div>

              {!isAnswerSubmitted ? (
                <button
                  onClick={handleSubmitAnswer}
                  disabled={selectedOptionIndex === null}
                  className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  Submit Answer
                </button>
              ) : (
                <button
                  onClick={handleNextQuestion}
                  className="inline-flex items-center gap-2 rounded-xl bg-slate-900 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-slate-800"
                >
                  {currentIndex + 1 < totalQuestions ? "Next Question" : "Complete Quiz"}
                  <ArrowRight className="h-4 w-4" />
                </button>
              )}
            </div>
          </div>
        ) : (
          /* Result Summary Screen */
          <div className="text-center py-6">
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-tr from-blue-600 to-indigo-600 text-white shadow-lg shadow-blue-500/30">
              <Award className="h-8 w-8" />
            </div>

            <h4 className="text-2xl font-black text-slate-900 mb-1">Quiz Completed!</h4>
            <p className="text-sm text-slate-500 mb-6">
              You scored <span className="font-bold text-blue-600">{userScore}</span> out of{" "}
              <span className="font-bold text-slate-800">{totalQuestions}</span> (
              {Math.round((userScore / totalQuestions) * 100)}%)
            </p>

            <div className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-4 py-2 text-xs font-semibold text-slate-700 mb-6">
              {userScore === totalQuestions
                ? "🌟 Outstanding! You mastered all concepts in this quiz."
                : userScore >= Math.ceil(totalQuestions / 2)
                ? "👍 Solid comprehension! Review the highlighted notes to reinforce gaps."
                : "📚 Great baseline check! Review the lesson content and try again."}
            </div>

            <div className="flex flex-wrap items-center justify-center gap-3">
              <button
                onClick={handleRestart}
                className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
              >
                <RotateCcw className="h-4 w-4" />
                Retake Quiz
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
