"use client";

import React from "react";

const takeaways = [
  "Linear regression is the simplest model that still teaches the full ML workflow: feature design, optimization, and evaluation.",
  "The model learns coefficients by minimizing squared error, which makes the objective differentiable and convex.",
  "Assumptions matter: linearity, independence, homoscedasticity, and low multicollinearity all affect reliability.",
];

export default function Page() {
  return (
    <div className="space-y-8 pb-10 text-slate-700">
      <header className="space-y-4">
        <div className="inline-flex items-center gap-2 rounded-full bg-emerald-100 px-3 py-1 text-[10px] font-bold uppercase tracking-[0.2em] text-emerald-700">
          Track 1 · Classical ML
        </div>
        <h1 className="text-4xl font-black tracking-tight text-slate-900">Linear Regression</h1>
        <p className="max-w-3xl text-lg leading-8 text-slate-600">
          Linear regression remains the foundation of predictive modeling. It is the clearest way to understand how
          machine learning turns data into a predictive function, how optimization works in practice, and how to judge
          a model by the quality of its error.
        </p>
      </header>

      <section className="grid gap-4 md:grid-cols-3">
        <div className="rounded-2xl border border-emerald-100 bg-emerald-50 p-4">
          <div className="text-xs font-semibold uppercase tracking-[0.2em] text-emerald-700">Use case</div>
          <p className="mt-3 text-sm leading-6 text-slate-700">
            Forecasting revenue, predicting housing prices, estimating demand, modeling risk, and deriving causal-like
            explanations from structured data.
          </p>
        </div>
        <div className="rounded-2xl border border-sky-100 bg-sky-50 p-4">
          <div className="text-xs font-semibold uppercase tracking-[0.2em] text-sky-700">Core idea</div>
          <p className="mt-3 text-sm leading-6 text-slate-700">
            Learn a coefficient vector w and bias b so that each prediction approximates the target as closely as possible.
          </p>
        </div>
        <div className="rounded-2xl border border-violet-100 bg-violet-50 p-4">
          <div className="text-xs font-semibold uppercase tracking-[0.2em] text-violet-700">Objective</div>
          <p className="mt-3 text-sm leading-6 text-slate-700">
            Minimize the squared residuals so the model balances bias and variance while staying differentiable.
          </p>
        </div>
      </section>

      <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm ring-1 ring-slate-100">
        <h2 className="mb-3 text-2xl font-bold text-slate-900">The model</h2>
        <p className="mb-4 text-slate-600">
          For a sample with feature vector x and target y, the model predicts:
        </p>
        <div className="rounded-2xl bg-slate-950 p-4 text-sm text-cyan-300">
          ŷ = wᵀx + b
        </div>
        <p className="mt-4 text-slate-600">
          The learning problem is to choose the parameters w and b that reduce the gap between predicted values and observed
          values on the training set.
        </p>
      </section>

      <section className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
        <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm ring-1 ring-slate-100">
          <h2 className="mb-4 text-2xl font-bold text-slate-900">Optimization target</h2>
          <p className="mb-4 text-slate-600">
            Linear regression commonly minimizes mean squared error (MSE):
          </p>
          <div className="rounded-2xl bg-slate-950 p-4 text-sm text-cyan-300">
            MSE = (1/n) Σ (yᵢ - (wᵀxᵢ + b))²
          </div>
          <p className="mt-4 text-slate-600">
            Squaring the error has an important benefit: it creates a smooth, convex objective, so gradient-based methods
            can reliably find the best-fitting parameters.
          </p>
        </div>

        <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm ring-1 ring-slate-100">
          <h2 className="mb-4 text-2xl font-bold text-slate-900">Why it matters</h2>
          <ul className="space-y-3 text-slate-600">
            {takeaways.map((point) => (
              <li key={point} className="flex gap-3">
                <span className="mt-1 h-2.5 w-2.5 rounded-full bg-emerald-500" />
                <span>{point}</span>
              </li>
            ))}
          </ul>
        </div>
      </section>

      <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm ring-1 ring-slate-100">
        <h2 className="mb-4 text-2xl font-bold text-slate-900">Example</h2>
        <div className="rounded-2xl bg-slate-950 p-4 text-sm text-slate-200">
          <pre className="overflow-x-auto whitespace-pre-wrap">
{`import numpy as np

X = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([2.1, 3.9, 5.8, 8.2])

# Fit ordinary least squares
w, *_ = np.linalg.lstsq(X, y, rcond=None)
print(f"Learned slope: {w[0]:.3f}")`}
          </pre>
        </div>
      </section>

      <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm ring-1 ring-slate-100">
        <h2 className="mb-4 text-2xl font-bold text-slate-900">Evaluation and assumptions</h2>
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <h3 className="mb-2 font-semibold text-slate-900">Common metrics</h3>
            <ul className="space-y-2 text-slate-600">
              <li>• MAE: average absolute deviation</li>
              <li>• RMSE: root mean squared error</li>
              <li>• R²: proportion of variance explained</li>
            </ul>
          </div>
          <div>
            <h3 className="mb-2 font-semibold text-slate-900">Key assumptions</h3>
            <ul className="space-y-2 text-slate-600">
              <li>• Linear relationship between features and target</li>
              <li>• Residuals are roughly independent and homoscedastic</li>
              <li>• Features are not dominated by multicollinearity</li>
            </ul>
          </div>
        </div>
      </section>

      <section className="rounded-3xl border border-emerald-200 bg-emerald-50 p-6">
        <h2 className="mb-3 text-2xl font-bold text-slate-900">Key takeaways</h2>
        <p className="text-slate-700">
          Linear regression is not just a model; it is a blueprint for machine learning. It teaches the core pattern of
          parameter learning, error minimization, and model validation that underpins far more complex systems such as
          logistic regression, neural networks, and deep learning models.
        </p>
      </section>
    </div>
  );
}

