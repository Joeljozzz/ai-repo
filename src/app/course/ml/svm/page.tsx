"use client";
import React from 'react';
import TerminalBlock from '@/components/TerminalBlock';
import InteractiveQuiz from '@/components/InteractiveQuiz';

export default function SVMPage() {
  const svmCode = `import numpy as np

class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        # Convert y to {-1, 1}
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    # Point is strictly outside the margin correctly (or on it)
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    # Point violates the margin (Support Vector / Misclassified)
                    dw = 2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    db = y_[idx]

                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)`;

  return (
    <div className="space-y-12 pb-24 text-slate-800 animate-in fade-in slide-in-from-bottom-4 duration-500">
      
      <header className="border-b border-slate-200 pb-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-teal-100 text-teal-700 rounded-full text-xs font-bold uppercase tracking-widest mb-4">
          <span className="w-2 h-2 rounded-full bg-teal-600 animate-pulse"></span>
          Stack 2: Classical Machine Learning
        </div>
        <h1 className="text-4xl font-extrabold tracking-tight text-slate-900 mb-6">2.2 Support Vector Machines</h1>
        <p className="text-xl text-slate-600 leading-relaxed max-w-3xl">
          Lagrange Multipliers, Margin Maximization, Mercer's Theorem, and the Kernel Trick.
        </p>
      </header>

      <section className="prose prose-slate max-w-none text-lg text-slate-600">
        <h2 className="text-3xl font-bold text-slate-900 mt-8 mb-4">1. Margin Maximization</h2>
        <p>
          Unlike standard linear classifiers that just seek <em>any</em> boundary that separates classes, Support Vector Machines (SVMs) seek the <strong>optimal separating hyperplane</strong>—the one that maximizes the geometric distance (margin) to the nearest data points (the Support Vectors).
        </p>

        <div className="bg-slate-50 border border-slate-200 rounded-2xl p-6 my-6 shadow-sm overflow-x-auto">
          <p className="font-mono text-sm text-slate-700 mb-2 font-bold">The Primal Optimization Problem:</p>
          <div className="font-mono text-sm text-slate-800 space-y-1">
            <p>Minimize: 1/2 ||w||²</p>
            <p>Subject to: y_i(w·x_i + b) ≥ 1  for all i</p>
          </div>
          <p className="text-sm text-slate-500 mt-3">Minimizing the norm of the weight vector <code>w</code> mathematically maximizes the margin <code>2 / ||w||</code>.</p>
        </div>

        <h2 className="text-3xl font-bold text-slate-900 mt-12 mb-4">2. The Dual Formulation & Lagrange Multipliers</h2>
        <p>
          To solve this constrained optimization problem, we construct the Lagrangian by introducing Lagrange multipliers <code>α_i ≥ 0</code>. Taking the partial derivatives with respect to <code>w</code> and <code>b</code> and setting them to zero gives us the <strong>Dual Formulation</strong>.
        </p>

        <div className="bg-slate-50 border border-slate-200 rounded-2xl p-6 my-6 shadow-sm overflow-x-auto">
          <p className="font-mono text-sm text-slate-700 mb-2 font-bold">The Dual Problem:</p>
          <div className="font-mono text-sm text-slate-800 space-y-1">
            <p>Maximize: L(α) = Σ α_i - 1/2 Σ Σ α_i α_j y_i y_j (x_i · x_j)</p>
            <p>Subject to: α_i ≥ 0,  Σ α_i y_i = 0</p>
          </div>
        </div>

        <p>
          <strong>Karush-Kuhn-Tucker (KKT) Conditions:</strong> In the final solution, most <code>α_i</code> will be exactly zero. The points where <code>α_i &gt; 0</code> are the <strong>Support Vectors</strong>. They are the only points that define the boundary.
        </p>

        <h2 className="text-3xl font-bold text-slate-900 mt-12 mb-4">3. The Kernel Trick (Mercer's Theorem)</h2>
        <p>
          Notice that in the dual formulation, the data points <code>x_i</code> only ever appear as <strong>dot products</strong> <code>(x_i · x_j)</code>. If our data is not linearly separable, we can apply a mapping function <code>Φ(x)</code> to project the data into a higher-dimensional space.
        </p>
        <p>
          Computing <code>Φ(x_i) · Φ(x_j)</code> explicitly is computationally intractable for infinite dimensions. However, according to <strong>Mercer's Theorem</strong>, if a Kernel function <code>K(x_i, x_j)</code> satisfies certain conditions (positive semi-definite), it perfectly computes the dot product in that higher dimension <em>without ever explicitly calculating the projection</em>.
        </p>

        <div className="bg-slate-50 border border-slate-200 rounded-2xl p-6 my-6 shadow-sm">
          <p className="font-mono text-sm text-slate-800 mb-2"><strong>Radial Basis Function (RBF) Kernel:</strong></p>
          <code className="bg-white p-2 rounded block font-mono text-sm border border-slate-200">
            K(x_i, x_j) = exp(-γ ||x_i - x_j||²)
          </code>
          <p className="text-sm text-slate-500 mt-2">Projects data into an infinite-dimensional Hilbert space.</p>
        </div>

        <div className="bg-white border border-slate-200 p-6 rounded-2xl my-8 shadow-lg not-prose">
            <h3 className="text-slate-900 font-bold mt-0 text-xl mb-4">Implementation: Linear SVM with Soft Margin Hinge Loss</h3>
            <TerminalBlock 
                language="python" 
                filename="svm_from_scratch.py" 
                code={svmCode}
            />
        </div>

        <InteractiveQuiz 
            question="In a Soft-Margin SVM, what happens to the geometric margin as the regularization hyperparameter C approaches infinity?"
            options={[
                "The margin becomes infinitely wide, misclassifying all points.",
                "The SVM becomes equivalent to a Hard-Margin SVM, severely penalizing any misclassifications and shrinking the margin.",
                "The Lagrange multipliers sum to zero, crashing the optimization.",
                "The Kernel trick bypasses C, making it irrelevant."
            ]}
            correctIndex={1}
            explanation="The parameter C acts as a penalty for misclassification. As C -> ∞, the model tolerates zero errors, forcing a Hard-Margin. This creates a very narrow margin and often leads to severe overfitting on noisy data."
        />
      </section>
    </div>
  );
}
