"use client";
import React from 'react';
import TerminalBlock from '@/components/TerminalBlock';
import InteractiveQuiz from '@/components/InteractiveQuiz';

export default function LinearModelsPage() {
  const linearCode = `import numpy as np

class PhDLinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent Optimization
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Compute Gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias`;

  return (
    <div className="space-y-12 pb-24 text-slate-800 animate-in fade-in slide-in-from-bottom-4 duration-500">
      
      <header className="border-b border-slate-200 pb-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-teal-100 text-teal-700 rounded-full text-xs font-bold uppercase tracking-widest mb-4">
          <span className="w-2 h-2 rounded-full bg-teal-600 animate-pulse"></span>
          Stack 2: Classical Machine Learning
        </div>
        <h1 className="text-4xl font-extrabold tracking-tight text-slate-900 mb-6">2.1 Linear & Distance Models</h1>
        <p className="text-xl text-slate-600 leading-relaxed max-w-3xl">
          OLS derivations, Maximum Likelihood Estimation in Logistic Regression, and Minkowski distance in KNN.
        </p>
      </header>

      <section className="prose prose-slate max-w-none text-lg text-slate-600">
        <h2 className="text-3xl font-bold text-slate-900 mt-8 mb-4">1. Ordinary Least Squares (OLS)</h2>
        <p>
          Linear Regression models the relationship between a scalar response and one or more explanatory variables. In the closed-form <strong>Normal Equation</strong>, we seek to minimize the Residual Sum of Squares (RSS).
        </p>
        
        <div className="bg-slate-50 border border-slate-200 rounded-2xl p-6 my-6 font-mono text-sm text-slate-700 shadow-sm overflow-x-auto">
          <p className="font-bold text-slate-900 mb-2">Mathematical Formulation:</p>
          <p>Let J(θ) = 1/2m * Σ (h_θ(x^(i)) - y^(i))²</p>
          <p>∇_θ J(θ) = 0</p>
          <p>θ = (X^T X)^(-1) X^T y</p>
        </div>
        
        <p>
          While the Normal Equation provides an exact analytical solution, computing the inverse matrix <code>(X^T X)^(-1)</code> scales at <strong>O(N^3)</strong>. For high-dimensional datasets, we fall back to numerical optimization via <strong>Gradient Descent</strong>, scaling at O(kN^2).
        </p>

        <h2 className="text-3xl font-bold text-slate-900 mt-12 mb-4">2. Logistic Regression & MLE</h2>
        <p>
          Despite its name, Logistic Regression is a classification algorithm. It predicts probabilities bounding the output between 0 and 1 using the Sigmoid (Logistic) function.
        </p>
        <p>
          Instead of minimizing MSE (which is non-convex for classification), it maximizes the <strong>Likelihood</strong> of the training data given the parameters. By taking the negative log, we derive the <strong>Log-Loss (Binary Cross-Entropy)</strong> cost function:
        </p>

        <div className="bg-slate-50 border border-slate-200 rounded-2xl p-6 my-6 font-mono text-sm text-slate-700 shadow-sm overflow-x-auto">
          <p>Cost(h_θ(x), y) = -y * log(h_θ(x)) - (1 - y) * log(1 - h_θ(x))</p>
        </div>

        <h2 className="text-3xl font-bold text-slate-900 mt-12 mb-4">3. K-Nearest Neighbors (KNN)</h2>
        <p>
          KNN is a non-parametric, lazy learning algorithm. It makes no underlying assumptions about the data distribution. Predictions are made by calculating distances in a generalized metric space using the <strong>Minkowski Distance</strong>:
        </p>

        <div className="bg-slate-50 border border-slate-200 rounded-2xl p-6 my-6 font-mono text-sm text-slate-700 shadow-sm overflow-x-auto">
          <p>D(X, Y) = ( Σ |x_i - y_i|^p )^(1/p)</p>
          <ul className="list-disc pl-5 mt-2">
            <li>If p = 1: Manhattan Distance (L1 norm)</li>
            <li>If p = 2: Euclidean Distance (L2 norm)</li>
            <li>If p = ∞: Chebyshev Distance</li>
          </ul>
        </div>

        <div className="bg-white border border-slate-200 p-6 rounded-2xl my-8 shadow-lg not-prose">
            <h3 className="text-slate-900 font-bold mt-0 text-xl mb-4">Implementation: Vectorized Gradient Descent</h3>
            <TerminalBlock 
                language="python" 
                filename="linear_regression.py" 
                code={linearCode}
            />
        </div>

        <InteractiveQuiz 
            question="Why is Mean Squared Error (MSE) an inappropriate cost function for Logistic Regression?"
            options={[
                "MSE requires the outputs to be strictly bounded between -1 and 1.",
                "MSE applied to the sigmoid hypothesis results in a non-convex function with multiple local minima.",
                "MSE is fundamentally incompatible with Maximum Likelihood Estimation.",
                "MSE penalizes correct classifications too heavily."
            ]}
            correctIndex={1}
            explanation="When the sigmoid function is plugged into the MSE cost function, the resulting surface is non-convex. Gradient descent might get stuck in local minima, failing to find the global optimum. Log-Loss ensures convexity."
        />
      </section>
    </div>
  );
}
