"use client";
import React from 'react';
import TerminalBlock from '@/components/TerminalBlock';
import InteractiveQuiz from '@/components/InteractiveQuiz';

export default function EnsemblesPage() {
  const ensembleCode = `import numpy as np
from sklearn.tree import DecisionTreeRegressor

class PhDGradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        
    def fit(self, X, y):
        # 1. Initialize f0(x) with the mean of y
        self.F0 = np.mean(y)
        Fm = np.full(len(y), self.F0)
        
        for m in range(self.n_estimators):
            # 2. Compute Pseudo-Residuals (Negative Gradient of MSE Loss)
            # Loss = 1/2 (y - F(x))^2  -->  dLoss/dF = -(y - F(x))
            residuals = y - Fm
            
            # 3. Fit a weak learner (stump) to the residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            # 4. Update the additive model
            update = tree.predict(X)
            Fm += self.learning_rate * update
            
            self.trees.append(tree)

    def predict(self, X):
        # Start with base prediction
        Fm = np.full(X.shape[0], self.F0)
        
        # Add contributions from all sequentially boosted trees
        for tree in self.trees:
            Fm += self.learning_rate * tree.predict(X)
            
        return Fm`;

  return (
    <div className="space-y-12 pb-24 text-slate-800 animate-in fade-in slide-in-from-bottom-4 duration-500">
      
      <header className="border-b border-slate-200 pb-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-teal-100 text-teal-700 rounded-full text-xs font-bold uppercase tracking-widest mb-4">
          <span className="w-2 h-2 rounded-full bg-teal-600 animate-pulse"></span>
          Stack 2: Classical Machine Learning
        </div>
        <h1 className="text-4xl font-extrabold tracking-tight text-slate-900 mb-6">2.4 Bagging & Boosting</h1>
        <p className="text-xl text-slate-600 leading-relaxed max-w-3xl">
          Bias-Variance Decomposition, Bootstrap Aggregation (Random Forests), and Gradient Boosting (XGBoost).
        </p>
      </header>

      <section className="prose prose-slate max-w-none text-lg text-slate-600">
        <h2 className="text-3xl font-bold text-slate-900 mt-8 mb-4">1. The Bias-Variance Decomposition</h2>
        <p>
          To understand Ensembles, we must decompose the expected prediction error of any ML model. Let <code>y = f(x) + ε</code> where <code>ε ~ N(0, σ²)</code> is irreducible noise.
        </p>
        
        <div className="bg-slate-50 border border-slate-200 rounded-2xl p-6 my-6 font-mono text-sm text-slate-800 shadow-sm overflow-x-auto">
          <p>Err(x) = E[(y - f̂(x))²]</p>
          <p>Err(x) = (Bias[f̂(x)])² + Var[f̂(x)] + σ²</p>
        </div>

        <ul className="list-disc pl-5 mt-4">
          <li><strong>Bias:</strong> Error from erroneous assumptions (e.g., fitting a linear model to a quadratic curve). Causes <em>Underfitting</em>.</li>
          <li><strong>Variance:</strong> Error from sensitivity to small fluctuations in the training set (e.g., an unpruned deep decision tree). Causes <em>Overfitting</em>.</li>
        </ul>
        <p>A single Decision Tree has low bias but astronomically high variance. Ensembles fix this.</p>

        <h2 className="text-3xl font-bold text-slate-900 mt-12 mb-4">2. Bagging (Bootstrap Aggregation)</h2>
        <p>
          <strong>Goal: Reduce Variance.</strong> Bagging trains `N` independent base models in <strong>parallel</strong>. It uses <em>Bootstrapping</em> (sampling with replacement) to create `N` slightly different datasets.
        </p>
        <div className="bg-slate-50 border-l-4 border-teal-500 p-6 my-6 shadow-sm">
          <h4 className="font-bold text-slate-900 mt-0">Random Forest Mathematics</h4>
          <p>If we average `B` independent random variables with variance `σ²`, the variance of the average is `σ² / B`. By decorrelating the trees (forcing them to split on random subsets of features), Random Forests achieve this variance reduction mathematically without increasing bias.</p>
        </div>

        <h2 className="text-3xl font-bold text-slate-900 mt-12 mb-4">3. Boosting (AdaBoost & Gradient Boosting)</h2>
        <p>
          <strong>Goal: Reduce Bias.</strong> Boosting trains `N` weak models <strong>sequentially</strong>. Each model focuses exclusively on correcting the errors of the combined ensemble that came before it.
        </p>
        
        <div className="bg-slate-50 border border-slate-200 rounded-2xl p-6 my-6 shadow-sm">
          <h4 className="font-bold text-slate-900 mt-0">Gradient Boosting as Functional Gradient Descent</h4>
          <p>Instead of optimizing parameters `θ` (like in Neural Networks), Gradient Boosting optimizes the actual function `F(x)`. It calculates the negative gradient of the Loss function (the pseudo-residuals) and fits a regression tree strictly to those residuals.</p>
          <div className="font-mono text-sm bg-white border border-slate-200 p-3 mt-3 rounded text-slate-800">
            r_im = - [ ∂L(y_i, F(x_i)) / ∂F(x_i) ]
          </div>
        </div>

        <p>
          <strong>XGBoost (Extreme Gradient Boosting):</strong> Revolutionized Kaggle by introducing a regularized objective function, handling missing values natively, and using second-order Taylor expansion (Hessians) to find optimal tree splits exponentially faster.
        </p>

        <div className="bg-white border border-slate-200 p-6 rounded-2xl my-8 shadow-lg not-prose">
            <h3 className="text-slate-900 font-bold mt-0 text-xl mb-4">Implementation: Gradient Boosting from Scratch</h3>
            <TerminalBlock 
                language="python" 
                filename="gradient_boosting.py" 
                code={ensembleCode}
            />
        </div>

        <InteractiveQuiz 
            question="In the XGBoost objective function, what is the mathematical purpose of using the second-order derivative (the Hessian) alongside the gradient during tree splitting?"
            options={[
                "It allows XGBoost to bypass the need for a learning rate (shrinkage) entirely.",
                "It enables parallelized construction of multiple trees simultaneously (like Bagging).",
                "It provides a more accurate quadratic approximation of the loss function, allowing faster convergence to the optimal leaf weight.",
                "It transforms a classification problem into a regression problem."
            ]}
            correctIndex={2}
            explanation="XGBoost uses a second-order Taylor approximation of the loss function. While standard GBM only uses the first derivative (gradient), adding the second derivative (Hessian) provides curvature information. This leads to a direct analytical solution for the optimal leaf weight and faster, more accurate convergence."
        />
      </section>
    </div>
  );
}
