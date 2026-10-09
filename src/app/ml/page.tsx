"use client";

import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

export default function ClassicalML() {
  const [activeModule, setActiveModule] = useState('mod2'); // Defaulting to SVM for emphasis

  const svmCode = `from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Initialize SVM with Radial Basis Function (RBF) Kernel
# C controls the penalty for misclassification (Regularization)
# gamma controls the 'spread' of the kernel
model = SVC(kernel='rbf', C=1.0, gamma='scale')

# 2. Train the model
model.fit(X_train, y_train)

# 3. Evaluate
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))`;

  const treeCode = `from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 1. Initialize Tree (Pruning with max_depth prevents overfitting)
tree = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=10)

# 2. Train
tree.fit(X_train, y_train)

# 3. Visualize the splits (Great for interpretability)
plt.figure(figsize=(20,10))
plot_tree(tree, filled=True, feature_names=feature_cols, class_names=class_names)
plt.show()`;

  const ensembleCode = `from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

# --- BAGGING: Random Forest ---
rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1)
rf.fit(X_train, y_train)

# --- BOOSTING: Scikit-Learn GBM ---
gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
gbm.fit(X_train, y_train)

# --- BOOSTING: XGBoost (Industry Standard) ---
# XGBoost is highly optimized, handles missing values natively, and is heavily parallelized.
xgb_model = xgb.XGBClassifier(
    n_estimators=200, 
    learning_rate=0.05, 
    max_depth=4,
    tree_method='hist', # Uses histogram-based split finding (much faster)
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10)
`;

  return (
    <div className="space-y-12 pb-16 text-slate-800">
      
      {/* Header */}
      <header className="border-b border-slate-200 pb-8">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-emerald-100 text-emerald-700 rounded-full text-xs font-bold uppercase tracking-widest mb-4">
          <span className="w-2 h-2 rounded-full bg-emerald-600 animate-pulse"></span>
          Advanced Certification Track
        </div>
        <h1 className="text-5xl font-extrabold tracking-tight text-slate-900 mb-4">Classical Machine Learning</h1>
        <p className="text-xl text-slate-600 leading-relaxed max-w-3xl">
          Master the mathematical and algorithmic foundations of AI. From drawing max-margin hyperplanes in SVMs to building massive XGBoost ensemble architectures used to dominate Kaggle competitions.
        </p>
      </header>

      {/* Course Modules Navigation */}
      <div className="flex border-b border-slate-200 gap-8 overflow-x-auto no-scrollbar">
        {[
          { id: 'mod1', title: 'Module 1', subtitle: 'Linear & Distance Models' },
          { id: 'mod2', title: 'Module 2', subtitle: 'Support Vector Machines' },
          { id: 'mod3', title: 'Module 3', subtitle: 'Decision Trees & Math' },
          { id: 'mod4', title: 'Module 4', subtitle: 'Bagging & Boosting (Ensembles)' },
        ].map((mod) => (
          <button 
            key={mod.id}
            onClick={() => setActiveModule(mod.id)}
            className={`pb-4 text-left transition-colors whitespace-nowrap min-w-[150px] ${
              activeModule === mod.id 
                ? 'border-b-2 border-emerald-600' 
                : 'text-slate-500 hover:text-slate-700'
            }`}
          >
            <div className={`text-xs font-bold uppercase tracking-wider mb-1 ${activeModule === mod.id ? 'text-emerald-600' : 'text-slate-400'}`}>
              {mod.title}
            </div>
            <div className={`text-base font-medium ${activeModule === mod.id ? 'text-slate-900' : ''}`}>
              {mod.subtitle}
            </div>
          </button>
        ))}
      </div>

      {/* Module 1: Linear & Distance */}
      {activeModule === 'mod1' && (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
          <section className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200">
            <h2 className="text-3xl font-bold mb-6 text-slate-900">Linear Regression, Logistic Regression & KNN</h2>
            <div className="grid md:grid-cols-2 gap-8">
              <div className="bg-slate-50 p-6 rounded-xl border border-slate-200">
                <h3 className="font-bold text-xl text-slate-800 mb-2">Linear Models</h3>
                <p className="text-sm text-slate-600 mb-4">
                  Assume a linear geometry in the feature space. Fast to train, highly interpretable, but prone to underfitting on complex datasets.
                </p>
                <ul className="text-sm text-slate-600 space-y-2 list-disc pl-5">
                  <li><strong>Linear Regression:</strong> Predicts continuous variables. Uses OLS (Ordinary Least Squares) to minimize MSE.</li>
                  <li><strong>Logistic Regression:</strong> Predicts binary outcomes. Passes a linear equation through a Sigmoid function to bound outputs between 0 and 1.</li>
                </ul>
              </div>
              <div className="bg-slate-50 p-6 rounded-xl border border-slate-200">
                <h3 className="font-bold text-xl text-slate-800 mb-2">K-Nearest Neighbors (KNN)</h3>
                <p className="text-sm text-slate-600 mb-4">
                  A non-parametric, "lazy" learning algorithm. It simply stores the training data and categorizes new points based on the majority class of its 'K' closest neighbors.
                </p>
                <ul className="text-sm text-slate-600 space-y-2 list-disc pl-5">
                  <li><strong>Distance Metrics:</strong> Euclidean (straight line), Manhattan (grid-like).</li>
                  <li><strong>Curse of Dimensionality:</strong> KNN performs terribly when there are too many features because distance becomes meaningless in high-dimensional space.</li>
                </ul>
              </div>
            </div>
          </section>
        </div>
      )}

      {/* Module 2: SVMs */}
      {activeModule === 'mod2' && (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
          <section className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200">
            <h2 className="text-3xl font-bold mb-4 text-slate-900">Support Vector Machines (SVMs)</h2>
            <p className="text-lg text-slate-600 mb-8">
              SVMs are powerful supervised models that find the <strong>optimal hyperplane</strong> which maximizes the margin (distance) between two classes. They are highly effective in high-dimensional spaces.
            </p>

            <div className="rounded-xl overflow-hidden shadow-lg border border-slate-800 mb-8">
              <div className="bg-slate-900 text-slate-400 text-xs px-4 py-2 font-mono border-b border-slate-800">
                svm_classifier.py
              </div>
              <SyntaxHighlighter language="python" style={vscDarkPlus} showLineNumbers customStyle={{margin: 0, padding: '1.5rem', backgroundColor: '#0f172a'}}>
                {svmCode}
              </SyntaxHighlighter>
            </div>

            <h3 className="text-xl font-bold mb-4 mt-12">Core Mathematical Concepts</h3>
            <div className="grid md:grid-cols-3 gap-6">
              <div className="p-5 bg-slate-50 rounded-xl border border-slate-200">
                <h4 className="font-bold text-slate-900 mb-2">The Margin & Support Vectors</h4>
                <p className="text-sm text-slate-600">The boundary is drawn solely based on the data points closest to the opposing class. These critical points are called "Support Vectors." All other points do not affect the boundary.</p>
              </div>
              <div className="p-5 bg-slate-50 rounded-xl border border-slate-200">
                <h4 className="font-bold text-slate-900 mb-2">The C Parameter (Soft Margin)</h4>
                <p className="text-sm text-slate-600">Controls the trade-off between a smooth decision boundary and classifying training points correctly. A high <code>C</code> aims for 100% accuracy (overfitting), a low <code>C</code> allows some misclassifications for better generalization.</p>
              </div>
              <div className="p-5 border-l-4 border-emerald-500 bg-emerald-50/50 rounded-r-lg">
                <h4 className="font-bold text-emerald-900 mb-2">The Kernel Trick</h4>
                <p className="text-sm text-emerald-800">When data isn't linearly separable, kernels (like RBF or Polynomial) mathematically project the data into a higher dimension where a straight plane CAN separate them—without ever actually computing the higher dimension.</p>
              </div>
            </div>
          </section>
        </div>
      )}

      {/* Module 3: Decision Trees */}
      {activeModule === 'mod3' && (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
          <section className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200">
            <h2 className="text-3xl font-bold mb-4 text-slate-900">Decision Trees & Information Theory</h2>
            <p className="text-lg text-slate-600 mb-6">
              Unlike regression, Decision Trees split the data space into orthogonal boxes by asking simple IF/ELSE questions. They require almost zero data preprocessing (no scaling needed!).
            </p>

            <div className="rounded-xl overflow-hidden shadow-lg border border-slate-800 mb-8">
              <div className="bg-slate-900 text-slate-400 text-xs px-4 py-2 font-mono border-b border-slate-800">
                decision_tree.py
              </div>
              <SyntaxHighlighter language="python" style={vscDarkPlus} showLineNumbers customStyle={{margin: 0, padding: '1.5rem', backgroundColor: '#0f172a'}}>
                {treeCode}
              </SyntaxHighlighter>
            </div>

            <div className="grid md:grid-cols-2 gap-8 mb-6">
              <div className="p-5 border-l-4 border-amber-500 bg-amber-50/50 rounded-r-lg">
                <h4 className="font-bold text-amber-900 text-lg mb-2">Gini Impurity</h4>
                <p className="text-amber-800 mb-2 text-sm">Default in Scikit-Learn. Measures how often a randomly chosen element would be incorrectly labeled if it was labeled randomly according to the distribution of labels in the node.</p>
                <code className="bg-white px-2 py-1 rounded text-amber-900 text-sm border border-amber-200">Gini = 1 - Σ (p_i)²</code>
              </div>

              <div className="p-5 border-l-4 border-indigo-500 bg-indigo-50/50 rounded-r-lg">
                <h4 className="font-bold text-indigo-900 text-lg mb-2">Entropy & Info Gain</h4>
                <p className="text-indigo-800 mb-2 text-sm">Derived from thermodynamics. Measures the "chaos" or impurity. The tree splits on the feature that reduces Entropy the most (Information Gain).</p>
                <code className="bg-white px-2 py-1 rounded text-indigo-900 text-sm border border-indigo-200">Entropy = -Σ p_i * log₂(p_i)</code>
              </div>
            </div>
          </section>
        </div>
      )}

      {/* Module 4: Ensembles */}
      {activeModule === 'mod4' && (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
          <section className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200">
            <h2 className="text-3xl font-bold mb-4 text-slate-900">Ensemble Methods: Bagging & Boosting</h2>
            <p className="text-lg text-slate-600 mb-8">
              A single decision tree overfits. Ensembles combine hundreds of "weak learners" to form a single, robust "strong learner." This is the gold standard for structured/tabular data.
            </p>

            <div className="rounded-xl overflow-hidden shadow-lg border border-slate-800 mb-8">
              <div className="bg-slate-900 text-slate-400 text-xs px-4 py-2 font-mono border-b border-slate-800">
                ensembles_xgboost.py
              </div>
              <SyntaxHighlighter language="python" style={vscDarkPlus} showLineNumbers customStyle={{margin: 0, padding: '1.5rem', backgroundColor: '#0f172a'}}>
                {ensembleCode}
              </SyntaxHighlighter>
            </div>

            <div className="grid md:grid-cols-2 gap-8">
              <div className="bg-slate-50 p-6 rounded-xl border border-slate-200">
                <div className="bg-blue-100 text-blue-800 text-xs font-bold uppercase px-2 py-1 rounded inline-block mb-3">Bagging (Parallel)</div>
                <h3 className="font-bold text-xl text-slate-800 mb-2">Random Forests</h3>
                <p className="text-sm text-slate-600 mb-4">
                  Trains hundreds of deep, overfitted trees in <strong>parallel</strong> on random subsets of data (Bootstrapping) and random subsets of features. Final prediction is an average/majority vote.
                </p>
                <ul className="text-sm text-slate-600 space-y-2 list-disc pl-5">
                  <li><strong>Goal:</strong> Reduces Variance (Prevents overfitting).</li>
                  <li><strong>Pro:</strong> Extremely hard to overfit; virtually no hyperparameter tuning needed.</li>
                </ul>
              </div>

              <div className="bg-slate-50 p-6 rounded-xl border border-slate-200">
                <div className="bg-rose-100 text-rose-800 text-xs font-bold uppercase px-2 py-1 rounded inline-block mb-3">Boosting (Sequential)</div>
                <h3 className="font-bold text-xl text-slate-800 mb-2">AdaBoost, GBM, XGBoost</h3>
                <p className="text-sm text-slate-600 mb-4">
                  Trains shallow "stump" trees <strong>sequentially</strong>. Each new tree focuses exclusively on fixing the errors (residuals) made by the previous tree.
                </p>
                <ul className="text-sm text-slate-600 space-y-2 list-disc pl-5">
                  <li><strong>Goal:</strong> Reduces Bias (Increases accuracy).</li>
                  <li><strong>Pro:</strong> Produces the highest accuracy models for tabular data (e.g., XGBoost). Requires careful learning-rate tuning to prevent overfitting.</li>
                </ul>
              </div>
            </div>
          </section>
        </div>
      )}

    </div>
  );
}
