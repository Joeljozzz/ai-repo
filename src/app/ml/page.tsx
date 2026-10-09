"use client";

import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

export default function ClassicalML() {
  const [activeTab, setActiveTab] = useState('regression');

  const rfCode = `from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load data
# X, y = load_your_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Initialize the model (100 trees)
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# 3. Train the model
rf.fit(X_train, y_train)

# 4. Predict and evaluate
preds = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.2f}")

# 5. Extract Feature Importance
for name, score in zip(feature_names, rf.feature_importances_):
    print(f"{name}: {score:.4f}")`;

  return (
    <div className="space-y-12 pb-16 text-slate-800">
      
      {/* Header */}
      <header className="border-b border-slate-200 pb-8">
        <div className="inline-block px-3 py-1 bg-emerald-100 text-emerald-700 rounded-full text-xs font-bold uppercase tracking-widest mb-4">
          Module 1
        </div>
        <h1 className="text-5xl font-extrabold tracking-tight text-slate-900 mb-4">Classical Machine Learning</h1>
        <p className="text-xl text-slate-600 leading-relaxed max-w-3xl">
          The foundation of AI. Before jumping into deep learning, mastering statistical modeling and tree-based ensembles is crucial for handling structured, tabular data in the real world.
        </p>
      </header>

      {/* Navigation Tabs */}
      <div className="flex border-b border-slate-200 gap-8 overflow-x-auto">
        <button 
          onClick={() => setActiveTab('regression')}
          className={`pb-4 text-lg font-medium whitespace-nowrap transition-colors ${activeTab === 'regression' ? 'border-b-2 border-emerald-600 text-emerald-600' : 'text-slate-500 hover:text-slate-700'}`}
        >
          Linear & Logistic
        </button>
        <button 
          onClick={() => setActiveTab('trees')}
          className={`pb-4 text-lg font-medium whitespace-nowrap transition-colors ${activeTab === 'trees' ? 'border-b-2 border-emerald-600 text-emerald-600' : 'text-slate-500 hover:text-slate-700'}`}
        >
          Decision Trees
        </button>
        <button 
          onClick={() => setActiveTab('ensemble')}
          className={`pb-4 text-lg font-medium whitespace-nowrap transition-colors ${activeTab === 'ensemble' ? 'border-b-2 border-emerald-600 text-emerald-600' : 'text-slate-500 hover:text-slate-700'}`}
        >
          Random Forests
        </button>
      </div>

      {/* Regression Tab */}
      {activeTab === 'regression' && (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
          <section className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200">
            <h2 className="text-3xl font-bold mb-6 text-slate-900">Linear vs. Logistic Regression</h2>
            
            <div className="grid md:grid-cols-2 gap-8">
              {/* Linear */}
              <div className="bg-slate-50 p-6 rounded-xl border border-slate-200">
                <div className="bg-blue-100 text-blue-800 text-xs font-bold uppercase px-2 py-1 rounded inline-block mb-3">Continuous Output</div>
                <h3 className="font-bold text-xl text-slate-800 mb-2">Linear Regression</h3>
                <p className="text-sm text-slate-600 mb-4">Predicts a continuous numerical value (e.g., predicting house prices, temperature) assuming a linear relationship.</p>
                <div className="font-mono bg-slate-900 text-slate-100 p-3 rounded-lg text-sm mb-4">
                  y = β₀ + β₁x₁ + ... + βₙxₙ + ε
                </div>
                <ul className="text-sm text-slate-600 space-y-2">
                  <li>✅ <strong>DO:</strong> Scale/normalize your features.</li>
                  <li>❌ <strong>DON'T:</strong> Ignore outliers; they skew the regression line heavily.</li>
                </ul>
              </div>

              {/* Logistic */}
              <div className="bg-slate-50 p-6 rounded-xl border border-slate-200">
                <div className="bg-purple-100 text-purple-800 text-xs font-bold uppercase px-2 py-1 rounded inline-block mb-3">Binary Output</div>
                <h3 className="font-bold text-xl text-slate-800 mb-2">Logistic Regression</h3>
                <p className="text-sm text-slate-600 mb-4">Binary classification (e.g., spam vs. not spam). It outputs probabilities bounded between 0 and 1.</p>
                <div className="font-mono bg-slate-900 text-slate-100 p-3 rounded-lg text-sm mb-4">
                  P(y=1|X) = 1 / (1 + e^-(βX))
                </div>
                <ul className="text-sm text-slate-600 space-y-2">
                  <li>✅ <strong>DO:</strong> Use as a strong baseline for classification.</li>
                  <li>❌ <strong>DON'T:</strong> Use for non-linear decision boundaries.</li>
                </ul>
              </div>
            </div>
          </section>
        </div>
      )}

      {/* Trees Tab */}
      {activeTab === 'trees' && (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
          <section className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200">
            <h2 className="text-3xl font-bold mb-4 text-slate-900">Decision Trees</h2>
            <p className="text-lg text-slate-600 mb-6">
              A highly interpretable model that splits data into branches based on feature conditions. Excellent for non-linear data boundaries.
            </p>

            <div className="grid md:grid-cols-2 gap-8 mb-6">
              <div className="p-5 border-l-4 border-emerald-500 bg-emerald-50/50 rounded-r-lg">
                <h4 className="font-bold text-emerald-900 text-lg mb-2">Gini Impurity</h4>
                <p className="text-emerald-800 mb-2 text-sm">Measures the probability of incorrectly classifying a randomly chosen element.</p>
                <code className="bg-white px-2 py-1 rounded text-emerald-900 text-sm border border-emerald-100">Gini = 1 - Σ (p_i)²</code>
              </div>

              <div className="p-5 border-l-4 border-indigo-500 bg-indigo-50/50 rounded-r-lg">
                <h4 className="font-bold text-indigo-900 text-lg mb-2">Entropy (Information Gain)</h4>
                <p className="text-indigo-800 mb-2 text-sm">Measures the amount of disorder or uncertainty in the data.</p>
                <code className="bg-white px-2 py-1 rounded text-indigo-900 text-sm border border-indigo-100">Entropy = -Σ p_i * log₂(p_i)</code>
              </div>
            </div>

            <div className="bg-slate-50 p-6 rounded-xl border border-slate-200">
              <h3 className="font-bold text-slate-800 mb-3">The Overfitting Problem</h3>
              <p className="text-sm text-slate-600">
                A standard decision tree will continue splitting until every single leaf node contains exactly 1 sample. This means it memorizes the training data perfectly (100% accuracy) but fails completely on new data. <strong>Always prune your trees</strong> (e.g., `max_depth=5`, `min_samples_split=10`).
              </p>
            </div>
          </section>
        </div>
      )}

      {/* Ensemble Tab */}
      {activeTab === 'ensemble' && (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
          <section className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200">
            <h2 className="text-3xl font-bold mb-4 text-slate-900">Random Forests (Bagging Ensemble)</h2>
            <p className="text-lg text-slate-600 mb-8">
              When a single Decision Tree overfits, a Random Forest stabilizes predictions by training hundreds of independent trees on random subsets of data and features, then taking a majority vote.
            </p>

            <h3 className="text-xl font-bold mb-4">Scikit-Learn Implementation</h3>
            <div className="rounded-xl overflow-hidden shadow-lg border border-slate-800 mb-8">
              <SyntaxHighlighter language="python" style={vscDarkPlus} showLineNumbers customStyle={{margin: 0, padding: '1.5rem'}}>
                {rfCode}
              </SyntaxHighlighter>
            </div>

            <h3 className="text-xl font-bold mb-4">Why Random Forests are King of Tabular Data</h3>
            <div className="grid md:grid-cols-3 gap-6">
              <div className="p-5 bg-slate-50 rounded-xl border border-slate-200">
                <h4 className="font-bold text-slate-900 mb-2">1. Low Variance</h4>
                <p className="text-sm text-slate-600">Averaging 100 trees drastically reduces the variance (overfitting) compared to a single tree.</p>
              </div>
              <div className="p-5 bg-slate-50 rounded-xl border border-slate-200">
                <h4 className="font-bold text-slate-900 mb-2">2. No Scaling Needed</h4>
                <p className="text-sm text-slate-600">Because it relies on split points rather than distance geometry, you don't need to normalize your data.</p>
              </div>
              <div className="p-5 bg-slate-50 rounded-xl border border-slate-200">
                <h4 className="font-bold text-slate-900 mb-2">3. Feature Importance</h4>
                <p className="text-sm text-slate-600">Out of the box, it tells you exactly which features contributed the most to the predictions.</p>
              </div>
            </div>
          </section>
        </div>
      )}

    </div>
  );
}
