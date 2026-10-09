"use client";

import React from 'react';
import TerminalBlock from '@/components/TerminalBlock';

export default function ClassicalML() {
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
    <div className="space-y-16 pb-24 text-slate-800">
      
      {/* Header */}
      <header className="border-b border-slate-200 pb-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-emerald-100 text-emerald-700 rounded-full text-xs font-bold uppercase tracking-widest mb-4">
          <span className="w-2 h-2 rounded-full bg-emerald-600 animate-pulse"></span>
          Advanced Certification Track
        </div>
        <h1 className="text-5xl font-extrabold tracking-tight text-slate-900 mb-6">Classical Machine Learning</h1>
        <p className="text-xl text-slate-600 leading-relaxed max-w-3xl">
          Master the mathematical and algorithmic foundations of AI. From drawing max-margin hyperplanes in SVMs to building massive XGBoost ensemble architectures used to dominate Kaggle competitions.
        </p>
      </header>

      {/* Module 1: Linear & Distance */}
      <section id="linear" className="scroll-mt-12">
        <div className="flex items-center gap-4 mb-6">
          <div className="w-10 h-10 rounded-xl bg-emerald-100 text-emerald-600 flex items-center justify-center font-bold text-xl">1</div>
          <h2 className="text-3xl font-bold text-slate-900">Linear Regression, Logistic Regression & KNN</h2>
        </div>
        <div className="prose prose-slate max-w-none mb-8 text-lg text-slate-600">
          <p>
            Before exploring non-linear boundaries, you must understand linear geometry in the feature space. These models are fast to train and highly interpretable, but prone to underfitting on complex datasets.
          </p>
        </div>
        <div className="grid md:grid-cols-2 gap-8">
          <div className="bg-slate-50 p-6 rounded-2xl border border-slate-200">
            <h3 className="font-bold text-xl text-slate-800 mb-3">Linear Models</h3>
            <ul className="text-sm text-slate-600 space-y-3 list-disc pl-5">
              <li><strong>Linear Regression:</strong> Predicts continuous variables. Uses OLS (Ordinary Least Squares) to minimize Mean Squared Error (MSE).</li>
              <li><strong>Logistic Regression:</strong> Predicts binary outcomes. Passes a linear equation through a Sigmoid function to bound outputs between 0 and 1.</li>
            </ul>
          </div>
          <div className="bg-slate-50 p-6 rounded-2xl border border-slate-200">
            <h3 className="font-bold text-xl text-slate-800 mb-3">K-Nearest Neighbors (KNN)</h3>
            <ul className="text-sm text-slate-600 space-y-3 list-disc pl-5">
              <li><strong>Lazy Learning:</strong> It simply stores the training data and categorizes new points based on the majority class of its 'K' closest neighbors.</li>
              <li><strong>Curse of Dimensionality:</strong> KNN performs terribly when there are too many features because distance becomes meaningless in high-dimensional space.</li>
            </ul>
          </div>
        </div>
      </section>

      <hr className="border-slate-200" />

      {/* Module 2: SVMs */}
      <section id="svm" className="scroll-mt-12">
        <div className="flex items-center gap-4 mb-6">
          <div className="w-10 h-10 rounded-xl bg-emerald-100 text-emerald-600 flex items-center justify-center font-bold text-xl">2</div>
          <h2 className="text-3xl font-bold text-slate-900">Support Vector Machines (SVMs)</h2>
        </div>
        <div className="prose prose-slate max-w-none text-lg text-slate-600 mb-8">
          <p>
            SVMs are powerful supervised models that find the <strong>optimal hyperplane</strong> which maximizes the margin (distance) between two classes. They are highly effective in high-dimensional spaces.
          </p>
          <div className="bg-emerald-50 border-l-4 border-emerald-500 p-6 rounded-r-xl my-6">
            <h4 className="font-bold text-emerald-900 text-lg mb-2 mt-0">The Kernel Trick</h4>
            <p className="text-emerald-800 m-0 text-base">When data isn't linearly separable, kernels (like RBF or Polynomial) mathematically project the data into a higher dimension where a straight plane CAN separate them—without ever actually computing the higher dimension.</p>
          </div>
        </div>

        <h3 className="text-2xl font-bold mb-4 mt-8">Python Implementation</h3>
        <TerminalBlock language="python" filename="svm_classifier.py" code={svmCode} />
      </section>

      <hr className="border-slate-200" />

      {/* Module 3: Decision Trees */}
      <section id="trees" className="scroll-mt-12">
        <div className="flex items-center gap-4 mb-6">
          <div className="w-10 h-10 rounded-xl bg-emerald-100 text-emerald-600 flex items-center justify-center font-bold text-xl">3</div>
          <h2 className="text-3xl font-bold text-slate-900">Decision Trees & Information Theory</h2>
        </div>
        <div className="prose prose-slate max-w-none text-lg text-slate-600 mb-8">
          <p>
            Unlike regression, Decision Trees split the data space into orthogonal boxes by asking simple IF/ELSE questions. They require almost zero data preprocessing (no scaling needed!).
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8 mb-8">
          <div className="p-6 border border-slate-200 bg-white shadow-sm rounded-2xl">
            <h4 className="font-bold text-slate-900 text-xl mb-3">Gini Impurity</h4>
            <p className="text-slate-600 mb-4 text-sm">Measures how often a randomly chosen element would be incorrectly labeled if it was labeled randomly according to the distribution of labels in the node.</p>
            <code className="block bg-slate-100 p-3 rounded-lg text-slate-800 text-sm font-mono border border-slate-200">Gini = 1 - Σ (p_i)²</code>
          </div>

          <div className="p-6 border border-slate-200 bg-white shadow-sm rounded-2xl">
            <h4 className="font-bold text-slate-900 text-xl mb-3">Entropy & Info Gain</h4>
            <p className="text-slate-600 mb-4 text-sm">Derived from thermodynamics. Measures "chaos". The tree splits on the feature that reduces Entropy the most (Information Gain).</p>
            <code className="block bg-slate-100 p-3 rounded-lg text-slate-800 text-sm font-mono border border-slate-200">Entropy = -Σ p_i * log₂(p_i)</code>
          </div>
        </div>

        <h3 className="text-2xl font-bold mb-4">Python Implementation</h3>
        <TerminalBlock language="python" filename="decision_tree.py" code={treeCode} />
      </section>

      <hr className="border-slate-200" />

      {/* Module 4: Ensembles */}
      <section id="ensembles" className="scroll-mt-12">
        <div className="flex items-center gap-4 mb-6">
          <div className="w-10 h-10 rounded-xl bg-emerald-100 text-emerald-600 flex items-center justify-center font-bold text-xl">4</div>
          <h2 className="text-3xl font-bold text-slate-900">Ensemble Methods: Bagging & Boosting</h2>
        </div>
        <div className="prose prose-slate max-w-none text-lg text-slate-600 mb-8">
          <p>
            A single decision tree overfits easily. <strong>Ensembles</strong> combine hundreds of "weak learners" to form a single, robust "strong learner." This is the gold standard for structured/tabular data.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8 mb-8">
          <div className="bg-slate-50 p-8 rounded-2xl border border-slate-200">
            <div className="bg-blue-100 text-blue-800 text-xs font-bold uppercase px-3 py-1.5 rounded-md inline-block mb-4">Bagging (Parallel)</div>
            <h3 className="font-bold text-2xl text-slate-800 mb-3">Random Forests</h3>
            <p className="text-slate-600 mb-4">
              Trains hundreds of deep, overfitted trees in <strong>parallel</strong> on random subsets of data (Bootstrapping) and random subsets of features. Final prediction is an average/majority vote.
            </p>
            <p className="text-sm font-semibold text-slate-700">Goal: Reduces Variance (Prevents overfitting).</p>
          </div>

          <div className="bg-slate-50 p-8 rounded-2xl border border-slate-200">
            <div className="bg-rose-100 text-rose-800 text-xs font-bold uppercase px-3 py-1.5 rounded-md inline-block mb-4">Boosting (Sequential)</div>
            <h3 className="font-bold text-2xl text-slate-800 mb-3">AdaBoost, GBM, XGBoost</h3>
            <p className="text-slate-600 mb-4">
              Trains shallow "stump" trees <strong>sequentially</strong>. Each new tree focuses exclusively on fixing the errors (residuals) made by the previous tree.
            </p>
            <p className="text-sm font-semibold text-slate-700">Goal: Reduces Bias (Increases accuracy).</p>
          </div>
        </div>

        <h3 className="text-2xl font-bold mb-4">Python Implementation (XGBoost)</h3>
        <TerminalBlock language="python" filename="ensembles_xgboost.py" code={ensembleCode} />
      </section>

    </div>
  );
}
