import React from 'react';

export default function ClassicalML() {
  return (
    <div className="space-y-12 pb-12">
      <header className="border-b pb-6">
        <h1 className="text-4xl font-extrabold tracking-tight text-gray-900">Classical Machine Learning</h1>
        <p className="mt-4 text-lg text-gray-600">
          The foundation of AI. Before diving into Deep Learning, understanding statistical modeling and classical algorithms is crucial for structured, tabular data.
        </p>
      </header>

      {/* Linear Regression */}
      <section className="space-y-4">
        <h2 className="text-3xl font-bold text-gray-800">1. Linear Regression</h2>
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h3 className="font-semibold text-blue-900">When to use:</h3>
          <p className="text-blue-800 text-sm mt-1">Predicting a continuous numerical value (e.g., house prices, temperature) when you assume a linear relationship between input features and the output.</p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
          <div className="bg-white p-5 rounded-lg border shadow-sm">
            <h4 className="font-bold text-gray-700 mb-2">The Formula</h4>
            <p className="font-mono text-sm bg-gray-100 p-3 rounded">
              y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
            </p>
            <ul className="mt-3 text-sm text-gray-600 list-disc pl-5 space-y-1">
              <li><strong>y</strong>: Dependent variable (Target)</li>
              <li><strong>β₀</strong>: Intercept</li>
              <li><strong>β₁, β₂</strong>: Coefficients (Weights)</li>
              <li><strong>x₁, x₂</strong>: Independent variables (Features)</li>
              <li><strong>ε</strong>: Error term</li>
            </ul>
          </div>
          <div className="bg-white p-5 rounded-lg border shadow-sm">
            <h4 className="font-bold text-gray-700 mb-2">Do's & Don'ts</h4>
            <ul className="text-sm text-gray-600 space-y-2">
              <li>✅ <strong>DO:</strong> Check for multicollinearity (correlated features).</li>
              <li>✅ <strong>DO:</strong> Scale/normalize your features.</li>
              <li>❌ <strong>DON'T:</strong> Use for non-linear relationships without feature transformations (like polynomial features).</li>
              <li>❌ <strong>DON'T:</strong> Ignore outliers; they skew the regression line heavily.</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Logistic Regression */}
      <section className="space-y-4">
        <h2 className="text-3xl font-bold text-gray-800">2. Logistic Regression (Classification)</h2>
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h3 className="font-semibold text-blue-900">When to use:</h3>
          <p className="text-blue-800 text-sm mt-1">Binary classification problems (e.g., spam vs. not spam, churn vs. retain). It outputs probabilities between 0 and 1.</p>
        </div>
        
        <div className="bg-white p-5 rounded-lg border shadow-sm mt-4">
          <h4 className="font-bold text-gray-700 mb-2">The Formula (Sigmoid Function)</h4>
          <p className="font-mono text-sm bg-gray-100 p-3 rounded">
            P(y=1|X) = 1 / (1 + e^-(β₀ + β₁x₁ + ...))
          </p>
          <p className="mt-3 text-sm text-gray-600">Converts the linear output into a probability curve (S-curve).</p>
        </div>
      </section>

      {/* Decision Trees */}
      <section className="space-y-4">
        <h2 className="text-3xl font-bold text-gray-800">3. Decision Trees</h2>
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h3 className="font-semibold text-blue-900">When to use:</h3>
          <p className="text-blue-800 text-sm mt-1">When you need a highly interpretable model that handles both numerical and categorical data without scaling.</p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
          <div className="bg-white p-5 rounded-lg border shadow-sm">
            <h4 className="font-bold text-gray-700 mb-2">How it works</h4>
            <p className="text-sm text-gray-600">
              Splits the data into branches based on feature conditions. It uses metrics like <strong>Gini Impurity</strong> or <strong>Entropy (Information Gain)</strong> to find the best splits.
            </p>
            <p className="font-mono text-xs bg-gray-100 p-2 mt-3 rounded">
              Gini = 1 - Σ (p_i)² <br/>
              Entropy = -Σ p_i * log₂(p_i)
            </p>
          </div>
          <div className="bg-white p-5 rounded-lg border shadow-sm">
            <h4 className="font-bold text-gray-700 mb-2">Do's & Don'ts</h4>
            <ul className="text-sm text-gray-600 space-y-2">
              <li>✅ <strong>DO:</strong> Use for non-linear data boundaries.</li>
              <li>✅ <strong>DO:</strong> Prune the tree (limit max depth) to prevent overfitting.</li>
              <li>❌ <strong>DON'T:</strong> Use for extrapolating numerical values outside the training range.</li>
              <li>❌ <strong>DON'T:</strong> Leave them unconstrained; they will overfit perfectly to training data.</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Random Forests */}
      <section className="space-y-4">
        <h2 className="text-3xl font-bold text-gray-800">4. Random Forests (Ensemble)</h2>
        <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
          <h3 className="font-semibold text-purple-900">When to use:</h3>
          <p className="text-purple-800 text-sm mt-1">Almost always a great baseline for tabular data. When Decision Trees overfit, Random Forests stabilize predictions by averaging multiple trees.</p>
        </div>
        
        <div className="bg-white p-5 rounded-lg border shadow-sm mt-4">
          <h4 className="font-bold text-gray-700 mb-2">How it works (Bagging)</h4>
          <p className="text-sm text-gray-600 mb-3">
            Builds hundreds of Decision Trees independently on random subsets of the data (bootstrapping) and random subsets of features. The final prediction is a majority vote (classification) or average (regression).
          </p>
          <ul className="text-sm text-gray-600 list-disc pl-5 space-y-1">
            <li>Reduces variance and overfitting.</li>
            <li>Handles missing values well.</li>
            <li>Provides "Feature Importance" scores out-of-the-box.</li>
          </ul>
        </div>
      </section>

    </div>
  );
}
