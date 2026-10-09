"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

export default function Page() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      {/* ── Track Badge ── */}
      <div className="not-prose mb-4">
        <span className="inline-block rounded-full bg-emerald-100 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-emerald-700">
          Track 1 · Classical Machine Learning
        </span>
      </div>

      <h1>Data Preprocessing</h1>

      <p className="lead">
        Raw data is almost never model-ready. Scaling inconsistencies, missing
        values, mixed feature types, and extreme observations can silently
        sabotage every algorithm you deploy. This lecture builds the full
        preprocessing toolkit—theory, mathematics, and production-grade
        scikit-learn pipelines—so that cleaning data becomes a deliberate
        engineering discipline rather than an afterthought.
      </p>

      {/* ── Learning Objectives ── */}
      <h2>Learning Objectives</h2>
      <ul>
        <li>
          Explain geometrically why unscaled features slow or derail gradient
          descent, and why tree models are immune to monotone feature
          transformations.
        </li>
        <li>
          Derive and apply Min-Max Scaling and Z-score Standardisation,
          choosing correctly based on data distribution and downstream
          algorithm.
        </li>
        <li>
          Distinguish MCAR, MAR, and MNAR missingness mechanisms and select an
          appropriate imputation strategy (mean, median, KNN, MICE) for each.
        </li>
        <li>
          Implement Ordinal, One-Hot, Target, and Frequency encoding with full
          awareness of the dummy-variable trap and target-leakage risk.
        </li>
        <li>
          Apply IQR/Z-score outlier removal, Winsorisation, and log transforms,
          matching the treatment to the skew of the feature.
        </li>
        <li>
          Construct a leak-free scikit-learn <code>Pipeline</code> with{" "}
          <code>ColumnTransformer</code> that fits on training data only and
          transforms validation and test splits consistently.
        </li>
      </ul>

      {/* ════════════════════════════════════════════════════
          SECTION 1 — Why Preprocessing Matters
      ════════════════════════════════════════════════════ */}
      <h2>1 · Why Preprocessing Matters</h2>

      <p>
        Imagine a dataset with two features: annual income in dollars
        (range 20,000–200,000) and age in years (range 18–90). Logistic
        regression and neural networks learn by gradient descent—they move
        in the direction of steepest loss decrease. When feature scales
        differ by orders of magnitude, the loss surface becomes a highly
        elongated ellipse whose long axis aligns with the small-scale
        feature. Gradient steps point across the valleys instead of toward
        the minimum, forcing the optimizer to zigzag inefficiently and
        requiring far more iterations to converge.
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 overflow-x-auto">
        <p className="text-slate-400 mb-2">{/* Gradient-descent intuition */}</p>
        <p>Loss contour (unscaled):</p>
        <p className="ml-4 text-yellow-300">L ≈ (θ₁ · 100000)² + (θ₂ · 50)²</p>
        <p className="mt-2 text-slate-400">→ ellipse with axes ratio ~2000:1</p>
        <p className="mt-2">Loss contour (scaled, both ∈ [0, 1]):</p>
        <p className="ml-4 text-green-300">L ≈ (θ₁)² + (θ₂)²</p>
        <p className="mt-2 text-slate-400">→ circular bowl, gradient always points at minimum</p>
      </div>

      <p>
        Distance-based algorithms (k-NN, SVM with RBF kernel, k-Means) are
        equally sensitive: a feature with a wider numerical range dominates
        the Euclidean distance metric, effectively making smaller-range
        features invisible. Regularisation (L2/L1) penalises large
        coefficients; unscaled features force artificially small weights on
        high-magnitude columns, distorting the regularisation effect.
      </p>

      <p>
        <strong>Tree models are the notable exception.</strong> Decision
        trees and their ensembles (Random Forest, XGBoost) split on
        thresholds, not magnitudes. Any strictly monotone transformation of
        a feature produces identical splits. You may still apply scaling for
        interpretability or memory efficiency, but it will not change
        predictive performance.
      </p>

      {/* ════════════════════════════════════════════════════
          SECTION 2 — Min-Max Scaling
      ════════════════════════════════════════════════════ */}
      <h2>2 · Min-Max Scaling (Normalisation)</h2>

      <p>
        Min-Max scaling linearly maps every value in a feature column to the
        closed interval [0, 1] (or any arbitrary [a, b] range).
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 overflow-x-auto">
        <p className="text-slate-400 mb-1">{/* Min-Max formula */}</p>
        <p>x_scaled = (x - x_min) / (x_max - x_min)</p>
        <p className="mt-3 text-slate-400">To map to arbitrary range [a, b]:</p>
        <p className="mt-1">x_scaled = a + (x - x_min) · (b - a) / (x_max - x_min)</p>
        <p className="mt-3 text-slate-400">Inverse transform (recover original):</p>
        <p className="mt-1">x = x_scaled · (x_max - x_min) + x_min</p>
      </div>

      <p>
        <strong>When to use it:</strong> Neural networks with sigmoid or
        tanh activations benefit from inputs bounded in [0, 1] because those
        activations saturate outside that range. Image pixel values (0–255)
        are almost universally divided by 255—a special case of Min-Max
        scaling.
      </p>

      <p>
        <strong>Critical weakness—sensitivity to outliers.</strong> A single
        extreme observation sets x_min or x_max, compressing all other
        values into a narrow sub-range. For example, if 99 values lie in
        [10, 50] but one outlier sits at 5000, the scaled values for the
        99 normal points cluster near zero. The scaler has learned the
        outlier, not the typical distribution. Always remove or cap
        outliers <em>before</em> fitting a MinMaxScaler.
      </p>

      {/* ════════════════════════════════════════════════════
          SECTION 3 — Standardisation (Z-score)
      ════════════════════════════════════════════════════ */}
      <h2>3 · Standardisation (Z-score Scaling)</h2>

      <p>
        Standardisation centres a feature at zero and scales it to unit
        variance using the training set's mean (μ) and standard deviation
        (σ). The output has mean 0 and standard deviation 1, but is
        unbounded—unlike Min-Max, extreme values remain visible as large
        absolute z-scores.
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 overflow-x-auto">
        <p className="text-slate-400 mb-1">{/* Z-score formula */}</p>
        <p>x_std = (x - μ) / σ</p>
        <p className="mt-2 text-slate-400">where:</p>
        <p className="ml-4">μ  = (1/n) · Σ xᵢ          (sample mean)</p>
        <p className="ml-4">σ  = sqrt[ (1/n) · Σ (xᵢ - μ)² ]  (population std)</p>
        <p className="ml-4 text-slate-400">{/* scikit-learn uses population std by default */}</p>
        <p className="mt-3 text-slate-400">Result: E[x_std] = 0,  Var[x_std] = 1</p>
        <p className="mt-3 text-slate-400">Inverse transform:</p>
        <p className="mt-1">x = x_std · σ + μ</p>
      </div>

      <p>
        <strong>Standardisation vs Min-Max — decision guide:</strong>
      </p>

      <div className="not-prose overflow-x-auto my-6">
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="bg-slate-100 text-slate-700">
              <th className="border border-slate-200 px-4 py-2 text-left">Criterion</th>
              <th className="border border-slate-200 px-4 py-2 text-left">Min-Max</th>
              <th className="border border-slate-200 px-4 py-2 text-left">Standardisation</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-slate-200 px-4 py-2">Output range</td>
              <td className="border border-slate-200 px-4 py-2">[0, 1] (bounded)</td>
              <td className="border border-slate-200 px-4 py-2">(-∞, +∞) (unbounded)</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="border border-slate-200 px-4 py-2">Outlier sensitivity</td>
              <td className="border border-slate-200 px-4 py-2">High — outliers compress range</td>
              <td className="border border-slate-200 px-4 py-2">Moderate — outliers inflate σ</td>
            </tr>
            <tr>
              <td className="border border-slate-200 px-4 py-2">Preserves outlier signal</td>
              <td className="border border-slate-200 px-4 py-2">No — squashed near 0 or 1</td>
              <td className="border border-slate-200 px-4 py-2">Yes — high z-scores retained</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="border border-slate-200 px-4 py-2">Best for</td>
              <td className="border border-slate-200 px-4 py-2">CNNs, image pixels, bounded activations</td>
              <td className="border border-slate-200 px-4 py-2">Linear models, SVMs, PCA, k-NN</td>
            </tr>
            <tr>
              <td className="border border-slate-200 px-4 py-2">Distribution assumption</td>
              <td className="border border-slate-200 px-4 py-2">None</td>
              <td className="border border-slate-200 px-4 py-2">None (but works best near-Gaussian)</td>
            </tr>
          </tbody>
        </table>
      </div>

      <p>
        For PCA, standardisation is mandatory: PCA finds directions of
        maximum variance, so features with larger raw variance dominate the
        first principal components regardless of their actual information
        content. After standardisation, each feature contributes equally to
        the variance calculation.
      </p>

      {/* ════════════════════════════════════════════════════
          SECTION 4 — Missing Data
      ════════════════════════════════════════════════════ */}
      <h2>4 · Missing Data: Mechanisms and Imputation</h2>

      <h3>4.1 The Missingness Taxonomy</h3>

      <p>
        Rubin (1976) formalised three mechanisms of missing data. Choosing
        the wrong imputation strategy for the wrong mechanism introduces bias
        that no model can overcome.
      </p>

      <div className="not-prose overflow-x-auto my-6">
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="bg-slate-100 text-slate-700">
              <th className="border border-slate-200 px-4 py-2 text-left">Mechanism</th>
              <th className="border border-slate-200 px-4 py-2 text-left">Definition</th>
              <th className="border border-slate-200 px-4 py-2 text-left">Example</th>
              <th className="border border-slate-200 px-4 py-2 text-left">Safe to impute?</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-slate-200 px-4 py-2 font-semibold">MCAR</td>
              <td className="border border-slate-200 px-4 py-2">Missing Completely At Random — missingness unrelated to any data</td>
              <td className="border border-slate-200 px-4 py-2">Lab technician randomly drops 2% of samples</td>
              <td className="border border-slate-200 px-4 py-2 text-green-700">Yes — any imputation valid</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="border border-slate-200 px-4 py-2 font-semibold">MAR</td>
              <td className="border border-slate-200 px-4 py-2">Missing At Random — missingness depends on observed variables only</td>
              <td className="border border-slate-200 px-4 py-2">Women less likely to report salary, but salary itself doesn't drive missingness</td>
              <td className="border border-slate-200 px-4 py-2 text-yellow-700">Yes — conditional imputation (KNN, MICE)</td>
            </tr>
            <tr>
              <td className="border border-slate-200 px-4 py-2 font-semibold">MNAR</td>
              <td className="border border-slate-200 px-4 py-2">Missing Not At Random — missingness depends on the unobserved value itself</td>
              <td className="border border-slate-200 px-4 py-2">High earners refuse to disclose their salary</td>
              <td className="border border-slate-200 px-4 py-2 text-red-700">Risky — bias cannot be fully removed</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h3>4.2 Imputation Strategies</h3>

      <p>
        <strong>Mean/Median/Mode imputation</strong> replaces every missing
        value with a single global statistic computed from the training set.
        Mean is appropriate for symmetric distributions; median for
        right-skewed data (income, house prices) because the mean is pulled
        toward outliers; mode for categorical columns. The downside: all
        imputed values are identical, artificially reducing variance and
        distorting correlations.
      </p>

      <p>
        <strong>KNN imputation</strong> finds the k nearest neighbours (in
        feature space) to each row with a missing value and replaces the
        missing entry with the weighted average of those neighbours' observed
        values. It respects local correlation structure far better than
        global statistics, but scales as O(n²) for large datasets.
      </p>

      <p>
        <strong>MICE — Multiple Imputation by Chained Equations</strong>{" "}
        (scikit-learn: <code>IterativeImputer</code>) iterates a sequence of
        regression models. For each feature with missing values, it trains a
        model using all other features as predictors, then predicts and fills
        the missing values. The process repeats for multiple cycles until
        convergence. This is the gold standard for MAR data—it models the
        full joint distribution of the features.
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 overflow-x-auto">
        <p className="text-slate-400 mb-2">{/* MICE algorithm sketch */}</p>
        <p>Initialise: fill all NaN with column means (M⁰)</p>
        <p className="mt-2">For iteration t = 1, 2, ..., T:</p>
        <p className="ml-4">  For each feature j with missing values:</p>
        <p className="ml-8">    fit model:  X_j_obs ~ f(X_-j_obs)</p>
        <p className="ml-8">    predict:    X_j_miss ← f̂(X_-j_miss)</p>
        <p className="ml-8">    update M^t[:, j]</p>
        <p className="mt-2">Convergence: ||M^t - M^(t-1)||_F {"<"} ε</p>
      </div>

      <p>
        <strong>When to drop rows:</strong> If fewer than 5% of rows are
        missing in a column and the mechanism is MCAR, listwise deletion
        (dropping rows) is statistically valid and loses minimal information.
        For MNAR data where the missingness itself carries signal, consider
        adding a binary indicator column (is_missing) before imputing, so
        the model can learn that the fact of being missing is predictive.
      </p>

      {/* ════════════════════════════════════════════════════
          SECTION 5 — Categorical Encoding
      ════════════════════════════════════════════════════ */}
      <h2>5 · Categorical Encoding</h2>

      <p>
        Machine learning models operate on real-valued tensors. Transforming
        categorical strings into numbers requires careful choices—the wrong
        encoding injects false ordinal relationships or leaks target
        information.
      </p>

      <h3>5.1 Ordinal Encoding</h3>
      <p>
        Assigns a monotone integer label: Low → 0, Medium → 1, High → 2.
        Use <em>only</em> when a true ordinal relationship exists in the
        domain. Never apply ordinal encoding to nominal categories (e.g.,
        city names) because the integer differences (New York – London = 3)
        become features that most models will try to exploit.
      </p>

      <h3>5.2 One-Hot Encoding (OHE)</h3>
      <p>
        Creates one binary column per category. For a feature with k unique
        values, OHE produces k columns. Each row has exactly one 1 and
        (k - 1) zeros.
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 overflow-x-auto">
        <p className="text-slate-400 mb-2">{/* OHE example */}</p>
        <p>Original: Color ∈ {"{Red, Green, Blue}"}</p>
        <p className="mt-2">One-Hot:</p>
        <p className="ml-4">Red   → [1, 0, 0]</p>
        <p className="ml-4">Green → [0, 1, 0]</p>
        <p className="ml-4">Blue  → [0, 0, 1]</p>
        <p className="mt-3 text-yellow-300">Dummy variable trap:</p>
        <p className="ml-4 text-slate-300">Red + Green + Blue = 1  (perfect multicollinearity)</p>
        <p className="ml-4 text-slate-300">Fix: drop_first=True → use k-1 columns only</p>
      </div>

      <p>
        The <em>dummy variable trap</em> occurs when all k columns are
        retained: their sum equals 1 for every row, making one column a
        linear combination of the others. This causes perfect
        multicollinearity in linear models, making coefficients
        non-identifiable. scikit-learn's <code>OneHotEncoder</code>{" "}
        handles this with <code>drop='first'</code> or{" "}
        <code>drop='if_binary'</code>. Tree models are immune.
      </p>

      <p>
        <strong>High-cardinality issue:</strong> A category with 10,000
        unique values produces 10,000 binary columns. The resulting sparse
        matrix is memory-expensive and may degrade model performance. Use
        target encoding or frequency encoding instead.
      </p>

      <h3>5.3 Target Encoding</h3>
      <p>
        Replaces each category value with the mean of the target variable
        for that category in the training set.
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 overflow-x-auto">
        <p className="text-slate-400 mb-1">{/* Target encoding formula */}</p>
        <p>encode(cᵢ) = mean(y | X_cat = cᵢ)  over training set</p>
        <p className="mt-3 text-slate-400">Smoothed variant (prevents overfitting on rare categories):</p>
        <p className="mt-1">encode(cᵢ) = (n_cᵢ · mean_cᵢ + λ · global_mean) / (n_cᵢ + λ)</p>
        <p className="ml-4 text-slate-400">n_cᵢ = count of category cᵢ in training set</p>
        <p className="ml-4 text-slate-400">λ    = smoothing strength (hyperparameter, typically 10–30)</p>
      </div>

      <p>
        <strong>Data leakage risk:</strong> If you compute target means on
        the full dataset before splitting, the encoded training features
        contain information from the test set. Always compute target
        encoding statistics on the training fold only, and apply them to the
        validation/test fold. Alternatively, use cross-validated target
        encoding (leave-one-out or k-fold within training) for production
        pipelines.
      </p>

      <h3>5.4 Frequency Encoding</h3>
      <p>
        Replaces each category with its relative frequency in the training
        set: <code>encode(cᵢ) = count(cᵢ) / n</code>. It is fast,
        handles high cardinality, and carries no target leakage. Rare and
        common categories become distinguishable to any model.
      </p>

      {/* ════════════════════════════════════════════════════
          SECTION 6 — Outlier Treatment
      ════════════════════════════════════════════════════ */}
      <h2>6 · Outlier Detection and Treatment</h2>

      <h3>6.1 IQR-Based Removal</h3>
      <p>
        The Interquartile Range (IQR) method uses the box-plot fence rule.
        It is robust because it relies on order statistics, not the mean—so
        it is not itself corrupted by outliers.
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 overflow-x-auto">
        <p className="text-slate-400 mb-1">{/* IQR fences */}</p>
        <p>Q1  = 25th percentile of feature</p>
        <p>Q3  = 75th percentile of feature</p>
        <p>IQR = Q3 - Q1</p>
        <p className="mt-2">Lower fence = Q1 - 1.5 · IQR</p>
        <p>Upper fence = Q3 + 1.5 · IQR</p>
        <p className="mt-2 text-yellow-300">Flag/remove observations outside [Lower fence, Upper fence]</p>
        <p className="mt-2 text-slate-400">Extreme outliers: use multiplier 3.0 instead of 1.5</p>
      </div>

      <h3>6.2 Z-Score Based Removal</h3>
      <p>
        Flag observations where |z| {">"} threshold (commonly 3). This assumes
        approximate normality. Under a Gaussian distribution, only 0.27% of
        observations fall beyond ±3σ, so any observation with |z| {">"} 3 is
        statistically anomalous.
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 overflow-x-auto">
        <p>z_i = (x_i - μ) / σ</p>
        <p className="mt-2 text-yellow-300">Flag: |z_i| {">"} 3  (or 2.5 for stricter threshold)</p>
        <p className="mt-2 text-slate-400">Note: μ and σ are corrupted by outliers themselves.</p>
        <p className="text-slate-400">For robustness, substitute Modified Z-Score:</p>
        <p className="mt-1">M_i = 0.6745 · (x_i - median) / MAD</p>
        <p className="ml-4 text-slate-400">MAD = median(|x_i - median(x)|)</p>
      </div>

      <h3>6.3 Winsorisation (Capping)</h3>
      <p>
        Rather than dropping outliers—which removes entire rows and may
        discard valid information in other columns—Winsorisation caps
        extreme values at a specified percentile. Values below the 1st
        percentile are set to the 1st-percentile value; values above the
        99th are set to the 99th-percentile value. This preserves sample
        size while limiting leverage.
      </p>

      <h3>6.4 Log Transform for Right-Skewed Data</h3>
      <p>
        Income, transaction amounts, and population counts are
        right-skewed: a long tail of very large values dwarfs the
        majority. Taking log(x + 1) compresses the tail, making the
        distribution closer to Gaussian and reducing the effective
        magnitude of outliers. The +1 offset handles zero values, since
        log(0) is undefined.
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 overflow-x-auto">
        <p className="text-slate-400 mb-1">{/* Transform options for skew */}</p>
        <p>Right-skewed (positive):  x_new = log(x + 1)     or  x_new = sqrt(x)</p>
        <p className="mt-1">Left-skewed  (negative):  x_new = x²              or  x_new = exp(x)</p>
        <p className="mt-3 text-slate-400">Box-Cox generalises both (requires x {">"} 0):</p>
        <p className="mt-1">x_bc = (x^λ - 1) / λ   if λ ≠ 0</p>
        <p>       x_bc = log(x)           if λ = 0</p>
        <p className="ml-4 text-slate-400">λ is estimated via MLE; λ=1 → no transform; λ=0 → log</p>
      </div>

      {/* ════════════════════════════════════════════════════
          SECTION 7 — Feature Engineering
      ════════════════════════════════════════════════════ */}
      <h2>7 · Feature Engineering</h2>

      <p>
        Feature engineering is the craft of creating new input variables
        from existing ones to better expose the underlying structure of the
        problem to the model. It is where domain knowledge delivers the most
        leverage—no algorithm can recover a signal that is not represented
        in the features.
      </p>

      <h3>7.1 Polynomial Features and Interaction Terms</h3>
      <p>
        Linear models cannot capture non-linear relationships natively.
        Adding polynomial terms (x², x³) or interaction terms (x₁ · x₂)
        lets linear models approximate curvature and joint effects. For two
        features x₁ and x₂, degree-2 polynomial expansion produces:
        [1, x₁, x₂, x₁², x₁x₂, x₂²]—six features from two.
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 overflow-x-auto">
        <p className="text-slate-400 mb-1">{/* Polynomial expansion */}</p>
        <p>Degree-2 expansion of [x₁, x₂]:</p>
        <p className="ml-4">[1,  x₁,  x₂,  x₁²,  x₁x₂,  x₂²]</p>
        <p className="mt-2 text-slate-400">Feature count grows as C(p + d, d) where p=features, d=degree:</p>
        <p className="ml-4">p=10, d=2  →  66 features</p>
        <p className="ml-4">p=10, d=3  →  286 features  (explosion!)</p>
        <p className="mt-2 text-yellow-300">Always combine with Lasso regularisation to kill redundant terms</p>
      </div>

      <h3>7.2 Date Decomposition</h3>
      <p>
        Raw timestamps are opaque integers to models. Decomposing a
        datetime into constituent parts exposes temporal patterns:
        year (long-term trend), month (seasonality), day-of-week
        (weekly rhythm), hour (intraday pattern), is_weekend (binary),
        days_since_event (recency). These features often dramatically
        improve time-series-aware models.
      </p>

      <h3>7.3 Ratio and Aggregation Features</h3>
      <p>
        Domain knowledge often reveals meaningful ratios: revenue per
        user, clicks per impression (CTR), debt-to-income ratio. These
        compress two columns into one signal-dense feature. Similarly,
        aggregations over groups (mean spend per customer segment,
        rolling 7-day average) capture context that single-row features
        cannot.
      </p>

      {/* ════════════════════════════════════════════════════
          SECTION 8 — Train/Validation/Test Split
      ════════════════════════════════════════════════════ */}
      <h2>8 · The Train / Validation / Test Split</h2>

      <p>
        The fundamental rule of machine learning generalisation is:{" "}
        <em>
          you may only use information from the training set to make
          decisions that affect what the model sees during evaluation.
        </em>{" "}
        Violating this rule is called <strong>data leakage</strong>, and it
        produces optimistic metrics that vanish in production.
      </p>

      <h3>8.1 Why Three Splits?</h3>
      <p>
        <strong>Training set</strong> (60–70%): fit model parameters and
        fit preprocessing transformers (scaler statistics, imputer means,
        encoder vocabularies).
      </p>
      <p>
        <strong>Validation set</strong> (10–20%): tune hyperparameters and
        architecture choices. Because you look at validation performance
        many times, some optimism leaks in—the validation metric is not a
        true out-of-sample estimate after hyperparameter search.
      </p>
      <p>
        <strong>Test set</strong> (10–20%): a single, untouched holdout
        used exactly once to report final performance. It must never be
        used to make modelling decisions. Looking at test performance and
        then adjusting the model is a subtle form of leakage.
      </p>

      <h3>8.2 The Preprocessing Leakage Trap</h3>
      <p>
        A common mistake: fit a StandardScaler on the entire dataset
        (train + val + test), then split. This means the scaler's mean and
        standard deviation were computed with test-set information. The
        test set is no longer held out—the model has implicitly seen it
        through the scaler's statistics. The correct protocol:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 overflow-x-auto">
        <p className="text-red-400 mb-2">❌ WRONG — leaks test statistics into training:</p>
        <p>scaler.fit(X_all)           # sees test set!</p>
        <p>X_all_scaled = scaler.transform(X_all)</p>
        <p>X_train, X_test = train_test_split(X_all_scaled)</p>
        <p className="mt-4 text-green-400">✅ CORRECT — fit only on training data:</p>
        <p>X_train, X_test = train_test_split(X_all)</p>
        <p>scaler.fit(X_train)         # never sees test set</p>
        <p>X_train_s = scaler.transform(X_train)</p>
        <p>X_test_s  = scaler.transform(X_test)  # apply learned params</p>
        <p className="mt-3 text-slate-400"># Pipelines enforce this automatically — use them.</p>
      </div>

      {/* ════════════════════════════════════════════════════
          SECTION 9 — Feature Selection
      ════════════════════════════════════════════════════ */}
      <h2>9 · Feature Selection</h2>

      <p>
        More features is not always better. Irrelevant features add noise,
        increase training time, inflate model complexity, and can trigger
        the curse of dimensionality in distance-based methods. Feature
        selection is the process of choosing the subset of features that
        best explains the target.
      </p>

      <h3>9.1 Filter Methods</h3>
      <p>
        Filter methods score features independently of any model, using
        statistical tests. They are fast but ignore feature interactions.
      </p>
      <ul>
        <li>
          <strong>Pearson correlation</strong>: for continuous features vs
          continuous target. Remove features with |r| {"<"} 0.05 (near-zero
          linear correlation with target) or |r| {">"} 0.95 with another
          feature (multicollinearity).
        </li>
        <li>
          <strong>Chi-square test</strong>: for categorical features vs
          categorical target. Tests independence; high χ² → feature is
          associated with the label.
        </li>
        <li>
          <strong>Mutual Information</strong>: measures any statistical
          dependence (not just linear). Works for both continuous and
          discrete variables.
        </li>
      </ul>

      <h3>9.2 Wrapper Methods — Recursive Feature Elimination (RFE)</h3>
      <p>
        RFE trains a model on all features, ranks features by their
        importance coefficients, removes the least important feature, and
        repeats until the desired feature count is reached. Computationally
        expensive (trains the model k times) but respects feature
        interactions because the model is always evaluated on the full
        remaining feature set.
      </p>

      <h3>9.3 Embedded Methods</h3>
      <p>
        Embedded methods perform selection as part of training.{" "}
        <strong>Lasso (L1 regularisation)</strong> drives irrelevant
        feature coefficients exactly to zero during fitting—the
        regularisation penalty is the selector. The strength is controlled
        by α: larger α → more features zeroed out.
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 overflow-x-auto">
        <p className="text-slate-400 mb-1">{/* Lasso objective */}</p>
        <p>Lasso objective:</p>
        <p className="ml-4 text-green-300">minimize: (1/2n) · ||y - Xβ||² + α · ||β||₁</p>
        <p className="mt-2 text-slate-400">L1 penalty (||β||₁ = Σ|βⱼ|) creates a diamond constraint</p>
        <p className="text-slate-400">region in β-space whose corners lie on axes → exact zeros</p>
        <p className="mt-2">Tree feature importance:</p>
        <p className="ml-4">importance(j) = Σ_t  p(t) · ΔImpurity(t)  for all splits on j</p>
        <p className="ml-4 text-slate-400">p(t) = fraction of samples reaching node t</p>
      </div>

      {/* ════════════════════════════════════════════════════
          SECTION 10 — Full scikit-learn Pipeline
      ════════════════════════════════════════════════════ */}
      <h2>10 · Complete scikit-learn Pipeline</h2>

      <p>
        The following code demonstrates a production-ready preprocessing and
        modelling pipeline. Every transformer is fitted exclusively on the
        training set; scikit-learn's <code>Pipeline</code> enforces this
        automatically by chaining <code>fit_transform</code> on train and{" "}
        <code>transform</code> on test. The{" "}
        <code>ColumnTransformer</code> applies different preprocessing
        recipes to numerical and categorical columns in parallel.
      </p>

      <TerminalBlock language="python">{`# ─────────────────────────────────────────────────────────────────
# Complete Data Preprocessing Pipeline
# Dataset: Titanic-style (numerical + categorical + missing values)
# ─────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,       # Z-score scaling for numerical features
    OneHotEncoder,        # Binary encoding for categorical features
    MinMaxScaler,         # Alternative: bounded [0,1] scaling
)
from sklearn.impute import (
    SimpleImputer,        # Mean/median/mode imputation
    KNNImputer,           # KNN-based imputation (optional comparison)
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# 1. Load data  (Titanic survival: binary classification)
# ─────────────────────────────────────────────────────────────────
titanic = fetch_openml("titanic", version=1, as_frame=True)
df = titanic.frame.copy()

# Keep a tractable feature subset for illustration
FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
TARGET   = "survived"

X = df[FEATURES]
y = (df[TARGET].astype(int))           # 0=died, 1=survived

print(f"Dataset shape: {X.shape}")
print(f"Missing values per column:\n{X.isnull().sum()}")
# age: ~20% missing, embarked: 2 missing → needs imputation

# ─────────────────────────────────────────────────────────────────
# 2. Train / Validation / Test split  — SPLIT BEFORE ANYTHING ELSE
#    Rationale: all preprocessing statistics must come from X_train
# ─────────────────────────────────────────────────────────────────
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.15,        # 15% held-out test set, touched once at the end
    random_state=42,
    stratify=y             # preserve class balance across splits
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.18,        # ~15% of original = validation set
    random_state=42,
    stratify=y_temp
)
print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

# ─────────────────────────────────────────────────────────────────
# 3. Define which columns are numerical vs categorical
# ─────────────────────────────────────────────────────────────────
NUMERICAL   = ["age", "fare", "sibsp", "parch"]   # continuous / count
CATEGORICAL = ["pclass", "sex", "embarked"]       # nominal strings / ints-as-labels

# ─────────────────────────────────────────────────────────────────
# 4. Build sub-pipelines for each column type
# ─────────────────────────────────────────────────────────────────

# Numerical pipeline:
#   Step 1 — Impute NaN with median (robust to outliers, unlike mean)
#   Step 2 — Standardise to zero mean, unit variance (required for LogReg/SVM)
numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

# Categorical pipeline:
#   Step 1 — Impute NaN with most-frequent value (mode)
#   Step 2 — One-Hot encode; handle_unknown='ignore' silences errors
#             for categories unseen during training (e.g. new ports)
#             drop='first' removes one column per feature to avoid
#             the dummy-variable trap (perfect multicollinearity)
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        drop="first",            # eliminates dummy trap
        handle_unknown="ignore", # unseen test categories → all zeros
        sparse_output=False,     # return dense array for compatibility
    )),
])

# ─────────────────────────────────────────────────────────────────
# 5. ColumnTransformer: applies different pipelines to different cols
#    remainder='drop' discards any column not mentioned above
# ─────────────────────────────────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_pipeline,   NUMERICAL),
    ("cat", categorical_pipeline, CATEGORICAL),
], remainder="drop")

# ─────────────────────────────────────────────────────────────────
# 6. Full Pipeline: preprocessor → classifier
#    Pipeline.fit(X_train) calls fit_transform on preprocessor,
#    then fit on the classifier using the transformed features.
#    Pipeline.predict(X_test) calls transform (NOT fit_transform)
#    on preprocessor, so test statistics never bleed into training.
# ─────────────────────────────────────────────────────────────────
full_pipeline = Pipeline(steps=[
    ("preprocessor",  preprocessor),
    ("classifier",    LogisticRegression(
        C=1.0,           # inverse regularisation strength; smaller C → stronger L2
        max_iter=1000,   # enough iterations for convergence after scaling
        random_state=42,
    )),
])

# ─────────────────────────────────────────────────────────────────
# 7. Fit on training data ONLY
# ─────────────────────────────────────────────────────────────────
full_pipeline.fit(X_train, y_train)

# ─────────────────────────────────────────────────────────────────
# 8. Evaluate on validation set (for hyperparameter decisions)
# ─────────────────────────────────────────────────────────────────
y_val_pred  = full_pipeline.predict(X_val)
y_val_proba = full_pipeline.predict_proba(X_val)[:, 1]

print("\n── Validation Performance ──")
print(classification_report(y_val, y_val_pred, target_names=["Died", "Survived"]))
print(f"Validation AUC-ROC: {roc_auc_score(y_val, y_val_proba):.4f}")

# ─────────────────────────────────────────────────────────────────
# 9. Cross-validation: more robust performance estimate on train set
#    Internally fits + transforms 5 separate folds — still leak-free
# ─────────────────────────────────────────────────────────────────
cv_scores = cross_val_score(
    full_pipeline, X_train, y_train,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,     # use all CPU cores
)
print(f"\n5-Fold CV AUC (train): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────────────────────────
# 10. Final test-set evaluation — done ONCE, after all decisions made
# ─────────────────────────────────────────────────────────────────
y_test_pred  = full_pipeline.predict(X_test)
y_test_proba = full_pipeline.predict_proba(X_test)[:, 1]

print("\n── FINAL Test Performance (one-time evaluation) ──")
print(classification_report(y_test, y_test_pred, target_names=["Died", "Survived"]))
print(f"Test AUC-ROC: {roc_auc_score(y_test, y_test_proba):.4f}")

# ─────────────────────────────────────────────────────────────────
# 11. Inspect what the preprocessor actually produced
# ─────────────────────────────────────────────────────────────────
# Get feature names after OHE expansion
ohe_feature_names = (
    full_pipeline
    .named_steps["preprocessor"]
    .named_transformers_["cat"]
    .named_steps["encoder"]
    .get_feature_names_out(CATEGORICAL)
)
all_feature_names = NUMERICAL + list(ohe_feature_names)
print(f"\nFeatures after preprocessing ({len(all_feature_names)}):")
print(all_feature_names)
# e.g. ['age', 'fare', 'sibsp', 'parch',
#        'pclass_3', 'sex_male', 'embarked_Q', 'embarked_S']

# ─────────────────────────────────────────────────────────────────
# 12. Outlier treatment demonstration (Winsorisation)
#     Applied before the pipeline, on the training set only
# ─────────────────────────────────────────────────────────────────
def winsorise(df_in: pd.DataFrame, cols: list, lower_pct=0.01, upper_pct=0.99):
    """
    Cap feature values at the given percentiles to limit outlier leverage.
    Percentile bounds must be computed on training data and applied uniformly.
    
    Parameters
    ----------
    df_in     : DataFrame to transform
    cols      : numerical columns to winsorise
    lower_pct : lower cap percentile (computed on training set)
    upper_pct : upper cap percentile (computed on training set)
    
    Returns (transformed_df, bounds_dict) — store bounds_dict for test set
    """
    df_out = df_in.copy()
    bounds = {}
    for col in cols:
        lo = df_in[col].quantile(lower_pct)  # fit on THIS df (train only)
        hi = df_in[col].quantile(upper_pct)
        bounds[col] = (lo, hi)
        df_out[col] = df_out[col].clip(lower=lo, upper=hi)  # cap values
    return df_out, bounds

def apply_winsorise(df_in: pd.DataFrame, bounds: dict):
    """Apply pre-computed winsorisation bounds to val/test sets."""
    df_out = df_in.copy()
    for col, (lo, hi) in bounds.items():
        df_out[col] = df_out[col].clip(lower=lo, upper=hi)
    return df_out

# Apply Winsorisation to 'fare' (heavy right skew in Titanic)
X_train_w, fare_bounds = winsorise(X_train, cols=["fare"])
X_val_w   = apply_winsorise(X_val,  fare_bounds)
X_test_w  = apply_winsorise(X_test, fare_bounds)

print(f"\nFare before Winsorisation — max: {X_train['fare'].max():.1f}")
print(f"Fare after  Winsorisation — max: {X_train_w['fare'].max():.1f}")

# ─────────────────────────────────────────────────────────────────
# 13. Feature selection with Lasso (embedded method)
# ─────────────────────────────────────────────────────────────────
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import LabelEncoder

# For feature selection demonstration, use the already-processed matrix
X_train_processed = (
    full_pipeline
    .named_steps["preprocessor"]
    .transform(X_train)       # use already-fitted preprocessor
)

# SelectFromModel wraps any estimator with coef_ or feature_importances_
# LassoCV auto-tunes alpha via cross-validation on training data
lasso_selector = SelectFromModel(
    LassoCV(cv=5, random_state=42),
    threshold="mean"      # keep features with |coef| >= mean(|coefs|)
)
lasso_selector.fit(X_train_processed, y_train)
selected_mask = lasso_selector.get_support()
selected_features = [f for f, s in zip(all_feature_names, selected_mask) if s]
print(f"\nLasso-selected features: {selected_features}")

# ─────────────────────────────────────────────────────────────────
# 14. Comparing two scalers: StandardScaler vs MinMaxScaler
#     Shows how the choice of scaler affects coefficient magnitudes
# ─────────────────────────────────────────────────────────────────
for scaler_name, scaler_obj in [("StandardScaler", StandardScaler()),
                                  ("MinMaxScaler",   MinMaxScaler())]:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  scaler_obj),
    ])
    preprocessor_alt = ColumnTransformer([
        ("num", num_pipe,              NUMERICAL),
        ("cat", categorical_pipeline,  CATEGORICAL),
    ], remainder="drop")
    pipe_alt = Pipeline([
        ("preprocessor", preprocessor_alt),
        ("classifier",   LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ])
    auc_scores = cross_val_score(
        pipe_alt, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1
    )
    print(f"{scaler_name:20s} — CV AUC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
`}</TerminalBlock>

      {/* ════════════════════════════════════════════════════
          SECTION 11 — Common Pitfalls
      ════════════════════════════════════════════════════ */}
      <h2>11 · Common Pitfalls</h2>

      <div className="not-prose overflow-x-auto my-6">
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="bg-rose-50 text-rose-800">
              <th className="border border-rose-100 px-4 py-2 text-left">Pitfall</th>
              <th className="border border-rose-100 px-4 py-2 text-left">Why It Hurts</th>
              <th className="border border-rose-100 px-4 py-2 text-left">Fix</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-slate-200 px-4 py-2 font-medium">Fitting scaler on full dataset</td>
              <td className="border border-slate-200 px-4 py-2">Test-set statistics contaminate training — overly optimistic metrics</td>
              <td className="border border-slate-200 px-4 py-2">Always fit transformers only on <code>X_train</code></td>
            </tr>
            <tr className="bg-slate-50">
              <td className="border border-slate-200 px-4 py-2 font-medium">Target encoding without cross-validation</td>
              <td className="border border-slate-200 px-4 py-2">Target mean computed with the row's own y — direct label leakage</td>
              <td className="border border-slate-200 px-4 py-2">Use OOF target encoding or <code>TargetEncoder</code> with <code>cv=5</code></td>
            </tr>
            <tr>
              <td className="border border-slate-200 px-4 py-2 font-medium">Dropping outliers from test set</td>
              <td className="border border-slate-200 px-4 py-2">Production data will contain outliers; model never sees them</td>
              <td className="border border-slate-200 px-4 py-2">Only remove outliers from train; Winsorise test</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="border border-slate-200 px-4 py-2 font-medium">Retaining all OHE columns</td>
              <td className="border border-slate-200 px-4 py-2">Perfect multicollinearity breaks linear model inversion</td>
              <td className="border border-slate-200 px-4 py-2"><code>drop='first'</code> in <code>OneHotEncoder</code></td>
            </tr>
            <tr>
              <td className="border border-slate-200 px-4 py-2 font-medium">Mean imputation for skewed features</td>
              <td className="border border-slate-200 px-4 py-2">Mean pulled by outliers → imputed value is unrepresentative</td>
              <td className="border border-slate-200 px-4 py-2">Use median for right-skewed features</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="border border-slate-200 px-4 py-2 font-medium">Feature selection before split</td>
              <td className="border border-slate-200 px-4 py-2">Selection criterion uses test-set labels — selection leakage</td>
              <td className="border border-slate-200 px-4 py-2">Embed <code>SelectFromModel</code> inside the <code>Pipeline</code></td>
            </tr>
            <tr>
              <td className="border border-slate-200 px-4 py-2 font-medium">Ignoring MNAR mechanism</td>
              <td className="border border-slate-200 px-4 py-2">Imputing values that are missing because they're extreme introduces systematic bias</td>
              <td className="border border-slate-200 px-4 py-2">Add <code>is_missing</code> indicator column before imputing</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* ════════════════════════════════════════════════════
          SECTION 12 — Further Reading
      ════════════════════════════════════════════════════ */}
      <h2>12 · Further Reading</h2>

      <ul>
        <li>
          <strong>Rubin, D. B. (1976).</strong> "Inference and Missing Data."{" "}
          <em>Biometrika</em>, 63(3), 581–592. — The foundational paper
          defining MCAR, MAR, and MNAR with formal probability definitions.
        </li>
        <li>
          <strong>van Buuren, S. &amp; Groothuis-Oudshoorn, K. (2011).</strong>{" "}
          "mice: Multivariate Imputation by Chained Equations in R."{" "}
          <em>Journal of Statistical Software</em>, 45(3), 1–67. — Canonical
          reference for MICE; the algorithm underpins scikit-learn's{" "}
          <code>IterativeImputer</code>.
        </li>
        <li>
          <strong>Géron, A. (2022).</strong>{" "}
          <em>
            Hands-On Machine Learning with Scikit-Learn, Keras, and
            TensorFlow
          </em>{" "}
          (3rd ed.), Chapter 2. O'Reilly. — Comprehensive end-to-end
          preprocessing walkthrough on the California housing dataset.
        </li>
        <li>
          <strong>Tibshirani, R. (1996).</strong> "Regression Shrinkage and
          Selection via the Lasso." <em>Journal of the Royal Statistical
          Society B</em>, 58(1), 267–288. — Original Lasso paper; explains
          why L1 geometry induces exact zeros in coefficient vectors.
        </li>
        <li>
          <strong>Micci-Barreca, D. (2001).</strong> "A Preprocessing Scheme
          for High-Cardinality Categorical Attributes in Classification and
          Prediction Problems." <em>ACM SIGKDD Explorations</em>, 3(1),
          27–32. — Introduces smoothed target encoding with the Bayesian
          shrinkage formula.
        </li>
        <li>
          <strong>scikit-learn documentation.</strong>{" "}
          <em>Preprocessing data</em> —{" "}
          <code>sklearn.preprocessing</code>,{" "}
          <code>sklearn.impute</code>,{" "}
          <code>sklearn.pipeline</code>. The official API reference with
          worked examples for every transformer discussed in this lecture.
        </li>
      </ul>
    </article>
  );
}
