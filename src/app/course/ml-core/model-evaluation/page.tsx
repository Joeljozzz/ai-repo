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

      <h1>Model Evaluation &amp; Metrics</h1>

      <p>
        Training a model is only half the job. The harder half is knowing
        whether the model is actually <em>good</em>—and good <em>for what</em>.
        A model that predicts "no fraud" for every transaction achieves 99.8 %
        accuracy on a typical fraud dataset, yet it is completely useless. This
        lecture builds a rigorous, ground-up understanding of every major
        evaluation tool used in production ML systems: from the confusion matrix
        and its derived metrics, through ROC and Precision-Recall curves, to
        regression error measures and the statistical machinery of
        cross-validation.
      </p>

      {/* ── Learning Objectives ── */}
      <h2>Learning Objectives</h2>
      <ul>
        <li>
          Construct a confusion matrix by hand and derive Accuracy, Precision,
          Recall, and F1 from first principles.
        </li>
        <li>
          Explain why accuracy is a misleading metric for imbalanced datasets
          and choose an appropriate alternative.
        </li>
        <li>
          Interpret an ROC curve and AUC score, and contrast it with a
          Precision-Recall curve for imbalanced problems.
        </li>
        <li>
          Derive the F-beta generalisation of F1 and choose the right beta for a
          given cost asymmetry.
        </li>
        <li>
          Select between RMSE and MAE for a regression task based on the
          sensitivity to outliers required.
        </li>
        <li>
          Implement stratified K-Fold cross-validation and interpret learning
          curves to diagnose overfitting and underfitting.
        </li>
      </ul>

      {/* ══════════════════════════════════════════════
          Section 1 · The Confusion Matrix
      ══════════════════════════════════════════════ */}
      <h2>1 · The Confusion Matrix</h2>

      <p>
        Every binary classifier makes exactly four kinds of decisions. Given a
        ground-truth label (Positive / Negative) and a predicted label, the
        outcome falls into one of four cells. The <strong>confusion matrix</strong>{" "}
        arranges these cells in a 2 × 2 grid:
      </p>

      {/* Confusion matrix table */}
      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full border border-slate-200 text-sm text-center">
          <thead className="bg-slate-100">
            <tr>
              <th className="border border-slate-200 px-4 py-2 text-left"></th>
              <th className="border border-slate-200 px-4 py-2">
                Predicted Positive
              </th>
              <th className="border border-slate-200 px-4 py-2">
                Predicted Negative
              </th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-slate-200 px-4 py-2 font-semibold text-left bg-slate-50">
                Actual Positive
              </td>
              <td className="border border-slate-200 px-4 py-2 bg-emerald-50 text-emerald-800 font-bold">
                TP (True Positive)
              </td>
              <td className="border border-slate-200 px-4 py-2 bg-rose-50 text-rose-800 font-bold">
                FN (False Negative)
              </td>
            </tr>
            <tr>
              <td className="border border-slate-200 px-4 py-2 font-semibold text-left bg-slate-50">
                Actual Negative
              </td>
              <td className="border border-slate-200 px-4 py-2 bg-rose-50 text-rose-800 font-bold">
                FP (False Positive)
              </td>
              <td className="border border-slate-200 px-4 py-2 bg-emerald-50 text-emerald-800 font-bold">
                TN (True Negative)
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <p>
        <strong>TP (True Positive)</strong>: The model correctly predicted
        Positive.{" "}
        <strong>TN (True Negative)</strong>: The model correctly predicted
        Negative.{" "}
        <strong>FP (False Positive)</strong>: The model wrongly predicted
        Positive (a "false alarm", Type I error).{" "}
        <strong>FN (False Negative)</strong>: The model wrongly predicted
        Negative (a "miss", Type II error).
      </p>

      <h3>1.1 · Building a Confusion Matrix by Hand</h3>
      <p>
        Consider a tiny email spam classifier tested on 10 emails. Ground truth
        and predictions are:
      </p>

      <div className="not-prose overflow-x-auto my-4">
        <table className="min-w-full border border-slate-200 text-sm text-center">
          <thead className="bg-slate-100">
            <tr>
              {["Email", "Actual", "Predicted", "Cell"].map((h) => (
                <th key={h} className="border border-slate-200 px-3 py-2">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[
              ["1", "Spam", "Spam", "TP"],
              ["2", "Spam", "Not Spam", "FN"],
              ["3", "Not Spam", "Not Spam", "TN"],
              ["4", "Spam", "Spam", "TP"],
              ["5", "Not Spam", "Spam", "FP"],
              ["6", "Not Spam", "Not Spam", "TN"],
              ["7", "Spam", "Spam", "TP"],
              ["8", "Not Spam", "Not Spam", "TN"],
              ["9", "Spam", "Not Spam", "FN"],
              ["10", "Not Spam", "Not Spam", "TN"],
            ].map(([id, act, pred, cell]) => (
              <tr key={id}>
                <td className="border border-slate-200 px-3 py-1">{id}</td>
                <td className="border border-slate-200 px-3 py-1">{act}</td>
                <td className="border border-slate-200 px-3 py-1">{pred}</td>
                <td
                  className={`border border-slate-200 px-3 py-1 font-semibold ${
                    cell === "TP" || cell === "TN"
                      ? "text-emerald-700"
                      : "text-rose-700"
                  }`}
                >
                  {cell}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p>
        Tallying: <strong>TP = 3, FN = 2, FP = 1, TN = 4</strong>. Total
        samples = 10. Every metric on the following sections derives directly
        from these four numbers.
      </p>

      {/* ══════════════════════════════════════════════
          Section 2 · Accuracy and Its Failure Mode
      ══════════════════════════════════════════════ */}
      <h2>2 · Accuracy — and Why It Lies</h2>

      <p>
        Accuracy is the first metric everyone reaches for, and the first one
        that misleads in practice.
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 leading-relaxed">
        <p className="text-slate-400 mb-2">// Accuracy</p>
        <p>
          Accuracy = (TP + TN) / (TP + TN + FP + FN)
        </p>
        <p className="mt-3 text-slate-400">// For our spam example:</p>
        <p>Accuracy = (3 + 4) / 10 = 0.70  →  70 %</p>
      </div>

      <h3>2.1 · The Imbalanced-Dataset Trap</h3>
      <p>
        Credit-card fraud datasets typically have a positive rate (fraud) of
        0.1–0.2 %. A classifier that <em>always predicts "not fraud"</em>{" "}
        achieves:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 leading-relaxed">
        <p className="text-slate-400 mb-1">// Assume 10,000 transactions, 20 are fraud</p>
        <p>TP = 0,  TN = 9980,  FP = 0,  FN = 20</p>
        <p className="mt-2">Accuracy = (0 + 9980) / 10000 = 99.80 %</p>
        <p className="mt-2 text-rose-400">
          // But Recall = 0/20 = 0.0  →  we catch ZERO fraudulent transactions
        </p>
      </div>

      <p>
        This is called the <strong>accuracy paradox</strong>. On imbalanced
        datasets the majority class dominates accuracy, masking total failure on
        the minority class. This is precisely why Precision, Recall, F1, and the
        PR curve exist.
      </p>

      {/* ══════════════════════════════════════════════
          Section 3 · Precision and Recall
      ══════════════════════════════════════════════ */}
      <h2>3 · Precision and Recall</h2>

      <h3>3.1 · Precision</h3>
      <p>
        Precision asks: <em>"Of everything I labelled Positive, how many were
        actually Positive?"</em> It measures the quality of positive
        predictions.
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 leading-relaxed">
        <p>Precision = TP / (TP + FP)</p>
        <p className="mt-2 text-slate-400">// Spam example: 3 / (3 + 1) = 0.75</p>
        <p className="mt-1 text-slate-400">
          // Of 4 emails flagged as spam, 3 were actually spam.
        </p>
      </div>

      <p>
        High precision is critical when <strong>false positives are costly</strong>.
        A spam filter with low precision annoys users by sending legitimate email
        to the spam folder. A cancer screening tool with low precision subjects
        healthy people to unnecessary biopsies.
      </p>

      <h3>3.2 · Recall (Sensitivity / True Positive Rate)</h3>
      <p>
        Recall asks: <em>"Of all actual Positives, how many did I catch?"</em>{" "}
        It measures completeness.
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 leading-relaxed">
        <p>Recall = TP / (TP + FN)</p>
        <p className="mt-2 text-slate-400">// Spam example: 3 / (3 + 2) = 0.60</p>
        <p className="mt-1 text-slate-400">
          // Of 5 actual spam emails, we caught only 3.
        </p>
      </div>

      <p>
        High recall is critical when <strong>false negatives are costly</strong>.
        A cancer screening test that misses actual cancers (high FN, low recall)
        is dangerous. A fraud detector that misses fraud (high FN) loses money.
      </p>

      <h3>3.3 · The Precision-Recall Trade-off</h3>
      <p>
        Precision and Recall are in fundamental tension. Lowering the
        classification threshold (flagging more samples as Positive) increases
        Recall (catch more true positives) but decreases Precision (more false
        alarms are included). Raising the threshold does the opposite. No single
        threshold simultaneously maximises both; instead, we trace the
        Precision-Recall curve across all thresholds.
      </p>

      {/* ══════════════════════════════════════════════
          Section 4 · F1-Score and F-beta
      ══════════════════════════════════════════════ */}
      <h2>4 · F1-Score — Harmonic Mean from First Principles</h2>

      <p>
        We need a single number that balances Precision (P) and Recall (R). The
        naive arithmetic mean <code>(P + R) / 2</code> fails because a model
        with P = 1.0 and R = 0.0 would score 0.5—even though it catches
        nothing. The <strong>harmonic mean</strong> solves this: it is heavily
        penalised when either value is low.
      </p>

      <h3>4.1 · Derivation</h3>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 leading-relaxed">
        <p className="text-slate-400 mb-2">// Harmonic mean of two values a, b:</p>
        <p>H(a, b) = 2 / (1/a + 1/b)</p>
        <p className="mt-3 text-slate-400">// Apply to Precision and Recall:</p>
        <p>F1 = 2 / (1/P + 1/R)</p>
        <p className="mt-2 text-slate-400">// Simplify by multiplying numerator and denominator by P·R:</p>
        <p>F1 = 2·P·R / (P + R)</p>
        <p className="mt-3 text-slate-400">// Spam example: 2 · 0.75 · 0.60 / (0.75 + 0.60)</p>
        <p>F1 = 0.90 / 1.35 ≈ 0.667</p>
        <p className="mt-3 text-rose-400">// Degenerate case: P=1.0, R=0.0</p>
        <p>F1 = 2·1·0 / (1+0) = 0.0  ← correctly penalised</p>
      </div>

      <h3>4.2 · The F-beta Generalisation</h3>
      <p>
        In practice the costs of false positives and false negatives are rarely
        equal. The F-beta score introduces a weight <code>β</code> that controls
        how much more we value Recall over Precision.
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 leading-relaxed">
        <p>F_β = (1 + β²) · P · R / (β² · P + R)</p>
        <p className="mt-3 text-slate-400">// β = 1  →  F1  (equal weight)</p>
        <p className="text-slate-400">// β = 2  →  F2  (Recall weighted 2× more than Precision)</p>
        <p className="text-slate-400">// β = 0.5 → F0.5 (Precision weighted 2× more than Recall)</p>
        <p className="mt-3 text-slate-400">
          // Intuition: β² represents how many times more important Recall is
        </p>
        <p className="text-slate-400">// than Precision in the numerator weighting.</p>
      </div>

      <p>
        <strong>Choosing β in practice:</strong> Medical diagnosis → use F2
        (missing a disease is worse than a false alarm). Document retrieval
        (precision matters more) → use F0.5.
      </p>

      {/* ══════════════════════════════════════════════
          Section 5 · ROC Curve and AUC
      ══════════════════════════════════════════════ */}
      <h2>5 · The ROC Curve and AUC</h2>

      <p>
        A binary classifier typically outputs a <em>score</em> (e.g. a
        probability), and a threshold converts that score into a class label.
        The <strong>ROC curve</strong> (Receiver Operating Characteristic)
        visualises classifier performance across <em>all</em> possible
        thresholds simultaneously.
      </p>

      <h3>5.1 · Axes Defined</h3>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 leading-relaxed">
        <p className="text-slate-400 mb-2">// Y-axis: True Positive Rate (Recall / Sensitivity)</p>
        <p>TPR = TP / (TP + FN)</p>
        <p className="mt-3 text-slate-400">// X-axis: False Positive Rate (1 - Specificity)</p>
        <p>FPR = FP / (FP + TN)</p>
        <p className="mt-3 text-slate-400">
          // As threshold ↓  (flag more samples Positive):
        </p>
        <p className="text-slate-400">
          //   TPR ↑ (we catch more true positives)
        </p>
        <p className="text-slate-400">
          //   FPR ↑ (we also flag more negatives incorrectly)
        </p>
      </div>

      <h3>5.2 · Interpreting AUC</h3>
      <p>
        The <strong>Area Under the ROC Curve (AUC-ROC)</strong> collapses the
        entire curve into a single number.
      </p>

      <div className="not-prose overflow-x-auto my-4">
        <table className="min-w-full border border-slate-200 text-sm">
          <thead className="bg-slate-100">
            <tr>
              {["AUC Value", "Interpretation"].map((h) => (
                <th key={h} className="border border-slate-200 px-4 py-2 text-left">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[
              ["0.5", "Random classifier — model has no discriminatory power"],
              ["0.5 – 0.7", "Poor — slight better than chance"],
              ["0.7 – 0.8", "Acceptable — useful for many problems"],
              ["0.8 – 0.9", "Good — strong discriminatory ability"],
              ["0.9 – 1.0", "Excellent — near-perfect separation"],
              ["1.0", "Perfect classifier — no threshold makes an error"],
            ].map(([val, desc]) => (
              <tr key={val}>
                <td className="border border-slate-200 px-4 py-2 font-mono font-semibold">
                  {val}
                </td>
                <td className="border border-slate-200 px-4 py-2">{desc}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p>
        <strong>Probabilistic interpretation:</strong> AUC equals the
        probability that a randomly chosen positive example receives a higher
        classifier score than a randomly chosen negative example. AUC = 0.5
        means the model ranks them randomly. AUC = 1.0 means every positive
        scores above every negative.
      </p>

      <h3>5.3 · Why AUC is Threshold-Independent</h3>
      <p>
        A single confusion matrix depends on a chosen threshold. AUC aggregates
        performance across all thresholds, making it useful when the operating
        threshold is unknown at evaluation time—e.g. when different deployments
        require different precision/recall trade-offs.
      </p>

      {/* ══════════════════════════════════════════════
          Section 6 · Precision-Recall Curve
      ══════════════════════════════════════════════ */}
      <h2>6 · The Precision-Recall Curve</h2>

      <p>
        The ROC curve can be overly optimistic on severely imbalanced datasets.
        When negatives vastly outnumber positives, even a bad model can maintain
        a low FPR (because TN is huge), inflating AUC-ROC. The{" "}
        <strong>Precision-Recall (PR) curve</strong> ignores true negatives
        entirely, making it the right tool for imbalanced problems.
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 leading-relaxed">
        <p className="text-slate-400 mb-2">// PR Curve: X = Recall, Y = Precision</p>
        <p>// Sweep threshold from 1.0 → 0.0:</p>
        <p>// threshold=1.0  →  nothing predicted Positive  →  P=1 (trivially), R=0</p>
        <p>// threshold=0.0  →  everything predicted Positive  →  P=base_rate, R=1</p>
        <p className="mt-3 text-slate-400">// Average Precision (AP):</p>
        <p>AP = Σ (R_n - R_(n-1)) · P_n</p>
        <p className="text-slate-400 mt-1">// i.e. the area under the PR curve (trapezoidal sum)</p>
        <p className="mt-3 text-slate-400">
          // Random baseline on imbalanced data (1% positives):
        </p>
        <p className="text-rose-400">// AUC-ROC ≈ 0.5  BUT  AP ≈ 0.01 (correctly shows failure)</p>
      </div>

      <p>
        The <strong>No-skill baseline</strong> on a PR curve is a horizontal
        line at <code>y = positive_rate</code>. A useful model must sit
        substantially above this line. Always report Average Precision alongside
        AUC-ROC for imbalanced tasks.
      </p>

      {/* ══════════════════════════════════════════════
          Section 7 · Regression Metrics: RMSE and MAE
      ══════════════════════════════════════════════ */}
      <h2>7 · Regression Metrics — RMSE vs MAE</h2>

      <p>
        Classification metrics do not apply to regression (continuous output).
        The two workhorses for regression evaluation differ in one critical
        property: sensitivity to outliers.
      </p>

      <h3>7.1 · Mean Absolute Error (MAE)</h3>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 leading-relaxed">
        <p>MAE = (1/n) · Σ |y_i - ŷ_i|</p>
        <p className="mt-2 text-slate-400">
          // Each error contributes linearly. An error of 10 is exactly 10×
        </p>
        <p className="text-slate-400">// worse than an error of 1.</p>
        <p className="mt-2 text-slate-400">// Units: same as target variable (e.g. dollars, metres)</p>
        <p className="mt-2 text-slate-400">// Robust to outliers — a single huge error doesn't dominate</p>
      </div>

      <h3>7.2 · Root Mean Squared Error (RMSE)</h3>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 leading-relaxed">
        <p>MSE  = (1/n) · Σ (y_i - ŷ_i)²</p>
        <p>RMSE = sqrt(MSE) = sqrt( (1/n) · Σ (y_i - ŷ_i)² )</p>
        <p className="mt-2 text-slate-400">
          // The squared term means an error of 10 contributes 100× more than
        </p>
        <p className="text-slate-400">
          // an error of 1. Large errors are heavily penalised.
        </p>
        <p className="mt-2 text-slate-400">// Units: same as target variable (sqrt restores units)</p>
        <p className="mt-2 text-slate-400">// Sensitive to outliers — prefer if large errors are very bad</p>
      </div>

      <h3>7.3 · Choosing Between RMSE and MAE</h3>

      <div className="not-prose overflow-x-auto my-4">
        <table className="min-w-full border border-slate-200 text-sm">
          <thead className="bg-slate-100">
            <tr>
              {["Situation", "Recommended Metric"].map((h) => (
                <th key={h} className="border border-slate-200 px-4 py-2 text-left">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[
              [
                "Outliers are expected and large errors should be punished severely",
                "RMSE",
              ],
              [
                "Outliers exist but shouldn't dominate the metric (robust evaluation)",
                "MAE",
              ],
              [
                "You need a metric that corresponds to optimising least-squares regression",
                "RMSE",
              ],
              [
                "You need a metric that corresponds to optimising a median-based model",
                "MAE",
              ],
              ["House price prediction (large errors matter a lot)", "RMSE"],
              ["Demand forecasting with noisy historical data", "MAE"],
            ].map(([sit, rec], i) => (
              <tr key={i}>
                <td className="border border-slate-200 px-4 py-2">{sit}</td>
                <td className="border border-slate-200 px-4 py-2 font-semibold font-mono">
                  {rec}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p>
        A complementary metric is <strong>R² (coefficient of
        determination)</strong>: the fraction of variance in the target that the
        model explains. R² = 1 is a perfect fit; R² = 0 means the model does no
        better than predicting the mean; R² can be negative (model is worse than
        the mean baseline).
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 leading-relaxed">
        <p>R² = 1 - SS_res / SS_tot</p>
        <p className="mt-1 text-slate-400">// where SS_res = Σ(y_i - ŷ_i)²  (residual sum of squares)</p>
        <p className="text-slate-400">// and   SS_tot = Σ(y_i - ȳ)²    (total sum of squares)</p>
      </div>

      {/* ══════════════════════════════════════════════
          Section 8 · Cross-Validation
      ══════════════════════════════════════════════ */}
      <h2>8 · Cross-Validation — Estimating True Generalisation Error</h2>

      <p>
        A single train/test split is noisy: the model's performance depends
        heavily on which samples happened to land in the test set. Cross-
        validation solves this by rotating which portion of the data acts as the
        test set.
      </p>

      <h3>8.1 · K-Fold Cross-Validation</h3>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 leading-relaxed">
        <p className="text-slate-400 mb-2">// Algorithm for K-Fold CV (K=5 shown):</p>
        <p>1. Split dataset into K equal folds: [F1, F2, F3, F4, F5]</p>
        <p>2. For i = 1 to K:</p>
        <p>     a. Hold out fold F_i as the validation set</p>
        <p>     b. Train on the remaining K-1 folds</p>
        <p>     c. Evaluate model → score_i</p>
        <p>3. Final score = mean(score_1, ..., score_K)</p>
        <p>   Uncertainty  = std(score_1,  ..., score_K)</p>
        <p className="mt-3 text-slate-400">
          // Every sample is used for validation exactly once.
        </p>
        <p className="text-slate-400">
          // Every sample is used for training K-1 times.
        </p>
      </div>

      <p>
        <strong>Stratified K-Fold</strong> is mandatory for classification on
        imbalanced datasets. It ensures each fold preserves the original class
        proportions, preventing folds where the minority class is absent or
        over-represented.
      </p>

      <h3>8.2 · Leave-One-Out CV (LOOCV)</h3>
      <p>
        LOOCV is K-Fold where K = n (total number of samples). Each fold holds
        out exactly one sample. This produces the lowest-bias estimate of
        generalisation error, but is computationally expensive (n model fits)
        and has high variance. Practical for small datasets (&lt; 1,000
        samples); avoid for large datasets.
      </p>

      <h3>8.3 · The Bias-Variance Decomposition of CV Error</h3>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 leading-relaxed">
        <p>Error = Bias² + Variance + Irreducible Noise</p>
        <p className="mt-2 text-slate-400">// K=5 or K=10: slight bias (training sets smaller than full data),</p>
        <p className="text-slate-400">//   moderate variance — good balance in practice.</p>
        <p className="mt-2 text-slate-400">// LOOCV:  near-zero bias, but HIGH variance between fold scores.</p>
        <p className="text-slate-400">//   Because training sets overlap almost entirely, fold scores</p>
        <p className="text-slate-400">//   are highly correlated — std is unreliable.</p>
      </div>

      {/* ══════════════════════════════════════════════
          Section 9 · Overfitting Detection
      ══════════════════════════════════════════════ */}
      <h2>9 · Overfitting Detection — Train vs Validation Curves</h2>

      <p>
        The most direct overfitting diagnostic is plotting train and validation
        loss against training epochs (for iterative models) or model complexity.
      </p>

      <div className="not-prose overflow-x-auto my-4">
        <table className="min-w-full border border-slate-200 text-sm">
          <thead className="bg-slate-100">
            <tr>
              {["Pattern", "Train Loss", "Val Loss", "Diagnosis"].map((h) => (
                <th key={h} className="border border-slate-200 px-4 py-2 text-left">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[
              ["Both high and flat", "High", "High", "Underfitting — model too simple"],
              ["Both low and close", "Low", "Low ≈ train", "Good fit"],
              [
                "Train low, val diverges upward",
                "Low",
                "High and rising",
                "Overfitting — model memorises training data",
              ],
              [
                "Both high even with more data",
                "High",
                "High",
                "Underfitting — need more features or complex model",
              ],
            ].map(([pat, tr, val, diag]) => (
              <tr key={pat}>
                <td className="border border-slate-200 px-4 py-2 font-medium">{pat}</td>
                <td className="border border-slate-200 px-4 py-2">{tr}</td>
                <td className="border border-slate-200 px-4 py-2">{val}</td>
                <td className="border border-slate-200 px-4 py-2 text-emerald-800">{diag}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3>9.1 · Learning Curves</h3>
      <p>
        A <strong>learning curve</strong> plots train and validation scores
        against the <em>number of training samples</em> (not epochs). This
        diagnostic distinguishes high bias from high variance:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 leading-relaxed">
        <p className="text-slate-400 mb-1">// High Variance (Overfitting) signature:</p>
        <p>  train score   ≈ 0.95   (high, nearly flat)</p>
        <p>  val   score   ≈ 0.70   (lower, large gap)</p>
        <p>  gap does NOT close as more data is added initially</p>
        <p className="mt-3 text-slate-400">// High Bias (Underfitting) signature:</p>
        <p>  train score   ≈ 0.72   (low plateau)</p>
        <p>  val   score   ≈ 0.70   (similar low plateau)</p>
        <p>  gap is small but both curves are stuck low</p>
        <p className="mt-3 text-slate-400">// Remedies:</p>
        <p>  Overfitting  → regularisation, dropout, more data, simpler model</p>
        <p>  Underfitting → richer features, larger model, reduce regularisation</p>
      </div>

      {/* ══════════════════════════════════════════════
          Section 10 · Full Python Implementation
      ══════════════════════════════════════════════ */}
      <h2>10 · Full Evaluation Pipeline in scikit-learn</h2>

      <p>
        The following code implements a complete evaluation pipeline on the
        breast cancer dataset—a real, moderately imbalanced binary classification
        problem. Every step is annotated to explain both the <em>what</em> and the{" "}
        <em>why</em>.
      </p>

      <TerminalBlock language="python">{`# ── Model Evaluation & Metrics ──────────────────────────────────────────────
# Full evaluation pipeline: confusion matrix, classification report,
# ROC curve, Precision-Recall curve, cross-validation, learning curves.
# Dataset: sklearn's breast cancer (569 samples, 30 features, binary labels).
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    ConfusionMatrixDisplay,
)

# ── 1. Load & split dataset ──────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
# target: 1 = malignant, 0 = benign  (212 malignant, 357 benign → slight imbalance)

# Stratify ensures each split has the same class proportions as the full dataset.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
print(f"Positive rate (test): {y_test.mean():.3f}")

# ── 2. Preprocess: standardise features ──────────────────────────────────────
# Logistic regression is sensitive to feature scale; StandardScaler makes
# each feature have mean=0 and std=1.
# IMPORTANT: fit only on training data to prevent data leakage.
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # learn μ, σ from train, then transform
X_test_sc  = scaler.transform(X_test)        # apply same μ, σ to test set

# ── 3. Train model ────────────────────────────────────────────────────────────
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_sc, y_train)

# ── 4. Predictions ───────────────────────────────────────────────────────────
y_pred      = clf.predict(X_test_sc)           # hard class labels
y_pred_prob = clf.predict_proba(X_test_sc)[:, 1]  # P(malignant) — needed for ROC/PR

# ── 5. Confusion Matrix ───────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
# cm[i, j] = # samples with true label i predicted as label j
# With binary labels: cm = [[TN, FP],
#                            [FN, TP]]
tn, fp, fn, tp = cm.ravel()   # unpack all four cells
print(f"\nConfusion Matrix:\n{cm}")
print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")

# ── 6. Derived metrics from scratch ──────────────────────────────────────────
accuracy  = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)       # of all we flagged positive, what fraction correct?
recall    = tp / (tp + fn)       # of all true positives, what fraction did we catch?
f1        = 2 * precision * recall / (precision + recall)   # harmonic mean
print(f"\nManual calculation:")
print(f"  Accuracy  = {accuracy:.4f}")
print(f"  Precision = {precision:.4f}")
print(f"  Recall    = {recall:.4f}")
print(f"  F1-Score  = {f1:.4f}")

# ── 7. Full classification report ────────────────────────────────────────────
# sklearn computes per-class metrics and macro/weighted averages.
# Macro average = unweighted mean of per-class metrics (treats classes equally).
# Weighted average = weighted by support (number of true instances per class).
print("\nsklearn classification_report:")
print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

# ── 8. ROC Curve & AUC ───────────────────────────────────────────────────────
# roc_curve sweeps all unique score thresholds and returns TPR, FPR at each.
fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_prob)
auc_roc = roc_auc_score(y_test, y_pred_prob)
print(f"AUC-ROC: {auc_roc:.4f}")

# ── 9. Precision-Recall Curve & Average Precision ────────────────────────────
# precision_recall_curve returns precision, recall, and the threshold that
# produced each (P, R) pair.
prec_curve, rec_curve, pr_thresholds = precision_recall_curve(y_test, y_pred_prob)
avg_precision = average_precision_score(y_test, y_pred_prob)
print(f"Average Precision: {avg_precision:.4f}")

# ── 10. Plot ROC and PR curves side by side ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROC Curve
axes[0].plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC-ROC = {auc_roc:.3f}")
axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.5)")  # diagonal baseline
axes[0].set_xlabel("False Positive Rate (FPR)")
axes[0].set_ylabel("True Positive Rate (Recall)")
axes[0].set_title("ROC Curve")
axes[0].legend()

# Precision-Recall Curve
# baseline = positive rate in test set (no-skill classifier)
baseline_pr = y_test.mean()
axes[1].step(rec_curve, prec_curve, color="darkorange", lw=2, where="post",
             label=f"AP = {avg_precision:.3f}")
axes[1].axhline(y=baseline_pr, color="k", linestyle="--", lw=1,
                label=f"No-skill baseline = {baseline_pr:.2f}")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")
axes[1].legend()

plt.tight_layout()
plt.savefig("roc_pr_curves.png", dpi=150)
plt.show()
print("Saved: roc_pr_curves.png")

# ── 11. Stratified K-Fold Cross-Validation ────────────────────────────────────
# StratifiedKFold preserves class proportions in each fold — essential for
# imbalanced datasets where a random fold might have zero minority samples.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# cross_val_score retrains the model K times on the raw (non-scaled) data.
# To avoid leakage inside CV, use a Pipeline that scales within each fold.
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("scaler", StandardScaler()),        # fitted only on the K-1 training folds
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])

# Evaluate with F1 (macro) — treats classes equally regardless of size
cv_f1 = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro")
cv_auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")

print(f"\n5-Fold CV Results:")
print(f"  F1 (macro): {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
print(f"  AUC-ROC:    {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
# The ± std gives us a confidence estimate for generalisation error.

# ── 12. Learning Curves (bias-variance diagnostic) ────────────────────────────
# learning_curve trains the model on increasing subsets of training data and
# records both training score and CV validation score at each size.
# This reveals whether the model is overfitting, underfitting, or just right.
train_sizes, train_scores, val_scores = learning_curve(
    pipe, X, y,
    train_sizes=np.linspace(0.10, 1.0, 10),   # 10%, 20%, ..., 100% of train data
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1,           # use all CPU cores
)

# Average scores across the K folds at each training size
train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, "o-", color="steelblue", label="Train F1")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                 alpha=0.15, color="steelblue")   # shaded 1-std band
plt.plot(train_sizes, val_mean, "o-", color="darkorange", label="Val F1 (CV)")
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                 alpha=0.15, color="darkorange")
plt.xlabel("Training Set Size")
plt.ylabel("F1 Score (macro)")
plt.title("Learning Curve — Logistic Regression on Breast Cancer Dataset")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=150)
plt.show()
print("Saved: learning_curve.png")

# ── 13. Regression metrics demo (simulated) ───────────────────────────────────
# Not all problems are classification. Here we demonstrate MAE, RMSE, and R²
# on a synthetic regression scenario.
rng = np.random.default_rng(42)
y_true_reg = rng.uniform(0, 100, size=200)
y_pred_reg = y_true_reg + rng.normal(0, 10, size=200)   # model with ±10 noise

# Add a few large outliers to illustrate MAE vs RMSE sensitivity
y_pred_reg[0:5] += 50   # five predictions are wildly off

mae  = mean_absolute_error(y_true_reg, y_pred_reg)
rmse = mean_squared_error(y_true_reg, y_pred_reg, squared=False)  # squared=False → RMSE
r2   = r2_score(y_true_reg, y_pred_reg)

print(f"\nRegression Metrics (with outliers injected):")
print(f"  MAE  = {mae:.2f}   ← robust, not dominated by the 5 outlier errors")
print(f"  RMSE = {rmse:.2f}  ← larger, inflated by squared outlier errors")
print(f"  R²   = {r2:.4f}  ← fraction of variance explained")
# RMSE >> MAE confirms outliers are present.
# If RMSE ≈ MAE, errors are approximately uniform (no extreme outliers).
`}</TerminalBlock>

      {/* ══════════════════════════════════════════════
          Section 11 · Common Pitfalls
      ══════════════════════════════════════════════ */}
      <h2>11 · Common Pitfalls</h2>

      <div className="not-prose space-y-4 my-4">
        {[
          {
            title: "Data Leakage in Cross-Validation",
            body:
              "Never fit a StandardScaler (or any preprocessing step) on the full training set before CV. The scaler must be inside the Pipeline so it is re-fit on each CV training fold. Fitting on all data before splitting leaks test-set information into training, artificially inflating CV scores.",
          },
          {
            title: "Using Accuracy on Imbalanced Data",
            body:
              "A model predicting the majority class always scores (1 - minority_rate) accuracy. Always report Precision, Recall, F1, and AUC alongside accuracy. For severe imbalance (< 5 % minority), prioritise PR curves and Average Precision.",
          },
          {
            title: "Threshold Confusion",
            body:
              "AUC-ROC is threshold-independent, but a deployed system needs a fixed threshold. Choose the threshold on the validation set, not the test set. Use the Youden Index (TPR - FPR maximised) or the business-defined cost ratio as your criterion.",
          },
          {
            title: "Optimising the Wrong Metric",
            body:
              "The metric you use for model selection (hyperparameter tuning) must match the metric you care about in production. Tuning for accuracy on an imbalanced fraud task will not produce the best F1 or Recall model.",
          },
          {
            title: "Reporting Only Mean CV Score",
            body:
              "Always report mean ± std across folds. A model with F1 = 0.92 ± 0.01 is far more trustworthy than one with F1 = 0.92 ± 0.08. High variance indicates the model is sensitive to which samples are in the training set.",
          },
          {
            title: "RMSE vs MSE Units",
            body:
              "MSE is in squared units (dollars², metres²), which is hard to interpret. Always take the square root to get RMSE in the same units as the target. Never compare MSE of two models with different target scales—they are not commensurable.",
          },
        ].map(({ title, body }) => (
          <div
            key={title}
            className="rounded-lg border border-amber-200 bg-amber-50 p-4"
          >
            <p className="font-semibold text-amber-900 mb-1">⚠ {title}</p>
            <p className="text-sm text-amber-800">{body}</p>
          </div>
        ))}
      </div>

      {/* ══════════════════════════════════════════════
          Section 12 · Further Reading
      ══════════════════════════════════════════════ */}
      <h2>12 · Further Reading</h2>

      <div className="not-prose overflow-x-auto my-4">
        <table className="min-w-full border border-slate-200 text-sm">
          <thead className="bg-slate-100">
            <tr>
              {["Reference", "Why Read It"].map((h) => (
                <th key={h} className="border border-slate-200 px-4 py-2 text-left">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[
              [
                "Fawcett (2006) — An Introduction to ROC Analysis. Pattern Recognition Letters 27(8)",
                "Definitive reference for ROC curves, AUC, and the probabilistic interpretation. Covers multi-class ROC.",
              ],
              [
                "Davis & Goadrich (2006) — The Relationship Between PR Curves and ROC Curves. ICML",
                "Proves why PR curves are preferred for imbalanced datasets and how to convert between them.",
              ],
              [
                "Hastie, Tibshirani, Friedman — Elements of Statistical Learning, Ch. 7 (Model Assessment & Selection)",
                "Thorough treatment of bias-variance trade-off, CV, AIC/BIC, and the bootstrap.",
              ],
              [
                "Arlot & Celisse (2010) — A Survey of Cross-Validation Procedures for Model Selection. Statistics Surveys",
                "Comprehensive comparison of CV variants; clarifies when LOOCV's high variance makes K-Fold preferable.",
              ],
              [
                "Saito & Rehmsmeier (2015) — The Precision-Recall Plot Is More Informative than the ROC Plot. PLOS ONE",
                "Empirical study showing AUC-ROC overestimates performance when negatives dominate.",
              ],
              [
                "scikit-learn docs — model_selection and metrics modules",
                "Canonical API reference and worked examples for all functions used in this lecture.",
              ],
            ].map(([ref, why]) => (
              <tr key={ref}>
                <td className="border border-slate-200 px-4 py-2 font-medium align-top">
                  {ref}
                </td>
                <td className="border border-slate-200 px-4 py-2 align-top">
                  {why}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </article>
  );
}
