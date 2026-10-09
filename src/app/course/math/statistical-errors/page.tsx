"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

export default function Page() {
  return (
    <article className="prose prose-slate max-w-none pb-24">

      {/* ── Track badge ── */}
      <div className="mb-4">
        <span className="inline-block rounded-full bg-sky-100 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-sky-700">
          Track 0 · Mathematical Foundations
        </span>
      </div>

      <h1>Statistical Errors &amp; Power</h1>

      <p>
        Every decision a model or analyst makes under uncertainty is a binary
        judgment: <em>is there an effect, or not?</em> The framework of
        hypothesis testing formalises that judgment—and makes explicit the two
        ways you can be wrong. Understanding Type I and Type II errors, the
        power of a test, effect size, and their machine-learning analogues
        (precision, recall, F1, AUC) is the bedrock of trustworthy model
        evaluation.
      </p>

      {/* ── Learning Objectives ── */}
      <h2>Learning Objectives</h2>
      <ul>
        <li>
          Enumerate all four outcomes of a hypothesis test and map them to the
          confusion matrix.
        </li>
        <li>
          Define Type I error (α) and Type II error (β) and explain how
          tightening one inflates the other.
        </li>
        <li>
          Calculate statistical power (1 − β) and explain why it depends on
          effect size, sample size, and α simultaneously.
        </li>
        <li>
          Compute Cohen's d and use it in a sample-size formula, showing a
          full worked numerical example.
        </li>
        <li>
          Derive the F1-score from first principles and justify the harmonic
          mean over the arithmetic mean.
        </li>
        <li>
          Interpret the ROC curve and AUC, and choose the right metric given
          class imbalance or asymmetric misclassification costs.
        </li>
        <li>
          Build a complete classification-evaluation pipeline using{" "}
          <code>scikit-learn</code>.
        </li>
      </ul>

      {/* ═══════════════════════════════════════════════════════════════════
          SECTION 1 – THE CONFUSION MATRIX
      ════════════════════════════════════════════════════════════════════ */}
      <h2>1 · The Confusion Matrix in a Hypothesis-Testing Context</h2>

      <p>
        A hypothesis test frames a question as two competing claims: the{" "}
        <strong>null hypothesis H₀</strong> (no effect, no difference, status
        quo) and the <strong>alternative hypothesis H₁</strong> (there is a
        real effect). After collecting data we either <em>reject</em> H₀ or{" "}
        <em>fail to reject</em> it. Reality is independent of our decision,
        giving four possible outcomes:
      </p>

      {/* Confusion-matrix table */}
      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full border border-slate-300 text-sm">
          <thead className="bg-slate-100 text-slate-700">
            <tr>
              <th className="border border-slate-300 px-4 py-2"></th>
              <th className="border border-slate-300 px-4 py-2">H₀ is TRUE (no real effect)</th>
              <th className="border border-slate-300 px-4 py-2">H₁ is TRUE (real effect exists)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-slate-300 px-4 py-2 font-semibold bg-slate-50">Fail to reject H₀</td>
              <td className="border border-slate-300 px-4 py-2 bg-green-50 text-green-800">
                <strong>True Negative (TN)</strong><br />
                Correct decision — we correctly conclude there is no effect.
              </td>
              <td className="border border-slate-300 px-4 py-2 bg-red-50 text-red-800">
                <strong>False Negative (FN)</strong><br />
                Type II Error — we miss a real effect. Probability = β.
              </td>
            </tr>
            <tr>
              <td className="border border-slate-300 px-4 py-2 font-semibold bg-slate-50">Reject H₀</td>
              <td className="border border-slate-300 px-4 py-2 bg-red-50 text-red-800">
                <strong>False Positive (FP)</strong><br />
                Type I Error — we claim an effect that does not exist. Probability = α.
              </td>
              <td className="border border-slate-300 px-4 py-2 bg-green-50 text-green-800">
                <strong>True Positive (TP)</strong><br />
                Correct detection. Probability = 1 − β = Power.
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <p>
        In machine-learning classification the same four cells appear when we
        compare a model's predicted label to the ground-truth label. A spam
        filter that flags a legitimate email commits a False Positive; one that
        lets spam through commits a False Negative. The <em>costs</em> of each
        error are domain-specific and must drive every metric choice you make.
      </p>

      {/* ═══════════════════════════════════════════════════════════════════
          SECTION 2 – TYPE I ERROR
      ════════════════════════════════════════════════════════════════════ */}
      <h2>2 · Type I Error (α) — The False Positive Rate</h2>

      <p>
        The <strong>significance level α</strong> is the probability that we
        reject a true null hypothesis. It is a <em>threshold we choose
        before seeing the data</em>, not a property of the data itself.
        Conventionally α = 0.05 is used, meaning we accept a 5 % chance of a
        spurious rejection across the long run of repeated experiments.
      </p>

      <p>
        Mechanically, we compute a <strong>p-value</strong>: the probability of
        obtaining a test statistic at least as extreme as the one observed,
        <em>assuming H₀ is true</em>. If p ≤ α we reject H₀.
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 leading-relaxed">
        <p className="text-slate-400 mb-2">// Type I error in terms of the test statistic Z (one-tailed)</p>
        <p>P(reject H₀ | H₀ true)  =  P( Z &gt; z_α | H₀ )  =  α</p>
        <br />
        <p className="text-slate-400">// For α = 0.05 (one-tailed), the critical value is:</p>
        <p>z_0.05  =  1.645</p>
        <br />
        <p className="text-slate-400">// For a two-tailed test (H₁: μ ≠ μ₀) we split α across both tails:</p>
        <p>z_0.025  =  1.960     (i.e., each tail carries 2.5 %)</p>
      </div>

      <p>
        Lowering α reduces false positives but—as we will see—necessarily
        increases false negatives (Type II error). There is no free lunch.
      </p>

      {/* ═══════════════════════════════════════════════════════════════════
          SECTION 3 – TYPE II ERROR & POWER
      ════════════════════════════════════════════════════════════════════ */}
      <h2>3 · Type II Error (β) and Statistical Power (1 − β)</h2>

      <p>
        The <strong>Type II error rate β</strong> is the probability of failing
        to reject H₀ when H₁ is actually true—a missed detection.{" "}
        <strong>Statistical power</strong> is the complement, 1 − β: the
        probability that the test correctly identifies a real effect.
      </p>

      <p>
        Power depends on <em>four factors</em> that are coupled in a single
        equation. You can control any three if you fix the fourth:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 leading-relaxed">
        <p className="text-slate-400 mb-2">// Power for a one-sample z-test (two-tailed)</p>
        <p>Power  =  1 − β</p>
        <p>       =  Φ( |μ₁ − μ₀| / (σ / sqrt(N)) − z_(α/2) )</p>
        <br />
        <p className="text-slate-400">// where:</p>
        <p>  Φ       = standard normal CDF</p>
        <p>  μ₁ − μ₀ = true difference between means (the "signal")</p>
        <p>  σ       = population standard deviation</p>
        <p>  N       = sample size</p>
        <p>  z_(α/2) = critical value for chosen α (e.g., 1.96 for α=0.05)</p>
      </div>

      <p>
        Intuitively: power grows when the signal is larger, variance is
        smaller, or we collect more data. Tightening α (demanding stronger
        evidence) moves the critical value outward, cutting power. This
        four-way coupling is why you <em>must</em> specify α, desired power,
        and expected effect size <strong>before</strong> collecting data—
        otherwise your study may be fatally underpowered.
      </p>

      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full border border-slate-300 text-sm">
          <thead className="bg-slate-100 text-slate-700">
            <tr>
              <th className="border border-slate-300 px-4 py-2">Change</th>
              <th className="border border-slate-300 px-4 py-2">Effect on Power</th>
              <th className="border border-slate-300 px-4 py-2">Effect on Type I Error</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-slate-300 px-4 py-2">Increase N</td>
              <td className="border border-slate-300 px-4 py-2">↑ Increases</td>
              <td className="border border-slate-300 px-4 py-2">Unchanged (α is preset)</td>
            </tr>
            <tr>
              <td className="border border-slate-300 px-4 py-2">Increase α (e.g., 0.05 → 0.10)</td>
              <td className="border border-slate-300 px-4 py-2">↑ Increases</td>
              <td className="border border-slate-300 px-4 py-2">↑ Increases</td>
            </tr>
            <tr>
              <td className="border border-slate-300 px-4 py-2">Decrease α (e.g., 0.05 → 0.01)</td>
              <td className="border border-slate-300 px-4 py-2">↓ Decreases</td>
              <td className="border border-slate-300 px-4 py-2">↓ Decreases</td>
            </tr>
            <tr>
              <td className="border border-slate-300 px-4 py-2">Larger effect size</td>
              <td className="border border-slate-300 px-4 py-2">↑ Increases</td>
              <td className="border border-slate-300 px-4 py-2">Unchanged</td>
            </tr>
            <tr>
              <td className="border border-slate-300 px-4 py-2">Reduce σ (better measurement)</td>
              <td className="border border-slate-300 px-4 py-2">↑ Increases</td>
              <td className="border border-slate-300 px-4 py-2">Unchanged</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* ═══════════════════════════════════════════════════════════════════
          SECTION 4 – EFFECT SIZE & SAMPLE SIZE
      ════════════════════════════════════════════════════════════════════ */}
      <h2>4 · Effect Size (Cohen's d) and Sample-Size Calculation</h2>

      <h3>4.1 Cohen's d</h3>
      <p>
        A p-value tells you <em>whether</em> an effect exists; it says nothing
        about <em>how big</em> it is. With a large enough N, even a trivially
        tiny difference will be statistically significant. Cohen's d
        standardises the magnitude of a difference relative to the variability
        in the data:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 leading-relaxed">
        <p className="text-slate-400 mb-2">// Cohen's d for two independent groups</p>
        <p>d  =  (μ₁ − μ₂) / s_pooled</p>
        <br />
        <p className="text-slate-400">// Pooled standard deviation</p>
        <p>s_pooled  =  sqrt( ((n₁−1)·s₁² + (n₂−1)·s₂²) / (n₁+n₂−2) )</p>
        <br />
        <p className="text-slate-400">// Conventional benchmarks (Cohen 1988):</p>
        <p>  d = 0.2  →  small effect</p>
        <p>  d = 0.5  →  medium effect</p>
        <p>  d = 0.8  →  large effect</p>
      </div>

      <h3>4.2 Sample-Size Formula (Worked Example)</h3>
      <p>
        For a two-sample z-test with equal group sizes, desired power{" "}
        <code>1−β</code>, and significance level α (two-tailed):
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 leading-relaxed">
        <p className="text-slate-400 mb-2">// Required N per group</p>
        <p>N  =  2 · ( (z_(α/2) + z_β) / d )²</p>
        <br />
        <p className="text-slate-400">// where z_β is the z-score for the desired power:</p>
        <p>  power = 0.80  →  z_β = 0.842</p>
        <p>  power = 0.90  →  z_β = 1.282</p>
        <p>  power = 0.95  →  z_β = 1.645</p>
        <br />
        <p className="text-slate-400 font-bold">// WORKED EXAMPLE</p>
        <p className="text-slate-400">// Scenario: A/B test for click-through rates.</p>
        <p className="text-slate-400">// Expected effect size d = 0.4 (medium-small),</p>
        <p className="text-slate-400">// α = 0.05 (two-tailed), desired power = 0.80.</p>
        <br />
        <p>z_(α/2)   =  z_0.025  =  1.960</p>
        <p>z_β       =  z_0.20   =  0.842</p>
        <p>d         =  0.4</p>
        <br />
        <p>N  =  2 · ( (1.960 + 0.842) / 0.4 )²</p>
        <p>   =  2 · ( 2.802 / 0.4 )²</p>
        <p>   =  2 · ( 7.005 )²</p>
        <p>   =  2 · 49.07</p>
        <p>   ≈  99  per group  (round up to 100)</p>
        <br />
        <p className="text-slate-400">// Interpretation: We need ~100 users per variant to have</p>
        <p className="text-slate-400">// an 80 % chance of detecting a d=0.4 effect at α=0.05.</p>
        <p className="text-slate-400">// Halving d to 0.2 would require ~394 per group — 4x more data.</p>
      </div>

      <p>
        This quadratic relationship between d and N is why small effects are
        expensive to detect reliably. Researchers who ignore power analysis
        routinely publish underpowered studies that fail to replicate.
      </p>

      {/* ═══════════════════════════════════════════════════════════════════
          SECTION 5 – PRECISION, RECALL & F1
      ════════════════════════════════════════════════════════════════════ */}
      <h2>5 · Precision, Recall, and the F1-Score</h2>

      <h3>5.1 The Precision–Recall Tradeoff</h3>
      <p>
        In a binary classifier, <strong>precision</strong> is the fraction of
        positive predictions that are actually positive (how trustworthy are
        your alarms?), while <strong>recall</strong> (also called sensitivity
        or True Positive Rate) is the fraction of actual positives that the
        model correctly detects (how complete is your coverage?).
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 leading-relaxed">
        <p className="text-slate-400 mb-2">// Confusion-matrix counts</p>
        <p>TP = true positives   FP = false positives</p>
        <p>TN = true negatives   FN = false negatives</p>
        <br />
        <p className="text-slate-400">// Core metrics</p>
        <p>Precision  (P)  =  TP / (TP + FP)</p>
        <p>Recall     (R)  =  TP / (TP + FN)   ← also: Sensitivity, TPR</p>
        <p>Specificity      =  TN / (TN + FP)   ← True Negative Rate</p>
        <p>Accuracy         =  (TP + TN) / (TP + TN + FP + FN)</p>
        <br />
        <p className="text-slate-400">// Direct analogy to hypothesis testing:</p>
        <p>  1 − Precision  →  False Discovery Rate  (≈ Type I error rate for positives)</p>
        <p>  1 − Recall     →  Miss Rate             (≈ Type II error rate for positives)</p>
      </div>

      <p>
        Adjusting the decision threshold (the probability cutoff above which a
        prediction is labelled "positive") traces a precision-recall curve.
        Lowering the threshold increases recall (more positives captured) but
        decreases precision (more false alarms)—the same tradeoff as relaxing
        α in hypothesis testing.
      </p>

      <h3>5.2 F1-Score — Derivation from First Principles</h3>
      <p>
        We need a single scalar that balances precision and recall. The{" "}
        <strong>arithmetic mean</strong> of P and R would be:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 leading-relaxed">
        <p className="text-slate-400 mb-2">// Why NOT the arithmetic mean?</p>
        <p>Arithmetic mean  =  (P + R) / 2</p>
        <br />
        <p className="text-slate-400">// Counterexample: a model predicts EVERY sample as positive.</p>
        <p>  Recall     = 1.0  (all positives captured)</p>
        <p>  Precision  = base_rate  (e.g., 0.01 for 1 % positive class)</p>
        <br />
        <p>  Arithmetic mean  =  (1.0 + 0.01) / 2  =  0.505  ← misleadingly high!</p>
        <br />
        <p className="text-slate-400">// The harmonic mean penalises extreme imbalance between P and R:</p>
        <p>  Harmonic mean  =  2 / (1/P + 1/R)  =  2PR / (P + R)</p>
        <p>               =  2·(1.0)·(0.01) / (1.0 + 0.01)</p>
        <p>               =  0.02 / 1.01  ≈  0.0198  ← correctly low!</p>
      </div>

      <p>
        The harmonic mean of two positive numbers is always ≤ their geometric
        mean ≤ their arithmetic mean (AM-GM-HM inequality). It is dominated by
        the <em>smaller</em> of the two values, so a model must do well on{" "}
        <strong>both</strong> precision and recall to achieve a high F1.
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 leading-relaxed">
        <p className="text-slate-400 mb-2">// F1-Score (β=1 gives equal weight to P and R)</p>
        <p>F1  =  2 · (Precision · Recall) / (Precision + Recall)</p>
        <p>    =  2TP / (2TP + FP + FN)</p>
        <br />
        <p className="text-slate-400">// General Fβ score — β² controls the relative weight of Recall vs Precision</p>
        <p>Fβ  =  (1 + β²) · (P · R) / (β²·P + R)</p>
        <br />
        <p className="text-slate-400">// β &gt; 1: penalise missing positives (recall matters more) — e.g., cancer screening</p>
        <p className="text-slate-400">// β &lt; 1: penalise false alarms (precision matters more) — e.g., spam detection</p>
      </div>

      {/* ═══════════════════════════════════════════════════════════════════
          SECTION 6 – ROC CURVE & AUC
      ════════════════════════════════════════════════════════════════════ */}
      <h2>6 · ROC Curve and AUC</h2>

      <p>
        The <strong>Receiver Operating Characteristic (ROC) curve</strong> plots
        the True Positive Rate (Recall) on the y-axis against the False Positive
        Rate (1 − Specificity) on the x-axis as the decision threshold is swept
        from 1 down to 0. Each threshold setting produces one (FPR, TPR) point;
        the curve connects all of them.
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 leading-relaxed">
        <p className="text-slate-400 mb-2">// Axis definitions</p>
        <p>TPR  =  TP / (TP + FN)    ← Sensitivity / Recall</p>
        <p>FPR  =  FP / (FP + TN)    ← 1 − Specificity</p>
        <br />
        <p className="text-slate-400">// Key reference points on the ROC plane:</p>
        <p>  (0, 0)  →  threshold = 1.0: classify everything as negative</p>
        <p>  (1, 1)  →  threshold = 0.0: classify everything as positive</p>
        <p>  (0, 1)  →  perfect classifier</p>
        <p>  diagonal (FPR = TPR) →  random guessing (AUC = 0.5)</p>
        <br />
        <p className="text-slate-400">// AUC = Area Under the ROC Curve</p>
        <p>AUC  =  integral from 0 to 1 of TPR(FPR) d(FPR)</p>
        <p>     =  P(score(positive) &gt; score(negative))</p>
        <br />
        <p className="text-slate-400">// The probabilistic interpretation: AUC equals the probability</p>
        <p className="text-slate-400">// that a randomly chosen positive example scores higher than</p>
        <p className="text-slate-400">// a randomly chosen negative example (Wilcoxon-Mann-Whitney statistic).</p>
      </div>

      <p>
        AUC is <strong>threshold-independent</strong> and{" "}
        <strong>class-imbalance robust</strong> compared to accuracy, making it
        ideal when you need to compare classifiers without committing to a
        specific threshold. However, when the positive class is very rare and
        false positives are costly, the precision-recall AUC can be more
        informative because it focuses exclusively on the positive class.
      </p>

      {/* ═══════════════════════════════════════════════════════════════════
          SECTION 7 – PYTHON IMPLEMENTATION
      ════════════════════════════════════════════════════════════════════ */}
      <h2>7 · Python Implementation — Full Evaluation Pipeline</h2>

      <p>
        The following script constructs a realistic imbalanced classification
        dataset, trains a logistic-regression model, and exhaustively evaluates
        it using every metric discussed in this lecture. Every line is annotated
        to explain both <em>what</em> it does and <em>why</em>.
      </p>

      <TerminalBlock language="python">{`# ──────────────────────────────────────────────────────────────────────────────
# statistical_errors_pipeline.py
#
# Full evaluation pipeline: confusion matrix, Type I/II errors, precision,
# recall, F1, ROC curve, AUC, and Cohen's d for effect-size analysis.
# ──────────────────────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

# ── 1. Synthetic dataset with class imbalance ─────────────────────────────────
# weights=[0.85, 0.15] gives a realistic 85/15 split — accuracy is misleading here.
X, y = make_classification(
    n_samples=5000,
    n_features=20,        # 20 input features
    n_informative=10,     # 10 of them actually carry signal
    n_redundant=5,        # 5 are linear combinations of informative ones
    weights=[0.85, 0.15], # 85 % negative, 15 % positive
    flip_y=0.02,          # 2 % label noise to make the problem realistic
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,       # 25 % held-out test set
    stratify=y,           # preserve class ratio in both splits
    random_state=42,
)

print(f"Train positives: {y_train.sum()} / {len(y_train)}")
print(f"Test  positives: {y_test.sum()}  / {len(y_test)}")

# ── 2. Train a logistic regression model ─────────────────────────────────────
# class_weight='balanced' re-weights the loss to compensate for imbalance;
# without it the model is incentivised to always predict the majority class.
clf = LogisticRegression(
    class_weight="balanced",  # up-weight minority class proportionally
    max_iter=1000,
    random_state=42,
)
clf.fit(X_train, y_train)

# Predicted class labels (threshold = 0.5 by default)
y_pred = clf.predict(X_test)

# Raw predicted probabilities for the POSITIVE class (needed for ROC/PR curves)
y_prob = clf.predict_proba(X_test)[:, 1]

# ── 3. Confusion matrix — map to the four hypothesis-test outcomes ─────────────
cm = confusion_matrix(y_test, y_pred)
# sklearn layout:  [[TN, FP],
#                   [FN, TP]]
TN, FP, FN, TP = cm.ravel()

print("\n── Confusion Matrix ──────────────────────────────")
print(f"  True  Negatives  (TN) : {TN}")
print(f"  False Positives  (FP) : {FP}  ← Type I  errors")
print(f"  False Negatives  (FN) : {FN}  ← Type II errors")
print(f"  True  Positives  (TP) : {TP}")

# ── 4. Derived metrics from first principles ──────────────────────────────────
precision  = TP / (TP + FP)          # of all positive predictions, how many correct?
recall     = TP / (TP + FN)          # of all actual positives, how many found?
specificity = TN / (TN + FP)         # of all actual negatives, how many correct?
accuracy   = (TP + TN) / cm.sum()    # fraction of all correct decisions

# F1 = harmonic mean of precision and recall (derived earlier)
f1 = 2 * (precision * recall) / (precision + recall)

# False Positive Rate (x-axis of ROC)
fpr_threshold = FP / (FP + TN)       # = 1 - specificity

print("\n── Manual Metrics ────────────────────────────────")
print(f"  Accuracy    : {accuracy:.4f}  ← deceptive with imbalance!")
print(f"  Precision   : {precision:.4f}")
print(f"  Recall      : {recall:.4f}")
print(f"  Specificity : {specificity:.4f}")
print(f"  F1-Score    : {f1:.4f}")
print(f"  FPR         : {fpr_threshold:.4f}")

# ── 5. sklearn's classification_report for per-class breakdown ────────────────
# This is the go-to for multi-class problems; supports macro/micro/weighted avg.
print("\n── Classification Report ─────────────────────────")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# ── 6. ROC curve and AUC ──────────────────────────────────────────────────────
# roc_curve returns (FPR, TPR, thresholds) for every unique score value.
fpr_curve, tpr_curve, thresholds_roc = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
print(f"\n── AUC-ROC : {auc_score:.4f}")
# AUC = P(positive scores higher than negative) — 0.5 = random, 1.0 = perfect.

# ── 7. Precision-Recall curve (better for severe class imbalance) ─────────────
pr_precision, pr_recall, thresholds_pr = precision_recall_curve(y_test, y_prob)
avg_precision = average_precision_score(y_test, y_prob)
print(f"── Avg Precision (PR-AUC): {avg_precision:.4f}")
# avg_precision = area under the PR curve — equivalent to the macro-averaged
# precision across recall levels; more informative than ROC-AUC when positives
# are rare because it does NOT factor in the large pool of true negatives.

# ── 8. Effect size: Cohen's d between class score distributions ───────────────
# This quantifies how separable the two score distributions are.
pos_scores = y_prob[y_test == 1]  # predicted probabilities for true positives
neg_scores = y_prob[y_test == 0]  # predicted probabilities for true negatives

mean_diff = pos_scores.mean() - neg_scores.mean()
# Pooled standard deviation (equal-variance formula for simplicity)
n1, n2 = len(pos_scores), len(neg_scores)
s_pooled = np.sqrt(
    ((n1 - 1) * pos_scores.std(ddof=1)**2 + (n2 - 1) * neg_scores.std(ddof=1)**2)
    / (n1 + n2 - 2)
)
cohens_d = mean_diff / s_pooled
print(f"\n── Cohen's d (score separation): {cohens_d:.3f}")
# Positive d means model assigns higher scores to actual positives — good.
# d > 0.8 is a large effect; the model has strong discriminative ability.

# ── 9. Visualise — ROC and PR curves side by side ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROC curve
axes[0].plot(fpr_curve, tpr_curve, color="steelblue", lw=2,
             label=f"Logistic Reg (AUC = {auc_score:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Random guess (AUC = 0.5)")
axes[0].set_xlabel("False Positive Rate  (1 − Specificity)")
axes[0].set_ylabel("True Positive Rate  (Recall / Sensitivity)")
axes[0].set_title("ROC Curve")
axes[0].legend(loc="lower right")
axes[0].grid(True, alpha=0.3)

# PR curve
axes[1].plot(pr_recall, pr_precision, color="darkorange", lw=2,
             label=f"Logistic Reg (AP = {avg_precision:.3f})")
# Baseline: a random classifier achieves precision = base rate
base_rate = y_test.mean()
axes[1].axhline(base_rate, color="k", linestyle="--", lw=1,
                label=f"Random classifier (P = {base_rate:.2f})")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")
axes[1].legend(loc="upper right")
axes[1].grid(True, alpha=0.3)

plt.suptitle("Statistical Errors & Power — Evaluation Pipeline", fontsize=13)
plt.tight_layout()
plt.savefig("evaluation_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("\\nSaved: evaluation_curves.png")

# ── 10. Sample-size calculator (function) ─────────────────────────────────────
from scipy import stats

def required_sample_size(effect_size: float, alpha: float = 0.05,
                         power: float = 0.80) -> int:
    """
    Compute N per group for a two-sample z-test.

    Parameters
    ----------
    effect_size : Cohen's d
    alpha       : significance level (two-tailed)
    power       : desired power = 1 − beta

    Returns
    -------
    N : int — sample size per group (rounded up)
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)   # critical value for alpha
    z_beta  = stats.norm.ppf(power)            # critical value for desired power
    N = 2 * ((z_alpha + z_beta) / effect_size) ** 2
    return int(np.ceil(N))                     # always round UP for safety

# Reproduce the worked example from the lecture
for d, label in [(0.2, "Small"), (0.4, "Med-Small"), (0.5, "Medium"), (0.8, "Large")]:
    n = required_sample_size(d, alpha=0.05, power=0.80)
    print(f"  d={d} ({label:10s}) → N={n:5d} per group  (total={2*n})")
`}</TerminalBlock>

      <p>
        Running the script prints a line like{" "}
        <code>AUC-ROC : 0.9234</code>, saves <code>evaluation_curves.png</code>
        , and outputs a sample-size table. The table illustrates the
        quadratic cost of detecting small effects: moving from d = 0.8 to d =
        0.2 requires 16× more data per group.
      </p>

      {/* ═══════════════════════════════════════════════════════════════════
          SECTION 8 – COMMON PITFALLS
      ════════════════════════════════════════════════════════════════════ */}
      <h2>8 · Common Pitfalls</h2>

      <h3>Pitfall 1 — Accuracy Is Deceptive with Imbalanced Classes</h3>
      <p>
        A dataset with 95 % negative examples lets a model that always predicts
        "negative" achieve 95 % accuracy while having 0 % recall. Accuracy is
        a valid metric <em>only</em> when classes are balanced. Use F1 or
        PR-AUC instead when the minority class is what you care about.
      </p>

      <h3>Pitfall 2 — Reporting p-values Without Effect Sizes</h3>
      <p>
        With N = 1 000 000, a difference of 0.001 standard deviations will be
        statistically significant at any reasonable α. Always report Cohen's d
        (or η², ω², etc.) alongside p-values. A "significant" finding may be
        practically irrelevant.
      </p>

      <h3>Pitfall 3 — Choosing F1 When You Should Use AUC</h3>
      <p>
        F1 evaluates performance at a <em>single</em> threshold (default 0.5).
        If your downstream use case will deploy at multiple thresholds, or if
        you want to compare model architectures in a threshold-agnostic way,
        use AUC-ROC or PR-AUC. Conversely, if you have a fixed operating point
        (a specific precision or recall target), F1 at that threshold is more
        interpretable.
      </p>

      <h3>Pitfall 4 — ROC-AUC Is Optimistic Under Severe Imbalance</h3>
      <p>
        Because ROC-AUC uses the large pool of true negatives (via specificity),
        it can be inflated when negatives vastly outnumber positives. A model
        can achieve AUC = 0.97 on a 1 % positive-class dataset while missing
        most actual positives. PR-AUC does not factor in true negatives and is
        therefore more diagnostic in this regime.
      </p>

      <h3>Pitfall 5 — Underpowered Studies and the Winner's Curse</h3>
      <p>
        Studies with low power that do find significant effects tend to
        <em>overestimate</em> the effect size (the "winner's curse"): only the
        most extreme random fluctuations clear the significance bar, so the
        published estimate is biased upward. Pre-registration and power analysis
        before data collection are the only defences.
      </p>

      <h3>Pitfall 6 — Multiple Comparisons Inflate Type I Errors</h3>
      <p>
        Running 20 independent tests at α = 0.05 gives an expected 1 false
        positive even when all null hypotheses are true. Apply a Bonferroni
        correction (α_corrected = α / m) or control the false discovery rate
        (Benjamini-Hochberg) when testing multiple hypotheses simultaneously.
      </p>

      {/* ═══════════════════════════════════════════════════════════════════
          SECTION 9 – FURTHER READING
      ════════════════════════════════════════════════════════════════════ */}
      <h2>9 · Further Reading</h2>

      <ul>
        <li>
          <strong>Cohen, J. (1988).</strong>{" "}
          <em>Statistical Power Analysis for the Behavioral Sciences</em> (2nd
          ed.). Routledge. — The canonical reference for effect sizes and power
          analysis; tables for every common test.
        </li>
        <li>
          <strong>Neyman, J., &amp; Pearson, E. S. (1933).</strong> "On the
          problem of the most efficient tests of statistical hypotheses."{" "}
          <em>Philosophical Transactions of the Royal Society A, 231</em>,
          289–337. — The foundational paper defining Type I and Type II errors
          formally.
        </li>
        <li>
          <strong>Davis, J., &amp; Goadrich, M. (2006).</strong> "The
          relationship between Precision-Recall and ROC curves." In{" "}
          <em>Proceedings of ICML 2006</em>. — Proves that PR-AUC is strictly
          more informative than ROC-AUC under class imbalance.
        </li>
        <li>
          <strong>Fawcett, T. (2006).</strong> "An introduction to ROC
          analysis." <em>Pattern Recognition Letters, 27</em>(8), 861–874. —
          Clear, practitioner-focused derivation of ROC curves and AUC from
          first principles.
        </li>
        <li>
          <strong>Gelman, A., &amp; Loken, E. (2014).</strong> "The statistical
          crisis in science." <em>American Scientist, 102</em>(6), 460–465. —
          Readable discussion of the winner's curse, p-hacking, and
          underpowered research.
        </li>
        <li>
          <strong>Button, K. S., et al. (2013).</strong> "Power failure: why
          small sample size undermines the reliability of neuroscience."{" "}
          <em>Nature Reviews Neuroscience, 14</em>, 365–376. — Empirical survey
          showing median power ≈ 8–31 % in a major sub-field; a cautionary tale
          applicable to ML benchmarking.
        </li>
      </ul>

    </article>
  );
}
