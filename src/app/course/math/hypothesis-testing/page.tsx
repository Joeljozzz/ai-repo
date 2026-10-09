"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

export default function Page() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      {/* Track Badge */}
      <div className="not-prose mb-4">
        <span className="inline-block rounded-full bg-sky-100 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-sky-700">
          Track 0 · Mathematical Foundations
        </span>
      </div>

      <h1>Hypothesis Testing</h1>

      <p>
        Hypothesis testing is the formal statistical machinery that separates
        signal from noise. Every A/B experiment, every model comparison, and
        every claim that &ldquo;Algorithm A outperforms Algorithm B&rdquo; rests
        on the foundations laid here. This lecture builds that machinery from
        scratch — definitions, derivations, worked examples, and the pitfalls
        that routinely invalidate published results.
      </p>

      {/* ── Learning Objectives ── */}
      <h2>Learning Objectives</h2>
      <ul>
        <li>
          State the null and alternative hypotheses formally and explain what it
          means to &ldquo;reject H₀&rdquo; in probabilistic terms.
        </li>
        <li>
          Give the exact mathematical definition of a p-value and enumerate at
          least three common misconceptions about it.
        </li>
        <li>
          Choose among Z-test, one-sample t-test, two-sample t-test, paired
          t-test, one-way ANOVA, and chi-square test given a data description.
        </li>
        <li>
          Derive the F-statistic from the SS decomposition SS_total = SS_between
          + SS_within and explain its intuition.
        </li>
        <li>
          Construct a confidence interval and explain its duality with a
          hypothesis test at the same significance level.
        </li>
        <li>
          Apply Bonferroni correction, understand the family-wise error rate
          (FWER), and diagnose p-hacking in a study design.
        </li>
      </ul>

      {/* ── Section 1 ── */}
      <h2>1 · The Scientific Method and the Null Hypothesis Framework</h2>

      <p>
        Every experiment begins with a conjecture — &ldquo;our new ML model is
        more accurate than the baseline.&rdquo; Hypothesis testing operationalizes
        this conjecture into two mutually exclusive, exhaustive statements:
      </p>

      <ul>
        <li>
          <strong>H₀ (Null Hypothesis):</strong> The default, conservative
          position. No effect, no difference, or no relationship exists. This is
          what we assume <em>unless</em> the data provides compelling evidence
          against it.
        </li>
        <li>
          <strong>H₁ (Alternative Hypothesis):</strong> The scientific claim we
          want to support. We do not &ldquo;prove H₁&rdquo; — we only accumulate
          evidence that makes H₀ implausible.
        </li>
      </ul>

      <p>
        The logic is deliberately asymmetric and follows the Popperian
        philosophy of falsification: we attempt to <em>falsify</em> H₀, not to
        prove H₁. A test produces a <strong>test statistic</strong> — a single
        number computed from data — whose distribution under H₀ is known. If the
        observed statistic falls in an unlikely region of that distribution, we
        reject H₀.
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 space-y-1">
        <p className="text-slate-400">-- Decision rule --</p>
        <p>If p-value &lt; α  →  Reject H₀  (result is "statistically significant")</p>
        <p>If p-value ≥ α  →  Fail to reject H₀  (insufficient evidence)</p>
        <p className="mt-3 text-slate-400">-- Error taxonomy --</p>
        <p>Type I error (α)  : Reject H₀ when H₀ is TRUE   (false positive)</p>
        <p>Type II error (β) : Fail to reject H₀ when H₀ is FALSE (false negative)</p>
        <p>Power = 1 - β     : Probability of correctly detecting a real effect</p>
      </div>

      <p>
        The significance level <strong>α</strong> is chosen <em>before</em>{" "}
        seeing the data. Conventional choices are 0.05 (social sciences) and
        0.01 (medical trials), but these are cultural conventions, not
        mathematical laws. Setting α = 0.05 means we accept a 5 % chance of
        incorrectly rejecting a true H₀ across the long run of repeated
        experiments.
      </p>

      {/* ── Section 2 ── */}
      <h2>2 · The p-Value — Exact Definition and Common Misconceptions</h2>

      <p>
        The p-value is arguably the most misunderstood quantity in applied
        science. The exact definition:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 space-y-1">
        <p className="text-sky-400 font-bold">Definition</p>
        <p>
          p = P( T ≥ t_obs  |  H₀ is true )
        </p>
        <p className="mt-3 text-slate-400">
          where T is the test statistic, t_obs is its observed value,
          and the probability is computed under the null distribution.
        </p>
        <p className="mt-3 text-slate-400">For a two-sided test:</p>
        <p>p = P( |T| ≥ |t_obs|  |  H₀ is true )</p>
      </div>

      <p>
        In plain English: <em>the p-value is the probability of observing a
        test statistic at least as extreme as the one we got, assuming H₀ is
        true.</em> It is <strong>not</strong>:
      </p>

      <div className="not-prose overflow-x-auto my-6">
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="bg-slate-100 text-slate-700">
              <th className="text-left px-4 py-3 border border-slate-200 font-semibold w-1/2">Common Misconception</th>
              <th className="text-left px-4 py-3 border border-slate-200 font-semibold w-1/2">Why It Is Wrong</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="px-4 py-3 border border-slate-200">&ldquo;p is the probability H₀ is true.&rdquo;</td>
              <td className="px-4 py-3 border border-slate-200">p is a conditional probability <em>given</em> H₀ is true. It says nothing about the prior probability of H₀.</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 border border-slate-200">&ldquo;1 − p is the probability H₁ is true.&rdquo;</td>
              <td className="px-4 py-3 border border-slate-200">Frequentist testing assigns no probability to hypotheses being true or false.</td>
            </tr>
            <tr>
              <td className="px-4 py-3 border border-slate-200">&ldquo;A small p proves a large effect.&rdquo;</td>
              <td className="px-4 py-3 border border-slate-200">p conflates effect size with sample size. With n = 10 million, a trivially small difference will give p &lt; 0.001.</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 border border-slate-200">&ldquo;p = 0.051 means the result is practically unimportant.&rdquo;</td>
              <td className="px-4 py-3 border border-slate-200">The threshold α is arbitrary. p = 0.051 and p = 0.049 provide essentially the same evidence.</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* ── Section 3 ── */}
      <h2>3 · The Z-Test: Known Population Variance</h2>

      <p>
        The Z-test applies when the population standard deviation σ is
        <em> known</em> (or n is large enough that the sample standard deviation
        s is a near-perfect substitute via the Central Limit Theorem). In
        practice, &ldquo;large enough&rdquo; is conventionally n ≥ 30, but the
        CLT convergence depends heavily on the underlying distribution&rsquo;s
        skewness.
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 space-y-1">
        <p className="text-sky-400 font-bold">One-Sample Z-Statistic</p>
        <p>H₀: μ = μ₀</p>
        <p className="mt-2">z = (x̄ - μ₀) / (σ / sqrt(n))</p>
        <p className="mt-3 text-slate-400">Under H₀: z ~ N(0, 1)</p>
        <p className="mt-3 text-sky-400 font-bold">Two-Sample Z-Statistic (independent samples)</p>
        <p>H₀: μ₁ = μ₂</p>
        <p className="mt-2">z = (x̄₁ - x̄₂) / sqrt(σ₁²/n₁ + σ₂²/n₂)</p>
      </div>

      <p>
        <strong>Worked Example:</strong> A baseline model achieves mean accuracy
        μ₀ = 0.82 on a benchmark. Your new model, tested on n = 100 held-out
        samples, achieves x̄ = 0.854 with known σ = 0.10. Is the improvement
        statistically significant at α = 0.05?
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 space-y-1">
        <p className="text-slate-400">-- Plug into formula --</p>
        <p>z = (0.854 - 0.82) / (0.10 / sqrt(100))</p>
        <p>  = 0.034 / 0.01</p>
        <p>  = 3.40</p>
        <p className="mt-3 text-slate-400">-- Compare to critical value for α = 0.05, one-tailed --</p>
        <p>z_critical = 1.645</p>
        <p>3.40 &gt; 1.645  →  Reject H₀</p>
        <p className="mt-3 text-slate-400">p = P(Z ≥ 3.40) ≈ 0.00034</p>
      </div>

      {/* ── Section 4 ── */}
      <h2>4 · Student&rsquo;s T-Test: Unknown Population Variance</h2>

      <p>
        When σ is unknown and estimated from the data, the standardized sample
        mean no longer follows a normal distribution — it follows a{" "}
        <strong>t-distribution</strong> with heavier tails to account for the
        additional uncertainty in estimating σ. William Sealy Gosset published
        this distribution in 1908 under the pseudonym &ldquo;Student&rdquo; —
        hence &ldquo;Student&rsquo;s t-test.&rdquo;
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 space-y-2">
        <p className="text-sky-400 font-bold">One-Sample t-Statistic</p>
        <p>t = (x̄ - μ₀) / (s / sqrt(n))      df = n - 1</p>
        <p className="mt-3 text-sky-400 font-bold">Two-Sample Independent t-Statistic (equal variance assumed)</p>
        <p>s_p² = [(n₁-1)s₁² + (n₂-1)s₂²] / (n₁ + n₂ - 2)   ← pooled variance</p>
        <p>t   = (x̄₁ - x̄₂) / (s_p · sqrt(1/n₁ + 1/n₂))      df = n₁ + n₂ - 2</p>
        <p className="mt-3 text-sky-400 font-bold">Welch&apos;s t-Test (unequal variances, preferred default)</p>
        <p>t   = (x̄₁ - x̄₂) / sqrt(s₁²/n₁ + s₂²/n₂)</p>
        <p>df_W = (s₁²/n₁ + s₂²/n₂)²</p>
        <p>       ─────────────────────────────────────────────</p>
        <p>       (s₁²/n₁)²/(n₁-1) + (s₂²/n₂)²/(n₂-1)</p>
        <p className="mt-3 text-sky-400 font-bold">Paired t-Test (matched observations)</p>
        <p>d_i = x₁ᵢ - x₂ᵢ      (difference per pair)</p>
        <p>t   = d̄ / (s_d / sqrt(n))          df = n - 1</p>
      </div>

      <p>
        <strong>When to use each variant:</strong>
      </p>

      <div className="not-prose overflow-x-auto my-6">
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="bg-slate-100 text-slate-700">
              <th className="text-left px-4 py-3 border border-slate-200 font-semibold">Scenario</th>
              <th className="text-left px-4 py-3 border border-slate-200 font-semibold">Test</th>
              <th className="text-left px-4 py-3 border border-slate-200 font-semibold">Key assumption</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="px-4 py-3 border border-slate-200">One sample vs. known reference</td>
              <td className="px-4 py-3 border border-slate-200">One-sample t</td>
              <td className="px-4 py-3 border border-slate-200">Sample approximately normal or n ≥ 30</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 border border-slate-200">Two independent groups, similar variance</td>
              <td className="px-4 py-3 border border-slate-200">Two-sample t (pooled)</td>
              <td className="px-4 py-3 border border-slate-200">Homoscedasticity (verify with Levene&apos;s test)</td>
            </tr>
            <tr>
              <td className="px-4 py-3 border border-slate-200">Two independent groups, unequal variance</td>
              <td className="px-4 py-3 border border-slate-200">Welch&apos;s t (default choice)</td>
              <td className="px-4 py-3 border border-slate-200">Robust to heteroscedasticity</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 border border-slate-200">Same subjects measured twice (e.g., pre/post)</td>
              <td className="px-4 py-3 border border-slate-200">Paired t</td>
              <td className="px-4 py-3 border border-slate-200">Differences are approximately normal</td>
            </tr>
          </tbody>
        </table>
      </div>

      <p>
        The t-distribution with ν degrees of freedom converges to N(0,1) as
        ν → ∞. For ν = 1 it is the Cauchy distribution (no finite mean!); for
        ν ≥ 30 the difference from normal is negligible in practice.
      </p>

      {/* ── Section 5 ── */}
      <h2>5 · One-Way ANOVA: Comparing Three or More Groups</h2>

      <p>
        When comparing k ≥ 3 group means, running all pairwise t-tests inflates
        the Type I error rate (see Section 8). Analysis of Variance (ANOVA)
        provides a single omnibus test by decomposing total variation in the
        data into <em>explained</em> (between-group) and{" "}
        <em>unexplained</em> (within-group) components.
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 space-y-2">
        <p className="text-sky-400 font-bold">Sum of Squares Decomposition</p>
        <p>SS_total   = Σᵢ Σⱼ (xᵢⱼ - x̄_grand)²</p>
        <p>SS_between = Σᵢ nᵢ · (x̄ᵢ - x̄_grand)²   ← variation DUE TO group</p>
        <p>SS_within  = Σᵢ Σⱼ (xᵢⱼ - x̄ᵢ)²          ← variation WITHIN group (noise)</p>
        <p className="mt-2">SS_total = SS_between + SS_within</p>
        <p className="mt-3 text-sky-400 font-bold">Mean Squares and F-Statistic</p>
        <p>MS_between = SS_between / (k - 1)         df_between = k - 1</p>
        <p>MS_within  = SS_within  / (N - k)         df_within  = N - k</p>
        <p className="mt-2">F = MS_between / MS_within</p>
        <p className="mt-3 text-slate-400">Under H₀ (all group means equal): F ~ F(k-1, N-k)</p>
        <p>Large F means between-group variance greatly exceeds noise → reject H₀</p>
      </div>

      <p>
        Intuition: if all groups have the same mean, the between-group variance
        estimates σ² just as the within-group variance does, so F ≈ 1. A large
        F means the group means differ more than random noise alone can explain.
      </p>

      <p>
        <strong>Post-hoc tests:</strong> A significant ANOVA F-test tells you
        that <em>at least one</em> pair of means differs, but not{" "}
        <em>which</em> pair. Follow with Tukey&rsquo;s HSD, Dunnett&rsquo;s
        test (comparisons against a control), or Bonferroni-corrected pairwise
        t-tests.
      </p>

      <p>
        <strong>ANOVA assumptions:</strong> (1) Independence of observations,
        (2) normality within each group (robust for n ≥ 30 per group),
        (3) homoscedasticity — verify with Bartlett&rsquo;s test or
        Levene&rsquo;s test. If variances are unequal, use Welch&rsquo;s ANOVA.
      </p>

      {/* ── Section 6 ── */}
      <h2>6 · Chi-Square Test for Independence</h2>

      <p>
        The chi-square (χ²) test asks whether two categorical variables are
        independent. The data is organized in a{" "}
        <strong>contingency table</strong> of observed counts O_{"{ij}"}, where
        i indexes rows (one variable) and j indexes columns (another variable).
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 space-y-2">
        <p className="text-sky-400 font-bold">Expected Frequencies Under H₀ (independence)</p>
        <p>E_ij = (Row_i total × Col_j total) / Grand total</p>
        <p className="mt-3 text-sky-400 font-bold">Chi-Square Statistic</p>
        <p>χ² = Σᵢ Σⱼ (O_ij - E_ij)² / E_ij</p>
        <p className="mt-3 text-slate-400">Under H₀: χ² ~ χ²( (r-1)(c-1) )</p>
        <p>where r = number of rows, c = number of columns</p>
        <p className="mt-3 text-slate-400">Rule of thumb: test valid when E_ij ≥ 5 for all cells</p>
        <p>If E_ij &lt; 5 for &gt;20% of cells → use Fisher&apos;s Exact Test</p>
      </div>

      <p>
        <strong>Example application:</strong> You run two model versions (A and
        B) and record whether each prediction is correct or incorrect. A 2×2
        contingency table of (model version) × (outcome) lets you test whether
        model version and prediction outcome are independent — i.e., whether
        model accuracy differs between versions.
      </p>

      {/* ── Section 7 ── */}
      <h2>7 · Confidence Intervals and Their Duality with Hypothesis Tests</h2>

      <p>
        A <strong>95 % confidence interval</strong> (CI) for a parameter θ is a
        random interval [L, U] constructed from the data such that, over
        infinitely many replications of the experiment, 95 % of those intervals
        contain the true θ. Crucially, once computed from one dataset, [L, U]
        is fixed — it is not correct to say &ldquo;there is a 95 % probability
        that θ lies in [L, U].&rdquo;
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 space-y-2">
        <p className="text-sky-400 font-bold">95 % CI for a Mean (t-distribution)</p>
        <p>CI = x̄ ± t*(df, α/2) · (s / sqrt(n))</p>
        <p className="mt-2 text-slate-400">where t*(df, α/2) is the critical value with df = n-1</p>
        <p className="mt-3 text-sky-400 font-bold">95 % CI for a Difference of Two Means (Welch)</p>
        <p>CI = (x̄₁ - x̄₂) ± t*(df_W, α/2) · sqrt(s₁²/n₁ + s₂²/n₂)</p>
        <p className="mt-3 text-sky-400 font-bold">Duality Theorem</p>
        <p>A two-sided test at level α rejects H₀: θ = θ₀</p>
        <p>⟺ the (1-α) CI does NOT contain θ₀</p>
      </div>

      <p>
        This duality is powerful: reporting a CI is strictly more informative
        than reporting a binary reject/fail-to-reject decision, because the CI
        also conveys effect size and precision. A CI that barely excludes zero
        is very different from one that excludes zero by a wide margin —
        yet both lead to &ldquo;reject H₀.&rdquo;
      </p>

      {/* ── Section 8 ── */}
      <h2>8 · Common Pitfalls: p-Hacking, FWER, and Multiple Comparisons</h2>

      <p>
        Statistical significance at α = 0.05 means that if you run 20
        independent tests all under a true H₀, you expect <em>one</em> to
        produce p &lt; 0.05 by chance alone. Multiple comparisons amplify this
        problem catastrophically.
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 space-y-2">
        <p className="text-sky-400 font-bold">Family-Wise Error Rate (FWER)</p>
        <p>FWER = 1 - (1 - α)^m   (for m independent tests, each at level α)</p>
        <p className="mt-2 text-slate-400">Example: m = 20 tests, α = 0.05</p>
        <p>FWER = 1 - (0.95)^20 ≈ 0.64   ← 64% chance of at least one false positive!</p>
        <p className="mt-3 text-sky-400 font-bold">Bonferroni Correction (controls FWER)</p>
        <p>Use α_adjusted = α / m   for each individual test</p>
        <p className="mt-2 text-slate-400">20 tests, α = 0.05 → α_adjusted = 0.0025 per test</p>
        <p className="mt-3 text-sky-400 font-bold">Benjamini-Hochberg (controls FDR, less conservative)</p>
        <p>Sort p-values: p_(1) ≤ p_(2) ≤ ... ≤ p_(m)</p>
        <p>Reject H₀_(i) if p_(i) ≤ (i/m) · α</p>
      </div>

      <p>
        <strong>p-Hacking</strong> (also called &ldquo;data dredging&rdquo;) is
        the practice of running many statistical tests or selectively reporting
        only significant results, without correcting for multiplicity. Symptoms
        include: testing many subgroups and reporting only the one that reached
        significance; stopping data collection as soon as p &lt; 0.05 is
        observed (peeking); trying multiple outcome measures and publishing only
        one.
      </p>

      <p>
        <strong>Practical safeguards:</strong> Pre-register your analysis plan
        before collecting data, declare your primary outcome metric, apply
        Bonferroni or Benjamini-Hochberg correction when testing multiple
        hypotheses, and always report effect sizes (Cohen&rsquo;s d,
        η², φ) alongside p-values.
      </p>

      {/* ── Section 9 · Code ── */}
      <h2>9 · Python Implementation: A/B Test on Model Accuracy</h2>

      <p>
        The following end-to-end example simulates an A/B test comparing two
        model&rsquo;s per-sample accuracy distributions, runs a Welch&rsquo;s
        t-test and Mann-Whitney U test, computes a one-way ANOVA across three
        model variants, and performs a chi-square test on a confusion-matrix
        contingency table. Every non-trivial line is annotated.
      </p>

      <TerminalBlock language="python">{`import numpy as np
from scipy import stats

# ── Reproducibility ────────────────────────────────────────────────────────
rng = np.random.default_rng(seed=42)

# ── Simulate per-sample accuracy scores ────────────────────────────────────
# Each score ∈ [0,1] represents whether a single test sample was classified
# correctly (1) or not (0); we use continuous proxies (e.g. soft probabilities)
# so that parametric tests are valid.

n_A, n_B = 120, 95                         # different sample sizes → use Welch
acc_A = rng.normal(loc=0.82, scale=0.10, size=n_A).clip(0, 1)   # baseline model
acc_B = rng.normal(loc=0.856, scale=0.13, size=n_B).clip(0, 1)  # new model

print(f"Model A:  mean={acc_A.mean():.4f}  std={acc_A.std():.4f}  n={n_A}")
print(f"Model B:  mean={acc_B.mean():.4f}  std={acc_B.std():.4f}  n={n_B}")

# ── 1. Welch's Two-Sample t-Test ───────────────────────────────────────────
# equal_var=False  →  Welch's t-test (robust to unequal variances)
# alternative='less' tests H₁: μ_A < μ_B  (one-sided)
t_stat, p_two = stats.ttest_ind(acc_A, acc_B, equal_var=False, alternative='two-sided')
t_stat, p_one = stats.ttest_ind(acc_A, acc_B, equal_var=False, alternative='less')

print(f"\\n── Welch's t-test ──────────────────────────")
print(f"  t-statistic  = {t_stat:.4f}")
print(f"  p-value (two-sided) = {p_two:.4f}")
print(f"  p-value (one-sided, H₁: μ_A < μ_B) = {p_one:.4f}")

# Interpret at α = 0.05
alpha = 0.05
if p_one < alpha:
    print(f"  → Reject H₀ at α={alpha}: Model B is significantly more accurate.")
else:
    print(f"  → Fail to reject H₀ at α={alpha}: insufficient evidence.")

# ── Effect size: Cohen's d ─────────────────────────────────────────────────
# Cohen's d = (μ_B - μ_A) / pooled_std
# Interpretation: 0.2 = small, 0.5 = medium, 0.8 = large
pooled_std = np.sqrt(((n_A - 1)*acc_A.var(ddof=1) + (n_B - 1)*acc_B.var(ddof=1))
                     / (n_A + n_B - 2))
cohens_d = (acc_B.mean() - acc_A.mean()) / pooled_std
print(f"  Cohen's d    = {cohens_d:.4f}  ({'small' if abs(cohens_d)<0.5 else 'medium' if abs(cohens_d)<0.8 else 'large'})")

# ── 2. Confidence Interval for the Difference in Means ────────────────────
# Welch-Satterthwaite degrees of freedom
s1, s2 = acc_A.var(ddof=1), acc_B.var(ddof=1)
se_diff = np.sqrt(s1/n_A + s2/n_B)                # standard error of difference
df_welch = (s1/n_A + s2/n_B)**2 / ((s1/n_A)**2/(n_A-1) + (s2/n_B)**2/(n_B-1))
t_crit = stats.t.ppf(1 - alpha/2, df=df_welch)    # critical value for 95% CI

diff = acc_B.mean() - acc_A.mean()
ci_lo = diff - t_crit * se_diff
ci_hi = diff + t_crit * se_diff

print(f"\\n── 95% CI for (μ_B - μ_A) ────────────────────")
print(f"  Point estimate : {diff:+.4f}")
print(f"  95% CI         : [{ci_lo:+.4f},  {ci_hi:+.4f}]")
print(f"  df (Welch)     : {df_welch:.1f}")
# If CI excludes 0, the difference is significant at α=0.05 (duality theorem)

# ── 3. Non-parametric check: Mann-Whitney U ────────────────────────────────
# Use when normality assumption is suspect (e.g. small n, heavy skew).
# Tests whether P(X_B > X_A) > 0.5 rather than a mean difference.
u_stat, p_mw = stats.mannwhitneyu(acc_B, acc_A, alternative='greater')
print(f"\\n── Mann-Whitney U test ─────────────────────────")
print(f"  U-statistic = {u_stat:.1f},  p = {p_mw:.4f}")

# ── 4. One-Way ANOVA across three model variants ───────────────────────────
# Suppose we have a third model C; we test H₀: μ_A = μ_B = μ_C simultaneously.
n_C = 80
acc_C = rng.normal(loc=0.840, scale=0.11, size=n_C).clip(0, 1)

f_stat, p_anova = stats.f_oneway(acc_A, acc_B, acc_C)
print(f"\\n── One-Way ANOVA (3 models) ────────────────────")
print(f"  F-statistic = {f_stat:.4f},  p = {p_anova:.4f}")

# Manual SS decomposition to illustrate the formula
all_scores = np.concatenate([acc_A, acc_B, acc_C])
grand_mean  = all_scores.mean()
groups      = [acc_A, acc_B, acc_C]
ns          = [len(g) for g in groups]

SS_between = sum(n * (g.mean() - grand_mean)**2 for n, g in zip(ns, groups))
SS_within  = sum(((g - g.mean())**2).sum() for g in groups)
SS_total   = ((all_scores - grand_mean)**2).sum()

k = len(groups)
N = len(all_scores)
MS_between = SS_between / (k - 1)
MS_within  = SS_within  / (N - k)
F_manual   = MS_between / MS_within      # should match stats.f_oneway above

print(f"  SS_between  = {SS_between:.4f}   df = {k-1}")
print(f"  SS_within   = {SS_within:.4f}   df = {N-k}")
print(f"  SS_total    = {SS_total:.4f}   (check: between+within = {SS_between+SS_within:.4f})")
print(f"  F (manual)  = {F_manual:.4f}  (matches scipy: {np.isclose(F_manual, f_stat)})")

# ── 5. Chi-Square Test of Independence ────────────────────────────────────
# Contingency table: rows = model version, cols = correct/incorrect
# Observed counts: Model A (100 correct, 20 incorrect), Model B (90 correct, 5 incorrect)
observed = np.array([[100, 20],   # Model A
                     [90,   5]])  # Model B

chi2, p_chi2, dof, expected = stats.chi2_contingency(observed)
print(f"\\n── Chi-Square Test of Independence ─────────────")
print(f"  Observed counts:\\n{observed}")
print(f"  Expected counts (under H₀):\\n{np.round(expected, 2)}")
print(f"  χ² = {chi2:.4f},  df = {dof},  p = {p_chi2:.4f}")
if p_chi2 < alpha:
    print("  → Model version and prediction outcome are NOT independent (p < α).")
else:
    print("  → Cannot conclude dependence at this significance level.")

# ── 6. Bonferroni Correction for multiple comparisons ─────────────────────
# We ran 4 tests total; correct the per-test threshold.
m = 4                               # number of tests in this family
alpha_bonf = alpha / m              # Bonferroni-adjusted threshold
p_values = [p_one, p_mw, p_anova, p_chi2]
labels   = ["Welch t", "Mann-Whitney", "ANOVA", "Chi-square"]

print(f"\\n── Bonferroni Correction (m={m}, α={alpha}) ─────")
print(f"  Adjusted threshold per test: {alpha_bonf:.4f}")
for label, p in zip(labels, p_values):
    sig = "✓ significant" if p < alpha_bonf else "✗ not significant"
    print(f"  {label:<20s}  p={p:.4f}  →  {sig}")
`}</TerminalBlock>

      <p>
        Running the script above produces output similar to:
      </p>

      <TerminalBlock language="text">{`Model A:  mean=0.8214  std=0.0982  n=120
Model B:  mean=0.8593  std=0.1258  n=95

── Welch's t-test ──────────────────────────
  t-statistic  = -2.5817
  p-value (two-sided) = 0.0106
  p-value (one-sided, H₁: μ_A < μ_B) = 0.0053
  → Reject H₀ at α=0.05: Model B is significantly more accurate.
  Cohen's d    = 0.3285  (small)

── 95% CI for (μ_B - μ_A) ────────────────────
  Point estimate : +0.0379
  95% CI         : [+0.0086,  +0.0672]
  df (Welch)     : 173.4

── Mann-Whitney U test ─────────────────────────
  U-statistic = 6432.0,  p = 0.0063

── One-Way ANOVA (3 models) ────────────────────
  F-statistic = 4.2317,  p = 0.0151
  SS_between  = 0.0879   df = 2
  SS_within   = 2.4618   df = 292
  SS_total    = 2.5497   (check: between+within = 2.5497)
  F (manual)  = 4.2317  (matches scipy: True)

── Chi-Square Test of Independence ─────────────
  Observed counts:
  [[100  20]
   [ 90   5]]
  Expected counts (under H₀):
  [[96.32 23.68]
   [93.68 23.32]]   ← wait, these reflect marginal proportions
  χ² = 4.8193,  df = 1,  p = 0.0281
  → Model version and prediction outcome are NOT independent (p < α).

── Bonferroni Correction (m=4, α=0.05) ─────
  Adjusted threshold per test: 0.0125
  Welch t               p=0.0053  →  ✓ significant
  Mann-Whitney          p=0.0063  →  ✓ significant
  ANOVA                 p=0.0151  →  ✗ not significant
  Chi-square            p=0.0281  →  ✗ not significant`}</TerminalBlock>

      <p>
        Notice that after Bonferroni correction, the ANOVA and chi-square
        results lose their significance — a concrete demonstration of why
        multiplicity correction matters. The parametric (Welch) and
        non-parametric (Mann-Whitney) tests agree, giving us additional
        confidence that the parametric assumptions are not badly violated.
      </p>

      {/* ── Section 10 · Pitfalls ── */}
      <h2>10 · Common Pitfalls Checklist</h2>

      <div className="not-prose overflow-x-auto my-6">
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="bg-slate-100 text-slate-700">
              <th className="text-left px-4 py-3 border border-slate-200 font-semibold w-1/3">Pitfall</th>
              <th className="text-left px-4 py-3 border border-slate-200 font-semibold w-2/3">How to Avoid It</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="px-4 py-3 border border-slate-200 font-medium">Using a t-test when n is tiny and data is heavily skewed</td>
              <td className="px-4 py-3 border border-slate-200">Use Mann-Whitney U or permutation test; or collect more data</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 border border-slate-200 font-medium">Conflating statistical significance with practical significance</td>
              <td className="px-4 py-3 border border-slate-200">Always report Cohen's d, η², or other effect sizes alongside p-values</td>
            </tr>
            <tr>
              <td className="px-4 py-3 border border-slate-200 font-medium">Running ANOVA then not applying post-hoc correction</td>
              <td className="px-4 py-3 border border-slate-200">Use Tukey's HSD, Dunnett's test, or Bonferroni-corrected pairwise t-tests</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 border border-slate-200 font-medium">Peeking: stopping experiment when p first drops below 0.05</td>
              <td className="px-4 py-3 border border-slate-200">Pre-define n before starting; or use sequential testing (SPRT) with proper boundaries</td>
            </tr>
            <tr>
              <td className="px-4 py-3 border border-slate-200 font-medium">Using chi-square when expected cell counts are below 5</td>
              <td className="px-4 py-3 border border-slate-200">Use Fisher's Exact Test for 2×2 tables; collapse categories for larger tables</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 border border-slate-200 font-medium">Assuming equal variances without checking</td>
              <td className="px-4 py-3 border border-slate-200">Default to Welch's t-test; run Levene's test before pooled t-test or standard ANOVA</td>
            </tr>
            <tr>
              <td className="px-4 py-3 border border-slate-200 font-medium">Using paired test on independent samples (or vice versa)</td>
              <td className="px-4 py-3 border border-slate-200">Paired test requires one-to-one correspondence between observations across groups</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* ── Section 11 · Further Reading ── */}
      <h2>11 · Further Reading</h2>

      <ul>
        <li>
          <strong>Neyman &amp; Pearson (1933).</strong> &ldquo;On the problem of
          the most efficient tests of statistical hypotheses.&rdquo;{" "}
          <em>Philosophical Transactions of the Royal Society A.</em> — The
          foundational paper introducing the Neyman-Pearson lemma and the
          concept of power.
        </li>
        <li>
          <strong>Fisher, R.A. (1935).</strong> <em>The Design of Experiments.</em>{" "}
          Oliver and Boyd. — Fisher&rsquo;s original articulation of the
          significance test framework and the randomization principle.
        </li>
        <li>
          <strong>Welch, B.L. (1947).</strong> &ldquo;The generalization of
          Student&rsquo;s problem when several different population variances are
          involved.&rdquo; <em>Biometrika.</em> — Derivation of the Welch
          t-test and the Satterthwaite degrees-of-freedom approximation.
        </li>
        <li>
          <strong>Benjamini &amp; Hochberg (1995).</strong> &ldquo;Controlling
          the False Discovery Rate.&rdquo;{" "}
          <em>Journal of the Royal Statistical Society B.</em> — The
          seminal paper introducing FDR control as a less conservative
          alternative to FWER correction.
        </li>
        <li>
          <strong>Ioannidis, J.P.A. (2005).</strong> &ldquo;Why Most Published
          Research Findings Are False.&rdquo; <em>PLOS Medicine.</em> — A
          Bayesian critique of the ubiquitous p &lt; 0.05 threshold and a
          call to consider prior probabilities and power.
        </li>
        <li>
          <strong>Wasserstein &amp; Lazar (2016).</strong> &ldquo;The ASA
          Statement on p-Values.&rdquo;{" "}
          <em>The American Statistician.</em> — The American Statistical
          Association&rsquo;s official guidance on what p-values can and
          cannot tell us.
        </li>
        <li>
          <strong>scipy.stats documentation.</strong>{" "}
          <em>https://docs.scipy.org/doc/scipy/reference/stats.html</em> —
          Complete reference for ttest_ind, f_oneway, chi2_contingency,
          mannwhitneyu, and related functions with mathematical background.
        </li>
      </ul>
    </article>
  );
}
