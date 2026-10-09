"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import numpy as np
from scipy.stats import norm

# ── Normal Distribution: PDF, CDF, and Quantiles ─────────
mu, sigma = 70, 15          # IQ distribution: mean=70 is hypothetical
x = 85

# P(X ≤ 85)
prob_leq_85 = norm.cdf(x, mu, sigma)
print(f"P(IQ ≤ 85) = {prob_leq_85:.4f}")

# P(X > 85) = 1 - CDF
prob_gt_85 = 1 - norm.cdf(x, mu, sigma)
print(f"P(IQ > 85) = {prob_gt_85:.4f}")

# Inverse CDF: what value corresponds to the top 5%?
top_5_pct = norm.ppf(0.95, mu, sigma)
print(f"Top 5% threshold: {top_5_pct:.2f}")

# ── Binomial: Exact Test ─────────────────────────────────
from scipy.stats import binom

# A model claims 60% accuracy. In 100 trials, it got 52 right.
# Is 52 unusual if the true rate is 60%?
n, p_claimed = 100, 0.60
k_observed = 52
p_val = binom.cdf(k_observed, n, p_claimed)  # P(X ≤ 52 | p=0.60)
print(f"\\nBinomial p-value (one-tailed): {p_val:.4f}")
# If p_val < 0.05 → performance is significantly below the claimed 60%

# ── Central Limit Theorem in Practice ────────────────────
np.random.seed(42)

# Parent distribution: heavily right-skewed Exponential
population = np.random.exponential(scale=10, size=100_000)
true_mean   = 10.0          # E[Exp(λ=0.1)] = 1/λ = 10
true_se     = 10.0 / np.sqrt(30)  # σ/√n for n=30

# Draw 10,000 samples of size 30 and compute each sample mean
sample_means = [np.mean(np.random.choice(population, size=30)) for _ in range(10_000)]
sample_means = np.array(sample_means)

print(f"\\nCentral Limit Theorem Demo (Exponential population)")
print(f"  True population mean:       {true_mean:.4f}")
print(f"  Mean of sample means:       {sample_means.mean():.4f}")
print(f"  Theoretical SE (σ/√n):      {true_se:.4f}")
print(f"  Empirical SE of sample means:{sample_means.std():.4f}")
# The empirical SE should closely match σ/√n = 10/√30 ≈ 1.826`;

export default function DistributionsPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">

      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-sky-100 text-sky-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-sky-500"></span>
          Track 0 · Mathematical Foundations
        </span>
      </div>

      <h1>Probability Distributions</h1>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Derive the parameters (mean and variance) of the Normal, Binomial, and Poisson distributions from first principles.</li>
        <li>Distinguish when each distribution is the correct model for a given data-generating process.</li>
        <li>Use the CDF to compute probabilities and the inverse CDF (PPF) to compute quantiles.</li>
        <li>State and apply the Central Limit Theorem with its exact conditions and implications.</li>
        <li>Identify the correct distribution to model ML-relevant phenomena (class imbalance, event counts, continuous predictions).</li>
      </ul>

      <h2>1 · The Normal (Gaussian) Distribution</h2>
      <p>
        The Normal distribution N(μ, σ²) is defined by its PDF:
      </p>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4">
        <p>f(x; μ, σ²) = (1 / (σ √(2π))) · exp( −(x − μ)² / (2σ²) )</p>
        <p className="text-slate-400 text-xs mt-2">Support: x ∈ (−∞, +∞)</p>
        <p className="text-slate-400 text-xs">Mean = μ,   Variance = σ²</p>
      </div>
      <p>
        The Normal distribution emerges naturally from the Central Limit Theorem and the principle
        of maximum entropy: given that you know only the mean and variance of a distribution, the
        distribution that makes the fewest additional assumptions (maximum entropy) is the Normal.
      </p>

      <h3>The Standard Normal N(0,1)</h3>
      <p>
        Any Normal distribution can be standardised to Z ~ N(0,1) using the Z-score:
      </p>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 space-y-1">
        <p>Z = (X − μ) / σ</p>
        <p></p>
        <p>The 68-95-99.7 rule (Empirical Rule):</p>
        <p>  P(|Z| ≤ 1) ≈ 0.6827    (1 standard deviation)</p>
        <p>  P(|Z| ≤ 2) ≈ 0.9545    (2 standard deviations)</p>
        <p>  P(|Z| ≤ 3) ≈ 0.9973    (3 standard deviations)</p>
      </div>
      <p>
        <strong>ML Relevance:</strong> Neural network weight initialisation (He, Xavier/Glorot) draws
        from Normal distributions. Linear regression residuals are assumed Normal. Gaussian Naive
        Bayes models feature likelihoods as Normal. Gaussian Processes use the Normal as the
        prior over functions.
      </p>

      <h2>2 · The Binomial Distribution</h2>
      <p>
        The Binomial(n, p) distribution models the number of successes k in n independent Bernoulli
        trials, each with success probability p:
      </p>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 space-y-1">
        <p>P(X = k) = C(n, k) · pᵏ · (1 − p)ⁿ⁻ᵏ</p>
        <p></p>
        <p>where C(n, k) = n! / (k!(n−k)!)  is the binomial coefficient</p>
        <p></p>
        <p>Mean:     E[X] = np</p>
        <p>Variance: Var[X] = np(1−p)</p>
      </div>

      <h3>When to Use It</h3>
      <ul>
        <li><strong>A/B testing sample sizes:</strong> How many trials to detect a 2% lift in conversion rate with 80% power?</li>
        <li><strong>Accuracy confidence intervals:</strong> Treating each correct/incorrect prediction as a Bernoulli trial.</li>
        <li><strong>Exact statistical tests:</strong> Binomial test for whether a model&apos;s accuracy differs from chance.</li>
      </ul>

      <h3>Binomial → Normal Approximation</h3>
      <p>
        When n is large and p is not too extreme (np ≥ 5 and n(1-p) ≥ 5), the Binomial converges
        to a Normal distribution by the CLT:
      </p>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4">
        <p>X ~ Binomial(n, p)  ≈  N(np, np(1−p))    as n → ∞</p>
      </div>

      <h2>3 · The Poisson Distribution</h2>
      <p>
        The Poisson(λ) distribution models the number of events occurring in a fixed interval of
        time or space, given an average rate λ:
      </p>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 space-y-1">
        <p>P(X = k) = (λᵏ · e⁻λ) / k!     for k = 0, 1, 2, …</p>
        <p></p>
        <p>Mean = Variance = λ   (a key property: if Var ≠ Mean, data is not Poisson)</p>
      </div>
      <p>
        The Poisson distribution is the limiting case of Binomial(n, p) as n → ∞ and p → 0 with
        the product λ = np held constant. This is why it models "rare events" well.
      </p>
      <p>
        <strong>ML Relevance:</strong> Modelling arrival rates of requests to a serving system,
        word counts in NLP (bag-of-words), fraud transaction rates, bug counts in code.
        Poisson regression is used when the output is a count variable.
      </p>

      <h2>4 · The Beta Distribution</h2>
      <p>
        The Beta(α, β) distribution models probabilities — values bounded in (0, 1). This makes it
        the natural prior in Bayesian analysis when the unknown parameter is itself a probability:
      </p>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 space-y-1">
        <p>f(x; α, β) = x^(α−1) (1−x)^(β−1) / B(α, β)    for x ∈ (0,1)</p>
        <p></p>
        <p>Mean:     α / (α + β)</p>
        <p>Variance: αβ / ((α+β)²(α+β+1))</p>
      </div>
      <p>
        In Bayesian A/B testing, if you observe s successes and f failures, the posterior
        distribution of the conversion rate is Beta(α + s, β + f) — an elegant closed-form update.
      </p>

      <h2>5 · The Central Limit Theorem (Rigorous Statement)</h2>
      <blockquote>
        <p>
          <strong>Theorem (Lindeberg-Lévy CLT):</strong> Let X₁, X₂, …, Xₙ be i.i.d. random
          variables with E[Xᵢ] = μ and Var[Xᵢ] = σ² &lt; ∞. Then as n → ∞:
        </p>
      </blockquote>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 space-y-1">
        <p>√n · (x̄ₙ − μ) / σ  →  N(0, 1)   in distribution</p>
        <p></p>
        <p>Equivalently: x̄ₙ ~ N(μ,  σ²/n)  approximately for large n</p>
        <p></p>
        <p>Standard Error of the Mean: SE = σ / √n</p>
      </div>

      <p>
        The "i.i.d." condition (independent, identically distributed) is important. The CLT holds
        for any distribution — exponential, uniform, Poisson — as long as the variance is finite.
        Heavy-tailed distributions (e.g., Cauchy, which has undefined variance) do not satisfy
        the CLT.
      </p>
      <p>
        <strong>Implication for ML:</strong> When you average predictions from multiple models
        (ensembling), the ensemble mean converges to a Normal distribution as the number of models
        grows. This is also why large-batch SGD behaves more deterministically — you are averaging
        many gradient estimates, each converging toward the true gradient.
      </p>

      <h2>6 · Distribution Reference Table</h2>
      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Distribution</th>
              <th className="px-4 py-3 text-left">Type</th>
              <th className="px-4 py-3 text-left">Parameter(s)</th>
              <th className="px-4 py-3 text-left">Mean</th>
              <th className="px-4 py-3 text-left">ML Use Case</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white"><td className="px-4 py-3">Normal</td><td className="px-4 py-3">Continuous</td><td className="px-4 py-3 font-mono">μ, σ²</td><td className="px-4 py-3 font-mono">μ</td><td className="px-4 py-3">Residuals, weight init</td></tr>
            <tr className="bg-slate-50"><td className="px-4 py-3">Binomial</td><td className="px-4 py-3">Discrete</td><td className="px-4 py-3 font-mono">n, p</td><td className="px-4 py-3 font-mono">np</td><td className="px-4 py-3">Classification accuracy</td></tr>
            <tr className="bg-white"><td className="px-4 py-3">Poisson</td><td className="px-4 py-3">Discrete</td><td className="px-4 py-3 font-mono">λ</td><td className="px-4 py-3 font-mono">λ</td><td className="px-4 py-3">Event counts, NLP</td></tr>
            <tr className="bg-slate-50"><td className="px-4 py-3">Beta</td><td className="px-4 py-3">Continuous</td><td className="px-4 py-3 font-mono">α, β</td><td className="px-4 py-3 font-mono">α/(α+β)</td><td className="px-4 py-3">Bayesian priors, CTR</td></tr>
            <tr className="bg-white"><td className="px-4 py-3">Exponential</td><td className="px-4 py-3">Continuous</td><td className="px-4 py-3 font-mono">λ</td><td className="px-4 py-3 font-mono">1/λ</td><td className="px-4 py-3">Time between events</td></tr>
            <tr className="bg-slate-50"><td className="px-4 py-3">Uniform</td><td className="px-4 py-3">Continuous</td><td className="px-4 py-3 font-mono">a, b</td><td className="px-4 py-3 font-mono">(a+b)/2</td><td className="px-4 py-3">Hyperparameter search</td></tr>
          </tbody>
        </table>
      </div>

      <h2>7 · Python Implementation</h2>
      <TerminalBlock language="python" filename="distributions.py" code={code} />

      <h2>8 · Common Pitfalls</h2>
      <ul>
        <li><strong>Applying Normal assumptions to count data.</strong> Count data (0, 1, 2, …) cannot be negative. Using a Normal model can produce negative predictions. Use Poisson regression (GLM) instead.</li>
        <li><strong>Ignoring over-dispersion in Poisson models.</strong> Poisson requires mean = variance. Real count data is often over-dispersed (variance &gt;&gt; mean). Use the Negative Binomial distribution in that case.</li>
        <li><strong>CLT sample size rule of thumb.</strong> "n ≥ 30 is always enough" is false. For very skewed or heavy-tailed data (e.g., income), you may need n ≥ 200 or more before the CLT kicks in reliably.</li>
      </ul>

      <h2>9 · Further Reading</h2>
      <ul>
        <li>DeGroot, M.H. & Schervish, M.J. (2012). <em>Probability and Statistics</em>, 4th ed.</li>
        <li>Bishop, C.M. (2006). <em>Pattern Recognition and Machine Learning</em>, Appendix B.</li>
        <li>SciPy <code>scipy.stats</code>: <code>norm</code>, <code>binom</code>, <code>poisson</code>, <code>beta</code>, <code>expon</code>.</li>
      </ul>

    </article>
  );
}
