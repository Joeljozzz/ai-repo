"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import numpy as np
from scipy.stats import norm, binom, poisson

# ── 1. Normal Distribution ────────────────────────────────
mu, sigma = 0, 1          # Standard Normal N(0,1)
x = np.linspace(-4, 4, 400)

pdf_values = norm.pdf(x, mu, sigma)
cdf_values = norm.cdf(x, mu, sigma)

# Probability of X being between -1 and 1 (the 68% rule)
p_within_1_sigma = norm.cdf(1) - norm.cdf(-1)
print(f"P(-1 < X < 1) = {p_within_1_sigma:.4f}")   # ≈ 0.6827

# ── 2. Binomial Distribution ─────────────────────────────
n, p = 20, 0.3            # 20 trials, 30% success probability
k = np.arange(0, 21)

pmf_values = binom.pmf(k, n, p)
mean_binom  = n * p
var_binom   = n * p * (1 - p)
print(f"Binomial  mean={mean_binom:.2f}, var={var_binom:.2f}")

# ── 3. Poisson Distribution ──────────────────────────────
lam = 5                   # avg 5 events per unit time
k_p = np.arange(0, 20)

pmf_poisson = poisson.pmf(k_p, lam)
print(f"Poisson   mean={lam}, var={lam}")  # mean == variance always

# ── 4. Central Limit Theorem (CLT) demonstration ─────────
# Sum of many i.i.d. random variables converges to Normal
sample_means = []
for _ in range(10_000):
    sample = np.random.exponential(scale=2, size=50)  # NOT Normal
    sample_means.append(np.mean(sample))

clt_mean = np.mean(sample_means)
clt_std  = np.std(sample_means)
print(f"\\nCLT Demo (Exponential samples, n=50)")
print(f"  Sample-mean distribution: mean={clt_mean:.3f}, std={clt_std:.3f}")
print(f"  Theoretical std (σ/√n):   {2 / np.sqrt(50):.3f}")`;

export default function ProbabilityPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">

      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-sky-100 text-sky-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-sky-500"></span>
          Track 0 · Mathematical Foundations
        </span>
      </div>

      <h1>Probability Theory</h1>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Define probability spaces, events, and axioms (Kolmogorov).</li>
        <li>Apply Bayes&apos; theorem to update beliefs given evidence.</li>
        <li>Distinguish discrete (PMF) from continuous (PDF/CDF) random variables.</li>
        <li>Derive mean and variance for the Normal, Binomial, and Poisson distributions.</li>
        <li>State the Central Limit Theorem and explain why it matters to ML at scale.</li>
      </ul>

      <h2>1 · Probability Spaces & Kolmogorov Axioms</h2>
      <p>
        A <strong>probability space</strong> is a triple (Ω, ℱ, P) where Ω is the sample space
        (all possible outcomes), ℱ is a σ-algebra of events (subsets of Ω), and P is a probability
        measure. Kolmogorov&apos;s three axioms are:
      </p>
      <ol>
        <li><strong>Non-negativity:</strong> P(A) ≥ 0 for every event A.</li>
        <li><strong>Normalisation:</strong> P(Ω) = 1 (something always happens).</li>
        <li><strong>Countable additivity:</strong> For mutually exclusive events A₁, A₂, … → P(∪ᵢ Aᵢ) = Σᵢ P(Aᵢ).</li>
      </ol>
      <p>
        Every other rule in probability (complement rule, inclusion–exclusion, etc.) is derived
        directly from these three axioms.
      </p>

      <h2>2 · Conditional Probability & Bayes&apos; Theorem</h2>
      <p>
        The conditional probability of A given B is the probability that A occurs, knowing B has
        already occurred:
      </p>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4">
        <p>P(A | B) = P(A ∩ B) / P(B),    provided P(B) &gt; 0</p>
      </div>
      <p>
        <strong>Bayes&apos; theorem</strong> inverts the conditioning — it tells us how to update our
        belief about a hypothesis H after observing evidence E:
      </p>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 space-y-2">
        <p>P(H | E) = [ P(E | H) · P(H) ] / P(E)</p>
        <p></p>
        <p>Posterior  = Likelihood × Prior / Evidence</p>
      </div>
      <p>
        This is the mathematical foundation of <strong>Bayesian inference</strong>, Naive Bayes
        classifiers, Bayesian optimisation for hyperparameter search, and probabilistic neural
        networks. The "prior" P(H) encodes what you believe before seeing data; the "posterior"
        P(H|E) is what you believe after.
      </p>

      <h2>3 · Random Variables & Distribution Functions</h2>

      <h3>3.1 Discrete Random Variables — PMF</h3>
      <p>
        A discrete random variable X takes a countable set of values. Its behaviour is fully
        described by the <strong>Probability Mass Function</strong>:
      </p>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 space-y-1">
        <p>PMF:  P(X = k) = p(k),   where Σₖ p(k) = 1</p>
      </div>

      <h3>3.2 Continuous Random Variables — PDF & CDF</h3>
      <p>
        A continuous random variable takes uncountably many values. Because P(X = x) = 0 for any
        specific x, we instead define a <strong>Probability Density Function</strong> (PDF) f(x)
        such that:
      </p>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 space-y-1">
        <p>P(a ≤ X ≤ b) = ∫ₐᵇ f(x) dx</p>
        <p></p>
        <p>Cumulative Distribution Function (CDF):</p>
        <p>F(x) = P(X ≤ x) = ∫₋∞ˣ f(t) dt</p>
      </div>
      <p>
        In practice, the CDF is what you use to compute p-values in hypothesis testing. The PDF
        gives the <em>shape</em>; the CDF gives <em>probabilities</em>.
      </p>

      <h2>4 · Key Distributions in ML</h2>

      <h3>4.1 The Normal (Gaussian) Distribution</h3>
      <p>
        The Normal distribution N(μ, σ²) is the most important distribution in statistics. Its PDF is:
      </p>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4">
        <p>f(x) = (1 / (σ√(2π))) · exp( −(x − μ)² / (2σ²) )</p>
      </div>
      <p>
        Why does it appear everywhere? The <strong>Central Limit Theorem</strong> (see §5).
        In ML it appears in: Gaussian Naive Bayes, the initialisation of neural network weights,
        the noise model in linear regression (OLS), and the evidence distribution in Gaussian
        Processes.
      </p>

      <h3>4.2 The Binomial Distribution</h3>
      <p>
        Models the number of successes in <em>n</em> independent Bernoulli trials, each with
        probability <em>p</em>:
      </p>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 space-y-1">
        <p>P(X = k) = C(n,k) · pᵏ · (1−p)ⁿ⁻ᵏ</p>
        <p>Mean:     μ = np</p>
        <p>Variance: σ² = np(1−p)</p>
      </div>
      <p>
        Used in: A/B test sample-size calculations, classification evaluation (number of correct
        predictions), training binary classifiers.
      </p>

      <h3>4.3 The Poisson Distribution</h3>
      <p>
        Models the number of events in a fixed interval of time or space, given an average rate λ:
      </p>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 space-y-1">
        <p>P(X = k) = (λᵏ · e⁻λ) / k!</p>
        <p>Mean = Variance = λ</p>
      </div>
      <p>
        The Poisson is the limit of Binomial(n, p) as n → ∞ and p → 0 with λ = np constant.
        Used in: fraud detection (events per minute), NLP (word frequency models), queuing theory
        for ML serving infrastructure.
      </p>

      <h2>5 · The Central Limit Theorem</h2>
      <p>
        The CLT is arguably the single most important theorem in statistics for data scientists.
        It states:
      </p>
      <blockquote>
        <p>
          Let X₁, X₂, …, Xₙ be i.i.d. random variables with mean μ and finite variance σ².
          As n → ∞, the distribution of the sample mean x̄ converges to N(μ, σ²/n),
          <em> regardless of the original distribution</em> of X.
        </p>
      </blockquote>
      <p>
        This is why we can apply t-tests (which assume normality) to sample means even when the raw
        data comes from an Exponential, Uniform, or any other distribution — so long as our sample
        size is large enough. In practice "large enough" is typically n ≥ 30, though this depends
        on how non-normal the underlying distribution is.
      </p>
      <p>
        The standard error of the mean is σ/√n — this tells you that quadrupling your dataset halves
        the uncertainty in your mean estimate. This principle drives the scaling laws behind LLMs.
      </p>

      <h2>6 · Python Implementation</h2>
      <TerminalBlock language="python" filename="probability_distributions.py" code={code} />

      <h2>7 · Common Pitfalls</h2>
      <ul>
        <li>
          <strong>Confusing PDF with probability.</strong> For a continuous distribution, f(x) can
          exceed 1 — it is a density, not a probability. The probability is the integral of f over
          an interval.
        </li>
        <li>
          <strong>Assuming normality without checking.</strong> Always plot a histogram and QQ-plot
          before assuming Gaussian. Use the Shapiro–Wilk test for small samples (&lt;50).
        </li>
        <li>
          <strong>Misinterpreting the CLT for small samples.</strong> The CLT kicks in asymptotically.
          For n = 10 from a heavily skewed distribution, the sample mean is still far from Normal.
        </li>
      </ul>

      <h2>8 · Further Reading</h2>
      <ul>
        <li>Kolmogorov, A.N. (1933). <em>Foundations of the Theory of Probability.</em></li>
        <li>Bishop, C.M. (2006). <em>Pattern Recognition and Machine Learning</em>, Chapter 1.</li>
        <li>SciPy <code>scipy.stats</code> — comprehensive distribution implementations.</li>
      </ul>

    </article>
  );
}
