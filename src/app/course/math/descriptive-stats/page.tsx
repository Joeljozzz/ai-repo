"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import numpy as np
from scipy import stats

# ── Dataset ───────────────────────────────────────────────
data = np.array([12, 15, 14, 10, 18, 21, 13, 17, 16, 14,
                 22, 11, 19, 23, 14, 12, 20, 15, 17, 13])

# ── Central Tendency ──────────────────────────────────────
mean   = np.mean(data)          # Arithmetic mean
median = np.median(data)        # 50th percentile (Q2)
mode   = stats.mode(data).mode  # Most frequent value

print(f"Mean:   {mean:.2f}")
print(f"Median: {median:.2f}")
print(f"Mode:   {mode}")

# ── Spread ────────────────────────────────────────────────
# ddof=1 → Bessel's correction → unbiased sample variance
variance = np.var(data, ddof=1)
std_dev  = np.std(data, ddof=1)
print(f"Variance (s²): {variance:.2f}")
print(f"Std Dev  (s):  {std_dev:.2f}")

# ── Shape ─────────────────────────────────────────────────
skewness = stats.skew(data, bias=False)
kurt     = stats.kurtosis(data, fisher=True, bias=False) # Excess kurtosis
print(f"Skewness: {skewness:.4f}")
print(f"Kurtosis: {kurt:.4f}")

# ── Box Plot Five-Number Summary ──────────────────────────
q1, q2, q3 = np.percentile(data, [25, 50, 75])
iqr         = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr
outliers    = data[(data < lower_fence) | (data > upper_fence)]

print(f"\\nFive-Number Summary")
print(f"  Min:    {data.min()}")
print(f"  Q1:     {q1}")
print(f"  Median: {q2}")
print(f"  Q3:     {q3}")
print(f"  Max:    {data.max()}")
print(f"  IQR:    {iqr}")
print(f"  Outliers: {outliers}")`;

export default function DescriptiveStats() {
  return (
    <article className="prose prose-slate max-w-none pb-24">

      {/* ── Track badge ── */}
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-sky-100 text-sky-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-sky-500"></span>
          Track 0 · Mathematical Foundations
        </span>
      </div>

      <h1>Descriptive Statistics</h1>

      <blockquote>
        <p>
          <em>"Statistics is the grammar of science."</em> — Karl Pearson
        </p>
      </blockquote>

      {/* ── Learning Objectives ── */}
      <h2>Learning Objectives</h2>
      <p>By the end of this module you will be able to:</p>
      <ul>
        <li>Compute and interpret the four statistical moments of any distribution.</li>
        <li>Distinguish mean, median and mode and know when each is the appropriate central-tendency estimator.</li>
        <li>Derive Tukey&apos;s fences from first principles and use them to detect outliers.</li>
        <li>Recognise left-skewed, right-skewed, leptokurtic, and platykurtic distributions from their moment values alone.</li>
        <li>Implement every computation in NumPy and SciPy and interpret the output.</li>
      </ul>

      {/* ── 1. Why descriptive stats matter ── */}
      <h2>1 · Why Descriptive Statistics Underpin All of ML</h2>
      <p>
        Before you fit a single model, you must <em>understand your data</em>.
        Descriptive statistics give you a compact, mathematically rigorous summary of a dataset's
        shape, centre, and spread. Skipping this step is one of the most common causes of silent
        model failure in industry — you build a regression on data you assumed was Gaussian, but it
        was log-normal; you scale features with the standard deviation, but your standard deviation
        was inflated by three extreme outliers.
      </p>
      <p>
        Descriptive statistics are also the prerequisite for inferential statistics (hypothesis
        testing), which governs how you decide whether a model improvement is real or random noise.
      </p>

      {/* ── 2. Measures of central tendency ── */}
      <h2>2 · Measures of Central Tendency</h2>
      <p>
        A measure of central tendency locates the "middle" of a distribution. There are three
        classical estimators, each optimal under different conditions.
      </p>

      <h3>2.1 The Arithmetic Mean</h3>
      <p>
        The arithmetic mean <strong>μ</strong> (population) or <strong>x̄</strong> (sample) is the
        sum of all observations divided by their count:
      </p>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 overflow-x-auto">
        <p>Population mean:  μ  = (1/N)  Σᵢ xᵢ</p>
        <p>Sample mean:      x̄  = (1/n)  Σᵢ xᵢ</p>
      </div>
      <p>
        The mean is the unique value that minimises the sum of squared deviations (i.e. it is the
        least-squares estimator of location). Its weakness: it is <em>sensitive to extreme values</em>.
        A single multi-billion-dollar income drags the arithmetic mean income of a country far above
        what the typical citizen earns.
      </p>

      <h3>2.2 The Median</h3>
      <p>
        The median is the value that splits the <em>sorted</em> dataset exactly in half — the 50th
        percentile. It is a <strong>robust</strong> estimator: it is immune to outliers because it
        depends only on rank, not magnitude.
      </p>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 overflow-x-auto">
        <p>If n is odd:   Median = x₍ₙ₊₁₎/₂</p>
        <p>If n is even:  Median = (x₍ₙ/₂₎ + x₍ₙ/₂₊₁₎) / 2</p>
      </div>
      <p>
        Use the median — not the mean — whenever your data could plausibly contain outliers or is
        clearly skewed. Real-world salary data, house prices, response times in production systems,
        and model inference latencies are all classically right-skewed.
      </p>

      <h3>2.3 The Mode</h3>
      <p>
        The mode is the most frequently occurring value. A distribution can be <em>unimodal</em>,
        <em>bimodal</em>, or <em>multimodal</em>. For continuous numerical data the mode is rarely
        useful; it is primarily used for <strong>categorical or ordinal data</strong> (e.g., "what
        category did customers click most?").
      </p>

      {/* ── 3. Measures of spread ── */}
      <h2>3 · Measures of Spread (Dispersion)</h2>

      <h3>3.1 Variance</h3>
      <p>
        Variance measures the average squared deviation from the mean. There are two formulas:
      </p>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 overflow-x-auto">
        <p>Population variance:  σ²  = (1/N)  Σᵢ (xᵢ − μ)²</p>
        <p>Sample variance:      s²  = (1/(n−1))  Σᵢ (xᵢ − x̄)²</p>
      </div>
      <p>
        The <strong>n−1</strong> denominator in the sample formula is <em>Bessel&apos;s correction</em>.
        When we don&apos;t know the true population mean μ and substitute x̄, we introduce a
        downward bias — we&apos;re underestimating the true spread because x̄ is always closer to the
        sample than μ would be. Dividing by n−1 corrects for this, making s² an unbiased estimator
        of σ².
      </p>

      <h3>3.2 Standard Deviation</h3>
      <p>
        The standard deviation σ (or s) is simply √σ². Its advantage over variance: it is expressed
        in the <em>same units as the data</em>, making it interpretable. For a Normal distribution,
        the 68–95–99.7 rule states:
      </p>
      <ul>
        <li>~68% of observations lie within <strong>1σ</strong> of the mean.</li>
        <li>~95% lie within <strong>2σ</strong> of the mean.</li>
        <li>~99.7% lie within <strong>3σ</strong> of the mean.</li>
      </ul>
      <p>
        In practice, data points beyond 3σ are treated as statistical outliers in many engineering
        systems — including anomaly detection pipelines.
      </p>

      {/* ── 4. Shape — the four moments ── */}
      <h2>4 · Shape of a Distribution — The Four Standardised Moments</h2>
      <p>
        The <em>moments</em> of a distribution characterise its shape with increasing granularity.
        A raw moment is E[Xⁿ]. A <em>central</em> moment normalises around the mean: E[(X − μ)ⁿ].
        A <em>standardised</em> moment further divides by σⁿ so the result is dimensionless.
      </p>

      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
          <thead className="bg-slate-100 text-slate-700 font-semibold">
            <tr>
              <th className="px-4 py-3 text-left">Moment</th>
              <th className="px-4 py-3 text-left">Name</th>
              <th className="px-4 py-3 text-left">Formula</th>
              <th className="px-4 py-3 text-left">Meaning</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr className="bg-white">
              <td className="px-4 py-3 font-mono">1st</td>
              <td className="px-4 py-3">Mean</td>
              <td className="px-4 py-3 font-mono">E[X]</td>
              <td className="px-4 py-3">Location / centre of mass</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-mono">2nd</td>
              <td className="px-4 py-3">Variance</td>
              <td className="px-4 py-3 font-mono">E[(X−μ)²]</td>
              <td className="px-4 py-3">Spread around the mean</td>
            </tr>
            <tr className="bg-white">
              <td className="px-4 py-3 font-mono">3rd</td>
              <td className="px-4 py-3">Skewness</td>
              <td className="px-4 py-3 font-mono">E[(X−μ)³] / σ³</td>
              <td className="px-4 py-3">Asymmetry of the distribution</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-mono">4th</td>
              <td className="px-4 py-3">Kurtosis</td>
              <td className="px-4 py-3 font-mono">E[(X−μ)⁴] / σ⁴ − 3</td>
              <td className="px-4 py-3">Tailedness / peak sharpness</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h3>4.1 Skewness in Depth</h3>
      <p>
        Skewness = 0 for a perfectly symmetric distribution. Skewness &gt; 0 implies a
        <strong> right (positive) skew</strong>: the tail extends to the right, and mean &gt; median
        &gt; mode. Skewness &lt; 0 implies a <strong>left (negative) skew</strong>.
      </p>
      <p>
        This matters in ML because many algorithms (linear regression, k-means) implicitly assume
        the data is approximately Gaussian. Applying a log-transform to right-skewed data (income,
        transaction amounts, user sessions) is one of the most effective preprocessing steps a data
        scientist can make.
      </p>

      <h3>4.2 Kurtosis in Depth</h3>
      <p>
        Fisher&apos;s definition of <em>excess kurtosis</em> subtracts 3 so the Normal distribution
        has kurtosis = 0 (called <em>mesokurtic</em>).
      </p>
      <ul>
        <li><strong>Leptokurtic</strong> (kurtosis &gt; 0): Heavy tails, sharp peak. Extreme events occur more often than a Normal distribution predicts. Financial returns are classically leptokurtic — "fat tails" are the root cause of unexpected crashes.</li>
        <li><strong>Platykurtic</strong> (kurtosis &lt; 0): Light tails, flat peak. Fewer extreme events than a Normal predicts.</li>
      </ul>

      {/* ── 5. Box Plots & IQR ── */}
      <h2>5 · Box Plots & Tukey&apos;s Fences</h2>
      <p>
        A box plot (or box-and-whisker plot) visually summarises the five-number summary and
        robustly detects outliers <em>without assuming any distribution</em>. John Tukey defined the
        whisker boundaries — known as Tukey&apos;s fences — as follows:
      </p>
      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-4 space-y-1">
        <p>Q1 = 25th percentile</p>
        <p>Q3 = 75th percentile</p>
        <p>IQR = Q3 − Q1</p>
        <p></p>
        <p>Lower fence = Q1 − 1.5 × IQR</p>
        <p>Upper fence = Q3 + 1.5 × IQR</p>
        <p></p>
        <p>Any point outside [Lower fence, Upper fence] is an outlier.</p>
      </div>
      <p>
        The 1.5 multiplier is not arbitrary. For a Normal distribution, this rule flags
        approximately 0.7% of observations as outliers — a reasonable threshold in most practical
        settings. You can tighten to 3.0 (flagging ~0.01%) for high-stakes financial anomaly
        detection where false positives are costly.
      </p>

      {/* ── 6. Implementation ── */}
      <h2>6 · Python Implementation</h2>
      <TerminalBlock language="python" filename="descriptive_stats.py" code={code} />

      {/* ── 7. Pitfalls ── */}
      <h2>7 · Common Pitfalls</h2>
      <ul>
        <li>
          <strong>Reporting only the mean on skewed data.</strong> Always report mean
          <em> and</em> median. If they diverge significantly, your data is skewed and the mean is
          misleading.
        </li>
        <li>
          <strong>Using population variance (1/N) on a sample.</strong> Almost all real-world
          datasets are samples, not full populations. Use Bessel&apos;s correction (1/(n−1)).
        </li>
        <li>
          <strong>Interpreting Pearson&apos;s kurtosis as peakedness.</strong> Modern statisticians
          agree kurtosis measures tail weight, not peak sharpness. A platykurtic distribution can
          have a sharper peak than a leptokurtic one.
        </li>
        <li>
          <strong>Assuming outliers are errors.</strong> A data point flagged by Tukey&apos;s fence
          may be the most important point in your dataset — a fraud transaction, a hardware failure,
          a viral post. Always investigate before removing.
        </li>
      </ul>

      {/* ── 8. Further reading ── */}
      <h2>8 · Further Reading</h2>
      <ul>
        <li>Tukey, J.W. (1977). <em>Exploratory Data Analysis.</em> Addison-Wesley.</li>
        <li>Pearson, K. (1895). "Contributions to the mathematical theory of evolution." <em>Philosophical Transactions of the Royal Society A</em>, 186.</li>
        <li>NumPy docs: <code>numpy.var</code>, <code>numpy.percentile</code></li>
        <li>SciPy docs: <code>scipy.stats.skew</code>, <code>scipy.stats.kurtosis</code></li>
      </ul>

    </article>
  );
}
