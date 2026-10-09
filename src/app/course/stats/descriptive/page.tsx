"use client";
import React from 'react';
import TerminalBlock from '@/components/TerminalBlock';
import InteractiveQuiz from '@/components/InteractiveQuiz';

export default function DescriptiveStatsPage() {
  const statsCode = `import numpy as np
from scipy.stats import skew, kurtosis

class DescriptiveAnalytics:
    def __init__(self, data):
        self.data = np.array(data)
        self.n = len(self.data)
        
    def calculate_moments(self):
        # 1st Moment: Mean (Location)
        mean = np.sum(self.data) / self.n
        
        # 2nd Moment: Variance (Spread)
        # Using ddof=0 for population, ddof=1 for sample
        variance = np.sum((self.data - mean)**2) / self.n
        std_dev = np.sqrt(variance)
        
        # 3rd Moment: Skewness (Asymmetry)
        data_skew = skew(self.data, bias=False)
        
        # 4th Moment: Kurtosis (Tailedness)
        # Fisher's definition (normal = 0.0)
        data_kurt = kurtosis(self.data, fisher=True, bias=False)
        
        return {
            "Mean (μ)": mean,
            "Variance (σ²)": variance,
            "Std Dev (σ)": std_dev,
            "Skewness": data_skew,
            "Kurtosis": data_kurt
        }

    def boxplot_analytics(self):
        # Quartiles
        q1 = np.percentile(self.data, 25)
        q2 = np.percentile(self.data, 50) # Median
        q3 = np.percentile(self.data, 75)
        
        # Interquartile Range (IQR)
        iqr = q3 - q1
        
        # Outlier Bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = self.data[(self.data < lower_bound) | (self.data > upper_bound)]
        
        return q1, q2, q3, iqr, outliers

# Example usage on a heavily skewed distribution
dataset = np.random.lognormal(mean=0, sigma=1, size=1000)
analytics = DescriptiveAnalytics(dataset)
print(analytics.calculate_moments())`;

  return (
    <div className="space-y-12 pb-24 text-slate-800 animate-in fade-in slide-in-from-bottom-4 duration-500">
      
      <header className="border-b border-slate-200 pb-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-xs font-bold uppercase tracking-widest mb-4">
          <span className="w-2 h-2 rounded-full bg-blue-600 animate-pulse"></span>
          Stack 0: Mathematics & Statistics
        </div>
        <h1 className="text-4xl font-extrabold tracking-tight text-slate-900 mb-6">0.1 Descriptive Statistics</h1>
        <p className="text-xl text-slate-600 leading-relaxed max-w-3xl">
          The mathematical Moments of a distribution: Mean, Variance, Skewness, Kurtosis, and rigorous Box Plot analytics.
        </p>
      </header>

      <section className="prose prose-slate max-w-none text-lg text-slate-600">
        <h2 className="text-3xl font-bold text-slate-900 mt-8 mb-4">1. The Standardized Moments of a Distribution</h2>
        <p>
          In probability theory, a "moment" is a specific quantitative measure of the shape of a set of points. To truly understand data, a Data Scientist must evaluate the first four standardized moments of the probability density function (PDF).
        </p>

        <div className="grid md:grid-cols-2 gap-8 my-8">
          <div className="bg-slate-50 border border-slate-200 rounded-2xl p-6 shadow-sm">
            <h3 className="text-xl font-bold text-slate-900 mt-0 mb-2">1st Moment: Mean (μ)</h3>
            <p className="text-sm text-slate-600 mb-4">Measures the central location of the data. Extremely sensitive to outliers.</p>
            <div className="bg-white p-3 rounded border border-slate-200 font-mono text-sm shadow-sm text-slate-800">
              μ = (1/N) Σ x_i
            </div>
          </div>

          <div className="bg-slate-50 border border-slate-200 rounded-2xl p-6 shadow-sm">
            <h3 className="text-xl font-bold text-slate-900 mt-0 mb-2">2nd Moment: Variance (σ²)</h3>
            <p className="text-sm text-slate-600 mb-4">Measures the geometric spread of the data around the mean (Central Moment).</p>
            <div className="bg-white p-3 rounded border border-slate-200 font-mono text-sm shadow-sm text-slate-800">
              σ² = (1/N) Σ (x_i - μ)²
            </div>
          </div>

          <div className="bg-slate-50 border border-slate-200 rounded-2xl p-6 shadow-sm">
            <h3 className="text-xl font-bold text-slate-900 mt-0 mb-2">3rd Moment: Skewness</h3>
            <p className="text-sm text-slate-600 mb-4">Measures the asymmetry of the probability distribution. A positive skew indicates a long tail to the right.</p>
            <div className="bg-white p-3 rounded border border-slate-200 font-mono text-sm shadow-sm text-slate-800">
              Skew = E[((X - μ) / σ)³]
            </div>
          </div>

          <div className="bg-slate-50 border border-slate-200 rounded-2xl p-6 shadow-sm">
            <h3 className="text-xl font-bold text-slate-900 mt-0 mb-2">4th Moment: Kurtosis</h3>
            <p className="text-sm text-slate-600 mb-4">Measures the "tailedness" of the distribution. High kurtosis means heavy tails (outliers are common).</p>
            <div className="bg-white p-3 rounded border border-slate-200 font-mono text-sm shadow-sm text-slate-800">
              Kurt = E[((X - μ) / σ)⁴] - 3
            </div>
            <p className="text-xs text-slate-500 mt-2">Note: Fisher's definition subtracts 3 so a Normal Distribution has Kurtosis = 0.</p>
          </div>
        </div>

        <h2 className="text-3xl font-bold text-slate-900 mt-12 mb-4">2. Non-Parametric Descriptors: Median, Mode & Box Plots</h2>
        <p>
          Moments are highly sensitive to outliers. When dealing with skewed data (like human income), we rely on rank-based (non-parametric) descriptors.
        </p>
        <ul className="list-disc pl-5">
          <li><strong>Median (Q2):</strong> The 50th percentile. The exact middle value when sorted. Completely robust to outliers.</li>
          <li><strong>Mode:</strong> The most frequent value (used primarily for categorical data).</li>
        </ul>

        <div className="bg-slate-50 border-l-4 border-blue-500 p-6 my-6 shadow-sm">
          <h4 className="font-bold text-slate-900 mt-0">The Mathematics of a Box Plot (Tukey's Fences)</h4>
          <p>A Box Plot maps the interquartile range (IQR) to detect statistical outliers without assuming a normal distribution:</p>
          <ul className="text-sm space-y-2 mt-3 font-mono text-slate-800">
            <li>1. Calculate Q1 (25th percentile) and Q3 (75th percentile).</li>
            <li>2. IQR = Q3 - Q1</li>
            <li>3. Lower Whisker Boundary = Q1 - (1.5 * IQR)</li>
            <li>4. Upper Whisker Boundary = Q3 + (1.5 * IQR)</li>
          </ul>
          <p className="text-sm mt-3">Any data point lying outside the whiskers is formally classified as a statistical outlier.</p>
        </div>

        <div className="bg-white border border-slate-200 p-6 rounded-2xl my-8 shadow-lg not-prose">
            <h3 className="text-slate-900 font-bold mt-0 text-xl mb-4">Implementation: Statistical Analysis with NumPy & SciPy</h3>
            <TerminalBlock 
                language="python" 
                filename="descriptive_stats.py" 
                code={statsCode}
            />
        </div>

        <InteractiveQuiz 
            question="If a dataset exhibits a very high positive Skewness (e.g., global wealth distribution), which relationship between the mean and median is statistically guaranteed to be true in almost all cases?"
            options={[
                "The Mean will be significantly lower than the Median.",
                "The Mean will be significantly higher than the Median.",
                "The Mean and Median will be exactly equal.",
                "The Variance will be equal to the Mean."
            ]}
            correctIndex={1}
            explanation="In a right-skewed (positive skew) distribution, there is a long tail on the right side. The massive values in that tail pull the arithmetic Mean significantly higher, while the Median (which just ranks the data) remains anchored at the 50th percentile."
        />
      </section>
    </div>
  );
}
