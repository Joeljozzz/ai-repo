"use client";
import React from 'react';
import TerminalBlock from '@/components/TerminalBlock';
import InteractiveQuiz from '@/components/InteractiveQuiz';

export default function HypothesisTestingPage() {
  const hypothesisCode = `import numpy as np
from scipy import stats

class HypothesisTesting:
    @staticmethod
    def independent_t_test(group1, group2, alpha=0.05):
        """
        Welch's t-test (assumes unequal variances).
        H0: The two population means are identical.
        H1: The two population means are significantly different.
        """
        # Calculate t-statistic and p-value
        t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
        
        reject_h0 = p_val < alpha
        
        return {
            "T-Statistic": t_stat,
            "P-Value": p_val,
            "Reject Null Hypothesis (H0)": reject_h0
        }

    @staticmethod
    def one_way_anova(*groups, alpha=0.05):
        """
        One-Way ANOVA (Analysis of Variance)
        H0: All group means are identical (mu1 = mu2 = mu3 ...)
        """
        # Calculate F-statistic and p-value
        f_stat, p_val = stats.f_oneway(*groups)
        
        reject_h0 = p_val < alpha
        
        return {
            "F-Statistic": f_stat,
            "P-Value": p_val,
            "Reject Null Hypothesis (H0)": reject_h0
        }

# Generate synthetic data for an A/B/C test
np.random.seed(42)
control_group = np.random.normal(loc=50, scale=10, size=100)
test_group_a = np.random.normal(loc=52, scale=12, size=100)
test_group_b = np.random.normal(loc=65, scale=9, size=100) # Notice the much higher mean

# Run tests
print("--- T-Test (Control vs Group A) ---")
print(HypothesisTesting.independent_t_test(control_group, test_group_a))

print("\\n--- ANOVA (Control vs A vs B) ---")
print(HypothesisTesting.one_way_anova(control_group, test_group_a, test_group_b))`;

  return (
    <div className="space-y-12 pb-24 text-slate-800 animate-in fade-in slide-in-from-bottom-4 duration-500">
      
      <header className="border-b border-slate-200 pb-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-xs font-bold uppercase tracking-widest mb-4">
          <span className="w-2 h-2 rounded-full bg-blue-600 animate-pulse"></span>
          Stack 0: Mathematics & Statistics
        </div>
        <h1 className="text-4xl font-extrabold tracking-tight text-slate-900 mb-6">0.3 Hypothesis Testing</h1>
        <p className="text-xl text-slate-600 leading-relaxed max-w-3xl">
          Statistical significance, p-values, Z-Tests, Student's T-Tests, and Analysis of Variance (ANOVA).
        </p>
      </header>

      <section className="prose prose-slate max-w-none text-lg text-slate-600">
        <h2 className="text-3xl font-bold text-slate-900 mt-8 mb-4">1. The Null Hypothesis Framework</h2>
        <p>
          In statistical engineering (e.g., A/B testing a new ML model in production), we must rigorously prove that observed improvements are not just random chance. We do this by formulating two hypotheses:
        </p>
        <ul className="list-disc pl-5">
          <li><strong>Null Hypothesis (H₀):</strong> The default assumption. E.g., "The new model has the same accuracy as the old model. Any observed difference is just noise."</li>
          <li><strong>Alternative Hypothesis (H₁):</strong> What we are trying to prove. E.g., "The new model is fundamentally more accurate."</li>
        </ul>

        <div className="bg-slate-50 border border-slate-200 rounded-2xl p-6 my-6 shadow-sm">
          <h3 className="font-bold text-slate-900 mt-0">The P-Value</h3>
          <p>The p-value is the probability of observing the data we collected (or something more extreme) <strong>ASSUMING the Null Hypothesis is completely true.</strong></p>
          <p className="mb-0">If the p-value is tiny (typically &lt; 0.05, known as our alpha <code>α</code> threshold), it means our observation is so incredibly rare under the Null Hypothesis that we reject H₀ and accept H₁.</p>
        </div>

        <h2 className="text-3xl font-bold text-slate-900 mt-12 mb-4">2. The Z-Test vs The T-Test</h2>
        <p>
          To calculate the p-value for comparing the means of two groups, we calculate a test statistic that follows a known probability distribution.
        </p>
        
        <div className="grid md:grid-cols-2 gap-8 my-8">
          <div className="bg-slate-50 border border-slate-200 rounded-2xl p-6 shadow-sm">
            <h3 className="text-xl font-bold text-slate-900 mt-0 mb-2">The Z-Test</h3>
            <p className="text-sm text-slate-600 mb-4">Used when the population variance is perfectly known, or the sample size is extremely large (N &gt; 30).</p>
            <div className="bg-white p-3 rounded border border-slate-200 font-mono text-sm shadow-sm text-slate-800">
              Z = (x̄ - μ) / (σ / √n)
            </div>
          </div>

          <div className="bg-slate-50 border border-slate-200 rounded-2xl p-6 shadow-sm">
            <h3 className="text-xl font-bold text-slate-900 mt-0 mb-2">Student's T-Test</h3>
            <p className="text-sm text-slate-600 mb-4">Used when sample sizes are small or population variance is unknown. It uses a T-distribution (which has fatter tails to account for uncertainty).</p>
            <div className="bg-white p-3 rounded border border-slate-200 font-mono text-sm shadow-sm text-slate-800">
              t = (x̄₁ - x̄₂) / √( s₁²/n₁ + s₂²/n₂ )
            </div>
            <p className="text-xs text-slate-500 mt-2">Formula for Welch's T-Test (assuming unequal variances).</p>
          </div>
        </div>

        <h2 className="text-3xl font-bold text-slate-900 mt-12 mb-4">3. ANOVA (Analysis of Variance)</h2>
        <p>
          If you want to compare the means of <strong>three or more groups</strong> (e.g., A/B/C/D testing four different ML models), you cannot just run multiple T-Tests. Running 10 T-Tests at a 5% error rate inflates your actual probability of making a false positive error to roughly 40% (the Family-wise Error Rate).
        </p>
        <p>
          Instead, we use a single Omnibus test called <strong>ANOVA</strong>. It calculates the <strong>F-Statistic</strong> by comparing the variance <em>between</em> the groups to the variance <em>within</em> the groups.
        </p>
        
        <div className="bg-slate-50 border border-slate-200 rounded-2xl p-6 my-6 font-mono text-sm text-slate-800 shadow-sm overflow-x-auto">
          <p>F = Variance Between Groups / Variance Within Groups</p>
          <p className="text-slate-500 mt-2">If F is very large, it means the groups are spread far apart relative to their internal noise, leading to a tiny p-value.</p>
        </div>

        <div className="bg-white border border-slate-200 p-6 rounded-2xl my-8 shadow-lg not-prose">
            <h3 className="text-slate-900 font-bold mt-0 text-xl mb-4">Implementation: Scipy.Stats Testing</h3>
            <TerminalBlock 
                language="python" 
                filename="hypothesis_testing.py" 
                code={hypothesisCode}
            />
        </div>

        <InteractiveQuiz 
            question="You run an ANOVA test comparing the performance of Models A, B, and C. The p-value comes back as 0.001. What exactly does this prove?"
            options={[
                "It proves that Model A is statistically significantly different from Model C.",
                "It proves that all three models have entirely different performance means.",
                "It proves that the probability of the Null Hypothesis being true is exactly 0.1%.",
                "It proves that AT LEAST ONE model's mean is statistically significantly different from the others, but doesn't tell you which one."
            ]}
            correctIndex={3}
            explanation="ANOVA is an omnibus test. A significant p-value only rejects the null hypothesis (that all means are equal). It tells you that at least one group is different, but you must run post-hoc tests (like Tukey's HSD) to find out exactly WHICH groups differ."
        />
      </section>
    </div>
  );
}
