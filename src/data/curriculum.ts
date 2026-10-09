export type Module = {
  id: string;
  title: string;
  desc: string;
};

export type Stack = {
  id: string;
  title: string;
  description: string;
  icon: string;
  color: string;
  modules: Module[];
};

export const curriculum: Stack[] = [
  {
    id: "stats",
    title: "Stack 0: Mathematics & Statistics",
    description: "The rigorous mathematical foundation required for PhD-level AI research.",
    icon: "Calculator",
    color: "blue",
    modules: [
      { id: "descriptive", title: "0.1 Descriptive Statistics", desc: "Moments of a distribution: Mean, Median, Mode, Variance, Kurtosis, Skewness, and Box Plot analytics." },
      { id: "distributions", title: "0.2 Probability Distributions", desc: "PDFs and PMFs. Normal, Binomial, Poisson, Gamma, and Beta distributions." },
      { id: "hypothesis", title: "0.3 Hypothesis Testing", desc: "Z-Test, T-Test (Student's), ANOVA (Analysis of Variance), and Chi-Square." },
      { id: "errors", title: "0.4 Statistical Errors", desc: "Type I (False Positive), Type II (False Negative), Statistical Power, and p-value mathematics." }
    ]
  },
  {
    id: "ml-foundations",
    title: "Stack 1: ML Fundamentals",
    description: "Core paradigms and problem formulations in machine learning.",
    icon: "Target",
    color: "emerald",
    modules: [
      { id: "paradigms", title: "1.1 Learning Paradigms", desc: "Mathematical distinctions between Supervised, Unsupervised, Semi-Supervised, and Reinforcement Learning." },
      { id: "problem-types", title: "1.2 Problem Formulations", desc: "Classification (Discrete space) vs Regression (Continuous space) vs Clustering." },
      { id: "evaluation", title: "1.3 Evaluation & Metrics", desc: "Confusion Matrix derivation, F1-Score (Harmonic Mean), ROC-AUC space, and Log-Loss bounds." }
    ]
  },
  {
    id: "ml",
    title: "Stack 2: Classical Machine Learning",
    description: "Deep dive into statistical modeling, geometric boundaries, and algorithms.",
    icon: "BookOpen",
    color: "teal",
    modules: [
      { id: "linear-models", title: "2.1 Linear & Distance Models", desc: "OLS derivations, Maximum Likelihood Estimation in Logistic Regression, Minkowski distance in KNN." },
      { id: "svm", title: "2.2 Support Vector Machines", desc: "Lagrange Multipliers, Margin Maximization, Mercer's Theorem, and the Kernel Trick (RBF/Poly)." },
      { id: "trees", title: "2.3 Decision Trees", desc: "Information Theory: Shannon Entropy, Gini Impurity, and CART split optimization." },
      { id: "ensembles", title: "2.4 Bagging & Boosting", desc: "Bias-Variance Tradeoff. Random Forests (Bootstrap Aggregation), AdaBoost, and XGBoost (Gradient Boosting)." }
    ]
  },
  {
    id: "dl",
    title: "Stack 3: Deep Learning",
    description: "Neural Networks, representation learning, and high-dimensional calculus.",
    icon: "Network",
    color: "purple",
    modules: [
      { id: "foundations", title: "3.1 MLP & Calculus", desc: "Universal Approximation Theorem, Chain Rule in Backpropagation, and Activation Jacobians." },
      { id: "cnn", title: "3.2 Computer Vision (CNN)", desc: "Cross-Correlation operators, Spatial Invariance, Pooling, and ResNet architectures." },
      { id: "rnn", title: "3.3 Sequence Modeling", desc: "BPTT (Backprop Through Time), Vanishing Gradients, and LSTM/GRU Gating Mathematics." },
      { id: "attention", title: "3.4 Attention Mechanisms", desc: "Seq2Seq models, Query-Key-Value matrices, and scaled dot-product attention." }
    ]
  },
  {
    id: "llm",
    title: "Stack 4: Generative AI & LLMs",
    description: "Modern Large Language Models, architectures, and fine-tuning.",
    icon: "MessageSquare",
    color: "indigo",
    modules: [
      { id: "transformers", title: "4.1 Transformers Architecture", desc: "Multi-Head Attention, Positional Encodings, and Masked Language Modeling." },
      { id: "fine-tuning", title: "4.2 PEFT & LoRA", desc: "Low-Rank Adaptation, QLoRA, and gradient updates in frozen networks." },
      { id: "rag", title: "4.3 Advanced RAG", desc: "High-dimensional Vector Databases, HNSW indexing, and Cross-Encoder Re-ranking." },
      { id: "agents", title: "4.4 Agentic Frameworks", desc: "ReAct loop reasoning, Tool/Function Calling schemas, and LangChain." },
      { id: "multi-agent", title: "4.5 Multi-Agent Systems", desc: "Cyclic directed graphs, LangGraph, and specialized autonomous agents." }
    ]
  }
];
