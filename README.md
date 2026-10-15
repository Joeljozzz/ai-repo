# 🧠 AI Engineer's Handbook

<div align="center">

[![Next.js](https://img.shields.io/badge/Next.js-16.4.0-black?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org/)
[![React](https://img.shields.io/badge/React-19.3.0-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-v4-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)
[![CI](https://github.com/Joeljozzz/ai-repo/actions/workflows/ci.yml/badge.svg)](https://github.com/Joeljozzz/ai-repo/actions/workflows/ci.yml)

**A Stanford-Grade, Full-Stack Interactive AI Curriculum & Engineering Reference**  
*From First-Principles Mathematical Foundations to Production-Grade Agent Swarms & RAG Architectures.*

[Explore Curriculum](#-curriculum-overview) • [Key Highlights](#-key-highlights) • [Interactive Components](#-interactive-learning-features) • [Getting Started](#-getting-started) • [Architecture](#-project-architecture)

</div>

---

## 🌟 Overview

The **AI Engineer's Handbook** is an interactive, content-dense engineering platform built to bridge the gap between academic research papers and production AI systems. Rather than high-level summaries or trivial tutorials, this repository presents rigorous mathematical proofs, architectural trade-offs, and complete, runnable Python/PyTorch implementations across **44 modules** and **6 core engineering tracks**.

Every topic is engineered from first principles:
- **Math is derived, not assumed** — complete step-by-step calculus, matrix differentials, and statistical variance proofs.
- **Implementations are production-ready** — pure NumPy and PyTorch implementations with vectorized operations, KV caching, rotary embeddings (RoPE), and gradient clipping.
- **Interactive UI with macOS aesthetic** — responsive sidebar navigation, real-time Recharts training dashboards, code terminals, concept cards, and self-assessment quizzes.

---

## ⚡ Key Highlights

- **44 In-Depth Curriculum Modules**: Spanning 6 progressive tracks from statistical foundations to autonomous agent swarms.
- **Rigorous Mathematical First-Principles**:
  - Exact derivation of backpropagation via multivariate chain rule and Jacobian matrices.
  - Variance scaling proof for Scaled Dot-Product Attention ($\frac{1}{\sqrt{d_k}}$).
  - Analytical proof of Adam's bias correction factors ($\frac{1}{1 - \beta^t}$) and AdamW decoupled weight decay.
  - Rotary Position Embeddings (RoPE) complex plane orthogonal transformations.
  - Pre-LayerNorm vs. Post-LayerNorm gradient stability analysis.
- **Production-Grade Implementations**:
  - Full Multi-Head, Multi-Query (MQA), and Grouped-Query Attention (GQA) mechanisms.
  - Decoder blocks with RMSNorm, SwiGLU activations, and KV-cache matrix management.
  - Recursive semantic chunkers with token-aware sliding window overlap.
  - Cyclic state machines using LangGraph StateGraph with human-in-the-loop checkpoints.
  - Multi-agent supervisor topologies, AutoGen conversation orchestrators, and MemGPT tiered virtual context paging.
- **Interactive Educational Engine**:
  - Real-time training and validation loss curve visualizer with epoch step simulator.
  - Multi-framework code syntax highlighting.
  - Interactive self-assessment quizzes with rationale breakdowns.

---

## 📚 Curriculum Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        AI ENGINEER'S CURRICULUM                            │
├───────────────┬───────────────┬────────────────┬───────────────┬───────────┤
│ 1. Foundations│ 2. Classical  │ 3. Deep        │ 4. Generative │ 5. System │
│    & Math     │    ML Core    │    Learning    │    AI & LLMs  │    & RAG  │
├───────────────┼───────────────┼────────────────┼───────────────┼───────────┤
│ • Lin. Algebra│ • Regressions │ • Neural Nets  │ • LLM Scaling │ • RAG Core│
│ • Calculus    │ • Trees & RF  │ • Backprop     │ • Embeddings  │ • Advanced│
│ • Probability │ • Boosting    │ • Optimizers   │ • Prompts     │ • Vectors │
│ • Statistics  │ • SVM & Bayes │ • Attention    │ • Fine-Tuning │ • RAG Eval│
│ • Hyp. Tests  │ • Clustering  │ • Transformers │ • PEFT / LoRA │ • Triad   │
└───────────────┴───────────────┴────────────────┴───────────────┴───────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │   6. Autonomous Agent Stack   │
                       ├───────────────────────────────┤
                       │ • ReAct Cognitive Loops       │
                       │ • Memory & MemGPT Paging      │
                       │ • LangChain & LCEL Runnables  │
                       │ • LangGraph Cyclic Graphs     │
                       │ • Multi-Agent Swarm Topologies│
                       │ • Production Guardrails & Eval│
                       └───────────────────────────────┘
```

### Track 1: Mathematical Foundations (`/course/math`)
| Module | Focus Areas | Key Formulas & Implementations |
| :--- | :--- | :--- |
| **Linear Algebra** | Vector spaces, dot products, matrix decomposition, eigenvalues | SVD, Eigendecomposition, Orthogonal projection |
| **Calculus** | Multivariable calculus, Jacobians, Hessians, Taylor series | Gradient vectors, directional derivatives, convexity |
| **Probability Theory** | Sample spaces, conditional probability, Bayes' theorem, independence | Bayes rule, Law of Total Probability, expectation |
| **Probability Distributions** | Normal, Bernoulli, Binomial, Poisson, Gaussian mixtures | PDF/CDF formulations, MLE parameter estimation |
| **Hypothesis Testing** | Null hypothesis, p-values, t-tests, ANOVA, Chi-Square | Two-tailed t-test, rejection regions, effect size |
| **Statistical Errors & Power** | Type I/II errors, significance ($\alpha$), statistical power ($1-\beta$) | Power curves, sample size estimation, confusion matrix |
| **Descriptive Statistics** | Central tendency, dispersion, skewness, kurtosis, correlation | Pearson $r$, Spearman $\rho$, IQR, variance |

### Track 2: Classical Machine Learning (`/course/ml-core`)
| Module | Focus Areas | Key Algorithms |
| :--- | :--- | :--- |
| **Learning Paradigms** | Supervised, unsupervised, semi-supervised, reinforcement learning | Inductive bias, PAC learning, empirical risk minimization |
| **Data Preprocessing** | Imputation, feature scaling, one-hot encoding, outlier handling | Standard/MinMax scalers, RobustScaler, power transforms |
| **Linear Regression** | Ordinary Least Squares (OLS), Ridge ($L_2$), Lasso ($L_1$), ElasticNet | Normal equation $(X^TX)^{-1}X^Ty$, subgradient descent |
| **Logistic Regression** | Sigmoid function, log-loss cross-entropy, decision boundaries | Maximum Likelihood Estimation, Newton-Raphson optimization |
| **Decision Trees** | Binary recursive splitting, CART algorithm, pruning | Gini impurity, Shannon entropy, information gain ratio |
| **Random Forests** | Bagging (Bootstrap Aggregation), feature subsampling, OOB error | Out-of-bag variance reduction, feature importance MDI |
| **Boosting Algorithms** | Sequential residual fitting, AdaBoost exponential loss | Exponential loss minimization, sample re-weighting |
| **XGBoost Deep Dive** | Second-order Taylor expansion loss, regularization ($\gamma, \lambda$) | Exact greedy & histogram-based split finding |
| **Support Vector Machines** | Maximum margin hyperplanes, soft margin ($C$), kernel trick | Dual Lagrangian formulation, RBF & polynomial kernels |
| **Naive Bayes** | Conditional independence assumption, Laplace smoothing | Gaussian NB, Multinomial NB, text classification |
| **K-Nearest Neighbors** | Instance-based learning, distance metrics ($L_1, L_2$, Minkowski) | KD-Tree and Ball-Tree search complexity |
| **Clustering Algorithms** | K-Means++, Hierarchical Ward, DBSCAN density reachability | Lloyd's algorithm, silhouette score, Davies-Bouldin index |
| **Dimensionality Reduction** | Principal Component Analysis (PCA), t-SNE, UMAP | Covariance matrix eigendecomposition, KL-divergence |
| **Model Evaluation** | Precision, Recall, F1, ROC-AUC, PR-AUC, Cross-Validation | K-Fold stratified cross-validation, confusion metrics |

### Track 3: Deep Learning Architectures (`/course/deep-learning`)
| Module | Focus Areas | Production Code & Proofs |
| :--- | :--- | :--- |
| **Neural Networks** | Perceptrons, MLPs, universal approximation theorem | Forward propagation, vectorized matrix multiplications |
| **Backpropagation** | Computational graphs, automatic differentiation, chain rule | Jacobian-vector products, analytical backward passes |
| **Optimization** | SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam, AdamW | Decoupled weight decay proofs, analytical bias correction |
| **Regularization** | Dropout, Batch Normalization, Layer Normalization, Weight Decay | Inverted dropout scaling, forward/backward batch norm equations |
| **CNNs** | Convolution kernels, padding, stride, pooling, receptive fields | Feature map spatial calculations, ResNet skip connections |
| **RNNs & LSTMs** | Hidden state recurrence, vanishing gradients, gating mechanisms | LSTM cell equations (input, forget, output gates, cell state) |
| **Attention Mechanisms** | Bahdanau additive attention, Luong multiplicative, Scaled Dot-Product | Softmax variance preservation proof, MHA vs. MQA vs. GQA |
| **Transformers** | Encoder-decoder stack, RoPE, RMSNorm, SwiGLU, KV caching | Complete causal self-attention decoder block in PyTorch |

### Track 4: Generative AI & Large Language Models (`/course/genai`)
| Module | Focus Areas | Key Topics |
| :--- | :--- | :--- |
| **LLM Foundations** | Autoregressive decoding, tokenization (BPE, WordPiece), scaling laws | Chinchilla compute-optimal scaling, temperature & top-p |
| **Vector Embeddings** | Word2Vec, contextual embeddings, bi-encoders, contrastive loss | InfoNCE loss formulation, cosine similarity, MTEB benchmarks |
| **Prompt Engineering** | Zero-shot, Few-shot, Chain-of-Thought (CoT), Tree-of-Thoughts | DSPy programmatic optimization, system prompt steering |
| **Fine-Tuning & PEFT** | Full fine-tuning vs. LoRA, QLoRA, Prefix Tuning, DPO/RLHF | Low-Rank decomposition $W = W_0 + B \cdot A$, 4-bit NormalFloat |

### Track 5: AI Systems & RAG Engineering (`/course/ai-systems`)
| Module | Focus Areas | System Implementations |
| :--- | :--- | :--- |
| **RAG Fundamentals** | Document ingestion, recursive chunking, embedding generation | Token-aware sliding chunker with metadata retention |
| **Advanced RAG** | HyDE (Hypothetical Document Embeddings), Parent-Child, Cross-encoders | Reciprocal Rank Fusion (RRF), re-ranking architectures |
| **Vector Databases** | HNSW graphs, IVFFlat, scalar quantization, product quantization | Vector indexing tradeoffs (latency vs. recall vs. memory) |
| **RAG Evaluation** | RAG Triad: Context Relevance, Groundedness, Answer Relevance | Automated LLM-as-a-judge evaluation harness |

### Track 6: Autonomous Agent Systems (`/course/agents`)
| Module | Focus Areas | Production Patterns |
| :--- | :--- | :--- |
| **Agent Fundamentals** | Autonomy taxonomy (Levels 1–4), perception-planning-action loop | Goal decomposition, tool registration, execution protocols |
| **ReAct Pattern** | Synergizing Reasoning + Acting in language models | Thought-Action-Observation cognitive loop implementation |
| **Agent Memory** | Short-term buffer, summary memory, vector memory, MemGPT paging | Stanford generative agents retrieval decay formula |
| **LangChain Deep Dive** | LangChain Expression Language (LCEL), Runnables, LangSmith | Streaming runnables, fallback handling, pipe composition |
| **LangGraph Systems** | Cyclic computation graphs, state schemas, conditional edges | Human-in-the-loop checkpoints, persistent state reducers |
| **Multi-Agent Swarms** | Supervisor pattern, peer-to-peer swarms, hierarchical teams | AutoGen conversation orchestrators, consensus routing |
| **Production Agents** | Guardrails, rate limiting, deterministic fallbacks, evals | Sandboxed code interpreters, latency optimization, observability |

### Track 7: Machine Learning for Beginners — 12-Week Masterclass (`/course/ml-beginners`)
*Converted from Microsoft's 12-week, 26-lesson, 52-quiz curriculum into a full interactive web experience.*

| Week | Focus Area | Lessons Included | Interactive Quizzes |
| :--- | :--- | :--- | :--- |
| **Week 1** | **Introduction & History** | 01. Intro to ML · 02. History of ML | Quizzes #1, #2, #3, #4 |
| **Week 2** | **Fairness & ML Techniques** | 03. Fairness in ML · 04. Techniques of ML | Quizzes #5, #6, #7, #8 |
| **Week 3** | **Tools & Data Preparation** | 05. Tools of the Trade · 06. Data Prep & EDA | Quizzes #9, #10, #11, #12 |
| **Week 4** | **Linear & Logistic Regression** | 07. Linear & Polynomial · 08. Logistic Regression | Quizzes #13, #14, #15, #16 |
| **Week 5** | **Web App Deployment** | 09. Web App Deployment with Flask & ONNX | Quizzes #17, #18 |
| **Week 6** | **Classification Part 1** | 10. Intro to Classification · 11. Linear SVC & K-NN | Quizzes #19, #20, #21, #22 |
| **Week 7** | **Classification Part 2 & Diagnostics** | 12. Decision Trees & Ensembles · 13. Diagnostics & ROC | Quizzes #23, #24, #25, #26 |
| **Week 8** | **Clustering & K-Means** | 14. Clustering Intro · 15. K-Means in Practice | Quizzes #27, #28, #29, #30 |
| **Week 9** | **NLP Foundations & Tasks** | 16. Intro to NLP · 17. POS Tagging, NER & TF-IDF | Quizzes #31, #32, #33, #34 |
| **Week 10** | **Sentiment Analysis & Hotel Reviews** | 18. VADER Sentiment · 19. Reviews NLP 1 · 20. Reviews NLP 2 | Quizzes #35, #36, #37, #38, #39, #40 |
| **Week 11** | **Time Series & ARIMA Forecasting** | 21. Time Series Intro · 22. ARIMA & SARIMAX · 23. SVR | Quizzes #41, #42, #43, #44, #51, #52 |
| **Week 12** | **Reinforcement Learning & Real-World ML** | 24. Q-Learning · 25. Gym CartPole · 26. Real-World Debugging | Quizzes #45, #46, #47, #48, #49, #50 |

---

## 💻 Interactive Learning Features

```text
ai-repo/src/components/
├── AlgorithmBlock.tsx         # Interactive algorithm step walkthroughs
├── ComparisonTable.tsx        # High-density feature & complexity matrix
├── ConceptCard.tsx            # First-principles conceptual takeaways
├── InteractiveQuiz.tsx        # Self-assessment module quizzes with immediate feedback
├── InteractiveQuizSuite.tsx   # 52-Quiz engine with question steppers, scorecards & persistence
├── LearningCurveDashboard.tsx # Recharts-powered real-time loss & accuracy simulator
├── MathBlock.tsx              # LaTeX mathematical derivation containers
└── TerminalBlock.tsx          # Multi-tab runnable code playground
```

### Visual Highlights
- **Real-Time Learning Curve Dashboard**: Simulates training convergence, validation divergence, and overfitting metrics using Recharts.
- **Mac-Inspired Native UI**: Clean macOS title bar with responsive multi-track sidebar navigation.
- **Dynamic Syntax Highlighter**: High-contrast dark terminal themes for NumPy, PyTorch, and Python code blocks.

---

## 🚀 Getting Started

### Prerequisites

- **Node.js**: `v20.x` or later
- **npm**: `v10.x` or later (or `pnpm` / `yarn`)

### Installation & Local Run

```bash
# 1. Clone the repository
git clone https://github.com/Joeljozzz/ai-repo.git

# 2. Navigate to project root
cd ai-repo

# 3. Install dependencies
npm install

# 4. Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 🛠️ Project Architecture

```text
ai-repo/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated GitHub Actions lint & build workflow
├── public/                      # Static assets & icons
├── src/
│   ├── app/
│   │   ├── course/
│   │   │   ├── agents/          # 7 Agentic AI modules
│   │   │   ├── ai-systems/      # 4 RAG & Vector DB modules
│   │   │   ├── deep-learning/   # 8 Deep Learning modules
│   │   │   ├── genai/           # 4 GenAI & LLM modules
│   │   │   ├── math/            # 7 Math Foundations modules
│   │   │   └── ml-core/         # 14 Classical ML modules
│   │   ├── globals.css          # Tailwind CSS v4 styling rules
│   │   ├── layout.tsx           # Responsive layout with dynamic icon sidebar
│   │   └── page.tsx             # Interactive curriculum landing dashboard
│   ├── components/              # Modular UI & interactive widget library
│   └── data/
│       └── curriculum.ts        # Strongly-typed curriculum metadata schema
├── eslint.config.mjs            # Flat ESLint configuration
├── next.config.ts               # Next.js 16 configuration with Turbopack
├── package.json                 # Project dependencies & scripts
├── tsconfig.json                # TypeScript configuration
└── LICENSE                      # MIT License
```

---

## 🧪 Available Scripts

| Command | Action |
| :--- | :--- |
| `npm run dev` | Starts local Next.js development server with Turbopack |
| `npm run build` | Compiles TypeScript and builds production-optimized static site |
| `npm run start` | Runs the compiled production build locally |
| `npm run lint` | Runs ESLint across all source files |

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more details.

---

<div align="center">

Crafted with first-principles rigor by [Joel Jose](https://github.com/Joeljozzz).

</div>
