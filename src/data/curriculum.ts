export type Module = {
  id: string;
  title: string;
  desc: string;
};

export type Stack = {
  id: string;
  label: string;
  title: string;
  description: string;
  icon: string;
  color: string;
  modules: Module[];
};

export const curriculum: Stack[] = [
  // ─────────────────────────────────────────────
  // TRACK 0: Mathematical Foundations
  // ─────────────────────────────────────────────
  {
    id: "math",
    label: "Track 0",
    title: "Mathematical Foundations",
    description: "The language of AI: linear algebra, calculus, probability, and optimization.",
    icon: "Calculator",
    color: "sky",
    modules: [
      { id: "descriptive-stats",   title: "Descriptive Statistics",     desc: "Moments, mean, median, mode, variance, skewness, kurtosis, IQR, boxplots." },
      { id: "probability",         title: "Probability Theory",         desc: "Sample spaces, Bayes theorem, conditional probability, PDFs, CDFs, PMFs." },
      { id: "distributions",       title: "Probability Distributions",  desc: "Normal, Binomial, Poisson, Beta, Gamma — derivations and real-world use cases." },
      { id: "linear-algebra",      title: "Linear Algebra for ML",     desc: "Vectors, matrices, dot products, eigenvalues, eigenvectors, SVD." },
      { id: "calculus",            title: "Calculus & Optimization",    desc: "Partial derivatives, gradients, chain rule, Jacobians, Hessians, convexity." },
      { id: "hypothesis-testing",  title: "Hypothesis Testing",         desc: "Z-test, T-test, ANOVA, Chi-Square, p-values, confidence intervals." },
      { id: "statistical-errors",  title: "Statistical Errors & Power", desc: "Type I/II errors, statistical power, effect size, multiple comparisons." },
    ],
  },

  // ─────────────────────────────────────────────
  // TRACK 1: Classical Machine Learning
  // ─────────────────────────────────────────────
  {
    id: "ml-core",
    label: "Track 1",
    title: "Classical Machine Learning",
    description: "Supervised, unsupervised learning, and the statistical models that power them.",
    icon: "GitBranch",
    color: "emerald",
    modules: [
      { id: "learning-paradigms",   title: "Learning Paradigms",         desc: "Supervised, Unsupervised, Semi-Supervised, and Reinforcement Learning — mathematically defined." },
      { id: "data-preprocessing",   title: "Data Preprocessing",         desc: "Normalization, standardization, encoding, missing value imputation, feature engineering." },
      { id: "model-evaluation",     title: "Model Evaluation & Metrics", desc: "Cross-validation, confusion matrix, precision, recall, F1, ROC-AUC, RMSE, log-loss from first principles." },
      { id: "linear-regression",    title: "Linear Regression",          desc: "OLS derivation, Normal Equation, gradient descent, regularization (Ridge, Lasso)." },
      { id: "logistic-regression",  title: "Logistic Regression",        desc: "Sigmoid function, MLE derivation, binary cross-entropy, multi-class (Softmax)." },
      { id: "knn",                   title: "K-Nearest Neighbors",        desc: "Minkowski distance, curse of dimensionality, KD-trees, computational complexity." },
      { id: "naive-bayes",          title: "Naive Bayes Classifiers",    desc: "Bayes theorem, Gaussian NB, Multinomial NB, Laplace smoothing." },
      { id: "svm",                   title: "Support Vector Machines",    desc: "Margin maximization, Lagrange multipliers, Mercer's theorem, RBF kernel, C and gamma." },
      { id: "decision-trees",       title: "Decision Trees",             desc: "CART, entropy, Gini impurity, information gain, pruning, depth control." },
      { id: "random-forests",       title: "Random Forests & Bagging",   desc: "Bootstrap aggregation, feature subsampling, out-of-bag error, feature importance." },
      { id: "boosting",             title: "Boosting: AdaBoost & GBM",   desc: "Additive modeling, weak learners, residual fitting, shrinkage, functional gradient descent." },
      { id: "xgboost",              title: "XGBoost & LightGBM",         desc: "Taylor expansion objective, Hessian weighting, histogram splits, LightGBM GOSS/EFB." },
      { id: "clustering",           title: "Clustering Algorithms",      desc: "K-Means (Lloyd's), DBSCAN, Hierarchical clustering, evaluation (silhouette, Davies-Bouldin)." },
      { id: "dim-reduction",        title: "Dimensionality Reduction",   desc: "PCA (eigendecomposition), t-SNE (KL divergence), UMAP (manifold learning), LDA." },
    ],
  },

  // ─────────────────────────────────────────────
  // TRACK 2: Deep Learning
  // ─────────────────────────────────────────────
  {
    id: "deep-learning",
    label: "Track 2",
    title: "Deep Learning",
    description: "Neural architectures from perceptrons to Transformers, and the calculus behind them.",
    icon: "Network",
    color: "purple",
    modules: [
      { id: "neural-networks",  title: "Neural Networks (MLP)",       desc: "Universal approximation, forward pass, activation functions (ReLU, sigmoid, tanh, GELU)." },
      { id: "backpropagation",  title: "Backpropagation & Calculus",  desc: "Chain rule derivation, computational graphs, vanishing/exploding gradients." },
      { id: "optimizers",       title: "Optimizers",                  desc: "SGD, Momentum, RMSProp, Adam, AdaGrad — update rules, convergence, learning rate schedules." },
      { id: "regularization",   title: "Regularization Techniques",   desc: "L1/L2, Dropout (inverted), Batch Normalization, Layer Normalization, Early Stopping." },
      { id: "cnn",              title: "Convolutional Networks (CNN)", desc: "Cross-correlation, feature maps, pooling, stride, padding, ResNets, EfficientNet." },
      { id: "rnn-lstm",         title: "Recurrent Networks & LSTMs",  desc: "Backprop through time (BPTT), vanishing gradients, LSTM gating math, GRUs." },
      { id: "attention",        title: "Attention Mechanisms",        desc: "Seq2Seq, alignment scores, scaled dot-product attention, multi-head attention." },
      { id: "transformers",     title: "The Transformer Architecture","desc": "Encoder-decoder, positional encodings, masked attention, BERT vs GPT." },
    ],
  },

  // ─────────────────────────────────────────────
  // TRACK 3: Generative AI & LLMs
  // ─────────────────────────────────────────────
  {
    id: "genai",
    label: "Track 3",
    title: "Generative AI & LLMs",
    description: "Large language models, fine-tuning, and the science of prompting.",
    icon: "MessageSquare",
    color: "indigo",
    modules: [
      { id: "llm-foundations",   title: "LLM Foundations",          desc: "Tokenization (BPE), scaling laws (Chinchilla), emergent capabilities, RLHF." },
      { id: "prompt-engineering","title": "Prompt Engineering",      desc: "Zero-shot, few-shot, Chain-of-Thought, Self-Consistency, Tree-of-Thought." },
      { id: "fine-tuning",       title: "Fine-Tuning & PEFT",        desc: "Full fine-tuning, LoRA (rank decomposition), QLoRA (4-bit), instruction tuning." },
      { id: "embeddings",        title: "Embeddings & Vector Spaces","desc": "Dense embeddings, cosine similarity, contrastive learning, Matryoshka embeddings." },
    ],
  },

  // ─────────────────────────────────────────────
  // TRACK 4: AI Systems & RAG
  // ─────────────────────────────────────────────
  {
    id: "ai-systems",
    label: "Track 4",
    title: "AI Systems & RAG Engineering",
    description: "Building production retrieval-augmented generation pipelines.",
    icon: "Database",
    color: "teal",
    modules: [
      { id: "rag-fundamentals",   title: "RAG Fundamentals",        desc: "Architecture, chunking strategies, embedding models, retrieval mechanisms." },
      { id: "vector-databases",   title: "Vector Databases",        desc: "HNSW indexing, Pinecone, Qdrant, Weaviate, FAISS — internals and tradeoffs." },
      { id: "advanced-rag",       title: "Advanced RAG Patterns",   desc: "Hybrid search (BM25 + dense), re-ranking, query expansion, HyDE, contextual compression." },
      { id: "evaluation-rag",     title: "Evaluating RAG Systems",  desc: "RAGAS framework, faithfulness, context precision, answer relevancy, trulens." },
    ],
  },

  // ─────────────────────────────────────────────
  // TRACK 5: Agentic AI
  // ─────────────────────────────────────────────
  {
    id: "agents",
    label: "Track 5",
    title: "Agentic AI & Orchestration",
    description: "Autonomous agent architectures, tool use, and multi-agent systems.",
    icon: "Cpu",
    color: "rose",
    modules: [
      { id: "agent-fundamentals",  title: "Agent Fundamentals",       desc: "What is an AI agent? Perception, memory, planning, action — the cognitive loop." },
      { id: "react-pattern",       title: "ReAct & Tool Calling",     desc: "Reason + Act pattern, JSON function schemas, structured outputs, error recovery." },
      { id: "memory-systems",      title: "Memory Systems",           desc: "Short-term (context), long-term (vector store), episodic vs semantic memory." },
      { id: "langchain",           title: "LangChain Deep Dive",      desc: "LCEL (LangChain Expression Language), chains, runnables, callbacks, tracing." },
      { id: "langgraph",           title: "LangGraph & Cyclic Agents","desc": "StateGraph, conditional edges, human-in-the-loop, time travel, checkpointing." },
      { id: "multi-agent",         title: "Multi-Agent Orchestration","desc": "Supervisor pattern, worker agents, message passing, AutoGen, CrewAI." },
      { id: "production-agents",   title: "Production AI Systems",    desc: "Observability (LangSmith), guardrails, rate limiting, cost control, deployment." },
    ],
  },

  // ─────────────────────────────────────────────
  // TRACK 6: ML for Beginners (12 Weeks · 26 Lessons)
  // ─────────────────────────────────────────────
  {
    id: "ml-beginners",
    label: "Track 6",
    title: "ML for Beginners (12 Weeks)",
    description: "Classic 12-week, 26-lesson, 52-quiz Machine Learning masterclass with interactive assessments.",
    icon: "GraduationCap",
    color: "amber",
    modules: [
      { id: "intro-to-ml",            title: "01. Intro to Machine Learning", desc: "Traditional rules vs learned models, Mitchell's formulation." },
      { id: "history-of-ml",          title: "02. History of Machine Learning", desc: "From Turing & Perceptrons to deep learning breakthroughs." },
      { id: "fairness-in-ml",         title: "03. Fairness & Responsible AI", desc: "Demographic parity, equal opportunity, disparate impact." },
      { id: "techniques-of-ml",       title: "04. Techniques of ML", desc: "Supervised, unsupervised, reinforcement, bias-variance tradeoff." },
      { id: "regression-tools",       title: "05. Tools of the Trade", desc: "Python, Scikit-learn, Pandas, reproducible pipelines." },
      { id: "regression-eda",         title: "06. Data Prep & EDA", desc: "US pumpkin prices, unit conversion, correlation analysis." },
      { id: "linear-regression-deep", title: "07. Linear & Polynomial", desc: "OLS derivations, normal equation, polynomial curves." },
      { id: "logistic-regression-deep",title: "08. Logistic Regression", desc: "Sigmoid mapping, log-odds, binary cross-entropy." },
      { id: "web-app-deployment",     title: "09. Web App Deployment", desc: "Flask model serving, ONNX serialization, client-side scoring." },
      { id: "intro-classification",   title: "10. Intro to Classification", desc: "Cuisine recipe dataset, multi-class strategies." },
      { id: "classifiers-1",          title: "11. Linear SVC & K-NN", desc: "Margin maximization hyperplanes, instance voting." },
      { id: "classifiers-2",          title: "12. Trees & Ensembles", desc: "Gini impurity, Random Forests, AdaBoost ensembles." },
      { id: "applied-classification", title: "13. Applied Diagnostics", desc: "Confusion matrix, ROC-AUC curves, Precision-Recall." },
      { id: "intro-clustering",       title: "14. Intro to Clustering", desc: "Unsupervised grouping, Spotify audio feature scaling." },
      { id: "kmeans-clustering",      title: "15. K-Means in Practice", desc: "Centroid updates, Elbow method, silhouette analysis." },
      { id: "intro-nlp",              title: "16. Intro to NLP", desc: "Tokenization, stop words, stemming vs lemmatization." },
      { id: "nlp-tasks",              title: "17. NLP Tasks & TF-IDF", desc: "POS tagging, NER, TF-IDF document similarity." },
      { id: "translation-sentiment",  title: "18. Translation & Sentiment", desc: "VADER rule-based sentiment, polarity vs subjectivity." },
      { id: "hotel-reviews-1",        title: "19. Hotel Reviews NLP 1", desc: "Cleaning European hotel reviews, noise removal." },
      { id: "hotel-reviews-2",        title: "20. Hotel Reviews NLP 2", desc: "Supervised sentiment models, n-gram vectorization." },
      { id: "intro-time-series",      title: "21. Intro to Time Series", desc: "Seasonality, trends, stationarity, ADF test." },
      { id: "arima-forecasting",      title: "22. ARIMA & SARIMAX", desc: "Autoregressive moving average, AIC minimization." },
      { id: "svr-time-series",        title: "23. SVR for Time Series", desc: "Sliding lag windows, non-linear kernel regression." },
      { id: "intro-reinforcement",    title: "24. Reinforcement & Q-Learning", desc: "MDPs, Bellman optimality, epsilon-greedy tabular updates." },
      { id: "gym-environments",       title: "25. Gym & CartPole Control", desc: "Gymnasium API, state discretization, pendulum balance." },
      { id: "real-world-debugging",   title: "26. Real-World Debugging", desc: "Data drift, concept drift, Population Stability Index (PSI)." },
    ],
  },
];
