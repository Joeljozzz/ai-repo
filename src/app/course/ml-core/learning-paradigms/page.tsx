"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

export default function Page() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      {/* ── Track Badge ── */}
      <div className="not-prose mb-4">
        <span className="inline-block rounded-full bg-emerald-100 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-emerald-700">
          Track 1 · Classical Machine Learning
        </span>
      </div>

      <h1>Learning Paradigms in Machine Learning</h1>

      <p>
        Machine learning is not a single algorithm — it is a <em>collection of
        paradigms</em>, each with a different assumption about what information
        is available at training time and what the learning objective is. A
        practitioner who cannot distinguish a supervised task from a
        self-supervised one will misapply methods and waste enormous effort.
        This lecture maps the full landscape: from Mitchell's formal definition
        of learning through supervised, unsupervised, semi-supervised,
        self-supervised, and reinforcement learning, culminating in the two
        theoretical frameworks that unify them all — the No Free Lunch theorem
        and the Bias-Variance tradeoff.
      </p>

      {/* ── Learning Objectives ── */}
      <h2>Learning Objectives</h2>
      <ul>
        <li>
          State Tom Mitchell's formal definition of machine learning and map the
          three components T, P, E to a concrete real-world system.
        </li>
        <li>
          Distinguish supervised classification from regression, and identify
          which loss functions belong to each.
        </li>
        <li>
          Explain what structure unsupervised learning seeks and name three
          algorithmic families that exploit it.
        </li>
        <li>
          Describe how self-supervised pretext tasks (masked language modelling,
          next-token prediction, contrastive pairs) create free supervision
          signals at internet scale.
        </li>
        <li>
          Define the reinforcement learning interaction loop — agent, state,
          action, reward, policy — and write out the Bellman equation from
          memory.
        </li>
        <li>
          Apply the No Free Lunch theorem to justify why model selection requires
          domain knowledge and understand why the Bias-Variance decomposition is
          the central diagnostic across every paradigm.
        </li>
      </ul>

      {/* ══════════════════════════════════════════ */}
      <h2>1 · What Is Machine Learning? Mitchell's Formal Definition</h2>

      <p>
        Tom Mitchell's 1997 textbook gives the most cited formal definition of
        machine learning:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 leading-relaxed">
        <span className="text-emerald-400">Definition (Mitchell 1997)</span>
        {"\n\n"}
        A computer program is said to <span className="text-yellow-300">learn</span> from{"\n"}
        {"  "}experience <span className="text-sky-300">E</span> with respect to some class of{"\n"}
        {"  "}tasks <span className="text-sky-300">T</span> and performance measure <span className="text-sky-300">P</span>,{"\n\n"}
        if its performance at tasks in <span className="text-sky-300">T</span>, as measured by <span className="text-sky-300">P</span>,{"\n"}
        improves with experience <span className="text-sky-300">E</span>.
      </div>

      <p>
        Each of the three primitives maps directly to engineering choices:
      </p>

      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full border border-slate-200 text-sm">
          <thead className="bg-slate-100 text-slate-700 uppercase text-xs tracking-wide">
            <tr>
              <th className="px-4 py-3 text-left border-b">Symbol</th>
              <th className="px-4 py-3 text-left border-b">Name</th>
              <th className="px-4 py-3 text-left border-b">Email-spam Example</th>
              <th className="px-4 py-3 text-left border-b">Self-driving Example</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr>
              <td className="px-4 py-3 font-mono font-bold text-sky-700">T</td>
              <td className="px-4 py-3">Task</td>
              <td className="px-4 py-3">Classify email as spam/ham</td>
              <td className="px-4 py-3">Choose steering angle</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3 font-mono font-bold text-sky-700">P</td>
              <td className="px-4 py-3">Performance measure</td>
              <td className="px-4 py-3">Precision, recall, F1</td>
              <td className="px-4 py-3">Miles driven without human override</td>
            </tr>
            <tr>
              <td className="px-4 py-3 font-mono font-bold text-sky-700">E</td>
              <td className="px-4 py-3">Experience</td>
              <td className="px-4 py-3">Labelled corpus of emails</td>
              <td className="px-4 py-3">Millions of simulator episodes</td>
            </tr>
          </tbody>
        </table>
      </div>

      <p>
        The definition is deliberately paradigm-agnostic. Whether your algorithm
        reads labeled examples, unlabeled pixels, or reward signals from a game
        engine — all three components are always present. What differs between
        paradigms is the <em>nature</em> of E and the structure of T.
      </p>

      {/* ══════════════════════════════════════════ */}
      <h2>2 · Supervised Learning: Labeled Experience</h2>

      <p>
        Supervised learning is the most commercially deployed paradigm. The
        distinguishing feature is a dataset of <em>input-output pairs</em>:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 leading-relaxed">
        <span className="text-emerald-400">Training set</span>
        {"\n"}
        {"  "}D = {"{"} (x_1, y_1), (x_2, y_2), ..., (x_n, y_n) {"}"}{"\n\n"}
        {"  "}x_i ∈ X   (feature space, e.g. R^d){"\n"}
        {"  "}y_i ∈ Y   (label space){"\n\n"}
        <span className="text-emerald-400">Goal</span>
        {"\n"}
        {"  "}Find h : X → Y  (hypothesis) such that h(x) ≈ y for unseen (x, y){"\n\n"}
        <span className="text-emerald-400">Empirical Risk Minimisation (ERM)</span>
        {"\n"}
        {"  "}h* = argmin_h  (1/n) * Σ_i  L( h(x_i), y_i ){"\n\n"}
        {"  "}L(ŷ, y) = loss function measuring prediction error
      </div>

      <p>
        ERM is the backbone of nearly every supervised algorithm. We choose a
        hypothesis class H (e.g., linear functions, depth-5 trees, neural
        networks), then search H for the member that minimises average loss on
        the training set. The philosophical risk is <em>overfitting</em>: the
        selected h* may memorise training labels rather than generalise — this
        tension is formalised in Section 9.
      </p>

      <h3>2.1 Classification vs. Regression</h3>

      <p>
        The label space Y determines whether a task is classification or
        regression.
      </p>

      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full border border-slate-200 text-sm">
          <thead className="bg-slate-100 text-slate-700 uppercase text-xs tracking-wide">
            <tr>
              <th className="px-4 py-3 text-left border-b">Property</th>
              <th className="px-4 py-3 text-left border-b">Classification</th>
              <th className="px-4 py-3 text-left border-b">Regression</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr>
              <td className="px-4 py-3">Output space Y</td>
              <td className="px-4 py-3">Discrete / categorical</td>
              <td className="px-4 py-3">Continuous (R or R^k)</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3">Canonical loss L</td>
              <td className="px-4 py-3">Cross-entropy: -Σ y log ŷ</td>
              <td className="px-4 py-3">Mean squared error: (ŷ - y)²</td>
            </tr>
            <tr>
              <td className="px-4 py-3">Evaluation metric</td>
              <td className="px-4 py-3">Accuracy, F1, AUC-ROC</td>
              <td className="px-4 py-3">RMSE, MAE, R²</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3">Industry example</td>
              <td className="px-4 py-3">Fraud detection, churn, diagnosis</td>
              <td className="px-4 py-3">House price, demand forecast, ETA</td>
            </tr>
            <tr>
              <td className="px-4 py-3">Common algorithms</td>
              <td className="px-4 py-3">Logistic regression, SVM, RF, XGBoost</td>
              <td className="px-4 py-3">Linear regression, Gradient boosting, MLP</td>
            </tr>
          </tbody>
        </table>
      </div>

      <p>
        A subtlety: the boundary is blurred by <em>ordinal regression</em> (ranking
        problems) and <em>multi-label classification</em> (each example has a
        subset of labels). Always inspect Y before choosing a loss.
      </p>

      {/* ══════════════════════════════════════════ */}
      <h2>3 · Unsupervised Learning: Learning Structure Without Labels</h2>

      <p>
        When labels are absent or prohibitively expensive to collect, unsupervised
        methods extract structure directly from the marginal distribution p(x).
        The dataset is simply:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 leading-relaxed">
        <span className="text-emerald-400">Unlabeled dataset</span>
        {"\n"}
        {"  "}D = {"{"} x_1, x_2, ..., x_n {"}"}{"\n\n"}
        <span className="text-emerald-400">Three families of structure to discover</span>
        {"\n\n"}
        {"  "}1. Clustering         — partition X into K coherent groups{"\n"}
        {"     "}K-Means objective: min Σ_k Σ_{"{x∈C_k}"} ||x - μ_k||²{"\n\n"}
        {"  "}2. Density estimation — learn p(x) explicitly{"\n"}
        {"     "}Gaussian Mixture:  p(x) = Σ_k π_k · N(x | μ_k, Σ_k){"\n\n"}
        {"  "}3. Dim. reduction     — find z ∈ R^d (d ≪ D) that preserves structure{"\n"}
        {"     "}PCA:  z = W^T x,   W = top-d eigenvectors of Cov(X)
      </div>

      <p>
        Unsupervised learning is intrinsically harder to evaluate: there is no
        ground-truth label to compare against. Practitioners use internal metrics
        (silhouette score for clustering, reconstruction error for autoencoders)
        or downstream task performance as proxies. A critically important use
        case: using a K-Means clustering of customer transactions to discover
        market segments that <em>nobody knew existed</em>.
      </p>

      {/* ══════════════════════════════════════════ */}
      <h2>4 · Semi-Supervised Learning: Cheap Unlabeled + Scarce Labeled</h2>

      <p>
        In most industrial settings, obtaining labels is expensive (radiologist
        reads, legal review, expert annotation) while raw data is abundant.
        Semi-supervised learning (SSL) exploits both:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 leading-relaxed">
        <span className="text-emerald-400">Dataset</span>
        {"\n"}
        {"  "}D_L = {"{"} (x_i, y_i) : i = 1..l {"}"} (l labeled, l &lt;&lt; n){"\n"}
        {"  "}D_U = {"{"} x_j : j = 1..u {"}"}          (u unlabeled, u &gt;&gt; l){"\n\n"}
        <span className="text-emerald-400">Three dominant approaches</span>
        {"\n\n"}
        {"  "}Self-training (pseudo-labeling){"\n"}
        {"    "}1. Train h on D_L{"\n"}
        {"    "}2. Predict labels ŷ_j = h(x_j) for D_U{"\n"}
        {"    "}3. Add high-confidence (x_j, ŷ_j) to D_L{"\n"}
        {"    "}4. Repeat{"\n\n"}
        {"  "}Label propagation{"\n"}
        {"    "}Build graph G: nodes = all x, edges = k-NN similarity{"\n"}
        {"    "}Diffuse labels from D_L over edges to D_U{"\n\n"}
        {"  "}Consistency regularisation (MixMatch, FixMatch){"\n"}
        {"    "}Enforce: h(augment(x)) ≈ h(x)  for all x ∈ D_U
      </div>

      <p>
        The core assumption underlying SSL is the <strong>cluster assumption</strong>:
        data points in the same high-density cluster share labels. When this is
        violated (e.g., two classes interleaved in feature space), pseudo-labeling
        can amplify errors catastrophically. Always sanity-check the geometry of
        your unlabeled data before applying SSL.
      </p>

      {/* ══════════════════════════════════════════ */}
      <h2>5 · Self-Supervised Learning: Supervision from the Data Itself</h2>

      <p>
        Self-supervised learning (SSL2) is the engine behind every modern large
        language model and vision foundation model. The key insight: we can
        construct a <em>pretext task</em> whose supervision signal comes from the
        raw data structure itself — no human annotations needed.
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 leading-relaxed">
        <span className="text-emerald-400">Three canonical pretext tasks</span>
        {"\n\n"}
        {"  "}1. BERT — Masked Language Modelling (MLM){"\n"}
        {"     "}Input:  "The [MASK] sat on the mat"{"\n"}
        {"     "}Target: predict "cat" at [MASK] position{"\n"}
        {"     "}Signal: cross-entropy vs. true token (free from raw text!){"\n\n"}
        {"  "}2. GPT — Autoregressive Next-Token Prediction{"\n"}
        {"     "}P(w_t | w_1, ..., w_{"{t-1}"}){"\n"}
        {"     "}Objective: L = -Σ_t log P(w_t | w_{"<t"}){"\n"}
        {"     "}Trained on ~10^12 tokens; no human label needed{"\n\n"}
        {"  "}3. SimCLR — Contrastive Learning (vision){"\n"}
        {"     "}For image x: generate two augmented views v_i, v_j{"\n"}
        {"     "}Encoder f(·) → representation space{"\n"}
        {"     "}NT-Xent loss: pull f(v_i), f(v_j) together,{"\n"}
        {"                    "}push apart all other pairs in batch
      </div>

      <p>
        Self-supervised learning is philosophically important: it demonstrates
        that rich, transferable representations can be learned from the
        statistical structure of raw data — predicting missing parts of the
        input forces the model to build an internal world model. The representations
        learned by GPT-4, CLIP, and DINO are then adapted to downstream
        supervised tasks with very few labeled examples (fine-tuning), giving
        rise to the modern <em>pre-train → fine-tune</em> paradigm.
      </p>

      {/* ══════════════════════════════════════════ */}
      <h2>6 · Reinforcement Learning: Learning from Interaction</h2>

      <p>
        When neither labeled pairs nor raw corpora are appropriate — when the
        system must <em>act</em> in an environment and learn from the
        consequences — reinforcement learning is the correct paradigm. The
        canonical formalism is the Markov Decision Process (MDP):
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 leading-relaxed">
        <span className="text-emerald-400">MDP tuple: (S, A, P, R, γ)</span>
        {"\n\n"}
        {"  "}S   — state space (e.g., board position, sensor readings){"\n"}
        {"  "}A   — action space (e.g., move left/right, set throttle){"\n"}
        {"  "}P   — transition model: P(s' | s, a)  (possibly unknown){"\n"}
        {"  "}R   — reward function: R(s, a, s') ∈ R{"\n"}
        {"  "}γ   — discount factor ∈ [0, 1): weight of future rewards{"\n\n"}
        <span className="text-emerald-400">Policy</span>
        {"\n"}
        {"  "}π : S → A  (deterministic) or  π(a | s) (stochastic){"\n\n"}
        <span className="text-emerald-400">State Value Function</span>
        {"\n"}
        {"  "}V^π(s) = E_π [ Σ_{"{t=0}"}^∞  γ^t · R_{"{t+1}"}  |  S_0 = s ]{"\n\n"}
        <span className="text-emerald-400">Bellman Expectation Equation</span>
        {"\n"}
        {"  "}V^π(s) = Σ_a π(a|s) · Σ_{"{s'}"} P(s'|s,a) · [ R(s,a,s') + γ·V^π(s') ]{"\n\n"}
        {"  "}↑ this recursive identity is the foundation of all DP/TD methods
      </div>

      <p>
        The Bellman equation is central because it converts the intractable sum
        over infinite trajectories into a tractable one-step recursive relation.
        Q-learning, SARSA, and deep RL methods (DQN, PPO, SAC) are all
        algorithms for solving or approximating versions of this equation.
      </p>

      <p>
        Notice the fundamental contrast with supervised learning: in RL there is
        no teacher providing correct actions. The agent must explore, receive
        delayed rewards, and infer which actions were responsible — the
        <em> credit assignment problem</em>. This is why RL is orders of magnitude
        harder than supervised learning for most real-world tasks.
      </p>

      {/* ══════════════════════════════════════════ */}
      <h2>7 · The No Free Lunch Theorem</h2>

      <p>
        Wolpert &amp; Macready (1997) proved a sobering result that every ML
        practitioner must internalize:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 leading-relaxed">
        <span className="text-emerald-400">No Free Lunch Theorem (informal statement)</span>
        {"\n\n"}
        {"  "}Averaged over ALL possible data-generating distributions,{"\n"}
        {"  "}every learning algorithm has the same expected generalisation error.{"\n\n"}
        <span className="text-emerald-400">Formal (binary classification, 0-1 loss)</span>
        {"\n\n"}
        {"  "}For any two algorithms A1, A2:{"\n\n"}
        {"  "}Σ_f  E[ L(A1, f) ] = Σ_f  E[ L(A2, f) ]{"\n\n"}
        {"  "}where the sum is over ALL target functions f : X → {"{"} 0,1 {"}"}
      </div>

      <p>
        The critical implication: there is <strong>no universally best algorithm</strong>.
        A deep neural network that dominates on natural images will underperform
        a simple linear model on a task where the true relationship is linear.
        Every practical ML project begins with an inductive bias — assumptions
        about the problem (smoothness, locality, compositionality) — and the job
        of model selection is to match that bias to the domain.
      </p>

      <p>
        NFL is not defeatist. It simply tells us that generalisation requires
        assumptions. The useful question is: <em>which assumptions are warranted
        by this domain?</em> Physics-informed neural networks embed PDEs.
        Convolutional networks embed spatial locality. Transformers embed
        permutation-equivariant attention. Each is a deliberate inductive bias
        justified by domain knowledge.
      </p>

      {/* ══════════════════════════════════════════ */}
      <h2>8 · The Bias-Variance Tradeoff: The Unifying Framework</h2>

      <p>
        The Bias-Variance decomposition gives the canonical diagnosis for any
        generalisation failure, regardless of paradigm. For a fixed test point x
        and squared-error loss:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6 leading-relaxed">
        <span className="text-emerald-400">Bias-Variance Decomposition</span>
        {"\n\n"}
        {"  "}E[ (h(x) - y)² ] = Bias²(h(x))  +  Var(h(x))  +  σ²{"\n\n"}
        {"  "}where:{"\n"}
        {"    "}Bias(h(x))  = E[h(x)] - f(x)         (systematic error){"\n"}
        {"    "}Var(h(x))   = E[ (h(x) - E[h(x)])² ] (sensitivity to training data){"\n"}
        {"    "}σ²          = irreducible noise in y   (cannot be eliminated){"\n\n"}
        <span className="text-emerald-400">Paradigm mapping</span>
        {"\n\n"}
        {"  "}High complexity model  →  Low bias,  High variance  (overfit){"\n"}
        {"  "}Low complexity model   →  High bias, Low variance   (underfit){"\n"}
        {"  "}Regularisation         →  Trades some bias for reduced variance{"\n"}
        {"  "}Ensemble (bagging)     →  Reduces variance, keeps bias{"\n"}
        {"  "}Boosting               →  Reduces bias sequentially, may increase var
      </div>

      <p>
        The tradeoff appears in every paradigm. In RL, a high-capacity policy
        network overfits to the replay buffer (high variance). In unsupervised
        clustering, K-Means with too many centroids memorises the training data
        distribution (high variance, low training error). In self-supervised
        learning, very small models underfit the pretext task (high bias). The
        decomposition is your diagnostic lens.
      </p>

      {/* ══════════════════════════════════════════ */}
      <h2>9 · Python Implementation: Supervised vs. Unsupervised on Iris</h2>

      <p>
        The following code trains a <strong>KNN classifier</strong> (supervised)
        and a <strong>K-Means clusterer</strong> (unsupervised) on the same Iris
        dataset, then compares what each one "knows" and how we evaluate them.
        Reading through the comments is mandatory — the conceptual differences
        manifest directly in the API.
      </p>

      <TerminalBlock language="python">{`# learning_paradigms_demo.py
# Demonstrates supervised vs unsupervised learning on the Iris dataset.
# Iris has 150 samples, 4 features (sepal/petal length & width), 3 classes.

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,          # for supervised: needs ground truth labels
    adjusted_rand_score,     # for unsupervised: compares cluster structure
    silhouette_score,        # for unsupervised: internal cohesion metric
    classification_report,
)

# ─────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────
iris = load_iris()
X = iris.data          # shape (150, 4) — feature matrix
y = iris.target        # shape (150,)   — integer class labels {0, 1, 2}
                       # NOTE: y is ONLY used by the supervised model and
                       # for evaluating the unsupervised one post-hoc.

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

# ─────────────────────────────────────────────────────────────
# 2. Supervised split
#    We must split data so the classifier NEVER sees test labels at
#    train time.  The unsupervised model doesn't need a split, but we
#    evaluate it on the same held-out indices for a fair comparison.
# ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,    # 30% held out — 45 samples
    random_state=42,   # reproducibility
    stratify=y,        # preserve class proportions in both splits
)
print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")

# ─────────────────────────────────────────────────────────────
# 3. Preprocessing — StandardScaler
#    K-Means and KNN are both distance-based; unscaled features with
#    larger numerical ranges would dominate the distance computation.
#    IMPORTANT: fit scaler ONLY on training data to prevent data leakage.
# ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # learn mean/std from training set
X_test_sc  = scaler.transform(X_test)        # apply SAME transformation to test set
X_all_sc   = scaler.transform(X)             # for K-Means on full unlabeled pool

# ─────────────────────────────────────────────────────────────
# 4. ── SUPERVISED ── K-Nearest Neighbours Classifier
#    The model has access to (X_train_sc, y_train) during fitting.
#    At inference time it assigns the majority class among the k=5 nearest
#    training points in Euclidean space.
# ─────────────────────────────────────────────────────────────
knn = KNeighborsClassifier(
    n_neighbors=5,       # vote among 5 nearest neighbours
    metric='euclidean',  # distance function in feature space
    weights='uniform',   # all neighbours vote equally
)

# .fit() consumes the LABELS — this is what makes it supervised
knn.fit(X_train_sc, y_train)

# Evaluate on held-out test set
y_pred_knn = knn.predict(X_test_sc)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print(f"\n── KNN Classifier (Supervised) ──")
print(f"Test Accuracy: {knn_accuracy:.3f}")
print(classification_report(y_test, y_pred_knn, target_names=iris.target_names))

# ─────────────────────────────────────────────────────────────
# 5. ── UNSUPERVISED ── K-Means Clustering
#    K-Means receives NO labels.  It partitions X into K=3 clusters
#    by minimising inertia: Σ_k Σ_{x in C_k} ||x - μ_k||²
#    We tell it K=3 (domain knowledge), but it does NOT know which
#    cluster corresponds to which species.
# ─────────────────────────────────────────────────────────────
kmeans = KMeans(
    n_clusters=3,      # we assume 3 groups — a strong inductive bias
    init='k-means++',  # smarter initialisation reduces convergence variance
    n_init=10,         # restart 10 times, keep best (lowest inertia) run
    random_state=42,
    max_iter=300,
)

# .fit() receives ONLY X — no y.  This is what makes it unsupervised.
kmeans.fit(X_all_sc)

# Cluster assignments for the entire dataset (no train/test split needed)
cluster_labels = kmeans.labels_   # integers in {0, 1, 2} — arbitrary cluster IDs

# ─────────────────────────────────────────────────────────────
# 6. Evaluating K-Means
#    a) Internal metric: silhouette score — needs only X and cluster IDs
#       Range [-1, 1]; higher = tighter clusters, better separation
#    b) External metric: Adjusted Rand Index — compares cluster structure
#       to ground-truth y (only valid as a post-hoc research tool;
#       in a real unsupervised use-case you wouldn't have y)
# ─────────────────────────────────────────────────────────────
silhouette = silhouette_score(X_all_sc, cluster_labels)
ari         = adjusted_rand_score(y, cluster_labels)   # post-hoc only

print(f"\n── K-Means Clustering (Unsupervised) ──")
print(f"Inertia (training objective): {kmeans.inertia_:.2f}")
print(f"Silhouette Score (internal):  {silhouette:.3f}  (no labels needed)")
print(f"Adjusted Rand Index (external, post-hoc): {ari:.3f}  (requires y — for research only)")

# ─────────────────────────────────────────────────────────────
# 7. Key contrast summary
# ─────────────────────────────────────────────────────────────
print("""
┌──────────────────────┬──────────────────────────┬──────────────────────────┐
│                      │  KNN (Supervised)         │  K-Means (Unsupervised)  │
├──────────────────────┼──────────────────────────┼──────────────────────────┤
│ Training input       │  (X_train, y_train)       │  X only                  │
│ Knows class names?   │  YES                      │  NO                      │
│ Needs train/test?    │  YES (prevents leakage)   │  Not required            │
│ Primary eval metric  │  Accuracy / F1            │  Silhouette / Inertia    │
│ Post-hoc eval (y)    │  Native                   │  ARI — optional check    │
└──────────────────────┴──────────────────────────┴──────────────────────────┘
""")
`}</TerminalBlock>

      <p>
        A typical run on Iris yields KNN test accuracy of{" "}
        <strong>~97%</strong> and K-Means ARI of <strong>~0.73</strong>. The
        gap illustrates a crucial lesson: unsupervised clustering on Iris is
        genuinely harder because two of the three species (Versicolor and
        Virginica) overlap in feature space, while the supervised model learns
        the decision boundary explicitly from labeled examples.
      </p>

      {/* ══════════════════════════════════════════ */}
      <h2>10 · Common Pitfalls</h2>

      <div className="not-prose overflow-x-auto my-6">
        <table className="min-w-full border border-slate-200 text-sm">
          <thead className="bg-slate-100 text-slate-700 uppercase text-xs tracking-wide">
            <tr>
              <th className="px-4 py-3 text-left border-b">Pitfall</th>
              <th className="px-4 py-3 text-left border-b">Paradigm</th>
              <th className="px-4 py-3 text-left border-b">Fix</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            <tr>
              <td className="px-4 py-3">Fitting the scaler on the full dataset before splitting</td>
              <td className="px-4 py-3">Supervised</td>
              <td className="px-4 py-3">Always fit scaler on training fold only; use Pipeline</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3">Using accuracy as the sole metric on imbalanced classes</td>
              <td className="px-4 py-3">Classification</td>
              <td className="px-4 py-3">Report precision, recall, F1, and AUC per class</td>
            </tr>
            <tr>
              <td className="px-4 py-3">Evaluating K-Means with accuracy (requires label alignment)</td>
              <td className="px-4 py-3">Unsupervised</td>
              <td className="px-4 py-3">Use silhouette score; ARI only for research benchmarking</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3">Pseudo-labeling with low-confidence predictions</td>
              <td className="px-4 py-3">Semi-supervised</td>
              <td className="px-4 py-3">Set a high confidence threshold (e.g., softmax &gt; 0.95) and use FixMatch</td>
            </tr>
            <tr>
              <td className="px-4 py-3">Applying self-supervised pretext task to tiny dataset</td>
              <td className="px-4 py-3">Self-supervised</td>
              <td className="px-4 py-3">Pretext tasks need large corpora; use supervised or transfer learning instead</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="px-4 py-3">Sparse rewards in RL causing no learning signal for thousands of steps</td>
              <td className="px-4 py-3">Reinforcement</td>
              <td className="px-4 py-3">Use reward shaping, curiosity-driven exploration, or hierarchical RL</td>
            </tr>
            <tr>
              <td className="px-4 py-3">Claiming one algorithm is universally best (violates NFL)</td>
              <td className="px-4 py-3">All</td>
              <td className="px-4 py-3">Always baseline multiple algorithms; justify inductive bias for your domain</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* ══════════════════════════════════════════ */}
      <h2>11 · Further Reading</h2>

      <ul>
        <li>
          <strong>Mitchell, T. (1997).</strong>{" "}
          <em>Machine Learning.</em> McGraw-Hill. — Chapter 1 defines T, P, E
          and gives the foundational decision-tree and neural network examples.
        </li>
        <li>
          <strong>Wolpert, D. &amp; Macready, W. (1997).</strong>{" "}
          "No Free Lunch Theorems for Optimization."{" "}
          <em>IEEE Transactions on Evolutionary Computation, 1(1).</em> — The
          original proof. Read Section II for the classification version.
        </li>
        <li>
          <strong>Devlin, J. et al. (2019).</strong>{" "}
          "BERT: Pre-training of Deep Bidirectional Transformers for Language
          Understanding." <em>NAACL 2019.</em> — The masked language model
          pretext task that launched the modern NLP era.
        </li>
        <li>
          <strong>Chen, T. et al. (2020).</strong>{" "}
          "A Simple Framework for Contrastive Learning of Visual
          Representations (SimCLR)." <em>ICML 2020.</em> — The clearest
          exposition of contrastive self-supervised learning with NT-Xent loss.
        </li>
        <li>
          <strong>Sutton, R. &amp; Barto, A. (2018).</strong>{" "}
          <em>Reinforcement Learning: An Introduction (2nd ed.).</em>{" "}
          MIT Press. — Free online. Chapters 3–4 cover MDPs and the Bellman
          equation rigorously.
        </li>
        <li>
          <strong>Geman, S. et al. (1992).</strong>{" "}
          "Neural Networks and the Bias/Variance Dilemma."{" "}
          <em>Neural Computation, 4(1).</em> — The paper that popularised the
          bias-variance decomposition in the ML community.
        </li>
        <li>
          <strong>van Engelen, J. &amp; Hoos, H. (2020).</strong>{" "}
          "A Survey on Semi-Supervised Learning."{" "}
          <em>Machine Learning, 109.</em> — Comprehensive taxonomy covering
          self-training, label propagation, and generative approaches.
        </li>
      </ul>
    </article>
  );
}
