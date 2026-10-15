import quizzesData from "./ml_beginners_quizzes.json";

export interface QuizOption {
  answerText: string;
  isCorrect: string | boolean;
}

export interface QuizQuestion {
  questionText: string;
  answerOptions: QuizOption[];
}

export interface QuizItem {
  id: number;
  title: string;
  quiz: QuizQuestion[];
}

export interface Lesson {
  id: string;
  week: number;
  lessonNumber: number;
  title: string;
  shortDesc: string;
  readTime: string;
  category: "Introduction" | "Regression" | "Web-App" | "Classification" | "Clustering" | "NLP" | "TimeSeries" | "Reinforcement" | "Real-World";
  preQuizId: number;
  postQuizId: number;
  objectives: string[];
  keyTerms: { term: string; definition: string }[];
  mathFormulas?: { name: string; formula: string; description: string }[];
  sections: {
    title: string;
    content: string;
    codeSnippet?: {
      language: string;
      title: string;
      code: string;
    };
  }[];
  assignment: {
    title: string;
    instructions: string;
  };
}

export interface Week {
  week: number;
  title: string;
  description: string;
  icon: string;
  lessons: Lesson[];
}

export const quizzes: QuizItem[] = quizzesData as unknown as QuizItem[];

export function getQuizById(id: number): QuizItem | undefined {
  return quizzes.find((q) => q.id === id);
}

export const mlBeginnersWeeks: Week[] = [
  // ── WEEK 1 ──
  {
    week: 1,
    title: "Introduction to Machine Learning & Its History",
    description: "Foundational definitions, the AI hype curve, biological vs statistical systems, and the pioneers from Turing to Deep Learning.",
    icon: "Brain",
    lessons: [
      {
        id: "intro-to-ml",
        week: 1,
        lessonNumber: 1,
        title: "Introduction to Machine Learning",
        shortDesc: "Understand what machine learning is, how it differs from traditional deterministic programming, and the core learning loop.",
        readTime: "15 min",
        category: "Introduction",
        preQuizId: 1,
        postQuizId: 2,
        objectives: [
          "Differentiate traditional rule-based programming from machine learning.",
          "State Tom Mitchell's formal definition of machine learning (Task T, Performance P, Experience E).",
          "Recognize why statistical patterns scale better than hardcoded heuristics for complex multidimensional problems.",
          "Identify the primary paradigms of ML: Supervised, Unsupervised, and Reinforcement Learning."
        ],
        keyTerms: [
          { term: "Machine Learning", definition: "A subfield of artificial intelligence that gives computers the ability to learn without being explicitly programmed (Arthur Samuel, 1959)." },
          { term: "Mitchell's Formulation", definition: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E." },
          { term: "Inductive Bias", definition: "The set of assumptions a machine learning algorithm uses to predict outputs for inputs it has not previously encountered." }
        ],
        mathFormulas: [
          {
            name: "Empirical Risk Minimization (ERM)",
            formula: "\\min_{f \\in \\mathcal{H}} R_{\\text{emp}}(f) = \\frac{1}{N} \\sum_{i=1}^N \\mathcal{L}(f(x_i), y_i)",
            description: "The principle where an algorithm chooses a model f from hypothesis set H that minimizes average loss across training samples."
          }
        ],
        sections: [
          {
            title: "1. The Fundamental Shift: Rules vs. Learned Patterns",
            content: "In classical software engineering, humans write explicit logic: Input Data + Rules = Output. If you want to detect spam, you might write regular expressions for words like 'lottery' or 'free cash'. However, as spammers change tactics, rule sets grow unwieldy and brittle. In Machine Learning, the paradigm inverts: Input Data + Known Outputs = Learned Rules. The algorithm discovers mathematical mappings that generalize to unseen instances."
          },
          {
            title: "2. The Learning Loop in Python",
            content: "Every standard supervised machine learning pipeline follows an identical cyclical architecture: ingestion, feature matrix X separation, target vector y extraction, splitting into train/test partitions, fitting the estimator, and validating against unseen holdout distributions.",
            codeSnippet: {
              language: "python",
              title: "standard_ml_loop.py",
              code: `import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Feature matrix X and target y
X = np.random.randn(200, 4)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# 2. Strict holdout partition to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training instances: {X_train.shape[0]} | Testing instances: {X_test.shape[0]}")`
            }
          }
        ],
        assignment: {
          title: "Identify ML in Your Daily Workflow",
          instructions: "List three software tools you use daily. Categorize whether each relies on deterministic heuristics or machine learning, and define Mitchell's (T, P, E) triplet for one of them."
        }
      },
      {
        id: "history-of-ml",
        week: 1,
        lessonNumber: 2,
        title: "History of Machine Learning",
        shortDesc: "Journey from Alan Turing and the 1956 Dartmouth Workshop to the Perceptron, AI Winters, SVMs, and the modern Deep Learning renaissance.",
        readTime: "18 min",
        category: "Introduction",
        preQuizId: 3,
        postQuizId: 4,
        objectives: [
          "Trace the chronological arc of AI and ML from 1950 to present day.",
          "Explain Frank Rosenblatt's Perceptron and its biological inspiration.",
          "Understand why Minsky and Papert's 1969 book caused the first AI Winter by exposing the XOR limitation.",
          "Identify the factors behind the 2012 Deep Learning breakthrough: big data, GPU parallelization, and deep architectures."
        ],
        keyTerms: [
          { term: "Dartmouth Workshop (1956)", definition: "The founding conference where John McCarthy coined the term 'Artificial Intelligence' and formulated early symbolic AI goals." },
          { term: "Perceptron", definition: "An algorithm for supervised learning of binary classifiers invented by Frank Rosenblatt in 1957; the simplest precursor to modern artificial neural networks." },
          { term: "AI Winter", definition: "Periods of reduced funding and public interest in AI research (notably 1974–1980 and 1987–1993) caused by overpromising and technical limitations." }
        ],
        mathFormulas: [
          {
            name: "Perceptron Decision Function",
            formula: "f(x) = \\begin{cases} 1 & \\text{if } w^T x + b \\ge 0 \\\\ 0 & \\text{otherwise} \\end{cases}",
            description: "A single hyper-plane partitioning feature space into two linearly separable half-spaces."
          }
        ],
        sections: [
          {
            title: "1. The Perceptron and The XOR Crisis",
            content: "In 1957, Frank Rosenblatt created the Perceptron at Cornell Aeronautical Laboratory. Running on custom hardware (the Mark I Perceptron), it learned weights automatically using mistake-driven updates. However, in 1969, Marvin Minsky and Seymour Papert published 'Perceptrons', mathematically demonstrating that a single linear unit cannot compute the XOR (Exclusive OR) logical function. The inability to train multi-layer networks efficiently precipitated the first AI Winter."
          },
          {
            title: "2. The Perceptron in Pure Python",
            content: "Here is Rosenblatt's original weight update rule implemented from scratch: w = w + lr * (y - y_hat) * x.",
            codeSnippet: {
              language: "python",
              title: "perceptron_from_scratch.py",
              code: `import numpy as np

class Perceptron:
    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = 1 if linear_output >= 0 else 0
                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)`
            }
          }
        ],
        assignment: {
          title: "Verify the XOR Failure",
          instructions: "Test the Perceptron code on the AND function versus the XOR function. Verify that AND achieves 100% accuracy while XOR fails to converge."
        }
      }
    ]
  },

  // ── WEEK 2 ──
  {
    week: 2,
    title: "Fairness, Ethics & Core Machine Learning Techniques",
    description: "Auditing algorithms for demographic parity and bias, alongside the bias-variance tradeoff and ML taxonomy.",
    icon: "Scale",
    lessons: [
      {
        id: "fairness-in-ml",
        week: 2,
        lessonNumber: 3,
        title: "Fairness and Responsible AI",
        shortDesc: "Understand systemic bias, demographic parity, equal opportunity, disparate impact, and ethical auditing in ML pipelines.",
        readTime: "16 min",
        category: "Introduction",
        preQuizId: 5,
        postQuizId: 6,
        objectives: [
          "Define how human, sampling, and historical bias enter algorithmic systems.",
          "Distinguish Demographic Parity from Equal Opportunity and Predictive Parity.",
          "Explain the Impossibility Theorem of Fairness.",
          "Implement bias audit metrics using Disparate Impact and Equalized Odds."
        ],
        keyTerms: [
          { term: "Demographic Parity", definition: "A fairness criterion requiring that the likelihood of receiving a positive prediction is identical across protected demographic groups." },
          { term: "Equal Opportunity", definition: "A fairness criterion requiring equal true positive rates across protected attributes (conditioned on the true positive class)." },
          { term: "Disparate Impact", definition: "The ratio of positive selection rates between the unprivileged group and the privileged group; ratios below 0.8 violate the EEOC 80% rule." }
        ],
        mathFormulas: [
          {
            name: "Disparate Impact Ratio",
            formula: "\\text{DIR} = \\frac{P(\\hat{Y} = 1 \\mid A = 0)}{P(\\hat{Y} = 1 \\mid A = 1)} \\ge 0.8",
            description: "Compares the favorable outcome rate of unprivileged group A=0 against privileged group A=1."
          }
        ],
        sections: [
          {
            title: "1. Sources of Algorithmic Bias",
            content: "Machine learning models do not invent bias; they reflect, encode, and amplify existing systemic inequalities present in training data. Common failure modes include historical bias (societal prejudice reflected in hiring records), representation bias (underrepresented demographic cohorts in training datasets), and proxy variables (using zip codes as an implicit proxy for race)."
          },
          {
            title: "2. Calculating Disparate Impact in Python",
            content: "Auditing a credit approval model across sensitive attribute A (e.g., age or gender):",
            codeSnippet: {
              language: "python",
              title: "fairness_audit.py",
              code: `import numpy as np

# Ground truth predictions and sensitive attributes (0: unprivileged, 1: privileged)
sensitive_attr = np.array([0, 0, 0, 0, 1, 1, 1, 1])
predictions = np.array([0, 1, 0, 0, 1, 1, 0, 1])

# Calculate selection rate per group
rate_unprivileged = np.mean(predictions[sensitive_attr == 0])
rate_privileged = np.mean(predictions[sensitive_attr == 1])

disparate_impact = rate_unprivileged / rate_privileged
print(f"Unprivileged Positive Rate: {rate_unprivileged:.2f}")
print(f"Privileged Positive Rate:   {rate_privileged:.2f}")
print(f"Disparate Impact Ratio:     {disparate_impact:.2f}")
print("Passes 80% rule?" , disparate_impact >= 0.8)`
            }
          }
        ],
        assignment: {
          title: "Datasheet for a Dataset",
          instructions: "Draft a 1-page Datasheet for a Dataset detailing motivation, composition, collection process, and recommended ethical boundaries for a hypothetical facial recognition training set."
        }
      },
      {
        id: "techniques-of-ml",
        week: 2,
        lessonNumber: 4,
        title: "Techniques and Paradigms of Machine Learning",
        shortDesc: "The bias-variance decomposition, supervised vs unsupervised learning, and the fundamental problem-framing decision tree.",
        readTime: "17 min",
        category: "Introduction",
        preQuizId: 7,
        postQuizId: 8,
        objectives: [
          "Decompose generalization error into bias, variance, and irreducible error.",
          "Categorize regression, classification, clustering, and dimensionality reduction.",
          "Select the appropriate ML paradigm based on label availability and problem constraints.",
          "Understand the No Free Lunch theorem."
        ],
        keyTerms: [
          { term: "Bias", definition: "Error resulting from erroneous assumptions in the learning algorithm; high bias causes underfitting." },
          { term: "Variance", definition: "Error resulting from sensitivity to small fluctuations in the training set; high variance causes overfitting." },
          { term: "No Free Lunch Theorem", definition: "If an algorithm performs well on a certain class of problems, it necessarily pays for that with degraded performance on the set of all remaining problems." }
        ],
        mathFormulas: [
          {
            name: "Bias-Variance Decomposition",
            formula: "\\mathbb{E}[(y - \\hat{f}(x))^2] = \\text{Bias}[\\hat{f}(x)]^2 + \\text{Var}[\\hat{f}(x)] + \\sigma^2",
            description: "Total expected prediction error decomposes into squared model bias, model variance, and noise variance."
          }
        ],
        sections: [
          {
            title: "1. The Bias-Variance Tradeoff",
            content: "As model complexity increases (e.g., adding polynomial features or tree depth), training error monotonically decreases while variance climbs. The optimal generalization point minimizes test error at the intersection of bias and variance curves."
          },
          {
            title: "2. The Problem Framing Decision Tree",
            content: "When confronted with an enterprise problem: 1. Do you have target labels y? If yes -> Supervised (Continuous: Regression; Categorical: Classification). 2. If no -> Unsupervised (Grouping: Clustering; Feature compression: PCA/UMAP). 3. Is the system interacting in an environment receiving reward signals? -> Reinforcement Learning."
          }
        ],
        assignment: {
          title: "Diagnosing Model Fits",
          instructions: "Given training error of 1% and test error of 18%, diagnose whether the model suffers from high bias or high variance, and propose three concrete remediation techniques."
        }
      }
    ]
  },

  // ── WEEK 3 ──
  {
    week: 3,
    title: "Data Preparation & Exploratory Data Analysis for Regression",
    description: "The data science toolkit (NumPy, Pandas, Matplotlib) and exploratory analysis on real-world US Pumpkin pricing datasets.",
    icon: "LineChart",
    lessons: [
      {
        id: "regression-tools",
        week: 3,
        lessonNumber: 5,
        title: "Tools of the Trade: Python, Pandas & Scikit-Learn",
        shortDesc: "Mastering the data manipulation and machine learning stack for data pipelines and feature extraction.",
        readTime: "14 min",
        category: "Regression",
        preQuizId: 9,
        postQuizId: 10,
        objectives: [
          "Set up reproducible Python environments with conda/pip.",
          "Manipulate structured DataFrames with Pandas indexing, grouping, and aggregations.",
          "Construct clean Scikit-Learn transformers and pipelines.",
          "Prevent data leakage across cross-validation splits."
        ],
        keyTerms: [
          { term: "Data Leakage", definition: "When information from outside the training dataset is inadvertently used to create the model, artificially inflating validation metrics." },
          { term: "Scikit-Learn Pipeline", definition: "A chain of transformers followed by a final estimator that guarantees transformations are fit strictly on training splits." }
        ],
        sections: [
          {
            title: "1. Structuring Robust Data Pipelines",
            content: "A professional data workflow separates raw data from processed feature vectors and relies on Scikit-Learn pipelines to ensure imputers and scalers are calculated only on training splits."
          },
          {
            title: "2. Building a Clean Data Pipeline",
            content: "Example pipeline combining standard scaling and polynomial features:",
            codeSnippet: {
              language: "python",
              title: "pipeline_setup.py",
              code: `import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge

# Pipeline encapsulates transformation parameters
pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('regressor', Ridge(alpha=1.0))
])

print("Pipeline constructed successfully:", [step[0] for step in pipe.steps])`
            }
          }
        ],
        assignment: {
          title: "Clean a Raw Dataset",
          instructions: "Load a CSV with missing values, encode two categorical features with OneHotEncoder, and construct a pipeline that fits without data leakage."
        }
      },
      {
        id: "regression-eda",
        week: 3,
        lessonNumber: 6,
        title: "Data Preparation & Exploratory Data Analysis",
        shortDesc: "Hands-on data cleaning, unit normalization, and correlation discovery on the classic US pumpkin market dataset.",
        readTime: "16 min",
        category: "Regression",
        preQuizId: 11,
        postQuizId: 12,
        objectives: [
          "Normalize disparate unit measurements (e.g. 1 1/9 bushel cartons vs 1/2 bushel crates).",
          "Extract temporal cyclical signals from raw date fields.",
          "Calculate Pearson and Spearman correlation coefficients between features and price.",
          "Visualize price distributions and detect outliers using IQR fences."
        ],
        keyTerms: [
          { term: "Exploratory Data Analysis (EDA)", definition: "An approach to analyzing datasets to summarize their main characteristics, often with visual methods, before modeling." },
          { term: "IQR Outlier Rule", definition: "Points outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are considered statistical outliers." }
        ],
        mathFormulas: [
          {
            name: "Pearson Correlation Coefficient",
            formula: "r_{xy} = \\frac{\\sum_{i=1}^n (x_i - \\bar{x})(y_i - \\bar{y})}{\\sqrt{\\sum_{i=1}^n (x_i - \\bar{x})^2} \\sqrt{\\sum_{i=1}^n (y_i - \\bar{y})^2}}",
            description: "Measures linear correlation between two continuous variables, bounded between -1 and +1."
          }
        ],
        sections: [
          {
            title: "1. The Pumpkin Pricing Case Study",
            content: "In agricultural markets, produce arrives in diverse package sizes: 24-inch bins, 1/2 bushel crates, and 1 1/9 bushel cartons. To predict price accurately, we first calculate the normalized price per bushel: normalized_price = (low_price + high_price) / 2 / package_bushel_capacity."
          },
          {
            title: "2. Unit Normalization Code",
            content: "Normalizing pumpkin bushel prices in Pandas:",
            codeSnippet: {
              language: "python",
              title: "normalize_bushels.py",
              code: `import pandas as pd

df = pd.DataFrame({
    'Package': ['1 1/9 bushel cartons', '1/2 bushel cartons', 'bushel baskets'],
    'Low Price': [15.0, 10.0, 20.0],
    'High Price': [17.0, 12.0, 22.0]
})

df['Price'] = (df['Low Price'] + df['High Price']) / 2

# Apply package normalization factors
conversion = {
    '1 1/9 bushel cartons': 1 + (1/9),
    '1/2 bushel cartons': 0.5,
    'bushel baskets': 1.0
}
df['Bushel Factor'] = df['Package'].map(conversion)
df['Normalized Price'] = df['Price'] / df['Bushel Factor']

print(df[['Package', 'Price', 'Normalized Price']])`
            }
          }
        ],
        assignment: {
          title: "Bushel Normalization Exercise",
          instructions: "Extend the conversion mapping to handle 24 inch bins and 36 inch bins, assuming 24 inch bins correspond to approximately 10.5 bushels."
        }
      }
    ]
  },

  // ── WEEK 4 ──
  {
    week: 4,
    title: "Linear, Polynomial & Logistic Regression",
    description: "Deriving the Ordinary Least Squares normal equation, polynomial feature engineering, and the Sigmoid logistic model.",
    icon: "TrendingUp",
    lessons: [
      {
        id: "linear-regression-deep",
        week: 4,
        lessonNumber: 7,
        title: "Linear and Polynomial Regression",
        shortDesc: "From simple OLS to polynomial feature expansion and regularized Ridge/Lasso models.",
        readTime: "20 min",
        category: "Regression",
        preQuizId: 13,
        postQuizId: 14,
        objectives: [
          "Derive the closed-form OLS normal equation beta = (X^T X)^(-1) X^T y.",
          "Compute and interpret Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.",
          "Generate polynomial feature spaces and observe the onset of Runge's phenomenon / overfitting.",
          "Apply Ridge (L2) and Lasso (L1) regularization to stabilize ill-conditioned matrix inversions."
        ],
        keyTerms: [
          { term: "Ordinary Least Squares (OLS)", definition: "A method for estimating unknown parameters in a linear regression model by minimizing the sum of squared differences between observed and predicted values." },
          { term: "R-Squared (Coefficient of Determination)", definition: "The proportion of variance in the dependent variable that is predictable from the independent variables: R^2 = 1 - (SS_res / SS_tot)." }
        ],
        mathFormulas: [
          {
            name: "Normal Equation",
            formula: "\\hat{\\beta} = (X^T X)^{-1} X^T y",
            description: "Analytical solution giving the global minimum MSE for unconstrained linear regression."
          },
          {
            name: "Ridge Objective Function",
            formula: "J(\\beta) = \\frac{1}{2N} \\|y - X\\beta\\|_2^2 + \\alpha \\|\\beta\\|_2^2",
            description: "Adds an L2 penalty on weight magnitudes to prevent explosive coefficients."
          }
        ],
        sections: [
          {
            title: "1. Closed-Form Derivation vs. Gradient Descent",
            content: "Setting the gradient of the MSE loss with respect to beta to zero yields the Normal Equation: d/dbeta [ (y - X beta)^T (y - X beta) ] = -2 X^T (y - X beta) = 0 => X^T X beta = X^T y => beta = (X^T X)^(-1) X^T y. For large feature spaces (d > 10,000), computing (X^T X)^(-1) is O(d^3), making iterative gradient descent far more efficient."
          },
          {
            title: "2. Fitting Polynomial Curves in Python",
            content: "Comparing linear vs degree-2 polynomial regression on non-linear price trends:",
            codeSnippet: {
              language: "python",
              title: "polynomial_fit.py",
              code: `import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([2.1, 3.9, 8.2, 15.8, 25.4, 37.1, 50.2, 65.0])

# Linear Model
lin = LinearRegression().fit(X, y)
r2_lin = r2_score(y, lin.predict(X))

# Quadratic Model
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression().fit(X_poly, y)
r2_poly = r2_score(y, poly_model.predict(X_poly))

print(f"Linear R2: {r2_lin:.3f} | Polynomial Degree-2 R2: {r2_poly:.3f}")`
            }
          }
        ],
        assignment: {
          title: "Regularize Polynomial Overfitting",
          instructions: "Fit an 8th-degree polynomial on 10 noisy data points. Compare unregularized linear regression against Ridge regression with alpha=10.0 and report test MSE."
        }
      },
      {
        id: "logistic-regression-deep",
        week: 4,
        lessonNumber: 8,
        title: "Logistic Regression & Classification",
        shortDesc: "Map unbounded real values to calibrated probabilities using the Sigmoid function and log-odds.",
        readTime: "18 min",
        category: "Regression",
        preQuizId: 15,
        postQuizId: 16,
        objectives: [
          "Explain why linear regression fails for binary classification (probabilities < 0 or > 1).",
          "Derive the Sigmoid activation function from log-odds.",
          "Formulate Binary Cross-Entropy loss via Maximum Likelihood Estimation.",
          "Interpret model odds ratios: exp(beta_j)."
        ],
        keyTerms: [
          { term: "Log-Odds (Logit)", definition: "The natural logarithm of the odds of an event: ln(p / (1 - p)) = w^T x + b." },
          { term: "Sigmoid Function", definition: "A mathematical function having a characteristic S-shaped curve, mapping any real number into the open interval (0, 1)." }
        ],
        mathFormulas: [
          {
            name: "Sigmoid Probability Function",
            formula: "P(Y=1 \\mid X) = \\sigma(w^T x + b) = \\frac{1}{1 + e^{-(w^T x + b)}}",
            description: "Transforms a linear combination of inputs into a valid probability."
          },
          {
            name: "Binary Cross-Entropy Loss",
            formula: "\\mathcal{L}_{\\text{BCE}}(w) = -\\frac{1}{N} \\sum_{i=1}^N \\left[ y_i \\ln \\hat{y}_i + (1 - y_i) \\ln(1 - \\hat{y}_i) \\right]",
            description: "Negative log-likelihood loss penalized heavily when predicting the wrong class with high confidence."
          }
        ],
        sections: [
          {
            title: "1. The Geometry of the Decision Boundary",
            content: "The decision threshold is typically set at P(Y=1|X) = 0.5. Since sigma(0) = 0.5, this corresponds exactly to the hyperplane where w^T x + b = 0. Points on one side are classified as Class 1; points on the opposite side are Class 0."
          },
          {
            title: "2. Training Logistic Regression on Pumpkin Varieties",
            content: "Predicting whether a pumpkin is an Orange variety or a White variety:",
            codeSnippet: {
              language: "python",
              title: "pumpkin_logistic.py",
              code: `import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Features: [Bushel Price, Diameter cm]
X = np.array([[12.5, 20], [14.0, 22], [18.0, 30], [25.0, 40], [27.0, 42]])
y = np.array([0, 0, 0, 1, 1]) # 0: Orange, 1: White

clf = LogisticRegression()
clf.fit(X, y)

test_sample = np.array([[22.0, 35]])
prob = clf.predict_proba(test_sample)[0]
print(f"Prediction: {'White' if prob[1] >= 0.5 else 'Orange'}")
print(f"Probabilities -> Orange: {prob[0]:.2%}, White: {prob[1]:.2%}")`
            }
          }
        ],
        assignment: {
          title: "Adjusting Decision Thresholds",
          instructions: "Modify the classification threshold from 0.5 to 0.2 and 0.8. Explain the trade-off in terms of False Positives versus False Negatives."
        }
      }
    ]
  },

  // ── WEEK 5 ──
  {
    week: 5,
    title: "Web App Deployment & Model Serving",
    description: "Exporting trained models, building lightweight inference endpoints with Flask, and client-side ONNX scoring.",
    icon: "Globe",
    lessons: [
      {
        id: "web-app-deployment",
        week: 5,
        lessonNumber: 9,
        title: "Deploying ML Models as Web Applications",
        shortDesc: "Package models with Joblib and ONNX, build interactive REST APIs with Flask, and run browser-side inference.",
        readTime: "17 min",
        category: "Web-App",
        preQuizId: 17,
        postQuizId: 18,
        objectives: [
          "Serialize models using Joblib and the open ONNX (Open Neural Network Exchange) format.",
          "Build a minimal, low-latency Python Flask inference server.",
          "Handle CORS, input sanitization, and JSON request/response contracts.",
          "Run client-side inference directly in browser JavaScript with ONNX Runtime Web."
        ],
        keyTerms: [
          { term: "ONNX", definition: "Open Neural Network Exchange — an open format built to represent machine learning models across diverse frameworks and runtimes." },
          { term: "Model Inference", definition: "The phase where a trained model processes new input features and produces predictions in production." }
        ],
        sections: [
          {
            title: "1. Production Serialization: Joblib vs ONNX",
            content: "While Python `pickle` and `joblib` are standard for rapid local persistence, they pose major security risks (arbitrary code execution upon unpickling) and are locked to the Python ecosystem. ONNX compiles computational graphs into an interoperable protobuf format that can execute in C++, Rust, mobile runtimes, or in-browser JavaScript via WebAssembly."
          },
          {
            title: "2. Minimal Flask Model Server",
            content: "Serving predictions over a JSON REST endpoint:",
            codeSnippet: {
              language: "python",
              title: "app.py",
              code: `from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
# model = joblib.load('pumpkin_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json(force=True)
    features = np.array(payload['features']).reshape(1, -1)
    
    # In production: prediction = model.predict(features)[0]
    prediction = int(features[0, 0] > 20)
    return jsonify({
        'prediction': prediction,
        'label': 'White' if prediction == 1 else 'Orange'
    })

if __name__ == '__main__':
    print("Flask server running on port 5000")`
            }
          }
        ],
        assignment: {
          title: "Build a Client-Side ONNX Predictor",
          instructions: "Convert a trained Scikit-Learn LogisticRegression estimator to `.onnx` format using `skl2onnx` and verify its output matches the Python model."
        }
      }
    ]
  },

  // ── WEEK 6 ──
  {
    week: 6,
    title: "Classification Foundations & Classic Classifiers",
    description: "The Pan-Asian cuisine recipe dataset, one-hot encoding, and support vector machines vs K-nearest neighbors.",
    icon: "Sparkles",
    lessons: [
      {
        id: "intro-classification",
        week: 6,
        lessonNumber: 10,
        title: "Introduction to Multi-Class Classification",
        shortDesc: "The culinary dataset case study: predicting Thai, Japanese, Chinese, Indian, and Italian cuisine from ingredients.",
        readTime: "15 min",
        category: "Classification",
        preQuizId: 19,
        postQuizId: 20,
        objectives: [
          "Distinguish binary, multi-class, and multi-label classification.",
          "Clean sparse ingredient matrices and remove non-informative features.",
          "Apply One-vs-Rest (OvR) and One-vs-One (OvO) multi-class strategies.",
          "Identify signature ingredients using class-conditional probability."
        ],
        keyTerms: [
          { term: "Multi-Class Classification", definition: "A classification task with more than two classes where each instance is assigned one and only one label." },
          { term: "One-vs-Rest (OvR)", definition: "A heuristic method of using binary classification algorithms for multi-class classification by training N independent binary models." }
        ],
        sections: [
          {
            title: "1. The Cuisine Recipe Dataset",
            content: "Given a list of ingredients (e.g. ['garlic', 'soy sauce', 'ginger', 'scallions']), predict the cuisine of origin. Ingredient lists naturally form binary one-hot encoded vectors where column j = 1 if ingredient j is present and 0 otherwise. This produces wide, sparse feature spaces."
          },
          {
            title: "2. Ingredient Frequency Inspection",
            content: "Filtering out ingredients present across all cuisines:",
            codeSnippet: {
              language: "python",
              title: "cuisine_inspection.py",
              code: `import pandas as pd

# Synthetic representation of ingredient occurrences
df = pd.DataFrame({
    'cuisine': ['thai', 'japanese', 'italian', 'mexican', 'indian'],
    'garlic': [1, 1, 1, 1, 1], # ubiquitous -> non-discriminative
    'fish_sauce': [1, 0, 0, 0, 0], # highly discriminative for Thai
    'wasabi': [0, 1, 0, 0, 0],     # highly discriminative for Japanese
    'basil': [0, 0, 1, 0, 0]       # discriminative for Italian
})

# Ubiquitous features offer low information gain
informative_cols = [c for c in df.columns if c != 'cuisine' and df[c].std() > 0.4]
print("Informative discriminative features:", informative_cols)`
            }
          }
        ],
        assignment: {
          title: "Discriminative Ingredient Discovery",
          instructions: "Calculate the top 3 most unique ingredients for Mexican cuisine versus Indian cuisine based on frequency ratio."
        }
      },
      {
        id: "classifiers-1",
        week: 6,
        lessonNumber: 11,
        title: "Classifiers 1: Linear SVC & K-Nearest Neighbors",
        shortDesc: "Margin maximization hyperplanes with LinearSVC and instance-based proximity voting with K-NN.",
        readTime: "18 min",
        category: "Classification",
        preQuizId: 21,
        postQuizId: 22,
        objectives: [
          "Formulate Support Vector Machines as a maximum margin optimization problem.",
          "Understand the role of the penalty parameter C in balancing margin width against training error.",
          "Explain the K-Nearest Neighbors voting rule and distance metrics.",
          "Recognize why K-NN suffers from the curse of dimensionality on wide sparse text matrices."
        ],
        keyTerms: [
          { term: "Support Vectors", definition: "The data points that lie closest to the decision surface; removing them changes the position of the dividing hyperplane." },
          { term: "Curse of Dimensionality", definition: "In high-dimensional space, the distance between any two points approaches the same value, degrading distance-based algorithms like K-NN." }
        ],
        mathFormulas: [
          {
            name: "Soft Margin SVM Primal Objective",
            formula: "\\min_{w, b, \\xi} \\frac{1}{2} \\|w\\|^2 + C \\sum_{i=1}^N \\xi_i \\quad \\text{s.t.} \\quad y_i(w^T x_i + b) \\ge 1 - \\xi_i",
            description: "Minimizes inverse margin while penalizing slack errors with weight C."
          }
        ],
        sections: [
          {
            title: "1. SVM vs. K-NN: Parametric vs. Non-Parametric",
            content: "Linear SVC is parametric: once training completes, we only store weight vector w and bias b. Prediction is an O(d) dot product. K-NN is non-parametric / lazy: it stores all training instances in memory and performs distance calculations O(N d) during inference."
          },
          {
            title: "2. Training Linear SVC on Sparse Data",
            content: "Benchmarking LinearSVC against KNeighborsClassifier:",
            codeSnippet: {
              language: "python",
              title: "svm_vs_knn.py",
              code: `from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

X_train = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]])
y_train = np.array([0, 1, 0, 1])

svc = LinearSVC(C=1.0, random_state=42).fit(X_train, y_train)
knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)

sample = np.array([[1, 0, 0]])
print(f"LinearSVC Prediction: {svc.predict(sample)[0]}")
print(f"K-NN Prediction:      {knn.predict(sample)[0]}")`
            }
          }
        ],
        assignment: {
          title: "K-NN Distance Metric Comparison",
          instructions: "Compare Euclidean versus Cosine distance on binary recipe vectors and explain why Cosine distance often outperforms Euclidean distance on sparse counts."
        }
      }
    ]
  },

  // ── WEEK 7 ──
  {
    week: 7,
    title: "Trees, Ensembles & Applied Model Diagnostics",
    description: "Decision tree impurity splits, Random Forests, AdaBoost ensembles, and the complete confusion matrix toolkit.",
    icon: "Network",
    lessons: [
      {
        id: "classifiers-2",
        week: 7,
        lessonNumber: 12,
        title: "Classifiers 2: Decision Trees & Ensembles",
        shortDesc: "Information gain, Gini impurity, bagging variance reduction in Random Forests, and sequential boosting.",
        readTime: "20 min",
        category: "Classification",
        preQuizId: 23,
        postQuizId: 24,
        objectives: [
          "Compute Gini Impurity and Shannon Entropy for prospective binary splits.",
          "State how Random Forests achieve variance reduction via bootstrap aggregation and feature subsampling.",
          "Distinguish Bagging (independent parallel trees) from Boosting (sequential residual correction).",
          "Interpret Mean Decrease in Impurity (MDI) feature importances."
        ],
        keyTerms: [
          { term: "Gini Impurity", definition: "A measure of how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset." },
          { term: "Random Forest", definition: "An ensemble learning method for classification and regression that operates by constructing a multitude of decision trees at training time." }
        ],
        mathFormulas: [
          {
            name: "Gini Impurity Formula",
            formula: "I_G(p) = 1 - \\sum_{i=1}^C p_i^2",
            description: "Equals 0 when a node is pure (all samples belong to one class)."
          },
          {
            name: "Shannon Entropy",
            formula: "H(p) = -\\sum_{i=1}^C p_i \\log_2(p_i)",
            description: "Information-theoretic measure of disorder and uncertainty in a partition."
          }
        ],
        sections: [
          {
            title: "1. Why Ensembles Beat Single Decision Trees",
            content: "Single decision trees are prone to high variance: a small change in training data can drastically alter split choices near the root. Random Forests solve this by generating B bootstrap datasets, training an unpruned tree on each with random feature subsets, and averaging predictions."
          },
          {
            title: "2. Random Forest Feature Importance Code",
            content: "Extracting feature importances from a fitted Random Forest:",
            codeSnippet: {
              language: "python",
              title: "forest_importance.py",
              code: `from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Synthetic features: [cumin, fish_sauce, parmesan, tortilla]
X = np.array([
    [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
    [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
])
y = np.array([0, 1, 2, 3, 0, 1, 2, 3]) # Indian, Thai, Italian, Mexican

rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X, y)

feature_names = ['cumin', 'fish_sauce', 'parmesan', 'tortilla']
for name, imp in zip(feature_names, rf.feature_importances_):
    print(f"Feature: {name:12s} | Importance: {imp:.3f}")`
            }
          }
        ],
        assignment: {
          title: "Compute Manual Gini Split",
          instructions: "Given a node with 4 positive and 6 negative samples, calculate Gini impurity before a split. If split A separates into [4+, 1-] and [0+, 5-], compute the weighted post-split Gini."
        }
      },
      {
        id: "applied-classification",
        week: 7,
        lessonNumber: 13,
        title: "Applied Classification & Model Diagnostics",
        shortDesc: "The complete evaluation matrix: precision, recall, F1, ROC-AUC curves, and diagnosing imbalanced data.",
        readTime: "18 min",
        category: "Classification",
        preQuizId: 25,
        postQuizId: 26,
        objectives: [
          "Construct and interpret a confusion matrix (True Positives, False Positives, False Negatives, True Negatives).",
          "Explain why high accuracy can be meaningless on class-imbalanced datasets (Accuracy Paradox).",
          "Calculate Precision, Recall, and the harmonic F1-score.",
          "Interpret the Receiver Operating Characteristic (ROC) curve and Area Under Curve (AUC)."
        ],
        keyTerms: [
          { term: "Precision", definition: "Out of all instances the model labeled positive, how many were actually positive: TP / (TP + FP)." },
          { term: "Recall (Sensitivity)", definition: "Out of all actual positive instances, how many did the model correctly catch: TP / (TP + FN)." },
          { term: "ROC-AUC", definition: "Plots True Positive Rate against False Positive Rate across all classification thresholds; 0.5 represents random guessing and 1.0 is perfect separation." }
        ],
        mathFormulas: [
          {
            name: "F1 Score (Harmonic Mean)",
            formula: "F_1 = 2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}} = \\frac{2 \\text{TP}}{2 \\text{TP} + \\text{FP} + \\text{FN}}",
            description: "Harmonic mean ensuring the score is penalized severely if either precision or recall is low."
          }
        ],
        sections: [
          {
            title: "1. The Accuracy Paradox in Fraud & Health Care",
            content: "If only 0.1% of transactions are fraudulent, a trivial model that always predicts 'Not Fraud' achieves 99.9% accuracy while having a Recall of 0%. In high-stakes applications, false negatives can be catastrophic, making Precision-Recall tracking mandatory."
          },
          {
            title: "2. Full Scikit-Learn Classification Report",
            content: "Computing multi-class precision, recall, and F1 metrics:",
            codeSnippet: {
              language: "python",
              title: "diagnostic_report.py",
              code: `from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
y_pred = np.array([0, 2, 2, 0, 1, 1, 0, 1, 2])

target_names = ['Indian', 'Thai', 'Italian']
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=target_names))`
            }
          }
        ],
        assignment: {
          title: "Optimize for Recall",
          instructions: "In a medical diagnostic test where missing a disease has a 10x higher cost than a false positive alarm, adjust the threshold to maximize recall while keeping precision above 60%."
        }
      }
    ]
  },

  // ── WEEK 8 ──
  {
    week: 8,
    title: "Clustering Algorithms & Unsupervised Learning",
    description: "Discovering latent natural groupings in high-dimensional space without labels using K-Means and the Elbow method on Spotify music features.",
    icon: "Boxes",
    lessons: [
      {
        id: "intro-clustering",
        week: 8,
        lessonNumber: 14,
        title: "Introduction to Clustering & Visualizations",
        shortDesc: "Exploring latent geometric groupings, the importance of feature standardization, and high-dimensional projections.",
        readTime: "15 min",
        category: "Clustering",
        preQuizId: 27,
        postQuizId: 28,
        objectives: [
          "Explain the difference between supervised classification and unsupervised clustering.",
          "Identify when Euclidean distance breaks down due to unscaled features.",
          "Interpret pairwise scatter plots and box plots for multidimensional cluster tendency.",
          "Formulate clustering use cases: customer segmentation, anomaly detection, document grouping."
        ],
        keyTerms: [
          { term: "Clustering", definition: "The unsupervised task of grouping a set of objects such that objects in the same group (cluster) are more similar to each other than to those in other groups." },
          { term: "Euclidean Distance", definition: "The straight-line distance between two points in Euclidean space: d(p, q) = sqrt(sum (p_i - q_i)^2)." }
        ],
        sections: [
          {
            title: "1. Why Scaling is Mandatory Before Clustering",
            content: "If one feature is Song Tempo (range 60–180 BPM) and another is Danceability (range 0.0–1.0), the unscaled Euclidean distance will be almost 100% dominated by the tempo column. StandardScaler transforms each feature to mean 0 and variance 1, restoring equal dimensional weighting."
          },
          {
            title: "2. Data Scaling for Clustering",
            content: "Applying StandardScaler before distance calculation:",
            codeSnippet: {
              language: "python",
              title: "scale_clustering.py",
              code: `import numpy as np
from sklearn.preprocessing import StandardScaler

raw_audio = np.array([
    [120.0, 0.85], # 120 BPM, high danceability
    [75.0,  0.20], # 75 BPM, low danceability
    [130.0, 0.90]  # 130 BPM, high danceability
])

scaler = StandardScaler()
scaled_audio = scaler.fit_transform(raw_audio)

print("Raw Matrix:\n", raw_audio)
print("Standardized Matrix (Mean 0, Var 1):\n", np.round(scaled_audio, 2))`
            }
          }
        ],
        assignment: {
          title: "Scale Sensitivity Audit",
          instructions: "Calculate the Euclidean distance between song 1 and song 3 before and after standard scaling. Note how the relative distance changes dramatically."
        }
      },
      {
        id: "kmeans-clustering",
        week: 8,
        lessonNumber: 15,
        title: "K-Means Clustering in Practice",
        shortDesc: "Lloyd's heuristic, centroid updates, the Elbow method for choosing k, and silhouette score analysis.",
        readTime: "19 min",
        category: "Clustering",
        preQuizId: 29,
        postQuizId: 30,
        objectives: [
          "State the steps of Lloyd's algorithm: Assignment step and Update step.",
          "Compute the Within-Cluster Sum of Squares (Inertia) objective.",
          "Identify the optimal number of clusters k using the Elbow Method.",
          "Evaluate cluster separation using the Silhouette Coefficient."
        ],
        keyTerms: [
          { term: "Inertia (WCSS)", definition: "Within-Cluster Sum of Squares — the sum of squared distances of samples to their closest cluster center." },
          { term: "K-Means++", definition: "A smart centroid initialization scheme that spreads initial centers far apart, speeding up convergence and avoiding bad local optima." },
          { term: "Silhouette Score", definition: "Measures how similar an object is to its own cluster compared to other clusters, ranging from -1 (poor) to +1 (dense, well-separated)." }
        ],
        mathFormulas: [
          {
            name: "K-Means Objective (WCSS)",
            formula: "J = \\sum_{k=1}^K \\sum_{x_i \\in S_k} \\|x_i - \\mu_k\\|^2",
            description: "Sum of squared Euclidean distances from each data point to its assigned centroid mu_k."
          },
          {
            name: "Silhouette Coefficient",
            formula: "s(i) = \\frac{b(i) - a(i)}{\\max(a(i), b(i))}",
            description: "a(i) is mean intra-cluster distance; b(i) is mean nearest-cluster distance."
          }
        ],
        sections: [
          {
            title: "1. The Iterative Lloyd's Cycle",
            content: "1. Initialize k centroids mu_1, ..., mu_k (using K-Means++). 2. Assignment: Assign each sample x_i to the closest centroid: c_i = argmin_j ||x_i - mu_j||^2. 3. Update: Recompute each centroid as the arithmetic mean of all assigned samples: mu_j = (1 / |S_j|) sum_{i in S_j} x_i. 4. Repeat until centroids stop shifting."
          },
          {
            title: "2. Finding the Elbow with Scikit-Learn",
            content: "Plotting inertia across multiple values of k:",
            codeSnippet: {
              language: "python",
              title: "kmeans_elbow.py",
              code: `from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

X = np.random.randn(150, 4)

inertias = []
silhouettes = []
k_range = range(2, 7)

for k in k_range:
    model = KMeans(n_clusters=k, init='k-means++', random_state=42)
    model.fit(X)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X, model.labels_))

for k, inrt, sil in zip(k_range, inertias, silhouettes):
    print(f"k={k} | Inertia: {inrt:.2f} | Silhouette Score: {sil:.3f}")`
            }
          }
        ],
        assignment: {
          title: "Cluster Music Genres",
          instructions: "Cluster the Nigerian Spotify dataset into 3 clusters. Print the average acousticness and danceability for each cluster center and describe the music profile."
        }
      }
    ]
  },

  // ── WEEK 9 ──
  {
    week: 9,
    title: "Natural Language Processing (NLP) Foundations",
    description: "Computational linguistics, text tokenization, stop word filtering, POS tagging, and TF-IDF matrices.",
    icon: "MessageSquare",
    lessons: [
      {
        id: "intro-nlp",
        week: 9,
        lessonNumber: 16,
        title: "Introduction to NLP & Text Processing",
        shortDesc: "How computers parse human language: tokenization, stop words, stemming (Porter) vs lemmatization (WordNet).",
        readTime: "16 min",
        category: "NLP",
        preQuizId: 31,
        postQuizId: 32,
        objectives: [
          "Explain the challenges of human language: ambiguity, polysemy, idiom, and context.",
          "Implement tokenization using whitespace, regex, and subword rules.",
          "Filter non-informative stop words safely.",
          "Distinguish rule-based algorithmic stemming from dictionary-based lemmatization."
        ],
        keyTerms: [
          { term: "Tokenization", definition: "The process of segmenting running text into discrete units (tokens) such as words, punctuation, or subwords." },
          { term: "Stemming", definition: "A crude heuristic process that chops off the ends of words in the hope of achieving a base form (e.g. 'running' -> 'run', 'studies' -> 'studi')." },
          { term: "Lemmatization", definition: "The algorithmic process of determining the lemma of a word based on its intended meaning and grammatical part of speech (e.g. 'better' -> 'good')." }
        ],
        sections: [
          {
            title: "1. Stemming vs. Lemmatization",
            content: "Stemmers like Porter or Snowball apply hardcoded phonetic string-chopping rules without morphological dictionaries. Lemmatizers use morphological vocabularies and part-of-speech context, ensuring words like 'was' reduce correctly to 'be', whereas stemmers fail."
          },
          {
            title: "2. Tokenization and Normalization in Python",
            content: "Comparing NLTK word tokenization and lemmatization:",
            codeSnippet: {
              language: "python",
              title: "nlp_preprocess.py",
              code: `import re

raw_text = "The quick brown foxes were jumping over 10 lazy dogs! Isn't that amazing?"

# 1. Regex tokenization
tokens = re.findall(r"\\b[A-Za-z]+\\b", raw_text.lower())
print("Lowercased Tokens:", tokens)

# 2. Stop words filtering
stop_words = {"the", "were", "over", "that", "is"}
filtered = [w for w in tokens if w not in stop_words]
print("After Stop Word Removal:", filtered)`
            }
          }
        ],
        assignment: {
          title: "Build a Custom Lemmatizer Pipeline",
          instructions: "Process a paragraph of text. Compare output vocabulary sizes between raw tokens, stemmed tokens, and lemmatized tokens."
        }
      },
      {
        id: "nlp-tasks",
        week: 9,
        lessonNumber: 17,
        title: "NLP Tasks: POS Tagging, NER & TF-IDF",
        shortDesc: "Extract semantic and syntactic structures with Part-of-Speech tags, Named Entity Recognition, and TF-IDF vectors.",
        readTime: "18 min",
        category: "NLP",
        preQuizId: 33,
        postQuizId: 34,
        objectives: [
          "Understand Part-of-Speech (POS) tagging and Named Entity Recognition (NER).",
          "Convert text corpora into numerical matrices using Bag-of-Words.",
          "Derive and compute Term Frequency-Inverse Document Frequency (TF-IDF).",
          "Calculate Cosine Similarity between document embedding vectors."
        ],
        keyTerms: [
          { term: "Named Entity Recognition (NER)", definition: "An information extraction task that seeks to locate and classify named entities in text into predefined categories (persons, organizations, locations)." },
          { term: "TF-IDF", definition: "A numerical statistic that reflects how important a word is to a document in a collection or corpus." }
        ],
        mathFormulas: [
          {
            name: "TF-IDF Formula",
            formula: "\\text{TF-IDF}(t, d, D) = \\text{TF}(t, d) \\times \\ln\\left(\\frac{1 + |D|}{1 + |\\{d \\in D : t \\in d\\}|}\\right) + 1",
            description: "Penalizes terms that appear uniformly across every document in the corpus D."
          },
          {
            name: "Cosine Similarity",
            formula: "\\text{CosineSim}(u, v) = \\frac{u \\cdot v}{\\|u\\| \\|v\\|} = \\frac{\\sum u_i v_i}{\\sqrt{\\sum u_i^2} \\sqrt{\\sum v_i^2}}",
            description: "Measures orientation angle between two non-zero vectors in high-dimensional space."
          }
        ],
        sections: [
          {
            title: "1. Why TF-IDF Beats Raw Word Counts",
            content: "Raw count matrices (Bag-of-Words) award the highest weights to frequent words ('said', 'would', 'also'). TF-IDF scales term frequency by the logarithm of inverse document frequency, elevating words that are uniquely characteristic of a specific document."
          },
          {
            title: "2. Scikit-Learn TF-IDF Implementation",
            content: "Computing TF-IDF vectors and pairwise document similarity:",
            codeSnippet: {
              language: "python",
              title: "tfidf_similarity.py",
              code: `from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus = [
    "Machine learning models require clean training datasets.",
    "Deep neural networks require GPU acceleration for fast training.",
    "Baking Italian sourdough bread requires flour, water, and yeast."
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
print("Similarity to Document 0 (ML):")
for idx, s in enumerate(sim[0]):
    print(f"Doc {idx}: {s:.3f}")`
            }
          }
        ],
        assignment: {
          title: "Document Search Engine",
          instructions: "Given a collection of 5 articles, build a search function that accepts a user query string, computes TF-IDF cosine similarity, and returns the top matching article."
        }
      }
    ]
  },

  // ── WEEK 10 ──
  {
    week: 10,
    title: "Translation & Sentiment Analysis with Real-World Review Data",
    description: "Rule-based sentiment with VADER, followed by training supervised classifiers on 500,000 European Hotel Reviews.",
    icon: "Smile",
    lessons: [
      {
        id: "translation-sentiment",
        week: 10,
        lessonNumber: 18,
        title: "Translation and Sentiment Analysis",
        shortDesc: "VADER rule-based sentiment scoring, polarity versus subjectivity in TextBlob, and translation concepts.",
        readTime: "16 min",
        category: "NLP",
        preQuizId: 35,
        postQuizId: 36,
        objectives: [
          "Trace the evolution from rule-based machine translation to statistical and neural models.",
          "Distinguish sentiment polarity (-1 to +1) from subjectivity (0 to 1).",
          "Apply VADER heuristic rules (capitalization, punctuation, intensifiers) to customer sentiment.",
          "Recognize the limitations of dictionary-based sentiment when encountering sarcasm or domain jargon."
        ],
        keyTerms: [
          { term: "Sentiment Analysis", definition: "The use of natural language processing to systematically identify, extract, and quantify affective states and subjective information." },
          { term: "VADER", definition: "Valence Aware Dictionary and sEntiment Reasoner — a lexicon and rule-based sentiment analysis tool specifically attuned to social media text." }
        ],
        sections: [
          {
            title: "1. Lexicon Sentiment: Polarity vs Subjectivity",
            content: "TextBlob evaluates text on two independent continuous axes: Polarity: [-1.0 (strongly negative) to +1.0 (strongly positive)]. Subjectivity: [0.0 (strictly objective fact) to 1.0 (pure personal opinion)]. For instance, 'The laptop weighs 1.2 kg' has polarity 0.0, subjectivity 0.0. 'The laptop is gorgeous and featherlight!' has polarity +0.8, subjectivity 0.9."
          },
          {
            title: "2. VADER Sentiment Scoring Code",
            content: "Using rule-based sentiment weighting:",
            codeSnippet: {
              language: "python",
              title: "vader_demo.py",
              code: `from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
# nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()
reviews = [
    "The room was pristine and the staff were exceptionally welcoming!",
    "Terrible customer service, noisy air conditioning, and overpriced food.",
    "The flight was on time."
]

for r in reviews:
    scores = sia.polarity_scores(r)
    print(f"Review: {r[:40]}... -> Compound: {scores['compound']:+.3f}")`
            }
          }
        ],
        assignment: {
          title: "Sarcasm Failure Modes",
          instructions: "Test VADER on three sarcastic reviews (e.g. 'Oh wonderful, another flight cancellation!'). Document why rule-based lexicons misclassify sarcastic statements."
        }
      },
      {
        id: "hotel-reviews-1",
        week: 10,
        lessonNumber: 19,
        title: "Hotel Reviews Sentiment Analysis Part 1",
        shortDesc: "Data cleaning and exploratory sentiment profiling on real European hotel reviews.",
        readTime: "17 min",
        category: "NLP",
        preQuizId: 37,
        postQuizId: 38,
        objectives: [
          "Clean large real-world text corpora (handling 'No Negative' / 'No Positive' placeholder entries).",
          "Filter out non-English reviews using language detection heuristics.",
          "Examine correlations between reviewer rating scores and text length.",
          "Generate frequency distributions of positive versus negative adjectives."
        ],
        keyTerms: [
          { term: "Corpus Cleaning", definition: "Removing boilerplate text, automated placeholders, HTML tags, and metadata artifacts from natural language datasets." }
        ],
        sections: [
          {
            title: "1. The European Hotel Reviews Dataset",
            content: "The dataset contains 515,000 luxury hotel reviews across Europe. A common data anomaly is that reviewers who had zero complaints often entered literal strings like 'No Negative' or 'None' in the negative review field. If untreated, models learn that 'Negative' is a negative predictor!"
          },
          {
            title: "2. Cleaning Review Placeholders",
            content: "Sanitizing filler entries in Pandas:",
            codeSnippet: {
              language: "python",
              title: "clean_hotel_reviews.py",
              code: `import pandas as pd

df = pd.DataFrame({
    'Negative_Review': ['No Negative', 'Room was tiny and dirty', 'None', 'Rude front desk'],
    'Reviewer_Score': [9.5, 4.0, 8.5, 3.5]
})

# Strip whitespace and replace common placeholders
placeholders = {'no negative', 'none', 'nothing', 'n a'}
df['Negative_Clean'] = df['Negative_Review'].apply(
    lambda x: '' if x.strip().lower() in placeholders else x
)

print(df[['Negative_Review', 'Negative_Clean']])`
            }
          }
        ],
        assignment: {
          title: "Compute Sentiment vs Score Correlation",
          instructions: "Calculate the correlation between review word count and the reviewer score. Test the hypothesis: 'Angry reviewers write longer reviews than satisfied reviewers'."
        }
      },
      {
        id: "hotel-reviews-2",
        week: 10,
        lessonNumber: 20,
        title: "Hotel Reviews Sentiment Analysis Part 2",
        shortDesc: "Train supervised classification models (Logistic Regression, Naive Bayes) to predict sentiment scores from review text.",
        readTime: "19 min",
        category: "NLP",
        preQuizId: 39,
        postQuizId: 40,
        objectives: [
          "Convert review text into TF-IDF feature matrices with n-gram ranges (1, 2).",
          "Train and compare Logistic Regression and Multinomial Naive Bayes classifiers.",
          "Inspect model coefficients to discover the most predictive positive and negative phrases.",
          "Evaluate model generalization with out-of-sample test splits and ROC-AUC."
        ],
        keyTerms: [
          { term: "N-Gram", definition: "A contiguous sequence of n items from a given sample of text or speech (e.g. unigram: 'not', bigram: 'not clean')." }
        ],
        sections: [
          {
            title: "1. The Power of Bigrams in Sentiment",
            content: "Unigram models ('bag of words') treat 'not clean' as two isolated tokens 'not' and 'clean', losing the negating syntactic structure. Using TfidfVectorizer(ngram_range=(1, 2)) captures bigrams like 'not clean', 'friendly staff', and 'poor value', substantially boosting sentiment accuracy."
          },
          {
            title: "2. Supervised Sentiment Pipeline in Python",
            content: "End-to-end TF-IDF and Logistic Regression training pipeline:",
            codeSnippet: {
              language: "python",
              title: "train_sentiment_pipeline.py",
              code: `from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

reviews = [
    "Amazing stay, loved the spacious room and friendly staff",
    "Disgusting bathroom, broken aircon, and zero customer support",
    "Clean quiet room in a central neighborhood, good breakfast",
    "Never coming back, waited two hours to check in and room smelled"
]
labels = [1, 0, 1, 0] # 1: Positive, 0: Negative

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', LogisticRegression())
])

pipeline.fit(reviews, labels)
test_review = ["The bed was comfortable and shower was great"]
pred = pipeline.predict(test_review)
print("Predicted Sentiment:", "Positive" if pred[0] == 1 else "Negative")`
            }
          }
        ],
        assignment: {
          title: "Inspect Top Logistic Coefficients",
          instructions: "Extract the top 10 most positive coefficients and top 10 most negative coefficients from the fitted model pipeline."
        }
      }
    ]
  },

  // ── WEEK 11 ──
  {
    week: 11,
    title: "Time Series Analysis & Forecasting",
    description: "Decomposing temporal trends, seasonality, stationarity, ARIMA models, and Support Vector Regressors (SVR).",
    icon: "Clock",
    lessons: [
      {
        id: "intro-time-series",
        week: 11,
        lessonNumber: 21,
        title: "Introduction to Time Series: Seasonality & Stationarity",
        shortDesc: "Understand temporal ordering, additive vs multiplicative decomposition, ACF/PACF, and the Augmented Dickey-Fuller test.",
        readTime: "18 min",
        category: "TimeSeries",
        preQuizId: 41,
        postQuizId: 42,
        objectives: [
          "Identify the core components of time series: Trend, Seasonality, Cycles, and Irregular residuals.",
          "Distinguish additive decomposition (Y = T + S + I) from multiplicative decomposition (Y = T * S * I).",
          "Define covariance stationarity and explain why non-stationary series invalidate standard statistical inference.",
          "Interpret the Augmented Dickey-Fuller (ADF) statistical test."
        ],
        keyTerms: [
          { term: "Stationarity", definition: "A time series property where mean, variance, and autocorrelation structure are constant over time." },
          { term: "Autocorrelation (ACF)", definition: "The correlation between values of the process at different times as a function of the two times or the time lag." },
          { term: "Augmented Dickey-Fuller Test", definition: "A statistical hypothesis test for the presence of a unit root (non-stationarity) in a time series." }
        ],
        mathFormulas: [
          {
            name: "Additive vs Multiplicative Decomposition",
            formula: "Y_t = T_t + S_t + I_t \\quad \\text{vs} \\quad Y_t = T_t \\times S_t \\times I_t",
            description: "Additive applies when seasonal fluctuations remain constant regardless of level; multiplicative applies when seasonal swings grow proportionally with trend."
          }
        ],
        sections: [
          {
            title: "1. Why Standard Cross-Validation Fails for Time Series",
            content: "Standard K-Fold cross-validation randomly shuffles data points, using future observations to predict the past. This causes massive data leakage. Time series models must use rolling-origin or Walk-Forward validation where the training set strictly precedes the validation window."
          },
          {
            title: "2. Stationarity Check with Augmented Dickey-Fuller",
            content: "Running the ADF statistical test in Python:",
            codeSnippet: {
              language: "python",
              title: "adf_test.py",
              code: `import numpy as np
from statsmodels.tsa.stattools import adfuller

# Generate a non-stationary random walk with trend
np.random.seed(42)
trend = np.linspace(10, 50, 100)
noise = np.random.randn(100)
non_stationary_series = trend + noise

result = adfuller(non_stationary_series)
print(f"ADF Statistic: {result[0]:.3f}")
print(f"p-value:       {result[1]:.4f}")
print("Is Stationary?" , result[1] <= 0.05)`
            }
          }
        ],
        assignment: {
          title: "Make a Series Stationary via Differencing",
          instructions: "Apply first-order differencing: Y'_t = Y_t - Y_{t-1}. Rerun the ADF test and confirm that the p-value drops below 0.05."
        }
      },
      {
        id: "arima-forecasting",
        week: 11,
        lessonNumber: 22,
        title: "Time Series Forecasting with ARIMA & SARIMAX",
        shortDesc: "Autoregressive Integrated Moving Average models, AIC minimization, and Walk-Forward validation.",
        readTime: "20 min",
        category: "TimeSeries",
        preQuizId: 43,
        postQuizId: 44,
        objectives: [
          "Deconstruct the three parameters of ARIMA(p, d, q).",
          "Identify autoregressive order p from PACF plots and moving average order q from ACF plots.",
          "Fit seasonal SARIMAX models with exogenous regressors.",
          "Evaluate forecasting accuracy using Mean Absolute Percentage Error (MAPE) and RMSE."
        ],
        keyTerms: [
          { term: "ARIMA(p, d, q)", definition: "Autoregressive (p lags), Integrated (d orders of differencing), Moving Average (q lagged forecast error terms)." },
          { term: "SARIMA", definition: "Seasonal ARIMA — incorporates seasonal autoregressive, differencing, and moving average terms with period s." },
          { term: "Akaike Information Criterion (AIC)", definition: "An estimator of out-of-sample prediction error; penalizes model complexity to prevent overfitting." }
        ],
        mathFormulas: [
          {
            name: "ARIMA(p, d, q) Mathematical Formulation",
            formula: "\\left(1 - \\sum_{i=1}^p \\phi_i B^i\\right) (1 - B)^d Y_t = c + \\left(1 + \\sum_{j=1}^q \\theta_j B^j\\right) \\epsilon_t",
            description: "B is the backshift lag operator where B^k Y_t = Y_{t-k}."
          }
        ],
        sections: [
          {
            title: "1. The Parameter Selection Recipe (p, d, q)",
            content: "1. Choose differencing order d until ADF p-value < 0.05. 2. Inspect PACF: if it cuts off sharply after lag p, choose AR(p). 3. Inspect ACF: if it cuts off sharply after lag q, choose MA(q). 4. Use auto_arima to verify model order that minimizes AIC."
          },
          {
            title: "2. Fitting an ARIMA Model in Python",
            content: "Training statsmodels ARIMA and generating confidence intervals:",
            codeSnippet: {
              language: "python",
              title: "fit_arima.py",
              code: `from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Stationary autoregressive synthetic series
np.random.seed(42)
series = np.sin(np.linspace(0, 20, 100)) + np.random.normal(0, 0.2, 100)

model = ARIMA(series, order=(2, 0, 1))
fitted = model.fit()

forecast_res = fitted.get_forecast(steps=5)
print("Forecast mean:", np.round(forecast_res.predicted_mean, 2))
print("Confidence interval:\n", np.round(forecast_res.conf_int(), 2))`
            }
          }
        ],
        assignment: {
          title: "Walk-Forward Validation Loop",
          instructions: "Implement an expanding window walk-forward validation loop that fits an ARIMA model and forecasts the next 1 step over 20 iterations, reporting overall RMSE."
        }
      },
      {
        id: "svr-time-series",
        week: 11,
        lessonNumber: 23,
        title: "Support Vector Regressors (SVR) for Time Series",
        shortDesc: "Transforming sequential data into supervised sliding window features and forecasting non-linear trends with SVR.",
        readTime: "18 min",
        category: "TimeSeries",
        preQuizId: 51,
        postQuizId: 52,
        objectives: [
          "Convert temporal sequences into supervised tabular matrices using sliding lag windows.",
          "Explain the epsilon-insensitive tube loss function in Support Vector Regression.",
          "Apply the non-linear Radial Basis Function (RBF) kernel to complex time series.",
          "Compare classical ARIMA against machine learning regressors on non-linear load demand."
        ],
        keyTerms: [
          { term: "Epsilon-Insensitive Loss", definition: "A loss function that penalizes errors only if they exceed a defined threshold epsilon, conferring robustness to noise." },
          { term: "Sliding Window Transformation", definition: "The process of framing historical sequential observations [y_{t-k}, ..., y_{t-1}] as feature vectors X to predict target y_t." }
        ],
        mathFormulas: [
          {
            name: "Epsilon-Insensitive Loss Function",
            formula: "L_\\epsilon(y, \\hat{y}) = \\max(0, |y - \\hat{y}| - \\epsilon)",
            description: "Ignores small prediction discrepancies within distance epsilon of the true target."
          }
        ],
        sections: [
          {
            title: "1. The Sliding Lag Window Technique",
            content: "To apply standard machine learning algorithms like SVR, Random Forest, or XGBoost to time series, we construct lag features. For lag=3: Row 1 uses [y1, y2, y3] to predict y4. Row 2 uses [y2, y3, y4] to predict y5."
          },
          {
            title: "2. Sliding Window & SVR in Python",
            content: "Building lag features and fitting an SVR model:",
            codeSnippet: {
              language: "python",
              title: "svr_forecasting.py",
              code: `import numpy as np
from sklearn.svm import SVR

# Generate continuous series
series = np.sin(np.linspace(0, 10, 50))

# Create lag-3 feature matrix
lag = 3
X, y = [], []
for i in range(len(series) - lag):
    X.append(series[i : i + lag])
    y.append(series[i + lag])

X, y = np.array(X), np.array(y)

# Fit non-linear SVR
svr = SVR(kernel='rbf', C=10.0, epsilon=0.05)
svr.fit(X, y)

next_pred = svr.predict(series[-lag:].reshape(1, -1))
print("Next predicted time step value:", float(next_pred[0]))`
            }
          }
        ],
        assignment: {
          title: "Multi-Step Forecasting Comparison",
          instructions: "Compare recursive single-step forecasting against direct multi-output forecasting using SVR over a 5-step horizon."
        }
      }
    ]
  },

  // ── WEEK 12 ──
  {
    week: 12,
    title: "Reinforcement Learning & Real-World Machine Learning",
    description: "Tabular Q-learning, the Bellman equation, OpenAI Gym CartPole balance control, and detecting production data drift.",
    icon: "Rocket",
    lessons: [
      {
        id: "intro-reinforcement",
        week: 12,
        lessonNumber: 24,
        title: "Reinforcement Learning & Q-Learning",
        shortDesc: "Agents, states, actions, rewards, the Bellman optimality equation, and epsilon-greedy exploration in a gridworld.",
        readTime: "20 min",
        category: "Reinforcement",
        preQuizId: 45,
        postQuizId: 46,
        objectives: [
          "Define the Markov Decision Process (MDP) tuple: (S, A, P, R, gamma).",
          "Differentiate model-free learning from model-based planning.",
          "State the Bellman Optimality Equation for state-action value functions Q*(s, a).",
          "Implement the tabular Q-Learning update rule with epsilon-greedy exploration."
        ],
        keyTerms: [
          { term: "Markov Decision Process (MDP)", definition: "A discrete-time stochastic control process providing a mathematical framework for modeling decision making where outcomes are partly random and partly under control of an agent." },
          { term: "Q-Learning", definition: "A model-free reinforcement learning algorithm used to find the optimal action-selection policy for any given finite MDP." },
          { term: "Epsilon-Greedy Strategy", definition: "Balancing exploration (choosing a random action with probability epsilon) with exploitation (choosing the best known action with probability 1 - epsilon)." }
        ],
        mathFormulas: [
          {
            name: "Bellman Optimality Equation",
            formula: "Q^*(s, a) = R(s, a) + \\gamma \\sum_{s'} P(s' \\mid s, a) \\max_{a'} Q^*(s', a')",
            description: "The optimal value of state-action pair (s, a) equals immediate reward plus discounted expected future return."
          },
          {
            name: "Q-Learning Update Rule",
            formula: "Q(s, a) \\leftarrow Q(s, a) + \\alpha \\left[ r + \\gamma \\max_{a'} Q(s', a') - Q(s, a) \\right]",
            description: "Updates current Q value toward the TD target by step size alpha."
          }
        ],
        sections: [
          {
            title: "1. The Perception-Action Reward Loop",
            content: "In Reinforcement Learning, the agent is not told which actions to take. Instead, it must discover which actions yield the highest reward by trying them. Discount factor gamma (0 < gamma < 1) balances immediate reward against distant long-term consequences."
          },
          {
            title: "2. Tabular Q-Learning in Python",
            content: "Simple 1D Gridworld Q-learning implementation:",
            codeSnippet: {
              language: "python",
              title: "q_learning_grid.py",
              code: `import numpy as np

# 5 states: [0, 1, 2, 3, 4 (Goal)]. Actions: 0: Left, 1: Right
Q = np.zeros((5, 2))
alpha = 0.1
gamma = 0.9
epsilon = 0.2

for episode in range(200):
    state = 0
    while state != 4:
        # Epsilon-greedy action selection
        action = np.random.choice([0, 1]) if np.random.rand() < epsilon else np.argmax(Q[state])
        next_state = max(0, state - 1) if action == 0 else min(4, state + 1)
        reward = 10.0 if next_state == 4 else -1.0

        # Temporal Difference update
        best_future_q = np.max(Q[next_state])
        Q[state, action] += alpha * (reward + gamma * best_future_q - Q[state, action])
        state = next_state

print("Learned Q-table (States x [Left, Right]):\n", np.round(Q, 2))`
            }
          }
        ],
        assignment: {
          title: "Epsilon Decay Policy",
          instructions: "Implement exponential decay on epsilon: epsilon = max(0.01, epsilon * 0.995). Compare convergence speed against a fixed exploration rate."
        }
      },
      {
        id: "gym-environments",
        week: 12,
        lessonNumber: 25,
        title: "Gym Environments & CartPole Control",
        shortDesc: "Standardizing RL with Gymnasium, state discretization, and balancing the inverted pendulum.",
        readTime: "18 min",
        category: "Reinforcement",
        preQuizId: 47,
        postQuizId: 48,
        objectives: [
          "Navigate the Gymnasium API: env.reset(), env.step(action), observation, reward, terminated, truncated.",
          "Discretize continuous observation spaces into discrete bins for tabular agents.",
          "Train a policy that successfully balances the CartPole for >200 steps.",
          "Identify when continuous state spaces mandate Deep Q-Networks (DQN) instead of Q-tables."
        ],
        keyTerms: [
          { term: "Gymnasium", definition: "A standard API for reinforcement learning environments created by Farama Foundation (formerly OpenAI Gym)." },
          { term: "State Discretization", definition: "The quantization of continuous observation variables into a finite number of intervals or bins." }
        ],
        sections: [
          {
            title: "1. The CartPole Balance Problem",
            content: "A pole is attached by an un-actuated joint to a cart moving along a frictionless track. The system state is 4 continuous values: [cart position, cart velocity, pole angle, pole tip velocity]. The agent can push the cart left (0) or right (1). A reward of +1 is awarded for every step the pole remains upright."
          },
          {
            title: "2. Observation Discretization Helper",
            content: "Quantizing 4 continuous variables into discrete bucket indices:",
            codeSnippet: {
              language: "python",
              title: "cartpole_discretizer.py",
              code: `import numpy as np

# Define discretization bins per state dimension
cart_bins = np.linspace(-2.4, 2.4, 6)
angle_bins = np.linspace(-0.25, 0.25, 6)

def discretize_state(cart_pos, pole_angle):
    c_idx = np.digitize(cart_pos, cart_bins)
    a_idx = np.digitize(pole_angle, angle_bins)
    return (c_idx, a_idx)

state_sample = (0.15, -0.04)
bucket = discretize_state(*state_sample)
print(f"Continuous state {state_sample} mapped to discrete bucket {bucket}")`
            }
          }
        ],
        assignment: {
          title: "Discretization Granularity Experiment",
          instructions: "Experiment with 4 bins per variable versus 10 bins per variable. Note how table size explodes from 4^4=256 states to 10^4=10,000 states."
        }
      },
      {
        id: "real-world-debugging",
        week: 12,
        lessonNumber: 26,
        title: "Real-World ML Applications & Model Debugging",
        shortDesc: "Detecting covariate shift, concept drift, calculating the Population Stability Index (PSI), and preventing silent model failures in production.",
        readTime: "19 min",
        category: "Real-World",
        preQuizId: 49,
        postQuizId: 50,
        objectives: [
          "Distinguish Data Drift (covariate shift) from Concept Drift.",
          "Compute the Population Stability Index (PSI) to detect distribution divergence.",
          "Implement the Kolmogorov-Smirnov test for continuous feature drift.",
          "Establish robust model monitoring, shadow deployment, and automated retraining triggers."
        ],
        keyTerms: [
          { term: "Data Drift (Covariate Shift)", definition: "When the distribution of input features P(X) changes over time, even if the underlying conditional probability P(Y|X) remains constant." },
          { term: "Concept Drift", definition: "When the relationship between input features and target labels P(Y|X) changes over time (e.g. consumer purchasing habits after an inflation surge)." },
          { term: "Population Stability Index (PSI)", definition: "A metric used to measure how much a variable has shifted in distribution between two samples over time." }
        ],
        mathFormulas: [
          {
            name: "Population Stability Index (PSI)",
            formula: "\\text{PSI} = \\sum_{b=1}^B \\left( \\text{Actual}_b - \\text{Expected}_b \\right) \\times \\ln\\left( \\frac{\\text{Actual}_b}{\\text{Expected}_b} \\right)",
            description: "PSI < 0.1: No significant change; 0.1 <= PSI < 0.25: Moderate shift; PSI >= 0.25: Significant drift requiring model retraining."
          }
        ],
        sections: [
          {
            title: "1. The Silent Failure Problem in Machine Learning",
            content: "Unlike traditional code that throws an exception when it crashes, a degraded machine learning model continues to return status code 200 with syntactically valid numbers, while making disastrously inaccurate business decisions. Continuous drift monitoring is the only defense."
          },
          {
            title: "2. Calculating PSI in Python",
            content: "Evaluating feature shift between training and live inference distributions:",
            codeSnippet: {
              language: "python",
              title: "calculate_psi.py",
              code: `import numpy as np

def calculate_psi(expected, actual, num_bins=10):
    # Determine bin thresholds on expected distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(expected, percentiles)
    bins[0], bins[-1] = -np.inf, np.inf

    # Calculate proportions
    expected_counts = np.histogram(expected, bins=bins)[0]
    actual_counts = np.histogram(actual, bins=bins)[0]

    expected_pct = np.clip(expected_counts / len(expected), 1e-4, 1.0)
    actual_pct = np.clip(actual_counts / len(actual), 1e-4, 1.0)

    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi_value

# Baseline training data vs drifted production data
train_data = np.random.normal(0, 1, 1000)
prod_drifted = np.random.normal(0.8, 1.2, 1000)

psi = calculate_psi(train_data, prod_drifted)
print(f"Calculated PSI: {psi:.3f}")
if psi >= 0.25:
    print("CRITICAL ALERT: Significant data drift detected! Retraining triggered.")`
            }
          }
        ],
        assignment: {
          title: "Design a Production ML Monitoring Plan",
          instructions: "Draft an operational Runbook defining latency SLAs, drift thresholds (PSI), data validation checks, and automatic fallback rules for an enterprise fraud classification model."
        }
      }
    ]
  }
];

export function getAllLessons(): Lesson[] {
  return mlBeginnersWeeks.flatMap((w) => w.lessons);
}

export function getLessonById(id: string): Lesson | undefined {
  return getAllLessons().find((l) => l.id === id);
}
