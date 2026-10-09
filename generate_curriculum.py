import os

CURRICULUM = {
    "ml": {
        "title": "Classical Machine Learning",
        "description": "Foundation of statistical modeling.",
        "icon": "BookOpen",
        "color": "emerald",
        "modules": [
            {"id": "linear-models", "title": "Linear & Distance Models", "desc": "Linear Regression, Logistic Regression, KNN."},
            {"id": "svm", "title": "Support Vector Machines", "desc": "Hyperplanes, Margins, and Kernel Tricks."},
            {"id": "trees", "title": "Decision Trees", "desc": "Gini, Entropy, and Information Gain."},
            {"id": "ensembles", "title": "Bagging & Boosting", "desc": "Random Forest, AdaBoost, and XGBoost."}
        ]
    },
    "dl": {
        "title": "Deep Learning",
        "description": "Neural Networks and Representation Learning.",
        "icon": "Network",
        "color": "purple",
        "modules": [
            {"id": "foundations", "title": "MLP Foundations", "desc": "Forward Pass, Backpropagation, and Activations."},
            {"id": "cnn", "title": "Computer Vision (CNN)", "desc": "Convolutions, Pooling, and ResNets."},
            {"id": "rnn", "title": "Sequence Modeling", "desc": "RNNs, LSTMs, and GRUs for time-series."},
            {"id": "attention", "title": "Attention Mechanisms", "desc": "Self-attention and the evolution before transformers."}
        ]
    },
    "llm": {
        "title": "Generative AI & LLMs",
        "description": "Modern Large Language Models and Applications.",
        "icon": "MessageSquare",
        "color": "indigo",
        "modules": [
            {"id": "transformers", "title": "Transformers Architecture", "desc": "Deep dive into Attention is All You Need."},
            {"id": "fine-tuning", "title": "LoRA & Fine-Tuning", "desc": "Parameter Efficient Fine Tuning (PEFT)."},
            {"id": "rag", "title": "Advanced RAG", "desc": "Semantic chunking, Vector DBs, Hybrid Search."},
            {"id": "agents", "title": "Agentic Frameworks", "desc": "ReAct, Tool Calling, and LangChain."},
            {"id": "multi-agent", "title": "LangGraph & Multi-Agent", "desc": "Cyclic graphs and specialized agents."}
        ]
    },
    "ops": {
        "title": "Evaluations & MLOps",
        "description": "Deploying and evaluating models in production.",
        "icon": "CheckSquare",
        "color": "rose",
        "modules": [
            {"id": "metrics", "title": "Classification Metrics", "desc": "Precision, Recall, F1, ROC-AUC."},
            {"id": "llm-evals", "title": "LLM Evaluations", "desc": "RAGAS, TruLens, and LLM-as-a-judge."},
            {"id": "deployment", "title": "Serving & Scaling", "desc": "vLLM, TensorRT-LLM, and Docker."}
        ]
    }
}

TEMPLATE = """\"\"\"use client\"\"\";
import React from 'react';
import TerminalBlock from '@/components/TerminalBlock';
import InteractiveQuiz from '@/components/InteractiveQuiz';

export default function Page() {{
  return (
    <div className="space-y-12 pb-24 text-slate-800 animate-in fade-in slide-in-from-bottom-4 duration-500">
      
      <header className="border-b border-slate-200 pb-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-{color}-100 text-{color}-700 rounded-full text-xs font-bold uppercase tracking-widest mb-4">
          <span className="w-2 h-2 rounded-full bg-{color}-600 animate-pulse"></span>
          {stack_title}
        </div>
        <h1 className="text-4xl font-extrabold tracking-tight text-slate-900 mb-6">{title}</h1>
        <p className="text-xl text-slate-600 leading-relaxed max-w-3xl">
          {desc}
        </p>
      </header>

      <section className="prose prose-slate max-w-none text-lg text-slate-600">
        <h2>Concept Overview</h2>
        <p>This interactive module covers the core concepts, mathematical foundations, and implementation details for {title}.</p>
        
        <div className="bg-slate-50 border border-slate-200 p-6 rounded-2xl my-8">
            <h3 className="text-slate-800 font-bold mt-0">Interactive Implementation</h3>
            <p className="text-sm">Explore the terminal block below for a production-ready implementation.</p>
            <TerminalBlock 
                language="python" 
                filename="implementation.py" 
                code={`# Implementation for {title}\\n# Coming in detailed expansion...\\nprint("Hello {title}")`}
            />
        </div>

        <InteractiveQuiz 
            question="Which scenario best fits the application of {title}?"
            options={{[
                "When latency is the only priority.",
                "When you need maximum accuracy with specific constraints.",
                "It should be avoided in production.",
                "When you have unlabelled data."
            ]}}
            correctIndex={{1}}
            explanation="Understanding the specific architectural tradeoffs is key to AI engineering."
        />
      </section>
    </div>
  );
}}
"""

def main():
    base_dir = "src/app/course"
    
    for stack_id, stack in CURRICULUM.items():
        for mod in stack["modules"]:
            mod_dir = os.path.join(base_dir, stack_id, mod["id"])
            os.makedirs(mod_dir, exist_ok=True)
            
            page_path = os.path.join(mod_dir, "page.tsx")
            content = TEMPLATE.format(
                color=stack["color"],
                stack_title=stack["title"],
                title=mod["title"],
                desc=mod["desc"]
            )
            
            # Use a simple string replacement for the react 'use client' workaround in python formatting
            content = content.replace('"""use client""";', '"use client";')
            
            with open(page_path, "w", encoding="utf-8") as f:
                f.write(content)
                
    print("Scaffolded all course pages successfully!")

if __name__ == "__main__":
    main()
