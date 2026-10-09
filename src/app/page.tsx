export default function Home() {
  return (
    <div className="space-y-6">
      <h1 className="text-4xl font-extrabold tracking-tight">AI & Machine Learning Hub</h1>
      <p className="text-lg text-gray-600">
        Welcome to the ultimate repository for AI concepts, from foundational statistical modeling to cutting-edge LangGraph agents.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-8">
        <a href="/ml" className="block p-6 border rounded-lg hover:border-blue-500 transition-colors">
          <h2 className="text-xl font-bold mb-2">Classical ML</h2>
          <p className="text-gray-600 text-sm">Linear Regression, Random Forests, Decision Trees, Statistical Modeling, and formulas.</p>
        </a>

        <a href="/deep-learning" className="block p-6 border rounded-lg hover:border-blue-500 transition-colors">
          <h2 className="text-xl font-bold mb-2">Deep Learning</h2>
          <p className="text-gray-600 text-sm">Neural Networks, backpropagation, CNNs, RNNs, and when to use them.</p>
        </a>

        <a href="/search-models" className="block p-6 border rounded-lg hover:border-blue-500 transition-colors">
          <h2 className="text-xl font-bold mb-2">Search & Game Theory</h2>
          <p className="text-gray-600 text-sm">State space search, Alpha-Beta pruning, Minimax, and strategic decision making.</p>
        </a>

        <a href="/evaluations" className="block p-6 border rounded-lg hover:border-blue-500 transition-colors">
          <h2 className="text-xl font-bold mb-2">Evaluations</h2>
          <p className="text-gray-600 text-sm">Classification matrices, ROC-AUC, F1-scores, and performance metrics.</p>
        </a>

        <a href="/llm-rag" className="block p-6 border rounded-lg hover:border-blue-500 transition-colors sm:col-span-2">
          <h2 className="text-xl font-bold mb-2">LLMs, LangChain & RAG</h2>
          <p className="text-gray-600 text-sm">Advanced prompt engineering, retrieval-augmented generation architectures, and agentic workflows using LangGraph.</p>
        </a>
      </div>
    </div>
  );
}
