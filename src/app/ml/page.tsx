export default function ClassicalML() {
  return (
    <div className="space-y-6 prose prose-slate max-w-none">
      <h1 className="text-3xl font-bold">Classical Machine Learning</h1>
      <p className="text-gray-600">Foundational models, statistical approaches, and regression techniques.</p>
      
      <div className="mt-8 space-y-8">
        <section>
          <h2 className="text-2xl font-semibold border-b pb-2">Linear Regression</h2>
          <p className="mt-2 text-sm text-gray-700"><strong>When to use:</strong> When predicting a continuous numerical value where the relationship between inputs and output is assumed to be linear.</p>
          <p className="mt-2 text-sm font-mono bg-gray-100 p-2 rounded">y = β0 + β1x1 + ... + βnxn + ε</p>
        </section>

        <section>
          <h2 className="text-2xl font-semibold border-b pb-2">Decision Trees</h2>
          <p className="mt-2 text-sm text-gray-700"><strong>When to use:</strong> When you need interpretable models for classification or regression, especially with non-linear relationships or mixed data types.</p>
        </section>

        <section>
          <h2 className="text-2xl font-semibold border-b pb-2">Random Forests</h2>
          <p className="mt-2 text-sm text-gray-700"><strong>When to use:</strong> When Decision Trees overfit. It's a robust, general-purpose ensemble method that handles tabular data exceptionally well.</p>
        </section>
      </div>
    </div>
  );
}
