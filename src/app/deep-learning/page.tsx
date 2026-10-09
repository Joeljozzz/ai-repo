export default function DeepLearning() {
  return (
    <div className="space-y-6 prose prose-slate max-w-none">
      <h1 className="text-3xl font-bold">Deep Learning</h1>
      <p className="text-gray-600">Neural networks, architectures, and deep feature extraction.</p>
      
      <div className="mt-8 space-y-8">
        <section>
          <h2 className="text-2xl font-semibold border-b pb-2">Multi-Layer Perceptrons (MLPs)</h2>
          <p className="mt-2 text-sm text-gray-700"><strong>When to use:</strong> General function approximation on structured data when classical ML isn't capturing complex enough interactions.</p>
        </section>

        <section>
          <h2 className="text-2xl font-semibold border-b pb-2">Convolutional Neural Networks (CNNs)</h2>
          <p className="mt-2 text-sm text-gray-700"><strong>When to use:</strong> Spatial data like images, audio spectrograms, and grid-like topology data.</p>
        </section>

        <section>
          <h2 className="text-2xl font-semibold border-b pb-2">Recurrent Neural Networks (RNNs & LSTMs)</h2>
          <p className="mt-2 text-sm text-gray-700"><strong>When to use:</strong> Sequential data like time-series or early NLP, where temporal dependencies matter.</p>
        </section>
      </div>
    </div>
  );
}
