import React from 'react';

export default function DeepLearning() {
  return (
    <div className="space-y-12 pb-12">
      <header className="border-b pb-6">
        <h1 className="text-4xl font-extrabold tracking-tight text-gray-900">Deep Learning</h1>
        <p className="mt-4 text-lg text-gray-600">
          Neural networks, backpropagation, and automatic feature extraction. Moving beyond classical ML into representational learning.
        </p>
      </header>

      {/* Neural Network Basics */}
      <section className="space-y-4">
        <h2 className="text-3xl font-bold text-gray-800">1. The Neural Network (MLP)</h2>
        <div className="bg-white p-5 rounded-lg border shadow-sm">
          <p className="text-sm text-gray-700 mb-4">
            A Multi-Layer Perceptron (MLP) consists of an input layer, one or more hidden layers, and an output layer. Neurons connect with learned <em>weights</em> and <em>biases</em>.
          </p>
          <h4 className="font-bold text-gray-700 mb-2">Forward Pass (The Math)</h4>
          <p className="font-mono text-sm bg-gray-100 p-3 rounded mb-3 text-center">
            z = W·x + b <br/>
            a = σ(z)
          </p>
          <ul className="text-sm text-gray-600 space-y-1">
            <li><strong>W:</strong> Weight matrix</li>
            <li><strong>x:</strong> Input vector</li>
            <li><strong>b:</strong> Bias vector</li>
            <li><strong>σ:</strong> Activation function (ReLU, Sigmoid, Tanh)</li>
            <li><strong>a:</strong> Output activation (passed to next layer)</li>
          </ul>
        </div>
      </section>

      {/* Backpropagation */}
      <section className="space-y-4">
        <h2 className="text-3xl font-bold text-gray-800">2. Backpropagation</h2>
        <div className="bg-white p-5 rounded-lg border shadow-sm">
          <p className="text-sm text-gray-700 mb-2">
            The algorithm that allows neural networks to learn. It calculates the gradient of the loss function with respect to every weight by applying the <strong>Chain Rule of Calculus</strong> backwards from the output to the input.
          </p>
          <p className="font-mono text-sm bg-gray-100 p-3 rounded text-center mb-3">
            W_new = W_old - (η * ∇Loss)
          </p>
          <p className="text-sm text-gray-600"><strong>η (eta)</strong> is the Learning Rate. If it's too high, the model overshoots the minimum. If too low, training takes forever.</p>
        </div>
      </section>

      {/* Architectures */}
      <section className="space-y-4">
        <h2 className="text-3xl font-bold text-gray-800">3. Architectures: When to use what?</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
          <div className="bg-white p-5 rounded-lg border shadow-sm border-t-4 border-t-blue-500">
            <h4 className="font-bold text-gray-800 text-lg mb-2">Dense (MLP)</h4>
            <p className="text-sm text-gray-600 mb-4">Every node connects to every node in the next layer.</p>
            <strong className="text-xs uppercase tracking-wider text-gray-500">When to use:</strong>
            <p className="text-sm text-gray-800 mt-1">General tabular data, or as the final classification head of CNNs/Transformers.</p>
          </div>

          <div className="bg-white p-5 rounded-lg border shadow-sm border-t-4 border-t-green-500">
            <h4 className="font-bold text-gray-800 text-lg mb-2">CNNs</h4>
            <p className="text-sm text-gray-600 mb-4">Uses convolutional filters to capture local spatial patterns.</p>
            <strong className="text-xs uppercase tracking-wider text-gray-500">When to use:</strong>
            <p className="text-sm text-gray-800 mt-1">Images, video frames, audio spectrograms (2D grid data).</p>
          </div>

          <div className="bg-white p-5 rounded-lg border shadow-sm border-t-4 border-t-purple-500">
            <h4 className="font-bold text-gray-800 text-lg mb-2">RNNs & LSTMs</h4>
            <p className="text-sm text-gray-600 mb-4">Maintains a hidden state to remember previous inputs.</p>
            <strong className="text-xs uppercase tracking-wider text-gray-500">When to use:</strong>
            <p className="text-sm text-gray-800 mt-1">Time series forecasting, older NLP tasks, sequence-to-sequence prediction.</p>
          </div>
        </div>
      </section>
    </div>
  );
}
