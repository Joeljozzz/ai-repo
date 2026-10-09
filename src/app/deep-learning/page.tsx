"use client";

import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

export default function DeepLearning() {
  const [activeTab, setActiveTab] = useState('theory');

  const cnnCode = `import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # 1 input channel (e.g. grayscale image), 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128) # Assuming 28x28 input image
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Convolution -> ReLU Activation -> Pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten for the dense layer
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()
print(model)`;

  const rnnCode = `import torch
import torch.nn as nn

class BasicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BasicLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer expects input of shape (batch, seq_length, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out`;

  return (
    <div className="space-y-12 pb-16 text-slate-800">
      
      {/* Header */}
      <header className="border-b border-slate-200 pb-8">
        <div className="inline-block px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-xs font-bold uppercase tracking-widest mb-4">
          Module 3
        </div>
        <h1 className="text-5xl font-extrabold tracking-tight text-slate-900 mb-4">Deep Learning Engineering</h1>
        <p className="text-xl text-slate-600 leading-relaxed max-w-3xl">
          A comprehensive guide to neural architectures, backpropagation, and PyTorch implementations. Learn exactly how deep learning extracts features hierarchically and when to apply which architecture in production.
        </p>
      </header>

      {/* Navigation Tabs */}
      <div className="flex border-b border-slate-200 gap-8">
        <button 
          onClick={() => setActiveTab('theory')}
          className={`pb-4 text-lg font-medium transition-colors ${activeTab === 'theory' ? 'border-b-2 border-blue-600 text-blue-600' : 'text-slate-500 hover:text-slate-700'}`}
        >
          Core Theory & Math
        </button>
        <button 
          onClick={() => setActiveTab('cnn')}
          className={`pb-4 text-lg font-medium transition-colors ${activeTab === 'cnn' ? 'border-b-2 border-blue-600 text-blue-600' : 'text-slate-500 hover:text-slate-700'}`}
        >
          CNNs (Vision)
        </button>
        <button 
          onClick={() => setActiveTab('rnn')}
          className={`pb-4 text-lg font-medium transition-colors ${activeTab === 'rnn' ? 'border-b-2 border-blue-600 text-blue-600' : 'text-slate-500 hover:text-slate-700'}`}
        >
          RNNs & LSTMs (Sequence)
        </button>
      </div>

      {/* Tab Content: Theory */}
      {activeTab === 'theory' && (
        <div className="space-y-12 animate-in fade-in slide-in-from-bottom-4 duration-500">
          
          <section className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200">
            <h2 className="text-3xl font-bold mb-6 text-slate-900">1. The Multi-Layer Perceptron (MLP)</h2>
            <p className="text-lg text-slate-600 mb-6 leading-relaxed">
              At its core, a neural network is a universal function approximator. An MLP stacks layers of linear transformations followed by non-linear activation functions.
            </p>
            
            <div className="grid md:grid-cols-2 gap-8">
              <div className="bg-slate-50 p-6 rounded-xl border border-slate-200">
                <h3 className="font-bold text-slate-800 mb-4">Forward Pass Equations</h3>
                <div className="font-mono bg-slate-900 text-slate-100 p-4 rounded-lg text-sm mb-4 overflow-x-auto">
                  <p>Z^[L] = W^[L] · A^[L-1] + b^[L]</p>
                  <p>A^[L] = g(Z^[L])</p>
                </div>
                <ul className="text-sm text-slate-600 space-y-2">
                  <li><strong>W:</strong> Weight Matrix (Parameters the model learns)</li>
                  <li><strong>b:</strong> Bias Vector (Shifts the activation)</li>
                  <li><strong>g:</strong> Non-linear Activation (ReLU, Sigmoid, Tanh)</li>
                </ul>
              </div>

              <div className="bg-slate-50 p-6 rounded-xl border border-slate-200">
                <h3 className="font-bold text-slate-800 mb-4">Why Non-Linearity?</h3>
                <p className="text-sm text-slate-600 mb-4">
                  Without an activation function like <code>ReLU(x) = max(0, x)</code>, stacking 100 layers is mathematically identical to a single linear regression layer. Non-linearities allow the network to warp space and draw complex decision boundaries.
                </p>
              </div>
            </div>
          </section>

          <section className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200">
            <h2 className="text-3xl font-bold mb-6 text-slate-900">2. Backpropagation & Optimization</h2>
            <p className="text-lg text-slate-600 mb-6 leading-relaxed">
              How does the network actually learn? We calculate the error (Loss) at the output, and use the <strong>Chain Rule of Calculus</strong> to pass that error backwards, adjusting every weight along the way.
            </p>

            <div className="space-y-6">
              <div className="p-5 border-l-4 border-blue-500 bg-blue-50/50 rounded-r-lg">
                <h4 className="font-bold text-blue-900 text-lg mb-2">Step 1: The Loss Function</h4>
                <p className="text-blue-800 mb-2">For classification, we use <strong>Cross-Entropy Loss</strong>. For regression, we use <strong>Mean Squared Error (MSE)</strong>.</p>
                <code className="bg-white px-2 py-1 rounded text-blue-900 text-sm border border-blue-100">Loss = -Σ y_true * log(y_pred)</code>
              </div>

              <div className="p-5 border-l-4 border-emerald-500 bg-emerald-50/50 rounded-r-lg">
                <h4 className="font-bold text-emerald-900 text-lg mb-2">Step 2: Gradient Descent</h4>
                <p className="text-emerald-800 mb-2">We subtract the gradient (the direction of steepest ascent) multiplied by the Learning Rate (α) from the current weights.</p>
                <code className="bg-white px-2 py-1 rounded text-emerald-900 text-sm border border-emerald-100">W_new = W_old - (α * ∂Loss/∂W)</code>
              </div>
            </div>
          </section>

        </div>
      )}

      {/* Tab Content: CNN */}
      {activeTab === 'cnn' && (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
          <section className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200">
            <h2 className="text-3xl font-bold mb-4 text-slate-900">Convolutional Neural Networks (CNNs)</h2>
            <p className="text-lg text-slate-600 mb-8">
              Unlike MLPs where every pixel connects to every neuron (causing parameter explosion), CNNs slide small "filters" (kernels) across the image to detect local patterns like edges, textures, and eventually full objects.
            </p>

            <h3 className="text-xl font-bold mb-4">PyTorch Implementation</h3>
            <div className="rounded-xl overflow-hidden shadow-lg border border-slate-800">
              <SyntaxHighlighter language="python" style={vscDarkPlus} showLineNumbers customStyle={{margin: 0, padding: '1.5rem'}}>
                {cnnCode}
              </SyntaxHighlighter>
            </div>

            <div className="grid md:grid-cols-3 gap-6 mt-8">
              <div className="p-5 bg-slate-50 rounded-xl border border-slate-200">
                <h4 className="font-bold text-slate-900 mb-2">1. Convolution</h4>
                <p className="text-sm text-slate-600">Extracts features by sliding a dot-product filter. Early layers find edges; deep layers find faces/cars.</p>
              </div>
              <div className="p-5 bg-slate-50 rounded-xl border border-slate-200">
                <h4 className="font-bold text-slate-900 mb-2">2. Max Pooling</h4>
                <p className="text-sm text-slate-600">Downsamples the spatial dimensions. Makes the network translation-invariant (a cat is a cat even if shifted).</p>
              </div>
              <div className="p-5 bg-slate-50 rounded-xl border border-slate-200">
                <h4 className="font-bold text-slate-900 mb-2">3. Fully Connected</h4>
                <p className="text-sm text-slate-600">Flattens the 2D feature maps into a 1D vector to perform the final classification vote.</p>
              </div>
            </div>
          </section>
        </div>
      )}

      {/* Tab Content: RNN */}
      {activeTab === 'rnn' && (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
          <section className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200">
            <h2 className="text-3xl font-bold mb-4 text-slate-900">Recurrent Neural Networks (LSTMs)</h2>
            <p className="text-lg text-slate-600 mb-8">
              Traditional networks have no memory. RNNs fix this by looping the output of the hidden layer back into itself for the next time step. Long Short-Term Memory (LSTMs) solve the "vanishing gradient" problem of vanilla RNNs using gating mechanisms.
            </p>

            <h3 className="text-xl font-bold mb-4">PyTorch Implementation</h3>
            <div className="rounded-xl overflow-hidden shadow-lg border border-slate-800">
              <SyntaxHighlighter language="python" style={vscDarkPlus} showLineNumbers customStyle={{margin: 0, padding: '1.5rem'}}>
                {rnnCode}
              </SyntaxHighlighter>
            </div>

            <div className="mt-8 bg-amber-50 border border-amber-200 p-6 rounded-xl">
              <h4 className="font-bold text-amber-900 mb-3 text-lg">The 3 Gates of an LSTM</h4>
              <ul className="space-y-4 text-amber-800">
                <li><strong className="text-amber-900">1. Forget Gate:</strong> Decides what information from the previous hidden state should be thrown away (e.g., forgetting a singular noun when a new plural noun appears).</li>
                <li><strong className="text-amber-900">2. Input Gate:</strong> Decides which values we will update in the cell state with new information.</li>
                <li><strong className="text-amber-900">3. Output Gate:</strong> Decides what the next hidden state should be, based on the filtered, newly updated cell state.</li>
              </ul>
            </div>
          </section>
        </div>
      )}

    </div>
  );
}
