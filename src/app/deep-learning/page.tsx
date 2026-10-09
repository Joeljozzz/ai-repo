"use client";

import React from 'react';
import TerminalBlock from '@/components/TerminalBlock';

export default function DeepLearning() {
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
    <div className="space-y-16 pb-24 text-slate-800">
      
      {/* Header */}
      <header className="border-b border-slate-200 pb-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-xs font-bold uppercase tracking-widest mb-4">
          <span className="w-2 h-2 rounded-full bg-purple-600 animate-pulse"></span>
          Advanced Certification Track
        </div>
        <h1 className="text-5xl font-extrabold tracking-tight text-slate-900 mb-6">Deep Learning Engineering</h1>
        <p className="text-xl text-slate-600 leading-relaxed max-w-3xl">
          A comprehensive guide to neural architectures, backpropagation, and PyTorch implementations. Learn exactly how deep learning extracts features hierarchically and when to apply which architecture in production.
        </p>
      </header>

      {/* Module 1: Theory */}
      <section id="mlp" className="scroll-mt-12">
        <div className="flex items-center gap-4 mb-6">
          <div className="w-10 h-10 rounded-xl bg-purple-100 text-purple-600 flex items-center justify-center font-bold text-xl">1</div>
          <h2 className="text-3xl font-bold text-slate-900">The Multi-Layer Perceptron (MLP)</h2>
        </div>
        <div className="prose prose-slate max-w-none text-lg text-slate-600 mb-8">
          <p>
            At its core, a neural network is a universal function approximator. An MLP stacks layers of linear transformations followed by non-linear activation functions.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8 mb-8">
          <div className="bg-slate-50 p-6 rounded-2xl border border-slate-200">
            <h3 className="font-bold text-xl text-slate-800 mb-4">Forward Pass Equations</h3>
            <div className="font-mono bg-slate-900 text-slate-100 p-4 rounded-lg text-sm mb-4 overflow-x-auto border border-slate-800">
              <p>Z^[L] = W^[L] · A^[L-1] + b^[L]</p>
              <p>A^[L] = g(Z^[L])</p>
            </div>
            <ul className="text-sm text-slate-600 space-y-2">
              <li><strong>W:</strong> Weight Matrix (Parameters the model learns)</li>
              <li><strong>b:</strong> Bias Vector (Shifts the activation)</li>
              <li><strong>g:</strong> Non-linear Activation (ReLU, Sigmoid, Tanh)</li>
            </ul>
          </div>

          <div className="bg-slate-50 p-6 rounded-2xl border border-slate-200">
            <h3 className="font-bold text-xl text-slate-800 mb-4">Why Non-Linearity?</h3>
            <p className="text-slate-600 mb-4">
              Without an activation function like <code>ReLU(x) = max(0, x)</code>, stacking 100 layers is mathematically identical to a single linear regression layer. Non-linearities allow the network to warp space and draw complex decision boundaries.
            </p>
          </div>
        </div>

        <div className="bg-white p-8 rounded-2xl border border-slate-200 shadow-sm">
          <h3 className="font-bold text-2xl text-slate-900 mb-6">Backpropagation & Optimization</h3>
          <p className="text-slate-600 mb-6">
            We calculate the error (Loss) at the output, and use the <strong>Chain Rule of Calculus</strong> to pass that error backwards, adjusting every weight along the way.
          </p>
          <div className="space-y-4">
            <div className="p-5 border-l-4 border-blue-500 bg-blue-50/50 rounded-r-xl">
              <h4 className="font-bold text-blue-900 text-lg mb-2">1. The Loss Function</h4>
              <p className="text-blue-800 text-sm mb-3">For classification, we use Cross-Entropy Loss. For regression, we use Mean Squared Error (MSE).</p>
              <code className="bg-white px-3 py-1.5 rounded-md text-blue-900 text-sm border border-blue-200 shadow-sm">Loss = -Σ y_true * log(y_pred)</code>
            </div>
            <div className="p-5 border-l-4 border-emerald-500 bg-emerald-50/50 rounded-r-xl">
              <h4 className="font-bold text-emerald-900 text-lg mb-2">2. Gradient Descent</h4>
              <p className="text-emerald-800 text-sm mb-3">We subtract the gradient (direction of steepest ascent) multiplied by the Learning Rate (α) from the current weights.</p>
              <code className="bg-white px-3 py-1.5 rounded-md text-emerald-900 text-sm border border-emerald-200 shadow-sm">W_new = W_old - (α * ∂Loss/∂W)</code>
            </div>
          </div>
        </div>
      </section>

      <hr className="border-slate-200" />

      {/* Module 2: CNNs */}
      <section id="cnn" className="scroll-mt-12">
        <div className="flex items-center gap-4 mb-6">
          <div className="w-10 h-10 rounded-xl bg-purple-100 text-purple-600 flex items-center justify-center font-bold text-xl">2</div>
          <h2 className="text-3xl font-bold text-slate-900">Convolutional Neural Networks (CNNs)</h2>
        </div>
        <div className="prose prose-slate max-w-none text-lg text-slate-600 mb-8">
          <p>
            Unlike MLPs where every pixel connects to every neuron (causing parameter explosion), CNNs slide small "filters" (kernels) across the image to detect local patterns like edges, textures, and eventually full objects.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-6 mb-8">
          <div className="p-6 bg-slate-50 rounded-2xl border border-slate-200">
            <h4 className="font-bold text-slate-900 text-lg mb-2">1. Convolution</h4>
            <p className="text-sm text-slate-600">Extracts features by sliding a dot-product filter. Early layers find edges; deep layers find faces/cars.</p>
          </div>
          <div className="p-6 bg-slate-50 rounded-2xl border border-slate-200">
            <h4 className="font-bold text-slate-900 text-lg mb-2">2. Max Pooling</h4>
            <p className="text-sm text-slate-600">Downsamples the spatial dimensions. Makes the network translation-invariant (a cat is a cat even if shifted).</p>
          </div>
          <div className="p-6 bg-slate-50 rounded-2xl border border-slate-200">
            <h4 className="font-bold text-slate-900 text-lg mb-2">3. Fully Connected</h4>
            <p className="text-sm text-slate-600">Flattens the 2D feature maps into a 1D vector to perform the final classification vote.</p>
          </div>
        </div>

        <h3 className="text-2xl font-bold mb-4">PyTorch Implementation (CNN)</h3>
        <TerminalBlock language="python" filename="cnn_vision.py" code={cnnCode} />
      </section>

      <hr className="border-slate-200" />

      {/* Module 3: RNNs */}
      <section id="rnn" className="scroll-mt-12">
        <div className="flex items-center gap-4 mb-6">
          <div className="w-10 h-10 rounded-xl bg-purple-100 text-purple-600 flex items-center justify-center font-bold text-xl">3</div>
          <h2 className="text-3xl font-bold text-slate-900">Recurrent Neural Networks (LSTMs)</h2>
        </div>
        <div className="prose prose-slate max-w-none text-lg text-slate-600 mb-8">
          <p>
            Traditional networks have no memory. RNNs fix this by looping the output of the hidden layer back into itself for the next time step. Long Short-Term Memory (LSTMs) solve the "vanishing gradient" problem of vanilla RNNs using gating mechanisms.
          </p>
        </div>

        <div className="bg-amber-50 border border-amber-200 p-8 rounded-2xl mb-8">
          <h4 className="font-bold text-amber-900 mb-4 text-xl">The 3 Gates of an LSTM</h4>
          <ul className="space-y-4 text-amber-800">
            <li className="flex gap-4">
              <span className="font-mono font-bold text-amber-600">1.</span>
              <span><strong>Forget Gate:</strong> Decides what information from the previous hidden state should be thrown away (e.g., forgetting a singular noun when a new plural noun appears).</span>
            </li>
            <li className="flex gap-4">
              <span className="font-mono font-bold text-amber-600">2.</span>
              <span><strong>Input Gate:</strong> Decides which values we will update in the cell state with new information.</span>
            </li>
            <li className="flex gap-4">
              <span className="font-mono font-bold text-amber-600">3.</span>
              <span><strong>Output Gate:</strong> Decides what the next hidden state should be, based on the filtered, newly updated cell state.</span>
            </li>
          </ul>
        </div>

        <h3 className="text-2xl font-bold mb-4">PyTorch Implementation (LSTM)</h3>
        <TerminalBlock language="python" filename="lstm_sequence.py" code={rnnCode} />
      </section>

    </div>
  );
}
