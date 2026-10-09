"use client";
import React from "react";
import TerminalBlock from "@/components/TerminalBlock";

const code = `import numpy as np

class LSTMCell:
    """
    Long Short-Term Memory (LSTM) cell (Hochreiter & Schmidhuber, 1997).
    Resolves the vanishing gradient problem in vanilla RNNs via additive cell state flow:
      f_t = σ(W_f · [h_{t-1}, x_t] + b_f)       # Forget gate (what to purge)
      i_t = σ(W_i · [h_{t-1}, x_t] + b_i)       # Input gate (what to update)
      c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)    # Candidate state
      c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t          # Additive cell state update
      o_t = σ(W_o · [h_{t-1}, x_t] + b_o)       # Output gate
      h_t = o_t ⊙ tanh(c_t)                    # Hidden state output
    """
    def __init__(self, d_in: int, d_hid: int):
        self.d_in = d_in
        self.d_hid = d_hid
        d_concat = d_in + d_hid

        # Xavier initialization for combined gates [f, i, c, o]
        scale = np.sqrt(2.0 / (d_concat + d_hid))
        self.W = np.random.randn(4 * d_hid, d_concat) * scale
        self.b = np.zeros((4 * d_hid, 1))

        # Forget gate bias initialized to +1.0 (Jozefowicz et al., 2015)
        # Prevents premature forgetting of long-term memories at training initialization
        self.b[:d_hid, :] = 1.0

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

    def forward_step(self, x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray):
        """
        x_t: (d_in, 1)
        h_prev: (d_hid, 1)
        c_prev: (d_hid, 1)
        """
        concat = np.vstack((h_prev, x_t))  # (d_in + d_hid, 1)
        gates = np.dot(self.W, concat) + self.b

        d = self.d_hid
        f_t = self._sigmoid(gates[0*d:1*d])
        i_t = self._sigmoid(gates[1*d:2*d])
        c_cand = np.tanh(gates[2*d:3*d])
        o_t = self._sigmoid(gates[3*d:4*d])

        # Additive cell state flow (constant error carousel)
        c_t = f_t * c_prev + i_t * c_cand
        h_t = o_t * np.tanh(c_t)

        return h_t, c_t, (f_t, i_t, c_cand, o_t)

class GRUCell:
    """
    Gated Recurrent Unit (Cho et al., 2014).
    Merges cell state and hidden state, replacing forget & input gates with a coupled update gate:
      z_t = σ(W_z · [h_{t-1}, x_t])             # Update gate
      r_t = σ(W_r · [h_{t-1}, x_t])             # Reset gate
      h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t])   # Candidate hidden state
      h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t     # Linear interpolation
    """
    def __init__(self, d_in: int, d_hid: int):
        self.d_in = d_in
        self.d_hid = d_hid
        d_concat = d_in + d_hid
        scale = np.sqrt(2.0 / (d_concat + d_hid))

        self.W_z = np.random.randn(d_hid, d_concat) * scale
        self.W_r = np.random.randn(d_hid, d_concat) * scale
        self.W_h = np.random.randn(d_hid, d_concat) * scale

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

    def forward_step(self, x_t: np.ndarray, h_prev: np.ndarray):
        concat = np.vstack((h_prev, x_t))
        z_t = self._sigmoid(np.dot(self.W_z, concat))
        r_t = self._sigmoid(np.dot(self.W_r, concat))

        concat_reset = np.vstack((r_t * h_prev, x_t))
        h_cand = np.tanh(np.dot(self.W_h, concat_reset))

        h_t = (1.0 - z_t) * h_prev + z_t * h_cand
        return h_t

if __name__ == "__main__":
    d_in, d_hid = 10, 32
    lstm = LSTMCell(d_in, d_hid)
    gru = GRUCell(d_in, d_hid)

    x_sample = np.random.randn(d_in, 1)
    h_init = np.zeros((d_hid, 1))
    c_init = np.zeros((d_hid, 1))

    h_lstm, c_lstm, _ = lstm.forward_step(x_sample, h_init, c_init)
    h_gru = gru.forward_step(x_sample, h_init)

    print(f"LSTM hidden state output: {h_lstm.shape} [Expected: (32, 1)]")
    print(f"LSTM cell state output:   {c_lstm.shape} [Expected: (32, 1)]")
    print(f"GRU hidden state output:  {h_gru.shape}  [Expected: (32, 1)]")
`;

export default function RNNLSTMPage() {
  return (
    <article className="prose prose-slate max-w-none pb-24">
      <div className="not-prose mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-100 text-purple-700 text-xs font-bold uppercase tracking-widest">
          <span className="w-1.5 h-1.5 rounded-full bg-purple-500"></span>
          Track 2 · Deep Learning
        </span>
      </div>

      <h1>Recurrent Networks, LSTMs, and GRUs: Backpropagation Through Time and Gated Dynamics</h1>

      <blockquote>
        <p>
          <em>"The Constant Error Carousel in an LSTM enforces an additive gradient flow through the cell state, preventing vanishing gradients and allowing memories to persist across thousands of temporal steps."</em> — Sepp Hochreiter &amp; Jürgen Schmidhuber (1997)
        </p>
      </blockquote>

      <h2>Learning Objectives</h2>
      <ul>
        <li>Derive Backpropagation Through Time (BPTT) and analyze the exponential Jacobian matrix chain <code>∏_(j=k+1)^T W_hh^T</code>.</li>
        <li>Explain the Constant Error Carousel (CEC) mechanism in LSTMs that breaks the vanishing gradient bottleneck.</li>
        <li>Derive the four mathematical gating operations: Forget, Input, Candidate, and Output gates.</li>
        <li>Contrast LSTMs with Gated Recurrent Units (GRUs): parameter count, coupled update gates, and speed trade-offs.</li>
        <li>Implement both an LSTM and GRU forward cell from scratch in NumPy with the Jozefowicz forget-gate bias trick.</li>
      </ul>

      <h2>1 · Backpropagation Through Time (BPTT) in Vanilla RNNs</h2>
      <p>
        In a vanilla Recurrent Neural Network, hidden state <code>h_t</code> is updated iteratively at every time step <code>t</code>:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>h_t = tanh( W_hh · h_(t-1) + W_xh · x_t + b_h )</p>
        <p>ŷ_t = Softmax( W_hy · h_t + b_y )</p>
      </div>

      <p>
        To compute the gradient of loss <code>L_T</code> at step <code>T</code> with respect to the recurrent weights <code>W_hh</code>, we apply the chain rule across all earlier time steps:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>∂L_T / ∂W_hh = Σ_(k=1)^T (∂L_T / ∂h_T) · (∂h_T / ∂h_k) · (∂h_k / ∂W_hh)</p>
        <br />
        <p>where the temporal Jacobian chain expands to:</p>
        <p>∂h_T / ∂h_k = ∏_(j=k+1)^T (∂h_j / ∂h_(j-1)) = ∏_(j=k+1)^T [ diag( 1 - tanh²(z_j) ) · W_hh^T ]</p>
      </div>

      <h3>The Mathematical Cause of Vanishing / Exploding Gradients</h3>
      <p>
        Let <code>λ_max</code> denote the largest eigenvalue of <code>W_hh</code>:
      </p>
      <ul>
        <li>Since <code>|1 - tanh²(z)| ≤ 1</code>, if <code>λ_max &lt; 1</code>, the product <code>(λ_max)^(T - k) → 0</code> decays exponentially as the temporal gap <code>T - k &gt; 15</code> increases (<strong>Vanishing Gradient</strong>). The network becomes amnesic to early context.</li>
        <li>If <code>λ_max &gt; 1</code> and tanh is unsaturated, the product explodes toward infinity (<strong>Exploding Gradient</strong>), causing numerical overflows (NaNs).</li>
      </ul>

      <h2>2 · The LSTM Architecture: The Constant Error Carousel</h2>
      <p>
        Hochreiter &amp; Schmidhuber (1997) resolved this by introducing the <strong>Cell State <code>c_t</code></strong>, which acts as an additive highway parallel to the hidden state:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p className="text-slate-400">// 1. Forget Gate: What information to discard from previous cell state</p>
        <p>f_t = σ( W_f · [h_(t-1), x_t] + b_f )</p>
        <br />
        <p className="text-slate-400">// 2. Input Gate &amp; Candidate State: What new information to store</p>
        <p>i_t = σ( W_i · [h_(t-1), x_t] + b_i )</p>
        <p>c̃_t = tanh( W_c · [h_(t-1), x_t] + b_c )</p>
        <br />
        <p className="text-slate-400">// 3. Additive Cell State Update (Constant Error Carousel):</p>
        <p>c_t = f_t ⊙ c_(t-1) + i_t ⊙ c̃_t</p>
        <br />
        <p className="text-slate-400">// 4. Output Gate &amp; Hidden State:</p>
        <p>o_t = σ( W_o · [h_(t-1), x_t] + b_o )</p>
        <p>h_t = o_t ⊙ tanh( c_t )</p>
      </div>

      <p>
        <strong>Why Gradients Do Not Vanish:</strong> In the cell state update equation <code>c_t = f_t ⊙ c_(t-1) + i_t ⊙ c̃_t</code>, the derivative is:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>∂c_t / ∂c_(t-1) = f_t</p>
      </div>

      <p>
        If the network learns to keep the forget gate open (<code>f_t ≈ 1.0</code>), the gradient propagates backward through time <strong>without multiplication by any weight matrix <code>W</code></strong>, enabling uninterrupted gradient flow across hundreds of steps.
      </p>

      <h2>3 · The Jozefowicz Forget-Gate Bias Trick (2015)</h2>
      <p>
        Rafal Jozefowicz et al. (ICML 2015) conducted an empirical exploration of thousands of RNN architectures and found that initializing the forget gate bias <code>b_f = +1.0</code> or <code>+2.0</code> consistently matches or beats complex variants.
      </p>
      <p>
        Because <code>σ(1.0) ≈ 0.73</code> and <code>σ(2.0) ≈ 0.88</code>, this ensures the model begins training with its memory gates open by default rather than discarding historical context.
      </p>

      <h2>4 · Gated Recurrent Unit (GRU): Cho et al. (2014)</h2>
      <p>
        The GRU couples the forget and input gates into a single <strong>Update Gate <code>z_t</code></strong>, eliminating the separate cell state:
      </p>

      <div className="not-prose bg-slate-900 rounded-xl p-5 font-mono text-slate-100 text-sm my-6">
        <p>z_t = σ( W_z · [h_(t-1), x_t] )    # Update gate</p>
        <p>r_t = σ( W_r · [h_(t-1), x_t] )    # Reset gate</p>
        <p>h̃_t = tanh( W_h · [r_t ⊙ h_(t-1), x_t] )</p>
        <p>h_t = (1 - z_t) ⊙ h_(t-1) + z_t ⊙ h̃_t</p>
      </div>

      <p>
        GRUs feature <strong>25% fewer parameters</strong> than standard LSTMs (3 weight matrices instead of 4), training faster while achieving comparable empirical performance on many sequence benchmarks.
      </p>

      <h2>5 · Python Implementation from First Principles</h2>
      <TerminalBlock language="python" filename="lstm_gru_scratch.py" code={code} />

      <h2>6 · Further Reading</h2>
      <ul>
        <li>Hochreiter, S., &amp; Schmidhuber, J. (1997). <em>Long Short-Term Memory</em>. Neural Computation, 9(8), 1735–1780.</li>
        <li>Cho, K., et al. (2014). <em>Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation</em>. EMNLP 2014.</li>
        <li>Jozefowicz, R., Zaremba, W., &amp; Sutskever, I. (2015). <em>An Empirical Exploration of Recurrent Network Architectures</em>. ICML 2015.</li>
      </ul>
    </article>
  );
}
