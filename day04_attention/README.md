# Day 4: Attention Mechanism (Numpy Implementation)

## 1. What is Attention?

Attention is a mechanism that allows a model to **dynamically focus on different parts of the input** when making predictions.

In traditional RNNs/LSTMs:
*   Information flows sequentially: $h_t$ depends on $h_{t-1}$.
*   Distant information must pass through many steps, causing the "vanishing gradient" problem.

In Attention:
*   **Every position can directly attend to every other position**.
*   The model learns **which past information is relevant** for the current prediction.

---

## 2. The Attention Formula (Scaled Dot-Product Attention)

Given:
*   **Query (Q)**: "What am I looking for?" (Current state)
*   **Key (K)**: "What information is available?" (All historical states)
*   **Value (V)**: "What is the actual signal?" (The content to retrieve)

The attention mechanism computes:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Where:
*   $QK^T$: Similarity scores between Query and all Keys (shape: `(seq_len, seq_len)`).
*   $\sqrt{d_k}$: Scaling factor (prevents softmax saturation when $d_k$ is large).
*   $\text{softmax}$: Converts scores to probabilities (attention weights).
*   Multiply by $V$: Weighted sum of Values based on attention weights.

---

## 3. Financial Interpretation of Q, K, V

In **Quantitative Finance**, attention can be understood as:

*   **Q (Query)**: "What market condition am I trying to understand **right now**?"
    *   Current market state, current asset price, current signal.
*   **K (Key)**: "What **historical events** or market states are stored in memory?"
    *   Past prices, past volatility regimes, past news events.
*   **V (Value)**: "What is the **signal/information** associated with each historical event?"
    *   Returns, volume, sentiment scores.

**Attention as Weighted Memory:**
*   The model learns to **retrieve relevant past information** based on the current query.
*   For example, during a volatility spike (high VIX), the model might attend more to past volatility spikes rather than calm periods.

---

## 4. Why Attention is Better Than MLP/LSTM for Time Series

### MLP:
*   Treats the input window as a "bag of features" (no notion of time).
*   Cannot distinguish between $x_{t-1}$ and $x_{t-10}$.

### LSTM:
*   Sequential processing (slow).
*   Information bottleneck (all history compressed into $h_t$).
*   Gradient issues for very long sequences.

### Attention:
*   **Parallel processing** (all positions computed at once).
*   **No information bottleneck** (direct access to all past states).
*   **Learns dynamic relevance** (automatically focuses on important historical periods).

---

## 5. Forward Pass Steps

1.  Compute $Q = XW_Q$, $K = XW_K$, $V = XW_V$.
2.  Compute attention scores: $\text{scores} = \frac{QK^T}{\sqrt{d_k}}$.
3.  Apply softmax: $\text{weights} = \text{softmax}(\text{scores})$.
4.  Compute output: $\text{output} = \text{weights} \cdot V$.

---

## 6. Backward Pass (Gradients)

Given upstream gradient $\frac{\partial L}{\partial \text{output}}$:

1.  **Gradient w.r.t. V**:
    $$\frac{\partial L}{\partial V} = \text{weights}^T \cdot \frac{\partial L}{\partial \text{output}}$$

2.  **Gradient w.r.t. weights**:
    $$\frac{\partial L}{\partial \text{weights}} = \frac{\partial L}{\partial \text{output}} \cdot V^T$$

3.  **Gradient w.r.t. scores** (softmax backward):
    $$\frac{\partial L}{\partial \text{scores}} = \text{weights} \odot \left(\frac{\partial L}{\partial \text{weights}} - \sum \frac{\partial L}{\partial \text{weights}} \odot \text{weights}\right)$$

4.  **Gradient w.r.t. Q and K**:
    *   $\frac{\partial L}{\partial Q} = \frac{1}{\sqrt{d_k}} \cdot \frac{\partial L}{\partial \text{scores}} \cdot K$
    *   $\frac{\partial L}{\partial K} = \frac{1}{\sqrt{d_k}} \cdot \frac{\partial L}{\partial \text{scores}}^T \cdot Q$

5.  **Gradient w.r.t. X** (via chain rule through $W_Q, W_K, W_V$).

---

## 7. Tasks

1.  Implement `ScaledDotProductAttention` with forward and backward passes.
2.  Visualize attention weights to see which past time steps the model focuses on.
3.  Compare this to the MLP from Day 3.
