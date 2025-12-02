# Day 1: Time Series Properties & Numpy Gradient Intuition

## 1. Core Concepts in Financial Time Series

### Non-stationarity (非平稳性)
Financial data distributions change over time. The mean and variance are not constant.
*   **Impact**: A model trained on 2010-2015 data might fail completely in 2020 because the market regime has changed.
*   **Deep Learning implication**: We often need techniques like LayerNorm, rolling normalization, or stationary transformations (returns instead of prices) to handle this.

### Auto-correlation (自相关性)
The correlation of a signal with a delayed copy of itself.
*   **Meaning**: Today's price is often highly correlated with yesterday's price.
*   **"Toxicity"**: High auto-correlation in raw prices can mislead models into just predicting $P_t = P_{t-1}$. This looks like high accuracy (low MSE) but has zero predictive power for returns.

### Fat-tails (肥尾效应)
Extreme events (3-sigma or more) happen far more frequently than a Normal (Gaussian) distribution predicts.
*   **Impact**: Using MSE (Mean Squared Error) assumes Gaussian noise. It might penalize outliers too much or fail to capture the true risk.

### Volatility Clustering (波动率聚集)
"Large changes tend to be followed by large changes, of either sign, and small changes tend to be followed by small changes."
*   **Impact**: Variance is not constant (Heteroskedasticity).

---

## 2. LSTM vs Transformer for Time Series

### Why LSTM struggles (relatively)
1.  **Sequential Processing**: $h_t$ depends on $h_{t-1}$. Cannot parallelize training. Slow on long sequences.
2.  **Information Bottleneck**: All history must be compressed into a fixed-size hidden state vector $h_t$. "Forgetting" happens for long sequences.
3.  **Gradient Flow**: Gradients must flow back through time (BPTT), leading to vanishing/exploding gradients (though LSTM gates help, they don't solve it for very long contexts).

### Why Transformer is naturally better
1.  **Global Attention**: Every time step can "look at" every other time step directly ($O(1)$ path length). No information bottleneck.
2.  **Parallelism**: The entire sequence is processed at once (during training).
3.  **Dynamic Context**: Attention weights allow the model to dynamically focus on relevant past events (e.g., "earnings call last year") regardless of how far back they are.
4.  **Positional Encoding**: Explicitly handles time, allowing it to learn temporal structures differently than just "next step".

---

## 3. Tasks

1.  **Generate AR(1) & Random Walk**: Understand the simplest time series models.
2.  **Gradient Check**: Implement numerical gradient checking to verify analytical gradients. This is the foundation of debugging backprop.
