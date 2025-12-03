# 📘 30 Days Quant Deep Learning — Handwritten Models

This repository is my **personal practice project** for reviewing and mastering the core deep learning concepts needed in **modern quantitative research**, especially:

* Attention
* Multi-Head Attention
* LayerNorm
* Positional/Temporal Encoding
* Transformer Encoder
* Time-Series Transformer Variants
* Training Stability (financial data is noisy & non-stationary)
* Multi-Asset Modeling
* Embedding Alignment & LLM Fusion (related to my research direction)

Everything is built **from scratch (numpy/PyTorch)** to ensure deep intuition—not just API usage.

This is not meant to be a production model zoo, but a **learning-by-writing** roadmap.

---

# 📅 30-Day Plan (Quant-Oriented, Handwritten Models)

This plan is upgraded to focus on **Time-Series Modeling, Multi-Asset Attention, and Financial Data Stability**, tailored for a **Quant + ML** career path.

---

# ⭐ Week 1 — Numpy Foundations (Time-Series Perspective)

**Goal:** Build mathematical intuition for attention and time-series (Essential for Quant).

---

## **Day 1 — Time Series Essence + Numpy Warmup**

**Topics:**
*   Non-stationarity
*   Sequence Toxicity (Auto-correlation)
*   Fat-tails, Volatility Clustering
*   Why LSTM fails vs. Why Transformer fits
*   Numpy Gradient Basics

**Tasks:**
*   Generate AR(1) / Random Walk code.
*   Perform a numerical gradient check.

---

## **Day 2 — Handwritten Linear + Activation (For Time-Series Features)**

**Focus:**
*   Feature scale in financial data.
*   Why LayerNorm is critical in finance.

**Tasks:**
*   Linear forward/backward.
*   ReLU, GELU derivatives.
*   MSE + MAE implementation.

---

## **Day 3 — Handwritten Time-Series MLP (Non-stationary Toy Series)**

**Data:**
*   Generate a toy price series (trend + noise).

**Tasks:**
*   Build a Window → Prediction MLP.
*   Train it to predict 1-step future return.
*   **Insight:** Realize how MLP fails to capture time dependencies (Transition to Attention).

---

## **Day 4 — Attention (Numpy Implementation)**

**Content:**
*   Financial meaning of Q, K, V:
    *   **Q:** Current market state / query.
    *   **K:** Historical events / keys.
    *   **V:** Signal / value.
*   Attention as "Weighted Memory".

**Tasks:**
*   Handwritten Scaled Dot-Product Attention (forward + backward).

---

## **Day 5 — Multi-Head Attention (Numpy)**

**Content:**
*   Multi-head = Multi-factor / Multi-frequency / Cross-asset expressivity.
*   Each head represents a specific "factor".

**Tasks:**
*   Implement Multi-head Q/K/V.
*   Concatenation + Linear Projection.

---

## **Day 6 — Positional Encoding (For Temporal Structure)**

**Focus:**
*   Trading calendar.
*   High/Low frequency rhythms.
*   Periodicity.
*   Timestamp embedding (Common in Quant).

**Tasks:**
*   Handwritten Sinusoidal + Trainable PE.
*   Visualize PE (Periodicity & Frequency).

---

## **Day 7 — Time-Series Transformer Encoder (Numpy)**

**Focus:**
*   Input shape: `(batch, seq, features)`.
*   Pre-norm LayerNorm.
*   FFN with GELU.
*   Skip connections.

**Tasks:**
*   Implement a 1-layer encoder (forward-only).

---

# ⭐ Week 2 — PyTorch Advanced + Time-Series Modeling

**Goal:** Build reusable blocks for future Quant models.

---

## **Day 8 — PyTorch Autograd + Tensor Operations**

**Tasks:**
*   Implement a trainable MLP (predicting returns).
*   Master Tensor broadcasting (Crucial).

---

## **Day 9 — Custom LayerNorm (Time-Series Style)**

**Content:**
*   Why LN > BN for Quant.
*   Stability on noisy data.

**Tasks:**
*   Handwritten LayerNorm.
*   Verify with PyTorch `gradcheck`.

---

## **Day 10 — Temporal Convolution (TCN Basics)**

**Content:**
*   Why CNN is insufficient compared to Attention.

**Tasks:**
*   1D Causal Convolution.
*   Compare Receptive Field.
*   Analyze failure on regime shifts.

---

## **Day 11 — Scaled Attention (PyTorch)**

**Tasks:**
*   Single-head Attention.
*   Multi-head Attention.
*   Causal Mask (for Autoregressive models).

---

## **Day 12 — Complete Transformer Encoder (PyTorch Handwritten)**

**Modules:**
*   MHA
*   Residual
*   LayerNorm
*   FFN

---

## **Day 13 — Time-Series Task: Predict Future Returns**

**Tasks:**
*   Use Transformer Encoder for:
    *   Next return prediction.
    *   Volatility prediction.
    *   Regime classification.
*   **Data:** Yahoo Finance (AAPL/ES/BTC).

---

## **Day 14 — Review + Financial Papers (Transformer Heavy)**

**Papers:**
*   PatchTST, iTransformer, FEDformer, TimesNet, Crossformer, Kite, TiDE, ForecastPFN.
*   **Insight:** All modern models are rooted in the Attention + Encoder backbone.

---

# ⭐ Week 3 — Advanced Quant Time-Series Modeling

**Goal:** Build a Quant-grade Transformer baseline.

---

## **Day 15 — Long Sequence Attention (Quant Specific)**

**Study:**
*   ProbSparse, AutoCorrelation, Nyströmformer, Performer.
*   **Insight:** Addressing computational cost for long sequences.

---

## **Day 16 — PatchTST Implementation (Mini)**

**Content:**
*   PatchTST (SOTA 2023 AI for Time-Series).

**Tasks:**
*   Time-series Tokenization.
*   Patch Embedding.
*   Linear Projector.
*   Encoder-only Transformer.

---

## **Day 17 — Cross-Asset Transformer**

**Tasks:**
*   Simulate Assets: ES, VIX, AAPL, BTC.
*   Multi-asset Sequence Concatenation.
*   Cross-Attention (Asset → Asset).
*   **Concepts:** Correlation breakdown, Risk-on/off, Regime shift detection.

---

## **Day 18 — Volatility-Aware Attention**

**Tasks:**
*   Add features to Input Embedding: ATR, Volatility, Realized Vol, RSI.
*   Implement **Volatility Gating Module** (e.g., Attention Logits * Vol Mask).

---

## **Day 19 — Train a Quant Transformer Baseline**

**Tasks:**
*   Data: Multi-asset OHLCV.
*   Labels: Next-day return, Binary direction, Volatility regime.
*   **Process:** Windowing, Train/Val/Test Split (No future leakage).

---

## **Day 20 — Model Evaluation (Quant Version)**

**Metrics:**
*   Accuracy (Direction).
*   MSE (Regression).
*   IC / RankIC.
*   Sharpe Ratio (Toy Backtest).
*   Turnover.
*   Prediction Stability.

---

# ⭐ Week 4 — LLM + FM + Transformer for Quant

**Goal:** Connect LLM/FM alignment, refiners, and embeddings to Time-Series tasks.

---

## **Day 21 — Multimodal "News + Market" (CLIP/SigLip/LLM Embedding)**

**Tasks:**
*   Embed historical news/headlines.
*   Fuse with market time-series via Attention.
*   **Result:** Multi-head attention aligning events with market reactions.

---

## **Day 22 — LLM for Regime Description**

**Tasks:**
*   Train an encoder to extract "Market Context Embedding".
*   Feed to LLM.
*   Generate natural language explanation (e.g., "Market is in risk-off mode...").

---

## **Day 23 — Build Your Market Transformer Template (Industrial Grade)**

**Structure:**
*   Embedding Layer → Patch Encoder → Cross-Asset MHA → Temporal Attention → Volatility Gate → Transformer Layers → Prediction Head.

---

## **Day 24 — Backtesting Integration**

**Tasks:**
*   Simple Backtest: Signal → Order.
*   Record: Equity Curve, Sharpe, Max Drawdown.
*   **Goal:** Understand model behavior in market terms.

---

## **Day 25 — Stability Techniques for Quant Transformers**

**Study:**
*   Gradient Explosion.
*   LayerNorm Placement (Pre-LN vs Post-LN).
*   Dropout / Temporal Dropout.
*   Embedding Noise.
*   Initialization Strategies.

---

## **Day 26 — Model Explainability (Attention Visualization)**

**Tasks:**
*   Visualize "What history is being attended to?".
*   Cross-Asset Attention Maps.

---

## **Day 27 — Temporal OT Alignment**

**Tasks:**
*   Apply OT (Optimal Transport) Alignment (from your FM research).
*   Align embeddings of different markets (e.g., ES ↔ NQ, BTC ↔ ETH).
*   **Goal:** Cross-market feature transfer.

---

## **Day 28 — Market State Encoder (ICL Pipeline Upgrade)**

**Tasks:**
*   Encoder extracts Market Embedding.
*   LLM performs Reasoning.
*   Cross-Attn performs Adjustment.

---

## **Day 29 — Reusable Quant TS Transformer Repo**

**Components:**
*   Data Loader, Patching, Cross-Asset Transformer, Volatility Gate, Backtest, Evaluation, Explainability, LLM Fusion, OT Alignment, Training Script, Config.

---

## **Day 30 — Final Project Summary**

**Outputs:**
*   Code Repository.
*   Baseline Model.
*   Backtest Curves.
*   Model Explanation & Attention Heatmaps.
*   Comparison with Traditional Models.

---

---

# 📂 Repo Structure (Planning)

```
quant-dl-handwritten/
│
├── day01_time_series_and_numpy/
│   ├── ar1_randomwalk.py
│   ├── gradient_check.py
│   └── README.md
│
├── day02_linear/
├── day03_mlp/
├── day04_mlp_mnist/
├── day05_trainingloop/
├── day06_framework/
├── day07_backprop_notes/
│
├── ... (Week 2–4 folders coming soon)
│
├── README.md
└── requirements.txt
```
