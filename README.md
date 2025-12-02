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

Below is the full 30-day curriculum that guides this repo’s structure (e.g., `day01_time_series_and_numpy/`, `day02_linear/`, ...).

---

# ⭐ Week 1 — Numpy Foundations (Handwritten Backprop)

**Goal:** Build a mental model of backpropagation and understand time series mathematically.

---

## **Day 1 — Numpy Basics & Autograd Intuition**

**Topics:**

* numpy gradient vs autograd
* Manual calculation of gradient for f(x)=x²
* MSE loss
* Sigmoid + derivative

**Example:**
`day01_time_series_and_numpy/`

---

## **Day 2 — Handwritten Linear Layer (forward + backward)**

* forward: `X @ W + b`
* backward: dW, db, dX
* Deriving gradient shapes

---

## **Day 3 — Two-Layer MLP from Scratch**

`X → Linear → ReLU → Linear → Softmax`

**Implementation:**

* forward
* backward
* sgd update

---

## **Day 4 — Train a Small MLP on MNIST (numpy)**

**Focus:** Ensure loss decreases.

---

## **Day 5 — Build a Training Loop**

* lr
* batching
* logging
* gradient clipping

---

## **Day 6 — Build a Mini Framework**

**Implementation:**

```python
class Layer:
    forward()
    backward()

class MLP:
    layers = [...]
    forward()
    backward()
```

---

## **Day 7 — Backprop Review**

Write a summary on "My understanding of Backpropagation".

---

# ⭐ Week 2 — PyTorch Foundations (Engineering for Quant)

**Goal:** Master modern DL engineering basics (crucial for modifying Transformers later).

---

## **Day 8 — PyTorch Tensor & Autograd**

* Write a PyTorch MLP
* Print computational graph gradient flow

---

## **Day 9 — Custom nn.Modules**

**Handwritten:**

* Linear
* Dropout
* LayerNorm (Understand why LN is better than BN for time series & finance)

---

## **Day 10 — Optimizer Mechanics**

**Handwritten:**

* SGD
* Adam (Key formulas)

---

## **Day 11 — CNN Basics**

(Theoretically unrelated to Quant, but helpful for understanding model architecture essence)

**Implementation:**

* Conv2d (forward only)
* MaxPool
* mini CNN classifier

---

## **Day 12 — Training Engineering**

* dataloader
* checkpoint
* device
* autocast(fp16)
* seed & reproducibility

---

## **Day 13 — Build a Mini ResNet Block**

Understand engineering details like skip/GELU/init/BN.

---

## **Day 14 — PyTorch Mini Model Quiz**

Write a FashionMNIST classifier and train it successfully.

---

# ⭐ Week 3 — Attention + Transformer (Core Week)

**Goal:** Master the Transformer encoder structure, the root of all Quant TS Transformers.

---

## **Day 15 — Scaled Dot-Product Attention**

**Handwritten:**

* Q = XWq
* K = XWk
* V = XWv
* score = softmax(QKᵀ / √d)
* output = score @ V

---

## **Day 16 — Multi-Head Attention**

* split heads
* concat
* projection
* Why multi-head = multi-factor/multi-frequency expression capability

---

## **Day 17 — Positional Encoding**

**Implementation:**

* sinusoidal
* trainable embedding

(And understand timestamp embedding in finance)

---

## **Day 18 — Transformer Encoder Block**

**Handwritten:**

* MHA
* residual
* LayerNorm
* FFN
* residual again

---

## **Day 19 — Build a Mini Transformer (2–4 layers)**

Build encoder-only (most commonly used in Quant).

---

## **Day 20 — Train a Character-Level Transformer**

Use tiny Shakespeare:
0.5M parameter small GPT to understand that *LLM is essentially Transformer*.

---

## **Day 21 — Transformer Deep Review**

Write a summary on "How I understand Transformer encoder".

---

# ⭐ Week 4 — LLM Engineering + Advanced Topics (Quant Needed)

**Goal:** Understand core engineering techniques for large model fine-tuning, stability, masking, and embedding.

---

## **Day 22 — Use LLM to Generate & Debug Code**

**Practice:**

* Identify hallucinations
* Identify logic errors
* How to fix prompt or code

---

## **Day 23 — Gradient Checking**

**Handwritten:**

* analytical grad
* numerical grad
* Compare errors

(Extremely useful for research and implementing attention backward)

---

## **Day 24 — Attention Mask (Essential)**

**Implementation:**

* causal mask
* padding mask
* Multi-mask combination

Master "Information Flow Control".

---

## **Day 25 — Initialization Principles**

* Xavier
* Kaiming
* LayerScale
* RMSNorm vs LN
* Why deep Transformers are unstable

---

## **Day 26 — Training Stability**

(This is the most important lesson for financial models)

**Study:**

* grad clip
* weight decay
* lr schedule
* warmup
* Role of LN in cross-attn

---

## **Day 27 — Train a Mini Transformer LM**

Train a toy LLM to feel:

* overfit
* underfit
* loss dynamics

---

## **Day 28 — Write “From Scratch GPT” Blog**

Organize learning outcomes.

---

## **Day 29 — Build a Personal Template Repo**

**Includes:**

* training loop
* dataset loader
* transformer block
* config system

Usable for future research (quant/LLM/FM).

---

## **Day 30 — Final Project: Build Your Own Mini GPT**

**Implementation:**

* tokenizer
* transformer
* training script
* inference script

**Objectives:**

* Debug large models
* Modify Transformer structures
* Fine-tune LLMs
* Build refiner / projector modules

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
