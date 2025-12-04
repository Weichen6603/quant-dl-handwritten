# Day 3: Handwritten Time-Series MLP

## 1. The Goal
We want to see if a simple Multi-Layer Perceptron (MLP) can predict a non-stationary time series.
*   **Input**: A window of past prices/returns $[x_{t-w}, ..., x_{t-1}]$.
*   **Output**: The next value $x_t$.

## 2. The Data: Non-stationary Toy Series
We will generate a synthetic series that combines:
1.  **Trend**: Linear growth (non-stationary mean).
2.  **Seasonality**: Sine wave.
3.  **Noise**: Random Gaussian noise.

$$ P_t = 0.05 \cdot t + \sin(0.1 \cdot t) + \epsilon_t $$

## 3. The MLP Architecture
$$ X \xrightarrow{Linear} H_1 \xrightarrow{ReLU} H_2 \xrightarrow{Linear} \hat{Y} $$

## 4. Why MLP Fails on Time Series (The Insight)
After training, you will likely observe:
1.  **Lag Effect**: The prediction $\hat{P}_{t+1}$ often looks like a shifted version of $P_t$. The model learns "the best guess for tomorrow is today's price" (Martingale property) rather than learning the structure.
2.  **No Temporal Context**: The MLP treats the input window as a "bag of features". It doesn't inherently understand that $x_{t-1}$ is closer in time than $x_{t-10}$.
3.  **Extrapolation Failure**: If the trend continues beyond the training range, the MLP (which is a universal approximator *within* the domain) often fails to extrapolate the trend correctly.

## 5. Tasks
1.  Run the code.
2.  Observe the "Predicted" vs "True" plot.
3.  Notice the lag and the failure to capture the exact turning points perfectly.
