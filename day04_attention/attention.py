import numpy as np
import matplotlib.pyplot as plt

class ScaledDotProductAttention:
    """
    Implements Scaled Dot-Product Attention:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    def __init__(self, d_k):
        """
        d_k: Dimension of the key (for scaling).
        """
        self.d_k = d_k
        self.scale = 1.0 / np.sqrt(d_k)
        
        # Cache for backward pass
        self.Q = None
        self.K = None
        self.V = None
        self.scores = None
        self.weights = None
        
    def forward(self, Q, K, V):
        """
        Q: (batch, seq_len, d_k)
        K: (batch, seq_len, d_k)
        V: (batch, seq_len, d_v)
        
        Returns: (batch, seq_len, d_v)
        """
        self.Q = Q
        self.K = K
        self.V = V
        
        # scores = Q @ K^T / sqrt(d_k)
        # Shape: (batch, seq_len, seq_len)
        self.scores = np.matmul(Q, K.transpose(0, 2, 1)) * self.scale
        
        # weights = softmax(scores)
        self.weights = self._softmax(self.scores)
        
        # output = weights @ V
        output = np.matmul(self.weights, V)
        
        return output
    
    def backward(self, d_output):
        """
        d_output: Gradient of loss w.r.t. output (batch, seq_len, d_v)
        
        Returns: dQ, dK, dV
        """
        batch_size, seq_len, d_v = d_output.shape
        
        # Step 1: dV = weights^T @ d_output
        dV = np.matmul(self.weights.transpose(0, 2, 1), d_output)
        
        # Step 2: d_weights = d_output @ V^T
        d_weights = np.matmul(d_output, self.V.transpose(0, 2, 1))
        
        # Step 3: Softmax backward
        d_scores = self._softmax_backward(d_weights, self.weights)
        
        # Step 4: dQ and dK from scores = Q @ K^T / sqrt(d_k)
        dQ = np.matmul(d_scores, self.K) * self.scale
        dK = np.matmul(d_scores.transpose(0, 2, 1), self.Q) * self.scale
        
        return dQ, dK, dV
    
    def _softmax(self, x):
        """
        Numerically stable softmax.
        x: (batch, seq_len, seq_len)
        """
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _softmax_backward(self, d_softmax, softmax_output):
        """
        Backward pass for softmax.
        d_softmax: upstream gradient (batch, seq_len, seq_len)
        softmax_output: forward pass output (batch, seq_len, seq_len)
        """
        # Jacobian of softmax: S_ij(δ_ij - S_ij)
        # Simplified: d_input = softmax * (d_softmax - sum(d_softmax * softmax))
        sum_term = np.sum(d_softmax * softmax_output, axis=-1, keepdims=True)
        d_input = softmax_output * (d_softmax - sum_term)
        return d_input


def numerical_gradient_check():
    """
    Verify attention backward pass using numerical gradients.
    """
    print("Running Numerical Gradient Check...")
    np.random.seed(42)
    
    batch = 2
    seq_len = 4
    d_k = 8
    d_v = 8
    
    Q = np.random.randn(batch, seq_len, d_k) * 0.1
    K = np.random.randn(batch, seq_len, d_k) * 0.1
    V = np.random.randn(batch, seq_len, d_v) * 0.1
    
    attention = ScaledDotProductAttention(d_k)
    
    # Forward
    output = attention.forward(Q, K, V)
    
    # Dummy upstream gradient
    d_output = np.random.randn(*output.shape)
    
    # Analytical gradients
    dQ_ana, dK_ana, dV_ana = attention.backward(d_output)
    
    # Numerical gradient for Q
    eps = 1e-5
    dQ_num = np.zeros_like(Q)
    
    for i in range(batch):
        for j in range(seq_len):
            for k in range(d_k):
                Q_plus = Q.copy()
                Q_plus[i, j, k] += eps
                output_plus = attention.forward(Q_plus, K, V)
                loss_plus = np.sum(output_plus * d_output)
                
                Q_minus = Q.copy()
                Q_minus[i, j, k] -= eps
                output_minus = attention.forward(Q_minus, K, V)
                loss_minus = np.sum(output_minus * d_output)
                
                dQ_num[i, j, k] = (loss_plus - loss_minus) / (2 * eps)
    
    # Compare
    diff = np.abs(dQ_ana - dQ_num).max()
    print(f"Max difference between analytical and numerical dQ: {diff}")
    
    if diff < 1e-5:
        print("✅ Gradient check PASSED!")
    else:
        print("❌ Gradient check FAILED. Check your backward implementation.")


def visualize_attention():
    """
    Visualize attention weights on a simple time series.
    """
    print("\nVisualizing Attention Weights on Time Series...")
    np.random.seed(42)
    
    seq_len = 20
    d_model = 16
    
    # Generate a simple time series (sine wave + noise)
    t = np.linspace(0, 4 * np.pi, seq_len)
    signal = np.sin(t) + 0.1 * np.random.randn(seq_len)
    
    # Create input (batch=1, seq_len=20, d_model=16)
    # We'll project the scalar signal into d_model dimensions
    X = signal.reshape(1, seq_len, 1) @ np.random.randn(1, d_model)
    
    # Initialize Q, K, V projection matrices
    W_Q = np.random.randn(d_model, d_model) * 0.1
    W_K = np.random.randn(d_model, d_model) * 0.1
    W_V = np.random.randn(d_model, d_model) * 0.1
    
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    
    attention = ScaledDotProductAttention(d_model)
    output = attention.forward(Q, K, V)
    
    # Attention weights shape: (1, seq_len, seq_len)
    attn_weights = attention.weights[0]  # Remove batch dimension
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Original signal
    axes[0].plot(signal, marker='o')
    axes[0].set_title("Input Time Series (Sine Wave + Noise)")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Value")
    axes[0].grid(True, alpha=0.3)
    
    # Attention heatmap
    im = axes[1].imshow(attn_weights, cmap='viridis', aspect='auto')
    axes[1].set_title("Attention Weights (Query vs Key)")
    axes[1].set_xlabel("Key Position")
    axes[1].set_ylabel("Query Position")
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('day04_attention_visualization.png')
    print("Plot saved to day04_attention_visualization.png")
    print("\nInterpretation:")
    print("- Each row shows: When processing position i, which past positions (columns) does it attend to?")
    print("- Bright colors = high attention weight (important)")
    print("- Dark colors = low attention weight (less relevant)")


def main():
    print("=" * 60)
    print("Day 4: Scaled Dot-Product Attention (Numpy Implementation)")
    print("=" * 60)
    
    # 1. Gradient Check
    numerical_gradient_check()
    
    # 2. Visualize Attention
    visualize_attention()
    
    print("\n" + "=" * 60)
    print("Day 4 Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Attention allows the model to focus on relevant historical information.")
    print("2. The attention weights are learned dynamically based on Q, K, V.")
    print("3. This is the foundation of Transformers (Day 7 will build on this).")
    print("\nNext: Day 5 - Multi-Head Attention (Multiple 'factors' or 'views').")


if __name__ == "__main__":
    main()
