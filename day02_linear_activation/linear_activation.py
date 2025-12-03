import numpy as np

class Linear:
    def __init__(self, input_dim, output_dim):
        # Xavier/Glorot Initialization
        # Keeps the scale of gradients roughly the same in all layers.
        limit = np.sqrt(6 / (input_dim + output_dim))
        self.W = np.random.uniform(-limit, limit, (input_dim, output_dim))
        self.b = np.zeros((1, output_dim))
        
        # Cache for backward pass
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        x: (batch_size, input_dim)
        """
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        """
        dout: Gradient of loss w.r.t output (batch_size, output_dim)
        """
        batch_size = dout.shape[0]
        
        # dW = X^T . dout
        self.dW = np.dot(self.x.T, dout)
        
        # db = sum(dout) across batch
        self.db = np.sum(dout, axis=0, keepdims=True)
        
        # dx = dout . W^T
        dx = np.dot(dout, self.W.T)
        
        return dx

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask

class GELU:
    """
    Gaussian Error Linear Unit.
    Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    def backward(self, dout):
        # Exact derivative of the approximation is complex, 
        # but for handwritten practice, let's implement the derivative of the approx.
        # d/dx [0.5x(1+tanh(y))] where y = sqrt(2/pi)(x + 0.044715x^3)
        
        x = self.input
        const_1 = np.sqrt(2 / np.pi)
        const_2 = 0.044715
        
        x_cubed = np.power(x, 3)
        inner = const_1 * (x + const_2 * x_cubed)
        tanh_inner = np.tanh(inner)
        
        # Derivative of tanh(u) is 1 - tanh^2(u)
        dtanh = 1 - tanh_inner**2
        
        # Derivative of inner w.r.t x
        d_inner = const_1 * (1 + 3 * const_2 * x**2)
        
        # Product rule: u'v + uv'
        # u = 0.5x, v = 1 + tanh(inner)
        # u' = 0.5
        # v' = dtanh * d_inner
        
        dx = 0.5 * (1 + tanh_inner) + 0.5 * x * (dtanh * d_inner)
        
        return dout * dx

def mse_loss(y_pred, y_true):
    """
    Mean Squared Error
    L = mean((y_pred - y_true)^2)
    dL/dy_pred = 2 * (y_pred - y_true) / N
    """
    N = y_pred.shape[0]
    loss = np.mean((y_pred - y_true)**2)
    d_pred = 2 * (y_pred - y_true) / N
    return loss, d_pred

def mae_loss(y_pred, y_true):
    """
    Mean Absolute Error
    L = mean(|y_pred - y_true|)
    dL/dy_pred = sign(y_pred - y_true) / N
    """
    N = y_pred.shape[0]
    loss = np.mean(np.abs(y_pred - y_true))
    d_pred = np.sign(y_pred - y_true) / N
    return loss, d_pred

def main():
    np.random.seed(42)
    
    # 1. Setup Data (Batch size 4, 3 features)
    X = np.array([
        [1.0, 2.0, -1.0],
        [0.5, -1.0, 0.0],
        [-2.0, 1.0, 1.0],
        [0.0, 0.0, 0.5]
    ])
    
    # Target (Batch size 4, 2 outputs)
    Y_true = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0]
    ])
    
    print("Input X shape:", X.shape)
    print("Target Y shape:", Y_true.shape)
    
    # 2. Initialize Layers
    linear1 = Linear(input_dim=3, output_dim=4)
    activation1 = GELU() # Try ReLU() here too
    linear2 = Linear(input_dim=4, output_dim=2)
    
    # 3. Forward Pass
    print("\n--- Forward Pass ---")
    out1 = linear1.forward(X)
    act1 = activation1.forward(out1)
    out2 = linear2.forward(act1)
    
    print("Layer 1 Output:\n", out1)
    print("Activation Output:\n", act1)
    print("Final Output:\n", out2)
    
    # 4. Loss Calculation
    loss, d_loss = mse_loss(out2, Y_true)
    print(f"\nMSE Loss: {loss:.6f}")
    
    # 5. Backward Pass
    print("\n--- Backward Pass ---")
    dout2 = linear2.backward(d_loss)
    dact1 = activation1.backward(dout2)
    dx = linear1.backward(dact1)
    
    print("Gradient dW2 shape:", linear2.dW.shape)
    print("Gradient db2 shape:", linear2.db.shape)
    print("Gradient dW1 shape:", linear1.dW.shape)
    print("Gradient dX shape:", dx.shape)
    
    # 6. Simple Update (SGD)
    lr = 0.01
    linear1.W -= lr * linear1.dW
    linear1.b -= lr * linear1.db
    linear2.W -= lr * linear2.dW
    linear2.b -= lr * linear2.db
    
    print("\nWeights updated. Checking loss reduction...")
    
    # Check new loss
    out1_new = linear1.forward(X)
    act1_new = activation1.forward(out1_new)
    out2_new = linear2.forward(act1_new)
    new_loss, _ = mse_loss(out2_new, Y_true)
    
    print(f"New MSE Loss: {new_loss:.6f}")
    if new_loss < loss:
        print("✅ Loss decreased! Backprop works.")
    else:
        print("❌ Loss did not decrease. Check gradients.")

if __name__ == "__main__":
    main()
