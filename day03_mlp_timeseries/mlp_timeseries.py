import numpy as np
import matplotlib.pyplot as plt

# --- 1. Reusing Components from Day 2 ---

class Linear:
    def __init__(self, input_dim, output_dim):
        # He Initialization for ReLU
        limit = np.sqrt(2 / input_dim)
        self.W = np.random.randn(input_dim, output_dim) * limit
        self.b = np.zeros((1, output_dim))
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0, keepdims=True)
        return np.dot(dout, self.W.T)

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask

def mse_loss(y_pred, y_true):
    N = y_pred.shape[0]
    loss = np.mean((y_pred - y_true)**2)
    d_pred = 2 * (y_pred - y_true) / N
    return loss, d_pred

# --- 2. The MLP Model ---

class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.layer1 = Linear(input_size, hidden_size)
        self.relu = ReLU()
        self.layer2 = Linear(hidden_size, output_size)
        
    def forward(self, x):
        out = self.layer1.forward(x)
        out = self.relu.forward(out)
        out = self.layer2.forward(out)
        return out
    
    def backward(self, d_loss):
        dout = self.layer2.backward(d_loss)
        dout = self.relu.backward(dout)
        dout = self.layer1.backward(dout)
        
    def update(self, lr):
        self.layer1.update(lr)
        self.layer2.update(lr)

# --- 3. Data Generation & Processing ---

def generate_data(n_steps=1000):
    t = np.arange(n_steps)
    # Trend + Seasonality + Noise
    trend = 0.05 * t
    seasonality = 10 * np.sin(0.1 * t)
    noise = np.random.normal(0, 1, n_steps)
    price = trend + seasonality + noise
    return price

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys).reshape(-1, 1)

# --- 4. Main Training Loop ---

def main():
    np.random.seed(42)
    
    # Hyperparameters
    SEQ_LENGTH = 10   # Lookback window
    HIDDEN_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 200
    
    # Data
    full_data = generate_data(n_steps=500)
    
    # Split Train/Test
    train_size = int(len(full_data) * 0.8)
    train_data = full_data[:train_size]
    test_data = full_data[train_size:]
    
    # Normalize (Crucial for MLP!)
    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data_norm = (train_data - mean) / std
    test_data_norm = (test_data - mean) / std # Note: using train stats
    
    # Create Sequences
    X_train, y_train = create_sequences(train_data_norm, SEQ_LENGTH)
    X_test, y_test = create_sequences(test_data_norm, SEQ_LENGTH)
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Initialize Model
    model = SimpleMLP(input_size=SEQ_LENGTH, hidden_size=HIDDEN_SIZE, output_size=1)
    
    # Training
    loss_history = []
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        # Forward
        y_pred = model.forward(X_train)
        
        # Loss
        loss, d_loss = mse_loss(y_pred, y_train)
        loss_history.append(loss)
        
        # Backward
        model.backward(d_loss)
        
        # Update
        model.update(LEARNING_RATE)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

    # Testing
    y_test_pred_norm = model.forward(X_test)
    
    # Inverse Transform for plotting
    y_test_pred = y_test_pred_norm * std + mean
    y_test_true = y_test * std + mean
    
    # Plotting
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(loss_history)
    plt.title("Training Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.subplot(2, 1, 2)
    plt.plot(y_test_true, label="True Price", color='black', alpha=0.7)
    plt.plot(y_test_pred, label="MLP Prediction", color='red', linestyle='--')
    plt.title(f"MLP Time Series Prediction (Window={SEQ_LENGTH})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('day03_mlp_result.png')
    print("\nPlot saved to day03_mlp_result.png")
    print("Observation: Look closely at the prediction. Does it just look like the True Price shifted to the right? (The 'Lag' effect)")

if __name__ == "__main__":
    main()
