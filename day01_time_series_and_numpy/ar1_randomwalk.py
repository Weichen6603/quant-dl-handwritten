import numpy as np
import matplotlib.pyplot as plt

def generate_ar1(phi, c, sigma, n_steps=1000, x0=0):
    """
    Generates an AR(1) process: X_t = c + phi * X_{t-1} + epsilon_t
    
    Args:
        phi: Autoregression coefficient.
             |phi| < 1 -> Stationary
             phi = 1   -> Random Walk (Non-stationary)
        c: Constant term (drift).
        sigma: Standard deviation of the noise (epsilon).
        n_steps: Number of time steps.
        x0: Initial value.
    """
    x = np.zeros(n_steps)
    x[0] = x0
    noise = np.random.normal(0, sigma, n_steps)
    
    for t in range(1, n_steps):
        x[t] = c + phi * x[t-1] + noise[t]
        
    return x

def main():
    np.random.seed(42)
    n_steps = 200
    
    # 1. Stationary AR(1)
    # Mean reverting to c / (1 - phi) = 0 / (1 - 0.8) = 0
    phi_stat = 0.8
    ar1_stat = generate_ar1(phi=phi_stat, c=0, sigma=1.0, n_steps=n_steps)
    
    # 2. Random Walk (Non-stationary)
    # phi = 1.0
    rw = generate_ar1(phi=1.0, c=0, sigma=1.0, n_steps=n_steps)
    
    # 3. Random Walk with Drift
    # phi = 1.0, c = 0.2
    rw_drift = generate_ar1(phi=1.0, c=0.2, sigma=1.0, n_steps=n_steps)

    # Plotting
    plt.figure(figsize=(12, 6))
    
    plt.subplot(3, 1, 1)
    plt.plot(ar1_stat, label=f'Stationary AR(1) phi={phi_stat}')
    plt.title('Stationary AR(1) Process')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(rw, label='Random Walk (phi=1.0)', color='orange')
    plt.title('Random Walk (Non-stationary)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.plot(rw_drift, label='Random Walk with Drift (c=0.2)', color='green')
    plt.title('Random Walk with Drift')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('day01_time_series_plot.png')
    print("Plot saved to day01_time_series_plot.png")

if __name__ == "__main__":
    main()
