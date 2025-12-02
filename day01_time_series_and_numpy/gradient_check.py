import numpy as np

def f(x):
    """Simple function: f(x) = x^2 + 3x"""
    return x**2 + 3*x

def analytical_gradient(x):
    """Analytical derivative: f'(x) = 2x + 3"""
    return 2*x + 3

def numerical_gradient(f, x, h=1e-5):
    """
    Numerical derivative using central difference:
    f'(x) approx (f(x+h) - f(x-h)) / (2h)
    """
    return (f(x + h) - f(x - h)) / (2 * h)

def check_gradient():
    x_test = 5.0
    
    grad_ana = analytical_gradient(x_test)
    grad_num = numerical_gradient(f, x_test)
    
    print(f"Testing function f(x) = x^2 + 3x at x = {x_test}")
    print(f"Analytical Gradient: {grad_ana}")
    print(f"Numerical Gradient:  {grad_num}")
    
    # Relative error
    error = abs(grad_ana - grad_num) / (abs(grad_ana) + abs(grad_num) + 1e-8)
    print(f"Relative Error:      {error}")
    
    if error < 1e-6:
        print("✅ Gradient Check Passed!")
    else:
        print("❌ Gradient Check Failed!")

if __name__ == "__main__":
    check_gradient()
