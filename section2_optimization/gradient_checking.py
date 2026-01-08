import numpy as np
from model import MLP
from losses import cross_entropy, softmax

def numerical_gradient(model, X, y, loss_fn, eps=1e-7):
    """
    Compute numerical gradients using two-sided difference
    Returns dictionary with same structure as model.parameters()
    """
    params = model.parameters()
    num_grads = {}
    
    for param_name in params:
        param = params[param_name]
        num_grad = np.zeros_like(param)
        
        # Iterate over all elements of the parameter matrix
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            original_value = param[idx]
            
            # f(x + ε)
            param[idx] = original_value + eps
            model.zero_grad()
            pred = model.predict_proba(X)
            loss_plus = loss_fn(pred, y)
            
            # f(x - ε)
            param[idx] = original_value - eps
            model.zero_grad()
            pred = model.predict_proba(X)
            loss_minus = loss_fn(pred, y)
            
            # Two-sided difference
            num_grad[idx] = (loss_plus - loss_minus) / (2 * eps)
            
            # Restore original value
            param[idx] = original_value
            it.iternext()
        
        num_grads[param_name] = num_grad
    
    return num_grads

def gradient_check(model, X, y, loss_fn, eps=1e-7, threshold=1e-7):
    """
    Compare analytical gradients with numerical gradients
    Returns maximum relative error and whether check passed
    """
    print("=" * 70)
    print("GRADIENT CHECKING")
    print("=" * 70)
    
    # Forward pass to compute analytical gradients
    pred = model.predict_proba(X)
    loss = loss_fn(pred, y)
    grad_loss = (pred - y) / y.shape[0]  # ∂L/∂Z2
    
    # Backward pass to compute analytical gradients
    model.backward(grad_loss)
    analytical_grads = model.gradients()
    
    # Compute numerical gradients
    numerical_grads = numerical_gradient(model, X, y, loss_fn, eps)
    
    # Compare gradients
    max_error = 0
    all_passed = True
    
    print("\nParameter-wise gradient comparison:")
    print("-" * 70)
    print(f"{'Parameter':<10} {'Analytical Norm':<20} {'Numerical Norm':<20} {'Relative Error':<20} {'Passed':<10}")
    print("-" * 70)
    
    for param_name in analytical_grads:
        a = analytical_grads[param_name]
        n = numerical_grads[param_name]
        
        # Compute relative error
        numerator = np.linalg.norm(a - n)
        denominator = np.linalg.norm(a) + np.linalg.norm(n)
        relative_error = numerator / denominator if denominator > 0 else 0
        
        max_error = max(max_error, relative_error)
        passed = relative_error < threshold
        
        if not passed:
            all_passed = False
            
        print(f"{param_name:<10} {np.linalg.norm(a):<20.6e} {np.linalg.norm(n):<20.6e} "
              f"{relative_error:<20.6e} {'✓' if passed else '✗':<10}")
    
    print("-" * 70)
    print(f"\nMaximum relative error: {max_error:.2e}")
    print(f"Threshold: {threshold:.2e}")
    print(f"Gradient check {'PASSED' if all_passed else 'FAILED'}!")
    print("=" * 70)
    
    return all_passed, max_error

def test_gradient_check():
    """Test gradient checking on a small dataset"""
    np.random.seed(42)
    
    # Create small dataset
    m, d, h, c = 10, 5, 3, 2
    X = np.random.randn(m, d)
    y = np.eye(c)[np.random.randint(0, c, m)]
    
    # Create model
    model = MLP(input_dim=d, hidden_dim=h, output_dim=c, activation="relu")
    
    # Run gradient check
    passed, max_error = gradient_check(model, X, y, cross_entropy)
    
    return passed, max_error

if __name__ == "__main__":
    # Run gradient check test
    test_gradient_check()