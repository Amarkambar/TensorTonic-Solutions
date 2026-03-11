def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    x = x0

    for _ in range(steps):
        grad = 2*a*x + b
        x = x - lr * grad

    return x
    
print(round(gradient_descent_quadratic(1, -4, 3, 0, 0.1, 50), 2))
print(round(gradient_descent_quadratic(0.5,-1, 0, -5, 0.2, 100), 2))