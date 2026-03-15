import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):

    # convert to numpy (prevents sequence × float error)
    param = np.array(param, dtype=float)
    grad = np.array(grad, dtype=float)
    m = np.array(m, dtype=float)
    v = np.array(v, dtype=float)

    # moment updates
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)

    # bias correction
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    # parameter update
    param_new = param - lr * (m_hat / (np.sqrt(v_hat) + eps))

    return param_new, m, v

param = 5.0
grad = 0.0
m = 0.0
v = 0.0

param_new, m, v = adam_step(param, grad, m, v, t=1)

print("grad=0 case →", param_new)