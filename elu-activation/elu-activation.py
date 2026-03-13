def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    result = []

    for v in x:
        if v > 0:
            result.append(v)
        else:
            result.append(alpha * ((2.71828 ** v) - 1))

    return result

print(elu([1.0, -1.0, 0.0, 2.0, -0.5], alpha = 1.0))
print(elu([-1.0, -2.0, -3.0], alpha = 2.0))