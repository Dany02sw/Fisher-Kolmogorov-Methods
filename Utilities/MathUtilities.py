import numpy as np

# Function to compute rates
def compute_rate(errors, refinements, i):
    return np.log(errors[i-1] / errors[i]) / np.log(refinements[i-1] / refinements[i])

# Function to compute and print exponential fit wrt polynomial degree
def compute_exponential_fit(errors, l_list):
    ls       = np.array(l_list, dtype=float)
    log_e    = np.log(np.array(errors))
    coeffs   = np.polyfit(ls, log_e, deg=1)
    beta     = -coeffs[0]
    fitted   = np.exp(np.polyval(coeffs, ls))
    residual = np.max(np.abs(log_e - np.polyval(coeffs, ls)))
    return beta, fitted, residual

# Function to compute the decimals of a quantity
def get_decimals(dt):
    s = f"{dt:.15f}".rstrip('0')
    if '.' in s:
        return len(s.split('.')[1])
    return 0