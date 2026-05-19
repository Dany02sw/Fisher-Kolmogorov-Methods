from Utilities.EnumUtilities       import SpaceMethod, TimeMethod
from Utilities.MathUtilities       import compute_rate, compute_exponential_fit
from Utilities.DictionaryUtilities import ERROR_LABELS_PRINT

# Function to print space convergence rates ____________________________________________________________________________________________________________
def print_space_rates(errors_c, errors_grad, hs, N_list, l, method=SpaceMethod.LDG):
    label_c, label_grad = ERROR_LABELS_PRINT[method]
    header = f" Space convergence rates for l = {l} (t = T) "
    print(f"\n{header:=^98}")
    print(f"\n{'N_from':>6} → {'N_to':>6} | {label_c+' from':>12} {label_c+' to':>12} {'rate':>6} {'(expc)':>5} | {label_grad+' from':>12} {label_grad+' to':>12} {'rate':>6} {'(expc)':>5}")
    print("-"*98)
    for i in range(1, len(N_list)):
        rate_c    = compute_rate(errors_c,    hs, i)
        rate_grad = compute_rate(errors_grad, hs, i)
        print(f"{N_list[i-1]:>6d} → {N_list[i]:>6d} | {errors_c[i-1]:>12.4e} {errors_c[i]:>12.4e} {rate_c:>6.2f} ({l+1:.2f}) | {errors_grad[i-1]:>12.4e} {errors_grad[i]:>12.4e} {rate_grad:>6.2f} ({l:.2f})")

# Function to print polynomial convergence rates _______________________________________________________________________________________________________
def print_polynomial_rates(errors_c, errors_grad, l_list, method=SpaceMethod.LDG):
    label_c,   label_grad            = ERROR_LABELS_PRINT[method]
    beta_c,    fitted_c,    res_c    = compute_exponential_fit(errors_c,    l_list)
    beta_grad, fitted_grad, res_grad = compute_exponential_fit(errors_grad, l_list)
    header = f" Polynomial degree convergence rates (t = T) "
    print(f"\n{header:=^60}")
    print(f"\n{'l':>4} | {label_c:>12} {'fit':>12} | {label_grad:>12} {'fit':>12}")
    print("-"*60)
    for i in range(len(l_list)):
        print(f"{l_list[i]:>4d} | {errors_c[i]:>12.4e} {fitted_c[i]:>12.4e} | {errors_grad[i]:>12.4e} {fitted_grad[i]:>12.4e}")
    print("-"*60)
    print(f"  {label_c   } ~ exp(-{beta_c:.2f} * l), residual = {res_c:.2e}")
    print(f"  {label_grad} ~ exp(-{beta_grad:.2f} * l), residual = {res_grad:.2e}")


# Function to print time convergence rates ____________________________________________________________________________________________________________
def print_time_rates(errors_c, errors_grad, dt_list, order, time_method=TimeMethod.THETA, space_method=SpaceMethod.LDG):
    label_c, label_grad = ERROR_LABELS_PRINT[space_method]
    expected   = order if time_method == TimeMethod.BDF else (2.0 if abs(order - 0.5) < 1e-10 else 1.0)
    method_str = f"BDF{order}" if time_method == TimeMethod.BDF else f"θ={order}"
    header = f" Time convergence rates for {method_str} (t = T) "
    print(f"\n{header:=^107}")
    print(f"\n{'dt_from':>10} → {'dt_to':>10} | {label_c+' from':>12} {label_c+' to':>12} {'rate':>6} {'(expc)':>5} | {label_grad+' from':>12} {label_grad+' to':>12} {'rate':>6} {'(expc)':>5}")
    print("-"*107)
    for i in range(1, len(dt_list)):
        rate_c    = compute_rate(errors_c,    dt_list, i)
        rate_grad = compute_rate(errors_grad, dt_list, i)
        print(f"{dt_list[i-1]:>10.4f} → {dt_list[i]:>10.4f} | {errors_c[i-1]:>12.4e} {errors_c[i]:>12.4e} {rate_c:>6.2f} ({expected:.2f}) | {errors_grad[i-1]:>12.4e} {errors_grad[i]:>12.4e} {rate_grad:>6.2f} ({expected:.2f})")

# Functions to print banners in mains _________________________________________________________________________________________________________________
def print_title(title, char="█", width=70):
    border = char * width
    inner = width - 2
    padding = max(0, inner - len(title) - 2)
    left = padding // 2
    right = padding - left
    print(f"\n{border}")
    print(f"{char}{char * left} {title} {char * right}{char}")
    print(f"{border}\n")

def print_subtitle(title, char="═", width=70):
    border = char * width
    padding = max(0, width - len(title))
    left = padding // 2
    right = padding - left
    print(f"\n{border}")
    print(f"{' ' * left}{title}{' ' * right}")
    print(f"{border}\n")