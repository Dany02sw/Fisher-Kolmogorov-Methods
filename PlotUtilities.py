import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# =============================================================================
# POLYNOMIAL CONVERGENCE PLOT
# =============================================================================
def plot_polynomial_convergence(errors_c, errors_sigma, l_list, h, save=False):
    """
    Function to plot polynomial convergence of both a base variable and a gradient-based one
    """

    # --- ℓ dynamically adapted ---
    L_ext = np.arange(l_list[0], l_list[-1] + 1)

    # --- Extract numerical values ---
    L_num = np.array(l_list)
    Ec_vals = np.array([errors_c[l-1] for l in l_list])
    Es_vals = np.array([errors_sigma[l-1] for l in l_list])

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- E_c ---
    ax.semilogy(
        L_num, Ec_vals, "-o", color="yellowgreen", linewidth=2,
        markersize=8, label=r"$E_c$-error"
    )

    # --- E_sigma ---
    ax.semilogy(
        L_num, Es_vals, "--s", color="lime", linewidth=2,
        markersize=8, label=r"$E_{\sigma}$-error"
    )

    # --- h^ℓ ---
    ref = Ec_vals[0] * (h ** (L_ext - l_list[0]))
    ax.semilogy(
        L_ext, ref, "k--", linewidth=3, label=r"$h^{\ell}$"
    )

    # --- Aesthetics ---
    ax.set_xlim(l_list[0], l_list[-1])
    ax.set_xticks(np.arange(l_list[0], l_list[-1]+1))

    ax.set_xlabel(r"$\ell$", fontsize=16)
    ax.set_ylabel("Error", fontsize=16)
    ax.set_title(r"Errors $E_c$ and $E_{\sigma}$", fontsize=20)
    ax.grid(True, which="both", linestyle=":", alpha=0.7)

    ax.legend(fontsize=14, loc="upper right")

    plt.tight_layout()

    # --- save figure ---
    if save:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        plot_dir = os.path.join(base_dir, "Plots")
        os.makedirs(plot_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        out_path = os.path.join(plot_dir, f"polynomial_convergence_{timestamp}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print("Polynomial convergence plot saved into Plots folder")
    plt.show()


# =============================================================================
# DIFFERENT COLORS FOR EACH l
# =============================================================================
POLY_COLORS = {
    1: "tab:blue",
    2: "tab:orange",
    3: "tab:green",
    4: "tab:red",
    5: "tab:purple",
}

# =============================================================================
# 1) SPACE CONVERGENCE PLOTS — E_c and E_sigma
# =============================================================================
def plot_spatial_convergence_with_sigma(h_dict, err_c_dict, err_s_dict, l_list):

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax_c, ax_s = axes

    # =====================  E_c  =====================
    for l in l_list:
        hs  = np.array(h_dict[l])
        Ec  = np.array(err_c_dict[l])
        col = POLY_COLORS.get(l, "black")

        # Plot
        ax_c.loglog(hs, Ec, "o-", color=col, label=fr"$\ell = {l}$")

        # riferimento teorico O(h^{l+1})
        ref = Ec[0] * (hs / hs[0]) ** (l + 1)
        ax_c.loglog(hs, ref, "--", color=col, alpha=0.5, label=fr"$O(h^{{{l+1}}})$")

    ax_c.set_title(r" Errors $E_c$ ", fontsize=16)
    ax_c.set_xlabel(r"$h[-]$", fontsize=14)
    ax_c.set_ylabel(
        r"$\||\, c(\cdot,T) - u(w_h^{(N)}) \,\||_{L^2(\Omega)}$",
        fontsize=15
    )
    ax_c.grid(True, which="both", linestyle=":")
    ax_c.invert_xaxis()
    ax_c.legend(fontsize=12, title_fontsize=12)

    # =====================  E_sigma  =====================
    for l in l_list:
        hs  = np.array(h_dict[l])
        Es  = np.array(err_s_dict[l])
        col = POLY_COLORS.get(l, "black")

        ax_s.loglog(hs, Es, "o-", color=col, label=fr"$\ell = {l}$")

        # riferimento teorico O(h^l)
        ref = Es[0] * (hs / hs[0]) ** (l)
        ax_s.loglog(hs, ref, "--", color=col, alpha=0.5, label=fr"$O(h^{{{l}}})$")

    ax_s.set_title(r"Errors $E_{\sigma}$ ", fontsize=16)
    ax_s.set_xlabel(r"$h[-]$", fontsize=14)
    ax_s.set_ylabel(
        r"$\||\, \nabla c(\cdot,T) + \sigma_h^{(N)} \,\||_{L^2(\Omega)^d}$",
        fontsize=15
    )
    ax_s.grid(True, which="both", linestyle=":")
    ax_s.invert_xaxis()
    ax_s.legend(fontsize=12, title_fontsize=12)

    plt.tight_layout()

    # --- save figure ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(base_dir, "Plots")
    os.makedirs(plot_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = os.path.join(plot_dir, f"space_convergence_{timestamp}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print("Space convergence plot saved into Plots folder!")
    plt.show()


# =============================================================================
# DIFFERENT COLORS FOR EACH nu
# =============================================================================
BDF_COLORS = {
    1: "yellowgreen",
    2: "deepskyblue",
    3: "blue",
    4: "magenta",
    5: "crimson",
    6: "darkorange",
}

# =============================================================================
# 3) TIME CONVERGENCE PLOTS — E_c and E_sigma
# =============================================================================
def plot_time_convergence_single(dt_list, errors, nu):
    """
    Plot della convergenza temporale per un solo ordine BDF–ν.
    dt_list  : lista dei passi temporali
    errors   : valori E_c corrispondenti
    nu       : ordine del metodo BDF
    """

    dt_list  = np.array(dt_list, dtype=float)
    err_vals = np.array(errors, dtype=float)

    col = BDF_COLORS.get(nu, "black")

    fig, ax = plt.subplots(figsize=(10, 6))

    # curva numerica
    ax.loglog(
        dt_list, err_vals, "o-", linewidth=2, markersize=8,
        color=col, label=fr"BDF{nu}"
    )

    # retta di riferimento O(tau^nu), stessa pendenza, stesso colore
    ref = err_vals[0] * (dt_list / dt_list[0])**nu
    ax.loglog(
        dt_list, ref, "--", linewidth=3, alpha=0.8,
        color=col, label=r"$\tau^{" + str(nu) + "}$"
    )

    ax.set_xlabel(r"$\tau\;[-]$", fontsize=16)
    ax.set_ylabel(
        r"$\|c(\cdot,T) - u(w_h^{(N)})\|_{L^2(\Omega)}$",
        fontsize=16
    )
    ax.set_title(fr"Errors $E_c$ using BDF{nu}", fontsize=20)

    ax.grid(True, which="both", linestyle=":", alpha=0.7)
    ax.legend(fontsize=14, loc="upper left")

    plt.tight_layout()

    # --- save figure ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(base_dir, "Plots")
    os.makedirs(plot_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = os.path.join(plot_dir, f"BDF_{nu}_{timestamp}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")

    plt.show()

# =============================================================================
# 4) PENALTY CONVERGENCE PLOTS — E_c and E_sigma
# =============================================================================
def plot_penalty_convergence(eps_list, errors_dict):
    """
    Plot della convergenza rispetto al parametro di penalizzazione epsilon.

    eps_list    : lista dei valori di epsilon
    errors_dict : dizionario tipo
                  {
                      "BDF1": [...],
                      "BDF6": [...]
                  }
    """

    eps = np.array(eps_list, dtype=float)

    fig, ax = plt.subplots(figsize=(10,6))

    COLORS = {
        "BDF1": "yellowgreen",
        "BDF6": "darkorange"
    }

    # ===================== numerical curves =====================
    for method, errors in errors_dict.items():

        err_vals = np.array(errors, dtype=float)

        ax.loglog(
            eps,
            err_vals,
            "o-",
            linewidth=2,
            markersize=7,
            color=COLORS.get(method,"black"),
            label=method
        )

    # ===================== reference lines =====================
    ref_point = list(errors_dict.values())[0][-1]

    ref_eps = ref_point * (eps/eps[-1])
    ref_sqrt = ref_point * np.sqrt(eps/eps[-1])

    ax.loglog(
        eps,
        ref_sqrt,
        "k:",
        linewidth=2.5,
        label=r"$\sim \sqrt{\epsilon}$"
    )

    ax.loglog(
        eps,
        ref_eps,
        "k--",
        linewidth=2.5,
        label=r"$\sim \epsilon$"
    )

    # ===================== aesthetics =====================
    ax.set_xlabel(r"$\epsilon$", fontsize=16)

    ax.set_ylabel(
        r"$\|c(\cdot,T) - u(w_h^{(N)})\|_{L^2(\Omega)}$",
        fontsize=16
    )

    ax.set_title(r"Errors $E_c$ against $\epsilon$", fontsize=20)

    ax.grid(True, which="both", linestyle=":", alpha=0.7)

    ax.legend(fontsize=14)

    plt.tight_layout()

    # --- save figure ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(base_dir, "Plots")
    os.makedirs(plot_dir, exist_ok=True)

    out_path = os.path.join(plot_dir, "penalty_convergence.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")

    plt.show()
