import matplotlib.pyplot as plt
import numpy as np
import os
from enum import Enum, auto
from datetime import datetime

VAR_MARKER = {
    "base": {"ls": "-",  "marker": "o"}, 
    "grad": {"ls": "--", "marker": "s"} 
}

POLY_COLORS = {
    1: "tab:blue",
    2: "tab:orange",
    3: "tab:green",
    4: "tab:red",
    5: "tab:purple",
    6: "tab:brown",
    7: "tab:pink",
    8: "tab:gray",
}

BDF_COLORS = {
    1: "yellowgreen",
    2: "deepskyblue",
    3: "blue",
    4: "magenta",
    5: "crimson",
    6: "darkorange"
}

##############################
# Enums for consistency      #
##############################
class ConvType(Enum):
    SPATIAL = auto()
    POLYNOMIAL = auto()
    TEMPORAL = auto()

class AnalysisType(Enum):
    RATE = auto() 
    SATURATION = auto()

##############################
# Class to select the styles #
##############################
class PlotTheme:
    @staticmethod
    def get_style(l=None, nu=None, var="base", mode="standard"):
        """
        Return the parameters for plt.plot() based on the dictionaries.
        mode: 'spatial_poly', 'bdf_saturation', 'temporal'
        """
        style = {"linewidth": 1.5, "markersize": 6}
        
        # The markers depend on the variable
        style.update(VAR_MARKER.get(var, VAR_MARKER["base"]))

        if mode == "spatial_poly":
            # Color based on polynomial degree l
            style["color"] = POLY_COLORS.get(l, "black")
            style["label"] = f"P{l} ({var})"
            
        elif mode == "bdf_saturation" or mode == "temporal":
            # Color based on the degree nu of BDF
            style["color"] = BDF_COLORS.get(nu, "black")
            style["label"] = f"BDF {nu} ({var})"
            
        return style

##############################
# Class to plot convergences #
##############################
class ConvergencePlotter:
    # Private: ______________________________________________________________________________ 
    # Constructor
    def __init__(self, title=""):
        self.title = title

    # Private method to set up the figure
    def _setup_figure(self, n_cols=2):
        fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6), constrained_layout=True)
        return fig, ([axes] if n_cols == 1 else axes)
    
    # Private method to setup axes
    def _format_axes(self, axes, conv_type, goal, n_cols):
        titles = ["Concentration", "Gradient"] if n_cols == 2 else [f"{conv_type.name.capitalize()} Convergence"]
        
        for i, ax in enumerate(axes):
            ax.set_title(titles[i])
            ax.grid(True, which="both", ls=":", alpha=0.6)
            ax.legend()
            
            # Labels for both axes
            if conv_type == ConvType.POLYNOMIAL:
                ax.set_xlabel("Polynomial Degree $l$")
            elif conv_type == ConvType.TEMPORAL:
                ax.set_xlabel("Time step $\Delta t$")
            else:
                ax.set_xlabel("Mesh size $h$")
            ax.set_ylabel("Error")

    # Private method to save the plot
    def _save_plot(self, filename_base):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        plot_dir = os.path.join(base_dir, "Plots")
        os.makedirs(plot_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        path = os.path.join(plot_dir, f"{filename_base}_{timestamp}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Plot saved: {path}")

    # Private method to plot reference lines (No more needed?)
    def _plot_reference_slope(self, ax, x, y, slope, label=None):
        """Write the reference lines."""
        x = np.array(x)
        # Use the first point to fix the line
        y_ref = y[0] * (x / x[0])**slope
        ax.loglog(x, y_ref, ls=':', color='black', alpha=0.8, label=label if label else f"slope {slope}")

    # ???
    def _apply_aesthetics(self, axes, conv_type):
        for ax in axes:
            ax.grid(True, which="both", linestyle=":", alpha=0.7)
            ax.legend(fontsize=12)
            if conv_type == ConvType.SPATIAL:
                ax.set_xlabel(r"$h[-]$")
                ax.invert_xaxis()
            elif conv_type == ConvType.TEMPORAL:
                ax.set_xlabel(r"$\tau [-]$")
            elif conv_type == ConvType.POLYNOMIAL:
                ax.set_xlabel(r"$\ell$")
    
    def _compute_limits(self, data_list):
        """Compute x and y limits."""
        all_x = []
        all_y = []
        
        for data in data_list:
            all_x.extend(data['x'])
            all_y.extend(data['y'])
            
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        ylim = (y_min/100, y_max*100)
        xlim = (x_min, x_max)
        
        return xlim, ylim
    
    # Public: ________________________________________________________________________________ 
    def plot(self, data_list, conv_type=ConvType.SPATIAL, goal=AnalysisType.RATE):
        """
        data_list: list of dict {'x':[], 'y':[], 'l':int, 'nu':int, 'var':'base'/'grad'}
        conv_type: SPATIAL, POLYNOMIAL, TEMPORAL
        goal: RATE, SATURATION (ignored for TEMPORAL)
        """
        # Set the layout and the setup the figure
        n_cols = 1 if conv_type == ConvType.TEMPORAL or (conv_type == ConvType.POLYNOMIAL and goal == AnalysisType.RATE) else 2
        fig, axes = self._setup_figure(n_cols)
        xlim, ylim = self._compute_limits(data_list)

        for data in data_list:
            l = data.get('l')
            nu = data.get('nu')
            var = data.get('var', 'base')
            ax = axes[0] if (n_cols == 1 or var == 'base') else axes[1]
            
            x = np.array(data['x'])
            y = np.array(data['y'])

            # --- Spatial convergence ---
            if conv_type == ConvType.SPATIAL:
                col = POLY_COLORS.get(l, "black")
                label = fr"$\ell = {l}$"
                ax.loglog(x, y, "o-", color=col, label=label)
                
                # Theoretical reference: h^(l+1) or h^l depending on the variable
                order = (l + 1) if var == 'base' else l
                ref = y[0] * (x / x[0])**order
                ax.loglog(x, ref, "--", color=col, alpha=0.5, label=fr"$O(h^{{{order}}})$")

            # --- Polynomial convergence ---
            elif conv_type == ConvType.POLYNOMIAL:
                col = "yellowgreen" if var == 'base' else "lime"
                fmt = "-o" if var == 'base' else "--s"
                ax.semilogy(x, y, fmt, color=col, linewidth=2, markersize=8, label=f"$E_{{{'c' if var=='base' else 'sigma'}}}$")
                
                # Reference h^l (only if var==base and h is provided)
                if var == 'base' and 'h_ref' in data:
                    h = data['h_ref']
                    ref = y[0]*(h**(x - x[0]))
                    ax.semilogy(x, ref, "k--", linewidth=3, label=r"$h^{\ell}$")

            # --- Temporal convergence ---
            elif conv_type == ConvType.TEMPORAL:
                col = BDF_COLORS.get(nu, "black")
                ax.loglog(x, y, "o-", color=col, linewidth=2, label=fr"BDF{nu}")
                
                # Theoretical reference: tau^nu
                ref = y[0] * (x / x[0])**nu
                ax.loglog(x, ref, "--", color=col, linewidth=3, alpha=0.8, label=fr"$\tau^{{{nu}}}$")

        # Apply limits
        for ax in axes:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        # Formatting axis
        self._format_axes(axes, conv_type, goal, n_cols)
        self._apply_aesthetics(axes, conv_type)
        self._save_plot(conv_type.name.lower())
            
        # Title and plot
        plt.suptitle(self.title)
        plt.show()