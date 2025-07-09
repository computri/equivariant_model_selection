import matplotlib.pyplot as plt
import json 
import numpy as np
import pickle

import matplotlib
import matplotlib.pyplot as plt


matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], "size": 20, })
matplotlib.rc('text', usetex=True)

cmap = plt.cm.viridis

# Generate 4 discrete colors
color_map = [cmap(i) for i in [0.2, 0.4, 0.6, 0.8]]
matplotlib.rcParams['axes.linewidth'] = 3

metric2title = {
    "brier": "Brier",
    "ECE": "ECE",
    "width_mean": "Mean set size",
    "post_train_log_likelihood": "Log Lik.",
    "post_log_marglik": "Log Marg. Lik.",
    "predictive_ll": "NLL"
}

target2title = {
    "mu": r"$\mu$",
    "alpha": r"$\alpha$",
    "homo": r"$\varepsilon_{HOMO}$",
    "lumo": r"$\varepsilon_{LUMO}$",
    "Cv": r"$C_{\nu}$"
}