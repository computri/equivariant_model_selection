import numpy as np

import matplotlib
import matplotlib.pyplot as plt


matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], "size": 20, })
matplotlib.rc('text', usetex=True)

cmap = plt.cm.viridis

# Generate 4 discrete colors
colors = [cmap(i) for i in [0.2, 0.4, 0.6, 0.8]]
matplotlib.rcParams['axes.linewidth'] = 3