import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def plot_histogram_by_category(data, ax, bin_edges=np.arange(100, 1001, 50)):
    """
    Plots histograms of sentence lengths for each category with Gaussian fit lines (scaled to counts).

    Args:
        data (dict): Keys are category names, values are lists of sentence lengths.
        ax (matplotlib.axes.Axes): The Axes object to draw on.
        bin_edges (array-like): Bin edges for the histogram.
    """
    colors = plt.cm.tab10.colors  # 10 distinct colors
    for i, (key, values) in enumerate(data.items()):
        if len(values) == 0:
            continue
        color = colors[i % len(colors)]

        # Histogram (normal counts)
        counts, bins, _ = ax.hist(values, bins=bin_edges, alpha=0.3, edgecolor='black', color=color)

        # Gaussian fit (scaled to match histogram max height)
        mu, std = np.mean(values), np.std(values)        
        x = np.linspace(0, bin_edges[-1], 1000)
        p = norm.pdf(x, mu, std)

        scale = max(counts) / np.max(p)
        ax.plot(x, p * scale, color=color, label=f'{key}')

    ax.legend()

def make_percent_category(data):
    result = {}
    for category, type_counts in data.items():
        total = sum(type_counts.values())
        result[category] = {k: (v / total * 100) for k, v in type_counts.items()} if total > 0 else {k: 0 for k in type_counts}
    return result

def plot_grouped_bar_chart(data, ax, width=0.25):
    """
    Plots a grouped bar chart for category-type counts.

    Args:
        data (dict): Nested dict with structure {category: {type: count}}.
        ax (matplotlib.axes.Axes): The Axes object to draw on.
        width (float): Width of each bar in the group.
    """
    data = make_percent_category(data)
    categories = list(data.keys())
    types = list(next(iter(data.values())).keys())
    x = np.arange(len(categories))

    if len(types) == 2:
        colors = ['#6084f7', '#d6425d']
    else:
        colors = None

    for i, t in enumerate(types):
        values = [data[cat][t] for cat in categories]
        ax.bar(x + i * width, values, width=width, label=t, color=colors[i] if colors else None)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(categories, rotation=5)
    ax.legend()


def plot_stacked_bar_chart(data, ax, height=0.5):
    """
    Horizontal stacked bar chart that writes each category label
    in the middle of the stacked bar.

    Args:
        data (dict): Nested dict with structure {category: {type: count}}.
        ax (matplotlib.axes.Axes): The Axes object to draw on.
        height (float): Height of each bar.
    """
    data = make_percent_category(data)
    categories = list(data.keys())
    types = list(next(iter(data.values())).keys())
    y = np.arange(len(categories))

    colors = ['#6084f7', '#d6425d'] if len(types) == 2 else None
    lefts = [0] * len(categories)

    for i, t in enumerate(types):
        values = [data[cat][t] for cat in categories]
        ax.barh(y, values, left=lefts, height=height,
                label=t, color=colors[i] if colors else None)
        lefts = [l + v for l, v in zip(lefts, values)]

    # --- label in the middle of each bar -------------------------------------
    for idx, cat in enumerate(categories):
        total = sum(data[cat].values())
        ax.text(total / 2, idx, cat,
                va='center', ha='center',
                color='white', weight='bold')

    # remove the default y-ticks/labels
    ax.set_yticks([], fontsize = 19)
    ax.set_yticklabels([])
    ax.set_xlim(0,100)
    ax.legend()
    