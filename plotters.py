# -*- coding: utf-8 -*-
"""
Functions for plotting distributions and performance metrics.

Used to visualise ideal observer model outputs and performance on the 
"ventriloquist effect" illustration project.

First released on Thu Nov 22 2023

@author: João Filipe Ferreira
"""

# Global imports
import numpy as np                  # Maths

import matplotlib.pyplot as plt     # Matplotlib (i.e. native) plotting package
from matplotlib import ticker       # Ticker class from Matplotlib
import seaborn as sns               # Seaborn plotting wrapper package

def gaussian(x, mu, sigma):
    """
    Generate a Gaussian distribution centred in mu and with standard deviation sigma.

    Parameters
    ----------
    mu, sigma
        mean and standard deviation of Gaussian distribution.
    """
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))

def find_nearest_idx(vector, value):
    """
    Finds index of element nearest to input value in a vector.

    Parameters
    ----------
    mu, sigma
        mean and standard deviation of Gaussian distribution.
    """    
    array = np.asarray(vector)
    idx = (np.abs(vector - value)).argmin()
    return idx

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def plot_estimates(mu_p, sigma_p, s_v, sigma_v, s_a, sigma_a, p_single_source, s_v_est, s_a_est):
    """
    Plots the prior, likelihoods, and angular estimates.
    
    Visualises the generative model distributions along with the ideal 
    observer's inferences for a given experimental trial.
    """

    # Generate x axis angle values for plotting
    x = np.linspace(-120, 120, 10000)

    prior              = gaussian(x, mu_p, sigma_p)
    ideal_posterior_v  = gaussian(x, s_v, sigma_v)
    ideal_posterior_a  = gaussian(x, s_a, sigma_a)

    # Create the graph using Seaborn
    sns.set(style="white")
    fig = plt.figure(figsize=(10, 6))    

    # Plot prior
    plt.plot(x, prior, label="Prior", linestyle="dotted", color='black')

    # Plot ideal visual sensation distribution
    plt.plot(x, ideal_posterior_v, label="Ideal Visual Posterior", color='blue')

    # Plot ideal auditory sensation distribution
    plt.plot(x, ideal_posterior_a, label="Ideal Auditory Posterior", color='green')

    # Plot ideal visual and auditory estimates
    idx_v = find_nearest_idx(x, s_v)
    plt.plot(x[idx_v], ideal_posterior_v[idx_v], label="Ideal Visual Estimate",
             linestyle='None', marker="o", markersize=7, alpha=.25, markeredgecolor="red", markerfacecolor="blue")
    idx_a = find_nearest_idx(x, s_a)
    plt.plot(x[idx_a], ideal_posterior_a[idx_a], label="Ideal Auditory Estimate",
             linestyle='None', marker="o", markersize=7, alpha=.25, markeredgecolor="red", markerfacecolor="green")      

    # Plot visual and auditory estimates given estimate C
    idx = find_nearest_idx(x, s_v_est)
    plt.plot(x[idx], ideal_posterior_v[idx_v], label="Actual Visual Estimate",
             linestyle='None', marker="*", markersize=10, markeredgecolor="red", markerfacecolor="blue")
    idx = find_nearest_idx(x, s_a_est)
    plt.plot(x[idx], ideal_posterior_a[idx_a], label="Actual Auditory Estimate",
             linestyle='None', marker="*", markersize=10, markeredgecolor="red", markerfacecolor="green")      

    # Customize the plot
    if p_single_source >= .5:
        plt.title(f"Case in which it is more likely that a single source exists [P(C=1)={round(p_single_source*100,2)}%] (\"ventriloquist effect\").")
    else:
        plt.title(f"Case in which it is more likely that two independent sources exist [P(C=2)={100-round(p_single_source*100,2)}%].")    
    plt.xlabel("Degrees (\u00b0)")  # \u00b0 is the Unicode for the degree symbol
    plt.ylabel("P")

    # Set the x-axis ticks to show only integer values
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))     

    plt.legend()

    # Show the plot
    plt.show()

def plot_simple_performance_metrics(conf_mat_C, mae_v, mae_a, mas_v, mas_a):
    """
    Plots performance metrics summarising ideal observer accuracy.
    
    Visualises confusion matrix for C and mean errors and sensory shifts across trials:
        conf_mat_C: Confusion matrix for estimating number of sources
        mae_v: Mean absolute error for visual location estimates
        mae_a: Mean absolute error for auditory location estimates
        mas_v: Mean absolute shift for visual location estimates 
        mas_a: Mean absolute shift for auditory location estimates
    """

    # Plot Confusion Matrix
    true        = ["C=1", "C=2"]
    estimated   = ["C=1", "C=2"]

    fig, ax = plt.subplots()

    im, cbar = heatmap(conf_mat_C, true, estimated, ax=ax,
                    cmap="YlGn", cbarlabel="Frequency (%)")
    texts = annotate_heatmap(im, valfmt="{x:.1f}%")

    ax.set_title("Confusion Matrix for C")
    ax.set_xlabel("Estimated Number of Sources")
    ax.set_ylabel("True Number of Sources")

    fig.tight_layout()
    plt.show()

    fig.tight_layout()
    plt.show()

    # Plot MAE values
    mae_values = [mae_v, mae_a]
    mae_labels = ["Visual", "Auditory"]

    plt.figure(figsize=(6, 8))
    ax = plt.gca()
    # Set the context with a specific size
    sns.set_context(rc={"figure.figsize": (3, 4)})
    sns.barplot(x=mae_labels, y=mae_values, palette=["blue", "green"])
    # add the annotation
    ax.bar_label(ax.containers[0], fmt='Mean: %.2f\u00b0', label_type='center', color='white')
    plt.title("Mean Absolute Errors (MAE)")
    plt.ylabel("Degrees (\u00b0)") # \u00b0 is the Unicode for the degree symbol
    plt.show()

    # Plot MAS values
    mas_values = [mas_v, mas_a]

    plt.figure(figsize=(6, 8))
    ax = plt.gca()
    # Set the context with a specific size
    sns.set_context(rc={"figure.figsize": (3, 4)})    
    sns.barplot(x=mae_labels, y=mas_values, palette=["blue", "green"])
    # add the annotation
    ax.bar_label(ax.containers[0], fmt='Mean: %.2f\u00b0', label_type='center', color='white')
    plt.title("Mean Absolute Shifts (MAS)")
    plt.ylabel("Degrees (\u00b0)") # \u00b0 is the Unicode for the degree symbol
    plt.show()
