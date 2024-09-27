import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotly.graph_objects as go
import seaborn as sns
import pandas as pd

from .afm_utils import convert_with_unit, convert_scan_setting, format_func
from PFM.domain_analysis import find_histogram_peaks
sys.path.append('../../../py-utils/src/')
from viz import scalebar

def visualize_afm_image(img, scan_size, sample_name,
                        colorbar_setting={'colorbar_setting': 'percent', 'colorbar_range': (0.2, 0.98)}, 
                        filename=None, fig=None, ax=None, figsize=(6, 4), title=None, 
                        zero_mean=False, save_plot=True, debug=False, printing=None):
    
    """
    Visualize AFM image with scalebar and colorbar.

    Parameters:
    -----------
    img : 2D numpy array
        AFM image.
    colorbar_setting['colorbar_range'] : tuple, optional
        Range of colorbar. Default is (0.2, 0.98).
    colorbar_range['colorbar_type'] : str, optional
        Type of colorbar scaling ('percent', 'value', 'adaptive').
    figsize : tuple, optional
        Size of the figure. Default is (6, 4).
    scalebar_setting : dict, optional
        Dictionary of scalebar parameters.
    filename : str, optional
        Filename to save the image.
    printing : object, optional
        Printing object for saving the figure.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on.
    fig : matplotlib.figure.Figure, optional
        Figure object.
    title : str, optional
        Title of the plot.
    units : str, optional
        Units for the colorbar (default: 'm').
    zero_mean : bool, optional
        Whether to subtract the mean from the image.
    show_plot : bool, optional
        Whether to show the plot.
    save_plot : bool, optional
        Whether to save the plot.
    debug : bool, optional
        If true, provides debug information for histogram peaks adjustment.

    Returns:
    --------
    fig, ax : Figure and Axes objects.
    """

    if zero_mean:
        if colorbar_setting['colorbar_type'] == 'percent':
            # Find histogram peaks and adjust image
            peaks, counts = find_histogram_peaks(img, num_peaks=1, distance=1, debug=debug)
            height_norm = img - peaks[0]
            peaks, counts = find_histogram_peaks(height_norm, num_peaks=1, distance=1, debug=debug)
            img = height_norm - peaks[0]
        else:
            img -= np.mean(img)

    if ax is None or fig is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    im = ax.imshow(img)

    if not isinstance(scan_size, dict):
        scan_size = convert_scan_setting(scan_size)
    scalebar(ax, image_size=scan_size['image_size'], scale_size=scan_size['scale_size'], units=scan_size['units'], loc='br')

    if colorbar_setting['colorbar_type'] == 'percent':
        vmin, vmax = np.percentile(img, colorbar_setting['colorbar_range'])
        im.set_clim(vmin, vmax)
    elif colorbar_setting['colorbar_type'] == 'value':
        im.set_clim(colorbar_setting['colorbar_range'])
    elif colorbar_setting['colorbar_type'] == 'adaptive':
        im.set_clim(np.min(img), np.max(img))
    else:
        if isinstance(colorbar_setting['colorbar_range'], (tuple, list)):
            im.set_clim(colorbar_setting['colorbar_range'])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Apply the dynamic formatter to the colorbar
    formatter = plt.FuncFormatter(format_func)
    fig.colorbar(im, cax=cax, format=formatter)
    
    ax.tick_params(which='both', bottom=False, left=False, right=False, top=False, labelbottom=False)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    
    if title:
        ax.set_title(title)
    elif sample_name:
        ax.set_title(f'Scan Result of {sample_name}')

    if save_plot and printing is not None and filename is not None:
        printing.savefig(fig, filename)

    if ax is None:
        plt.tight_layout()
        plt.show()



def show_peaks(x, z, peaks=None, valleys=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=z, mode='lines+markers', name='Original Plot'))
    if isinstance(peaks, np.ndarray):
        marker=dict(size=8, color='red', symbol='cross')
        fig.add_trace(go.Scatter(x=x[peaks], y=z[peaks], mode='markers', marker=marker, name='Detected Peaks'))
    if isinstance(valleys, np.ndarray):
        marker=dict(size=8, color='black', symbol='cross')
        fig.add_trace(go.Scatter(x=x[valleys], y=z[valleys], mode='markers', marker=marker, name='Detected Valleys'))
    fig.show()


def df_scatter(df1, df2, xaxis, yaxis, label_with, style, start_0 = (False, False), logscale=False):
    # df1 = df1
    # df2 = df2
    if style == 'simple' and df2 == None:
        plt.scatter(df1[xaxis], df1[yaxis])
        for i, name in enumerate(df1[label_with]):
            plt.text(df1[xaxis].iloc[i], df1[yaxis].iloc[i], name)

    if style == 'detail' and not isinstance(df2, type(None)):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        sns.scatterplot(data=df1, x=xaxis, y=yaxis, hue=label_with, legend=False, alpha=0.3)
        sns.scatterplot(data=df2, x=xaxis, y=yaxis, hue=label_with, legend=False)

        # sns.move_legend(ax, loc="upper right", ncol=2, frameon=True)
        if start_0[0]:
            plt.xlim(left=0)
        if start_0[1]:
            plt.ylim(bottom=0)

        for i, name in enumerate(df2[label_with]):
            plt.text(df2[xaxis].iloc[i], df2[yaxis].iloc[i], name, fontsize=8)

    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    if logscale:
        plt.yscale('log')
        if start_0[1]:
           plt.ylim(bottom=1)

    plt.show()



def violinplot_roughness(RMS_dict):
    # Convert dictionary to long-form d ataframe
    data_df = pd.DataFrame([(k, v) for k, vs in RMS_dict.items() for v in vs], columns=['Label', 'Value'])
    
    mean_list, std_list, max_list, length_list = [], [], [], []
    for i, (label, data) in enumerate(RMS_dict.items()):
        length_list.append(len(data))
        max_list.append(np.max(data))
        mean_list.append(np.mean(data))
        std_list.append(np.std(data))

    figure_width = np.min([18, len(RMS_dict.keys())*1.5])
    plt.figure(figsize=(figure_width, 4))
    ax = sns.violinplot(x='Label', y='Value', data=data_df, palette='viridis', scale='count')

    for i, (length, max, mean, std) in enumerate(zip(length_list, max_list, mean_list, std_list)):
        plt.text(i, max*1.1, f"n={length}\n {convert_with_unit(mean)}\n+-{convert_with_unit(std)}", ha='center')

    plt.title('Roughness')
    plt.show()