import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotly.graph_objects as go
import seaborn as sns
import pandas as pd
from scipy.stats import median_abs_deviation

from afm_learn.afm_utils import convert_with_unit, convert_scan_setting, format_func, define_percentage_threshold
from m3util.viz.layout import scalebar, layout_fig
from afm_learn.domain_analysis import find_histogram_peaks

class tip_potisition_analyzer:
    def __init__(self,):
        pass
        
    def show_tune(self, freq, amps, colors, positions):
        n = len(amps)
        
        fig, ax = layout_fig(1, 1, figsize=(n*3*0.9-0.4*(n-1), 2))
        for amp, color in zip(amps, colors):
            ax.plot(freq, amp, color=color)
        plt.xlabel('Frequency (kHz)')
        plt.ylabel('Amplitude (mV)')
        plt.xlim(30, 720)
        plt.legend(positions, loc='upper left')
        plt.show()
        
        fig, axes = layout_fig(2, 2, figsize=(n*3-0.3*(n-2), 2), subplot_style='gridspec', spacing=(0.2, 0.1))
        for amp, color in zip(amps, colors):
            axes[0].plot(freq, amp, color=color)
        axes[0].set_xlabel('Frequency (kHz)')
        axes[0].set_ylabel('Amplitude (mV)')
        axes[0].set_xlim(340, 390)
        axes[0].legend(positions, loc='upper left')
        
        for amp, color in zip(amps, colors):
            axes[1].plot(freq, amp, color=color)
        axes[1].set_xlabel('Frequency (kHz)')
        axes[1].set_ylabel('Amplitude (mV)')
        axes[1].set_xlim(630, 670)
        axes[1].legend(positions, loc='upper left')
        plt.show()

    def show_tip_scans(self, tip_imgs, positions, phase_imgs, amp_imgs, scan_size):
        n = len(tip_imgs)

        fig, axes = layout_fig(n*3, n, figsize=(n*3, 9), subplot_style='gridspec', spacing=(0.1, 0.1))
        for ax, img, position in zip(axes[:n], tip_imgs, positions):
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(position)
            
        afm_viz = AFMVisualizer(colorbar_setting={'colorbar_type': 'percent', 'colorbar_range': (0.2, 99.8), 'outliers_std': 5, 'symmetric_clim':False, 'visible': True}, zero_mean=False, scalebar=True, debug=True)
        
        for i, ax in enumerate(axes[n:n*2]):
            afm_viz.viz(img=phase_imgs[i], scan_size=scan_size, fig=fig, ax=ax, title=f'phase', cbar_unit='deg')
        
        for i, ax in enumerate(axes[n*2:n*3]):
            afm_viz.viz(img=amp_imgs[i], scan_size=scan_size, fig=fig, ax=ax, title=f'amplitudes', cbar_unit='pm')



def show_pfm_images(imgs, labels, cmap='viridis', clim_threshold=(2, 98), fig_name=None):

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, ax in enumerate(axes.flatten()):
        clim = define_percentage_threshold(imgs[:,:,i], percentage=clim_threshold)
        im = ax.imshow(imgs[:,:,i], cmap=cmap, vmin=clim[0], vmax=clim[1])
        ax.set_title(labels[i])
        plt.colorbar(im, ax=ax)

    if fig_name is not None:
        plt.savefig(fig_name, dpi=300)
    plt.show()


class AFMVisualizer:
    def __init__(self, colorbar_setting={'colorbar_type': 'percent', 'colorbar_range': (0.2, 0.98), 'outliers_std': 3, 'symmetric_clim':True, 'visible': True}, zero_mean=False, scalebar=True, debug=False):
        
        self.colorbar_setting = colorbar_setting
        self.zero_mean = zero_mean
        self.scalebar = scalebar
        self.debug = debug
            
    def add_colorbar(self, img, im, fig, ax, unit="nm"):
        
        # Handle outliers: Clip to mean ± n_std * std
        if self.colorbar_setting.get('outliers_std', None):  # Ensure key exists
            img = img.copy()
            
            # show_image_stats(img, n_std_list=[1,2,3], bins=100)
            # mean_val = np.mean(img)
            # std_val = np.std(img)
            # lower_bound = mean_val - self.colorbar_setting['outliers_std'] * std_val
            # upper_bound = mean_val + self.colorbar_setting['outliers_std'] * std_val
            # img = np.clip(img, lower_bound, upper_bound)
            # img -= np.mean(img) # Recenter the image around 0 for balanced clim
                
            median_val = np.median(img)
            mad_val = median_abs_deviation(img)
            lower_bound = median_val - self.colorbar_setting['outliers_std'] * mad_val
            upper_bound = median_val + self.colorbar_setting['outliers_std'] * mad_val
            img = np.clip(img, lower_bound, upper_bound)
            # img -= np.median(img)
            
            # show_image_stats(img, n_std_list=[1,2,3], bins=100)
                    
        # Set color limits based on colorbar type
        if self.colorbar_setting['colorbar_type'] == None:
            raise ValueError("Colorbar type must be specified.")
        
        elif self.colorbar_setting['colorbar_type'] == 'percent':
            vmin, vmax = np.percentile(img, self.colorbar_setting['colorbar_range'])
            
            if self.colorbar_setting['symmetric_clim']:
                absmax = np.max(np.abs([vmin, vmax]))
                vmin, vmax = -absmax, absmax
            im.set_clim(vmin, vmax)
            
        elif self.colorbar_setting['colorbar_type'] == 'value':
            im.set_clim(self.colorbar_setting['colorbar_range'])
            
        elif self.colorbar_setting['colorbar_type'] == 'adaptive':
            n_std = self.colorbar_setting['colorbar_range']
            vmin, vmax = np.mean(img) - n_std[0]*np.std(img), np.mean(img) + n_std[1]*np.std(img)
            im.set_clim(vmin, vmax)
            
        elif self.colorbar_setting['colorbar_type'] == 'minmax':
            im.set_clim(np.min(img), np.max(img))
            
        else:
            if isinstance(self.colorbar_setting['colorbar_range'], (tuple, list)):
                im.set_clim(self.colorbar_setting['colorbar_range'])
                
        from functools import partial
        
        if self.colorbar_setting['visible']:
            # Add colorbar to the figure
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            formatter = plt.FuncFormatter(lambda value, tick_number: format_func(value, unit=unit))

            # formatter = plt.FuncFormatter(partial(format_func, unit=unit))
            colorbar = fig.colorbar(im, cax=cax, format=formatter)
            
            # Adjust tick padding
            colorbar.ax.yaxis.set_tick_params(pad=1, labelsize=7, direction='in', length=2)  # Adjust spacing between ticks and labels
            # colorbar.ax.yaxis.set_tick_params(pad=-2.2, labelsize=7, right=False)  # Adjust spacing between ticks and labels
            # colorbar.ax.tick_params(direction='in', width=1)

            # Set unit label at the top of the colorbar
            cax.set_title(unit, loc='center', pad=1, fontsize=7)  # Adjust fontsize as needed

            # colorbar.ax.tick_params(direction='in')  # Set tick direction to 'in'

    def adjust_ticks(self, ax):
        # Remove axis ticks and labels
        ax.tick_params(which='both', bottom=False, left=False, right=False, top=False, labelbottom=False)
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])

    def preprocess_image(self, img):
        # Adjust image based on zero_mean and colorbar settings
        if self.zero_mean:
            if self.colorbar_setting['colorbar_type'] == 'percent':
                peaks, counts = find_histogram_peaks(img, num_peaks=1, distance=1, debug=self.debug)
                height_norm = img - peaks[0]
                peaks, counts = find_histogram_peaks(height_norm, num_peaks=1, distance=1, debug=self.debug)
                img = height_norm - peaks[0]
            else:
                img -= np.mean(img)
        return img

    def viz(self, img, scan_size, fig=None, ax=None, figsize=(6, 4), title=None, cbar_unit="nm"):
        
        # Preprocess image if needed
        img = self.preprocess_image(img)

        # Set up figure and axis if not provided
        if ax is None or fig is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        im = ax.imshow(img)

        # Convert scan size if necessary and add scalebar
        if self.scalebar:
            if not isinstance(scan_size, dict):
                scan_size = convert_scan_setting(scan_size)
            scalebar(ax, image_size=scan_size['image_size'], scale_size=scan_size['scale_size'], 
                     units=scan_size['units'], loc='br', text_fontsize=9)
    
        # Add colorbar
        if self.colorbar_setting['colorbar_type'] != None:
            self.add_colorbar(img, im, fig, ax, unit=cbar_unit)

        # Adjust ticks and labels
        self.adjust_ticks(ax)

        # Set title
        if title:
            ax.set_title(title)

        if fig is None and ax is None:
            plt.tight_layout()
            plt.show()
        
        return fig, ax



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