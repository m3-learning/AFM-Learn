import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import plotly.express as px

import sys
sys.path.append('../../../py-utils/src/')
from viz import show_images

def domain_rule(images_binary_dict, viz=False):
    shape = images_binary_dict['LatAmplitude'].shape
    lat_c_domain, lat_a_domain_c_tilt, vert_a_domain, vert_c_domain_a_tilt = np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            if images_binary_dict['LatAmplitude'][i,j] == 1:
                if images_binary_dict['LatPhase'][i,j] == 1:
                    lat_c_domain[i,j] = 1
                elif images_binary_dict['LatPhase'][i,j] == 0:
                    lat_c_domain[i,j] = -1

            if images_binary_dict['LatAmplitude'][i,j] == 0:
                if images_binary_dict['LatPhase'][i,j] == 1:
                    lat_a_domain_c_tilt[i,j] = 1
                elif images_binary_dict['LatPhase'][i,j] == 0:
                    lat_a_domain_c_tilt[i,j] = -1

            if images_binary_dict['Amplitude'][i,j] == 1:
                if images_binary_dict['Phase'][i,j] == 1:
                    vert_a_domain[i,j] = 1
                elif images_binary_dict['Phase'][i,j] == 0:
                    vert_a_domain[i,j] = -1

            if images_binary_dict['Amplitude'][i,j] == 0:
                if images_binary_dict['Phase'][i,j] == 1:
                    vert_c_domain_a_tilt[i,j] = 1
                elif images_binary_dict['Phase'][i,j] == 0:
                    vert_c_domain_a_tilt[i,j] = -1

    domains = [lat_c_domain, vert_c_domain_a_tilt, vert_a_domain, lat_a_domain_c_tilt]
    labels = ['Lat c Domain', 'Vert c Domain a Tilt?', 'Vert a Domain', 'Lat a Domain c Tilt?']
    clim_labels = [['c-', 0, 'c+'], ['a-', 0, 'a+'], ['a-', 0, 'a+'], ['c-', 0, 'c+']]

    if viz:
        fig, axes = plt.subplots(1, 4, figsize=(16,3))
        for i, (domain, title, cl) in enumerate(zip(domains, labels, clim_labels)):
            ax = axes.flatten()[i]
            im = ax.imshow(domain)
            ax.set_title(title)
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_ticks([-1, 0, 1])  # Set the positions of the ticks
            cbar.set_ticklabels(cl)  # Set the labels for the ticks
        # plt.show()
    return lat_c_domain, lat_a_domain_c_tilt, vert_a_domain, vert_c_domain_a_tilt


def convert_binary(image, real_value=False, debug=False):
    '''
    convert image to binary color blocks based on the color threshold.
    '''
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    uint8_image = normalized_image.astype(np.uint8)
    _, binary_image = cv2.threshold(uint8_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    output_image = (binary_image == 255).astype(np.float32)

    if real_value:
        c1 = image[output_image == 0]
        c2 = image[output_image == 1]
        output_image[output_image == 0] = c1.mean()
        output_image[output_image == 1] = c2.mean()

        if debug:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            vmin, vmax = define_percentage_threshold(image, (2, 98))
            im0 = axes[0].imshow(image, cmap='viridis', vmin=vmin, vmax=vmax)
            im0.set_clim(vmin, vmax)
            fig.colorbar(im0, ax=axes[0])
            axes[0].set_title('Original image')
            im1 = axes[1].imshow(output_image, cmap='viridis', vmin=vmin, vmax=vmax)
            im1.set_clim(vmin, vmax)
            fig.colorbar(im1, ax=axes[1])
            axes[1].set_title('Processed mask')
            plt.tight_layout()
            plt.show()
    else:
        if debug:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            im0 = axes[0].imshow(image, cmap='viridis')
            fig.colorbar(im0, ax=axes[0])
            axes[0].set_title('Original image')
            im1 = axes[1].imshow(output_image, cmap='viridis')
            fig.colorbar(im1, ax=axes[1])
            axes[1].set_title('Processed mask')
            plt.tight_layout()
            plt.show()

    return output_image

def show_interactive_image(image, cmap='Viridis', clim_threshold=(2, 98), **kwargs):
    # Plot the figure
    # clim = (np.min(image), np.max(image))
    clim = define_percentage_threshold(image, percentage=clim_threshold)
    fig = px.imshow(image, color_continuous_scale=cmap, range_color=clim, width=600, height=500)
    # Adjust layout to be tight
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='white', plot_bgcolor='white')
    fig.show()

def define_percentage_threshold(image, percentage=(2, 98)):
    low, high = np.percentile(image, percentage[0]), np.percentile(image, percentage[1])
    return low, high

def crop_image(image, cropx=None, cropy=None):
    if cropx is not None:
        image = image[cropy[0]:cropy[1]]
    if cropy is not None:
        image = image[:,cropx[0]:cropx[1]]
    return image

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

def shift_phase(phase_imgs, LatPhase_imgs, voltage_labels, n_viz_plots=20, viz=True):

    # find the peaks in the histogram of the phase images
    peaks_intensities, counts = find_histogram_peaks(phase_imgs, num_peaks=2)
    if len(peaks_intensities) < 2:
        raise ValueError('Could not find two peaks in the histogram of the lateral phase images')
    
    # calculate the number of frames to show
    every_frames = len(phase_imgs) // n_viz_plots + 1
    
    phase_imgs_ = shift_peak_batch(phase_imgs, peaks_intensities, shift_method='soft', debug=False)
    if viz:
        show_images(phase_imgs[::every_frames], img_per_row=10, img_height=0.6, labels=voltage_labels[::every_frames], 
                    show_colorbar=True, title='Vertical Phase (before)', hist_bins=100)
        show_images(phase_imgs_[::every_frames], img_per_row=10, img_height=0.6, labels=voltage_labels[::every_frames], 
                    show_colorbar=True, title='Vertical Phase (after)', hist_bins=100)

    # find the peaks in the histogram of the lateral phase images
    peaks_intensities, counts = find_histogram_peaks(LatPhase_imgs, num_peaks=2)
    if len(peaks_intensities) < 2:
        raise ValueError('Could not find two peaks in the histogram of the lateral phase images')
    
    LatPhase_imgs_ = shift_peak_batch(LatPhase_imgs, peaks_intensities, shift_method='hard', debug=False)
    if viz:
        show_images(LatPhase_imgs[::every_frames], img_per_row=10, img_height=0.6, labels=voltage_labels[::every_frames], 
                    show_colorbar=True, title='Lateral Phase (before)', hist_bins=100)
        show_images(LatPhase_imgs_[::every_frames], img_per_row=10, img_height=0.6, labels=voltage_labels[::every_frames], 
                    show_colorbar=True, title='Lateral Phase (after)', hist_bins=100)
    return phase_imgs_, LatPhase_imgs_


def shift_peak_batch(images, peaks_intensities, shift_method='soft', debug=False):
    '''
    Shift the images to align the peaks to the reference peaks found from all images, assuming the images should have two peaks

    Parameters:
    images: np.array, 3D array of images
    peaks_intensities: list, reference peaks intensities
    shift_method: str, 'soft' or 'hard', 'soft' method shifts the image peak based on approximity to reference peak, 
                'hard' method shifts the image peak based on the values: higher image peak will be shifted to higher reference peak, lower image peak will be shifted to lower reference peak,
    Returns:
    images: np.array, 3D array of shifted images
    '''
    # shift to origin 0
    peak_0 = np.min(peaks_intensities) # find the peak closest to 0
    images = images - peak_0
    peaks_intensities = [p-peak_0 for p in peaks_intensities]
    peaks_intensities = np.array(peaks_intensities)

    # process images to be 0 and 180 degrees
    for i in range(len(images)):
        images[i] = shift_peak(images[i], peaks_intensities, shift_method, debug=False)
    return images

def shift_peak(image, peaks_intensities, shift_method='soft', debug=False):

    if isinstance(peaks_intensities, list):
        peaks_intensities = np.array(peaks_intensities)

    peaks, _ = find_histogram_peaks(image, num_peaks=2, distance=90, debug=debug)

    if len(peaks) == 1:
        # print(i, peaks)
        peaks = [peaks[0], peaks[0]]
    elif len(peaks) < 2:

        peaks = [0, 0]
        
    img_peak = peaks[0]

    if debug:
        print('reference peaks:', peaks_intensities, 'image peaks:', peaks)

    if shift_method == 'hard':
        if peaks[0] > peaks[1]:
            peak_ref = np.max(peaks_intensities)
            diff = img_peak - peak_ref
            image = image - diff
        else:
            peak_ref = np.min(peaks_intensities)
            diff = img_peak - peak_ref
            image = image - diff

    elif shift_method == 'soft':
        peak_diff = np.abs(peaks_intensities - img_peak) # find the peak closest to the image peak
        # print(np.argmin(peak_diff), peaks_intensities)
        peak_ref = peaks_intensities[np.argmin(peak_diff)] # find the peak closest to the image peak
        # print(f"img_peak: {img_peak}, Peak_ref: {peak_ref}")
        image = image - (img_peak - peak_ref) # shift the image peak to the reference peak

    if debug:
        peaks, _ = find_histogram_peaks(image, num_peaks=2, distance=60, debug=debug)

        # peaks, _ = find_histogram_peaks(images[i], num_peaks=2, plot_histogram=False)
        # count = 2
        # while np.abs(peaks[0] - peaks[1]) < 90:
        #     count += 1
        #     peaks, _ = find_histogram_peaks(images[i], num_peaks=count, plot_histogram=False)
        #     peaks = [peaks[0], peaks[-1]]
        print('shifted peaks:', peaks)
    return image
        

def find_histogram_peaks(image, bins=256, num_peaks=2, distance='auto', threshold_factor=1.5, min_prominence=5, debug=False):
    """
    Find the largest 'num_peaks' peaks of the histogram of an image that are above a certain threshold and have a minimum prominence.
    This method is adjusted to ignore broader, less distinct peaks that do not meet the prominence criterion.

    Args:
        image (array): NumPy array of the image.
        bins (int): Number of bins to use for the histogram.
        num_peaks (int): Number of peaks to find.
        plot_histogram (bool): Whether to plot the histogram.
        threshold_factor (float): Factor to multiply with the median to set the frequency threshold.
        min_prominence (float): Minimum prominence required for peaks to be considered.

    Returns:
        peaks_values (list): Pixel intensity values corresponding to the peaks.
        peaks_frequencies (list): Counts of pixels at the peaks.
    """
    # convert nan to 0
    image = np.nan_to_num(image)

    # Calculate histogram
    histogram, bin_edges = np.histogram(image, bins=bins, range=[np.min(image), np.max(image)])
    # if debug:
    #     plt.plot(bin_edges[:-1], histogram)
    #     plt.show()


    median_freq = np.median(histogram)

    # Calculate frequency threshold
    threshold = threshold_factor * median_freq

    # Find peaks with prominence
    if distance == 'auto':
        distance = int((np.max(image) - np.min(image)) / (num_peaks * 5)) 
    peaks_indices, properties = find_peaks(histogram, distance=distance, height=threshold, prominence=min_prominence)

    if len(peaks_indices) > num_peaks:
        # Sort peaks by frequency and select the top 'num_peaks'
        sorted_indices = np.argsort(properties["prominences"])[-num_peaks:]
        top_peaks_indices = peaks_indices[sorted_indices][::-1]
    else:
        top_peaks_indices = peaks_indices

    peaks_values = bin_edges[top_peaks_indices] + np.diff(bin_edges)[0] / 2
    peaks_frequencies = histogram[top_peaks_indices]

    if debug:
        plt.figure(figsize=(10, 4))
        plt.bar(bin_edges[:-1], histogram, width=np.diff(bin_edges), edgecolor='black', align='edge')

        colors = [plt.get_cmap('viridis')(i) for i in np.linspace(0, 1, len(peaks_values))]
        for peak, freq, color in zip(peaks_values, peaks_frequencies, colors):
            plt.axvline(peak, color=color, linestyle='dashed', linewidth=1, label=f'Peak at {peak:.2f} with count {freq}')
        plt.title('Histogram of the Image')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

    return peaks_values.tolist(), peaks_frequencies.tolist()

# example
# # Find peaks, you can specify any number of peaks you want to find
# peaks_intensities, peaks_counts = find_histogram_peaks(phase_imgs, num_peaks=2, plot_histogram=True)
# print(f"The top {len(peaks_intensities)} peaks of the histogram are at intensity values {peaks_intensities} with counts {peaks_counts}.")