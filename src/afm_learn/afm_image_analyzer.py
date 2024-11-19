import numpy as np
import imutils
from matplotlib import (pyplot as plt, animation, colors, ticker, path, patches, patheffects)
import plotly.graph_objects as go
import scipy
from scipy import signal
from afm_learn.afm_utils import parse_ibw

def fft2d(image, viz=True):
    """
    Compute the 2D Fast Fourier Transform (FFT) of an image.
    
    Parameters:
    -----------
    image : 2D numpy array
        The input image (grayscale).
    visualize : bool, optional
        If True, display the magnitude and phase of the FFT (default is True).
    
    Returns:
    --------
    fft_result : 2D complex numpy array
        The FFT result of the image.
    fft_magnitude : 2D numpy array
        The magnitude spectrum of the FFT.
    fft_phase : 2D numpy array
        The phase spectrum of the FFT.
    """
    # Compute the 2D FFT of the image
    fft_result = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft_result)  # Shift zero frequency to center

    # Calculate magnitude and phase spectra
    fft_magnitude = np.abs(fft_shifted)  # Magnitude spectrum
    fft_phase = np.angle(fft_shifted)   # Phase spectrum

    # Visualize if needed
    if viz:
        plt.figure(figsize=(12, 5))

        # Original Image
        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap='viridis')
        plt.title("Original Image")
        plt.axis('off')

        # Magnitude Spectrum
        plt.subplot(1, 3, 2)
        plt.imshow(np.log(1 + fft_magnitude), cmap='viridis')  # Log scale for better visibility
        plt.title("FFT Magnitude Spectrum")
        plt.axis('off')

        # Phase Spectrum
        plt.subplot(1, 3, 3)
        plt.imshow(fft_phase, cmap='viridis')
        plt.title("FFT Phase Spectrum")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return fft_result, fft_magnitude, fft_phase



def polyfit2d(x, y, z, kx=3, ky=3, order=None):
    '''
    Two dimensional polynomial fitting by least squares.
    Fits the functional form f(x,y) = z.

    Notes
    -----
    Resultant fit can be plotted with:
    np.polynomial.polynomial.polygrid2d(x, y, soln.reshape((kx+1, ky+1)))

    Parameters
    ----------
    x, y: array-like, 1d
        x and y coordinates.
    z: np.ndarray, 2d
        Surface to fit.
    kx, ky: int, default is 3
        Polynomial order in x and y, respectively.
    order: int or None, default is None
        If None, all coefficients up to maxiumum kx, ky, ie. up to and including x^kx*y^ky, are considered.
        If int, coefficients up to a maximum of kx+ky <= order are considered.

    Returns
    -------
    Return paramters from np.linalg.lstsq.

    soln: np.ndarray
        Array of polynomial coefficients.
    residuals: np.ndarray
    rank: int
    s: np.ndarray

    '''

    # grid coords
    x, y = np.meshgrid(x, y)

    # coefficient array, up to x^kx, y^ky
    coeffs = np.ones((kx+1, ky+1))

    # solve array
    a = np.zeros((coeffs.size, x.size))

    # for each coefficient produce array x^i, y^j
    for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
        # do not include powers greater than order
        if order is not None and i + j > order:
            arr = np.zeros_like(x)
        else:
            arr = coeffs[i, j] * x**i * y**j
        a[index] = arr.ravel()

    # do leastsq fitting and return leastsq result
    return np.linalg.lstsq(a.T, np.ravel(z), rcond=None)



def fit_background(img, degrees=(3,3), viz=False):
    '''
    img: shape: H, W
    '''
    x, y = np.array(range(img.shape[0])), np.array(range(img.shape[1]))
    out = polyfit2d(x, y, img, kx=degrees[0], ky=degrees[1])
    background = np.polynomial.polynomial.polygrid2d(x, y, out[0].reshape((degrees[0]+1,degrees[1]+1)))

    if viz:
        fig, axes = plt.subplots(1, 3, figsize=(9, 2.5))
        im0 = axes[0].imshow(img)
        cbar = fig.colorbar(im0, ax=axes[0], format=ticker.ScalarFormatter(useMathText=True))
        axes[0].set_title('origninal')
        im1 = axes[1].imshow(background)
        cbar = fig.colorbar(im1, ax=axes[1], format=ticker.ScalarFormatter(useMathText=True))
        axes[1].set_title('background')
        im2 = axes[2].imshow(img-background)
        cbar = fig.colorbar(im2, ax=axes[2], format=ticker.ScalarFormatter(useMathText=True))
        axes[2].set_title('out')
        plt.suptitle('fit background:')
        plt.tight_layout()
        plt.show()
    return img-background, background

def remove_surface_particles(img, threshold=3, viz=False):
    mean, std = np.mean(img), np.std(img)
    out = np.copy(img)
    out[out<mean-threshold*std] = mean
    out[out>mean+threshold*std] = mean

    if viz:
        fig, axes = plt.subplots(1, 3, figsize=(9, 2.5))
        im0 = axes[0].imshow(img)
        cbar = fig.colorbar(im0, ax=axes[0], format=ticker.ScalarFormatter(useMathText=True))
        axes[0].set_title('origninal')
        im1 = axes[1].imshow(img-out)
        cbar = fig.colorbar(im1, ax=axes[1], format=ticker.ScalarFormatter(useMathText=True))
        axes[1].set_title('particles')
        im2 = axes[2].imshow(out)
        cbar = fig.colorbar(im2, ax=axes[2], format=ticker.ScalarFormatter(useMathText=True))
        axes[2].set_title('out')
        plt.suptitle('remove surface particles:')
        plt.tight_layout()
        plt.show()
    return out


def afm_RMS_roughness(height):
    if np.min(height) < 0:
       height = height + np.abs(np.min(height))

    avg = np.mean(height)
    n = height.shape[0]*height.shape[1]
    # RMS = [1/n * (x_1-x_avg)**2 + (x_3-x_avg)**2 + ... + (x_n-x_avg)**2] ** 1/2
    return (np.sum((height-avg)**2)/n)**(1/2)


def map_roughness(files, RMS_dict, threshold=3, bk_degrees=(3,3), viz=False):
    for file in files:
        sample_name = file.split('/')[-1][:-4]
        img, sample_name, labels, scan_size = parse_ibw(file)

        height = img[:,:,0]
        # remove surface particles
        if not isinstance(threshold, type(None)):
            height = remove_surface_particles(height, threshold, viz=viz)

        # fit background
        if not isinstance(bk_degrees, type(None)):
            height, bk = fit_background(height, degrees=bk_degrees, viz=viz)

        if sample_name[:6] not in RMS_dict.keys():
            RMS_dict[sample_name[:6]] = []

        # calculate the roughness RMS
        # print(sample_name, afm_RMS_roughness(img[:,:,0]))
        RMS_dict[sample_name[:6]] += [afm_RMS_roughness(height)]

    for k in RMS_dict:
        # remove duplicated records
        seen = set()
        uniq = [x for x in RMS_dict[k] if x not in seen and not seen.add(x)]   
        RMS_dict[k] = uniq

        # # remove outliers
        # if threshold:
        #     data = RMS_dict[k]
        #     mean = np.mean(data)
        #     std = np.std(data)
        #     # outliers = [value for value in data if (value - mean) / std > threshold]
        #     RMS_dict[k] = [value for value in data if (value - mean) / std <= threshold]

    return RMS_dict


def bresenham_line(point1, point2):
    """
    Generate the coordinates of points on a line between two given points 
    using Bresenham's line algorithm.

    Parameters
    ----------
    point1 : tuple of int
        The starting point of the line as (x0, y0).
    point2 : tuple of int
        The ending point of the line as (x1, y1).

    Returns
    -------
    np.ndarray
        An array of points representing the line, where each point is a tuple (x, y).
        The endpoint (x1, y1) is excluded from the returned array.
    
    Notes
    -----
    Bresenham's line algorithm is an efficient way to generate points on a straight line 
    between two given coordinates in a grid-based environment, such as for raster graphics.
    """

    # Unpack the coordinates of the starting and ending points
    x0, y0 = point1[0], point1[1]
    x1, y1 = point2[0], point2[1]

    # List to store the points along the line
    points = []

    # Compute the difference between the starting and ending points
    dx = abs(x1 - x0)  # Absolute difference in the x direction
    dy = abs(y1 - y0)  # Absolute difference in the y direction

    # Initialize the starting point
    x, y = x0, y0

    # Determine the step direction in x and y (positive or negative)
    sx = -1 if x0 > x1 else 1  # Step for x direction
    sy = -1 if y0 > y1 else 1  # Step for y direction

    # Determine whether the line is more horizontal or vertical
    if dx > dy:
        # More horizontal: use dx as the major axis
        err = dx / 2.0  # Initialize the error term

        # Iterate until we reach the end point along the x-axis
        while x != x1:
            points.append((x, y))  # Add the current point to the list
            err -= dy  # Update the error term
            if err < 0:  # If error exceeds threshold
                y += sy  # Move in y direction
                err += dx  # Adjust the error term
            x += sx  # Move in x direction
    else:
        # More vertical: use dy as the major axis
        err = dy / 2.0  # Initialize the error term

        # Iterate until we reach the end point along the y-axis
        while y != y1:
            points.append((x, y))  # Add the current point to the list
            err -= dx  # Update the error term
            if err < 0:  # If error exceeds threshold
                x += sx  # Move in x direction
                err += dy  # Adjust the error term
            y += sy  # Move in y direction

    # Add the final endpoint (x1, y1) to the list
    points.append((x, y))

    # Convert the list of points to a NumPy array, excluding the last point
    return np.array(points[:-1])

class afm_line_profiler():
    def __init__(self, imgs):
        self.imgs = imgs

    def draw_line(self, point1, point2, viz=False):
        line_points = bresenham_line(point1, point2)
        if viz:
            plt.imshow(self.imgs[0])
            plt.plot(line_points[:,0], line_points[:,1], color='red', linewidth=2)
            plt.title('Phase image')
            plt.show()
        return line_points

    def afm_line_profile(self, frame_index, point1, point2):
        line_points = self.draw_line(point1, point2)
        return self.imgs[frame_index][line_points[:,0], line_points[:,1]]

    def dynamic_afm_line_profile(self, point1, point2, x_values, label_str=None, step=1):

        intensities = [self.afm_line_profile(i, point1, point2) for i, img in enumerate(self.imgs[::step])]
        x_values = x_values[::step]

        norm = plt.Normalize(min(x_values), max(x_values))
        cmap = plt.get_cmap('viridis')
        colors = [cmap(norm(value)) for value in x_values]
        labels = [f'{label_str}: {x:.2f}' for x in x_values]
   
        fig, axes = plt.subplots(len(intensities), 1, figsize=(15, 1*len(intensities)))
        for i, value in enumerate(intensities):
            axes[i].plot(value, color=colors[i])
            axes[i].set_axis_off()
            axes[i].text(len(value), value[-1], labels[i])
        plt.show()
        
        return intensities, x_values, labels, colors
    
class afm_analyzer():
    """
    This class is designed to facilitate the analysis of an atomic force microscopy (AFM) substrate image. 
    The class includes methods for image rotation, coordinate transformation, peak detection, and step parameter calculation.
    """ 
    def __init__(self, img, pixels, size):
        '''
        img: the image to be analyzed
        pixels: the number of pixels in the image
        size: the size of the image in meters
        '''
        self.img = img
        self.pixels = pixels
        self.size = size
    
    def rotate_image(self, angle, colorbar_range=None, demo=True):
        '''
        angle: the angle to rotate the image in degrees
        '''
        rad = np.radians(angle)
        scale = 1/(np.abs(np.sin(rad)) + np.abs(np.cos(rad)))
        size_rot = self.size * scale

        img_rot = imutils.rotate(self.img, angle=angle, scale=scale)
        h, w = img_rot.shape[:2]

        if demo:
            plt.figure(figsize=(10, 8))
            im = plt.imshow(img_rot)
            plt.plot([0, w], [h//4, h//4], color='w')
            plt.plot([0, w], [h//2, h//2], color='w')
            plt.plot([0, w], [h*3//4, h*3//4], color='w')
            if colorbar_range:
                im.set_clim(colorbar_range) 
            plt.colorbar()
            plt.show()
        return img_rot, size_rot


    def rotate_xz(self, x, z, xz_angle):
        '''
        x: the x coordinates of the image
        z: the z coordinates of the image
        xz_angle: the angle to rotate the xz plane
        '''
        theta = np.radians(xz_angle)
        x_rot = x * np.cos(theta) - z * np.sin(theta)
        z_rot = x * np.sin(theta) + z * np.cos(theta)
        return x_rot, z_rot

    def show_peaks(self, x, z, peaks=None, valleys=None):
        '''
        x: the x-axis data
        z: the z-axis data - height
        peaks: the indices of the peaks
        valleys: the indices of the valleys
        '''
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=z, mode='lines+markers', name='Original Plot'))
        if isinstance(peaks, np.ndarray):
            marker=dict(size=8, color='red', symbol='cross')
            fig.add_trace(go.Scatter(x=x[peaks], y=z[peaks], mode='markers', marker=marker, name='Detected Peaks'))
        if isinstance(valleys, np.ndarray):
            marker=dict(size=8, color='black', symbol='cross')
            fig.add_trace(go.Scatter(x=x[valleys], y=z[valleys], mode='markers', marker=marker, name='Detected valleys'))
        fig.show()

    def slice_rotate(self, img_rot, size, j, prominence, width, xz_angle=0, demo=False):
        '''
        img_rot: the rotated image
        size: the size of the image in meters
        j: the column to slice
        xz_angle: the angle between the x and z axes in degrees
        '''
        i = np.linspace(0, self.pixels-1, self.pixels)
        x = i / self.pixels * size
        z = img_rot[np.argwhere(img_rot[:, j]!=0).flatten(), j]
        x = x[np.argwhere(img_rot[:, j]!=0).flatten()]
        peak_indices, _ = signal.find_peaks(z, prominence=prominence, width=width)
        valley_indices, _ = signal.find_peaks(-z, prominence=prominence, width=width)

        if xz_angle != 0:
            x_min, x_max, z_min, z_max = np.min(x), np.max(x), np.min(z), np.max(z)
            x_norm = (x - x_min) / (x_max - x_min)
            z_norm = (z - z_min) / (z_max - z_min)

            peak_indices, _ = signal.find_peaks(z_norm, prominence=prominence, width=width)
            valley_indices, _ = signal.find_peaks(-z_norm, prominence=prominence, width=width)

            # rotate the xz plane to level the step
            x_norm_rot, z_norm_rot = self.rotate_xz(x_norm, z_norm, xz_angle)
            x, z = x_norm_rot * (x_max - x_min) + x_min, z_norm_rot * (z_max - z_min) + z_min
        
        if demo:
            self.show_peaks(x, z, peak_indices, valley_indices)
        return x, z, peak_indices, valley_indices


    def calculate_simple(self, x, z, peak_indices, fixed_height=None, demo=False):
        '''
        Calculate the height, width, and miscut of the steps in a straight forward way.
        Calculate the height and width of each step from the rotated line profile.
        x: the x-axis data
        z: the z-axis data - height
        peak_indices: the indices of the peaks
        fixed_height: the height of the steps
        '''

        # find the level of z and step height and width
        step_widths = np.diff(x[peak_indices])
        if fixed_height:
            step_heights = np.full(len(step_widths), fixed_height)
        else:
            step_heights = z[peak_indices[1:]] - z[peak_indices[:-1]]
        miscut = np.degrees(np.arctan(step_heights/step_widths))
        
        if demo:
            for i in range(len(step_heights)):
                print(f"Step {i+1}: Height = {step_heights[i]:.2e}, Width = {step_widths[i]:.2e}, Miscut = {miscut[i]:.3f}°")
            print('Results:')
            print(f"  Average step height = {np.mean(step_heights):.2e}, Standard deviation = {np.std(step_heights):.2e}")
            print(f"  Average step width = {np.mean(step_widths):.2e}, Standard deviation = {np.std(step_widths):.2e}")
            print(f"  Average miscut = {np.mean(miscut):.3f}°, Standard deviation = {np.std(miscut):.3f}°")
        return step_heights, step_widths, miscut

    def calculate_fit(self, x, z, peak_indices, valley_indices, fixed_height, demo=False):
        '''
        calculate the step height, width and miscut angle. 
        The step height is calculated by the perpendicular distance between lower step bottom point (valley) and the fitting function of higher step edge (line between left peak and right peak). 
        x: the x-axis data
        z: the z-axis data - height
        peak_indices: the indices of the peaks
        valley_indices: the indices of the valleys
        fixed_height: the fixed step height
        demo: whether to show the demo plot
        '''
        # print(valley_indices)
        step_widths = []
        for i, v_ind in enumerate(valley_indices):
            x_valley, z_valley = x[v_ind], z[v_ind]

            # ignore if there's no peak on the left
            if x_valley < np.min(x[peak_indices]): continue
            # if there's no peak on the right, then the valley is the last one
            if x_valley > np.max(x[peak_indices]): continue

            # find the nearest peak on the left of the valley v_ind
            peaks_lhs = peak_indices[np.where(x[peak_indices] < x_valley)]
            left_peak_indice = peaks_lhs[np.argmax(peaks_lhs)]
            x_left_peak, z_left_peak = x[left_peak_indice], z[left_peak_indice]

            # find the nearest peak on the right of the valley v_ind
            peaks_rhs = peak_indices[np.where(x[peak_indices] > x_valley)]
            right_peak_indice = peaks_rhs[np.argmin(peaks_rhs)]
            x_right_peak, z_right_peak = x[right_peak_indice], z[right_peak_indice]

            # ignore if can't make a peak, valley, peak sequence
            if i!=0 and i!=len(valley_indices)-1:
                if  x[valley_indices[i-1]] > x_left_peak or x[valley_indices[i+1]] < x_right_peak:
                    continue
            
            # fit the linear function between the right peak and the valley
            m, b = scipy.stats.linregress(x=[x_right_peak, x_valley], y=[z_right_peak, z_valley])[0:2]
            m = (z_right_peak-z_valley)/(x_right_peak-x_valley)
            b = z_valley - m*x_valley

            # calculate the euclidean distance between the left peak and fitted linear function
            step_width = np.abs((m * x_left_peak - z_left_peak + b)) / (np.sqrt(m**2 + 1))
            step_widths.append(step_width)
            
            # print left peak, valley, right peak
            if demo:
                print(f'step {i}: step_width: {step_width:.2e}, left_peak: ({x_left_peak:.2e}, {z_left_peak:.2e}), valley: ({x_valley:.2e}, {z_valley:.2e}), right_peak: ({x_right_peak:.2e}, {z_right_peak:.2e})')
                
        step_heights = np.full(len(step_widths), fixed_height)
        miscut = np.degrees(np.arctan(step_heights/step_widths))
        
        if demo:
            print('Results:')
            print(f"  Average step height = {np.mean(step_heights):.2e}, Standard deviation = {np.std(step_heights):.2e}")
            print(f"  Average step width = {np.mean(step_widths):.2e}, Standard deviation = {np.std(step_widths):.2e}")
            print(f"  Average miscut = {np.mean(miscut):.3f}°, Standard deviation = {np.std(miscut):.3f}°")
        return step_heights, step_widths, miscut

    def clean_data(self, step_heights, step_widths, miscut, std_range=1, demo=False):
        '''
        step_heights: the heights of the steps
        step_widths: the widths of the steps
        miscut: the miscut of the steps
        std_range: the range of standard deviation to remove outliers
        demo: whether to show the cleaned results
        '''
        # remove outliers
        miscut = miscut[np.abs(miscut-np.mean(miscut))<std_range*np.std(miscut)]
        step_heights = step_heights[np.abs(step_heights-np.mean(step_heights))<std_range*np.std(step_heights)]
        step_widths = step_widths[np.abs(step_widths-np.mean(step_widths))<std_range*np.std(step_widths)]
        if demo:
            print('Cleaned results:')
            print(f"  Average step height = {np.mean(step_heights):.2e}, Standard deviation = {np.std(step_heights):.2e}")
            print(f"  Average step width = {np.mean(step_widths):.2e}, Standard deviation = {np.std(step_widths):.2e}")
            print(f"  Average miscut = {np.mean(miscut):.3f}°, Standard deviation = {np.std(miscut):.3f}°")
        return step_heights, step_widths, miscut

    def calculate_substrate_properties(self, image_rot, size_rot, xz_angle, prominence=1e-11, width=2, style='simple', fixed_height=None, std_range=1, demo=False):
        '''
        image_rot: the rotated image
        size_rot: the size of the rotated image in meters
        prominence: the prominence of the peaks
        width: the width of the peaks
        fixed_height: the height of the step, provide if can be acquired from literature
        std_range: the range of standard deviation to remove outliers
        '''
        step_heights_list, step_widths_list, miscut_list = [], [], []
        for j in range(self.pixels//4, self.pixels*3//4, 10):
            x, z, peak_indices, valley_indices = self.slice_rotate(image_rot, size_rot, j, prominence, width, xz_angle=xz_angle, demo=demo)
            
            if style == 'simple':
                step_heights, step_widths, miscut = self.calculate_simple(x, z, peak_indices, fixed_height=fixed_height, demo=demo)
            elif style == 'fit':
                step_heights, step_widths, miscut = self.calculate_fit(x, z, peak_indices, valley_indices, fixed_height=fixed_height, demo=demo)  
            step_heights_list.append(step_heights)
            step_widths_list.append(step_widths)
            miscut_list.append(miscut)
        
        step_heights = np.concatenate(step_heights_list)
        step_widths = np.concatenate(step_widths_list)
        miscut = np.concatenate(miscut_list)
        substrate_properties = {'step_heights': step_heights, 'step_widths': step_widths, 'miscut': miscut}

        print(f"Step height = {np.mean(step_heights):.2e} +- {np.std(step_heights):.2e}")
        print(f"Step width = {np.mean(step_widths):.2e} +- {np.std(step_widths):.2e}")
        print(f"Miscut = {np.mean(miscut):.3f}° +- {np.std(miscut):.3f}°")
        return substrate_properties
    
    
    
