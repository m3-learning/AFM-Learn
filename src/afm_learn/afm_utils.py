import re
import numpy as np
from igor import binarywave, igorpy
from matplotlib import pyplot as plt

def load_waves(file_ibw):
    obj = binarywave.load(file_ibw)
    labels = obj['wave']['labels']
    data = obj['wave']['wData']
    return data, labels

    
def parse_ibw(file, mode=None):

    obj = binarywave.load(file)
    imgs = obj['wave']['wData']

    # swap h and w to match the software
    imgs = np.swapaxes(imgs, 0, 1)
    imgs = np.flip(imgs, 0)

    sample_name = obj['wave']['wave_header']['bname'].decode()

    # decode the notes
    notes_byte = obj['wave']['note']
    notes = notes_byte.decode('latin-1')

    pattern = r'([\w\s]+):\s*([^\r]+)'
    notes_dict = {}
    counter = {}
    for key, value in re.findall(pattern, notes):
        key = re.sub(r'\r', '', key)
        if key in counter:
            counter[key] += 1
            key = f"{key}_{counter[key]}"
        else:
            counter[key] = 0
        notes_dict[key] = value
    
    # print(mode, notes_dict['ImagingMode'])
    if mode == None:
        mode = notes_dict['ImagingMode']

    # decode the current image labels
    labels_byte = obj['wave']['labels'][-2][1:]
    labels_current = [l.decode() for l in labels_byte]
    for i, l in enumerate(labels_current):
        index = l.find("Retrace")
        new_string = l[:index]
        labels_current[i] = new_string

    # print(mode, notes_dict['ImagingMode'], labels_current)

    # correct order of labels
    if mode == 'PFM Mode':
        if len(labels_current) == 6: # dual frequency
            labels_correct = [notes_dict['Channel1DataType_1'], notes_dict['Channel2DataType'], notes_dict['Channel3DataType'], 
                    notes_dict['Channel4DataType'], notes_dict['Channel5DataType'], notes_dict['Channel6DataType']]
            if labels_correct == ['LatAmplitude', 'LatPhase', 'Height', 'Amplitude', 'Phase', 'ZSensor']:
                labels_correct = ['Height', 'LatAmplitude', 'LatPhase', 'ZSensor', 'Amplitude', 'Phase']
        elif len(labels_current) == 4: # single frequency
            # labels_correct = [notes_dict['Channel1DataType'], notes_dict['Channel2DataType'], 
            #                   notes_dict['Channel3DataType'], notes_dict['Channel4DataType']]
            labels_correct = ['Height', 'Deflection', 'Amplitude', 'Phase']
        
    elif mode == 'AC Mode':
        labels_correct = ['Height', 'Amplitude', 'Phase', 'ZSensor']

    # print(labels_correct, labels_current)
    # match images order:
    new_indices = [labels_current.index(lc) for lc in labels_correct]
    imgs = imgs[:, :, new_indices]
    imgs = imgs.astype(np.float32)

    # read ScanSize
    scan_size = np.float32(notes_dict['ScanSize'])

    return imgs, sample_name, labels_correct, scan_size


def convert_with_unit(value):
    units = {
            'pm': 1e-12,
            'nm': 1e-9,
            'um': 1e-6,
            'mm': 1e-3,
            }

    for unit, factor in units.items():
        if abs(value) < 1e3 * factor:
            return f"{value / factor:.2f} {unit}"

    return f"{value:.2e} m"  # fallback to meters if no appropriate unit found


def convert_scan_setting(scan_size):
    scan_size = flexible_round(scan_size)

    scale_ranges = [
        (2e-5, 5e-5, {'scale_size': 5, 'units': 'µm'}), # 20-30 um
        (1e-5, 2e-5, {'scale_size': 2, 'units': 'µm'}), # 10-20 um
        (3e-6, 1e-5, {'scale_size': 1, 'units': 'µm'}), # 3-10 um
        (2e-6, 3e-6, {'scale_size': 500, 'units': 'nm'}), # 2-3 um
        (1e-6, 2e-6, {'scale_size': 200, 'units': 'nm'}), # 1-2 um
        (5e-7, 1e-6, {'scale_size': 100, 'units': 'nm'}), # 0.1-1 um
        (2e-7, 5e-7, {'scale_size': 50, 'units': 'nm'}), # 200-500 nm
        (1e-7, 2e-7, {'scale_size': 20, 'units': 'nm'}), # 100-200 nm
        (5e-8, 1e-7, {'scale_size': 10, 'units': 'nm'}), # 50-100 nm
        (3e-8, 5e-8, {'scale_size': 5, 'units': 'nm'}), # 30-50 nm
        (2e-8, 3e-8, {'scale_size': 3, 'units': 'nm'}), # 20-30 nm
        (1e-8, 2e-8, {'scale_size': 2, 'units': 'nm'}), # 10-20 nm
        (5e-9, 1e-8, {'scale_size': 1, 'units': 'nm'}), # 5-10 nm
        (3e-9, 5e-9, {'scale_size': 500, 'units': 'pm'}), # 3-5 nm
        (2e-9, 3e-9, {'scale_size': 300, 'units': 'pm'}), # 2-3 nm
        (1e-9, 2e-9, {'scale_size': 200, 'units': 'pm'}), # 1-2 nm
        (5e-10, 1e-9, {'scale_size': 100, 'units': 'pm'}), # 0.1-1 nm
        (3e-10, 5e-10, {'scale_size': 50, 'units': 'pm'}), # 300-500 pm
        (2e-10, 3e-10, {'scale_size': 30, 'units': 'pm'}), # 200-300 pm
        (1e-10, 2e-10, {'scale_size': 20, 'units': 'pm'}), # 100-200 pm
        (1e-11, 1e-10, {'scale_size': 10, 'units': 'pm'}), # 50-100 pm
    ]
    magnitude_dict = {
                        'pm': 1e-12,
                        'nm': 1e-9,
                        'µm': 1e-6,
                        'mm': 1e-3,
                        }

    for min_val, max_val, scale_params in scale_ranges:
        # print(min_val, max_val, scale_params, scan_size)
        if min_val <= abs(scan_size) <= max_val:
            scale_params['image_size'] = scan_size/magnitude_dict[scale_params['units']]
            return scale_params
    raise ValueError(f'Provided scan size {scan_size} is not in dict keys.')


def flexible_round(value, sig_digits=1):
    """
    Rounds a number flexibly based on its magnitude to retain significant digits.
    
    Parameters:
    value : float
        The number to round.
    sig_digits : int, optional
        The number of significant digits to retain (default is 1).
    
    Returns:
    str : The rounded number in scientific notation.
    """
    # Calculate the order of magnitude (e.g., -9 for 1.000001e-9)
    if value == 0:
        return value
    
    magnitude = np.floor(np.log10(abs(value)))
    factor = 10 ** magnitude
    
    # Round the value to the specified number of significant digits
    rounded_value = round(value / factor, sig_digits) * factor
    
    return rounded_value

    
def format_func(value, tick_number=None):
    """
    Format the colorbar ticks to only display numeric values without units.
    Dynamically adjust the scale of values based on magnitude.
    """
    value = flexible_round(value)
    
    # Format based on magnitude without units
    if abs(value) >= 1:
        return f'{value:.1e}'
    elif 1e-3 <= abs(value) < 1:
        scaled_value = value * 1e3
        return f'{scaled_value:.1f}' if scaled_value % 1 != 0 else f'{int(scaled_value)}'
    elif 0.1e-6 <= abs(value) < 0.1e-3:
        scaled_value = value * 1e6
        return f'{scaled_value:.1f}' if scaled_value % 1 != 0 else f'{int(scaled_value)}'
    elif 0.1e-9 <= abs(value) < 1e-6:
        scaled_value = value * 1e9
        return f'{scaled_value:.1f}' if scaled_value % 1 != 0 else f'{int(scaled_value)}'
    elif 0.1e-12 <= abs(value) < 0.1e-9:
        scaled_value = value * 1e12
        return f'{scaled_value:.1f}' if scaled_value % 1 != 0 else f'{int(scaled_value)}'
    elif 0.1e-15 <= abs(value) < 0.1e-12:
        scaled_value = value * 1e15
        return f'{scaled_value:.1f}' if scaled_value % 1 != 0 else f'{int(scaled_value)}'
    else:
        return f'{value:.1f}' if value % 1 != 0 else f'{int(value)}'


def show_image_stats(image, n_std_list=[1,2,3], bins=100):
    """
    Display the histogram of an image with lines showing mean ± n_std*std for multiple values of n_std.
    
    Parameters:
    -----------
    image : numpy array
        The input image (2D or 3D for RGB, automatically flattened for the histogram).
    n_std_list : list of int or float, optional
        List of multipliers for standard deviation to show as vertical lines (default is [1, 2]).
    bins : int, optional
        Number of bins to use for the histogram (default is 50).
    """
    # Flatten the image to get pixel values for the histogram
    if image.ndim == 3:  # Convert RGB to grayscale
        image = np.mean(image, axis=-1)
    pixel_values = image.flatten()
    
    # Calculate mean and std
    mean_val = np.mean(pixel_values)
    std_val = np.std(pixel_values)

    # Plot the histogram
    plt.figure(figsize=(8, 3))
    plt.hist(pixel_values, bins=bins, color='gray', alpha=0.7, edgecolor='black')
    
    # Plot mean and ±n_std lines for each value in n_std_list
    for n_std in n_std_list:
        lower_bound = mean_val - n_std * std_val
        upper_bound = mean_val + n_std * std_val
        plt.axvline(lower_bound, color='blue', linestyle='--', label=f'Mean - {n_std}*Std: {lower_bound:.2e}')
        plt.axvline(upper_bound, color='green', linestyle='--', label=f'Mean + {n_std}*Std: {upper_bound:.2e}')
    
    # Plot the mean line
    plt.axvline(mean_val, color='red', linestyle='-', label=f'Mean: {mean_val:.2e}')
    
    # Labeling
    plt.title("Image Histogram with Mean ± Multiple Std Lines", fontsize=12)
    plt.xlabel("Pixel Intensity", fontsize=10)
    plt.ylabel("Frequency", fontsize=10)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

