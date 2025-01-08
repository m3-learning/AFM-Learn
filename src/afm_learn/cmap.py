import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def define_white_viridis(viz=False):

    # Get the viridis colormap
    viridis = plt.cm.get_cmap('viridis', 256)
    data = np.linspace(0, 1, 100).reshape(10, 10)

    # Define a custom colormap that goes from white to viridis
    colors = [(1, 1, 1), viridis(0.75), viridis(0.5), viridis(0.25), viridis(0)]  # white to full viridis
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', colors, N=256)

    if viz:
        # Plot with the new colormap
        plt.imshow(data, cmap=white_viridis)
        plt.colorbar()
        plt.show()
    
    return white_viridis


class ScientificColourMaps:
    def __init__(self, cmap_dir):
        self.cmap_dir = cmap_dir
        if self.cmap_dir[-1] != '/':
            self.cmap_dir += '/'
        if not os.path.exists(self.cmap_dir):
            raise FileNotFoundError(f"Directory {self.cmap_dir} not found.")

        # Check that all files in the directory are .txt files
        self.cmap_dict = {}
        cmap_folders = glob.glob(self.cmap_dir+'/*')
        for cmap_folder in cmap_folders:
            if os.path.isdir(cmap_folder):
                file = glob.glob(cmap_folder + '/*.txt')
                if file != []:
                    cmap_name = os.path.basename(file[0]).split('.')[0]
                    self.cmap_dict[cmap_name] = file[0]

    def list_colormaps(self):
        print(f'Available colors: {list(self.cmap_dict.keys())}')
    
    def load_cmap(self, cmap_name, demo=False):
        if cmap_name not in self.cmap_dict:
            raise ValueError(f"Colormap {cmap_name} not found.")
        cm_data = np.loadtxt(self.cmap_dict[cmap_name])
        cmap = LinearSegmentedColormap.from_list(cmap_name, cm_data)

        if demo:
            x = np.linspace(0, 100, 100)[None, :]
            plt.figure(figsize=(6, 1))
            plt.imshow(x, aspect='auto', cmap=cmap)
            plt.axis('off')
            plt.show()
        return cmap

if __name__ == '__main__':
    cmap_loader = ScientificColourMaps('../../../../ScientificColourMaps8/')
    cmap_loader.list_colormaps()
    cmap_loader.load_cmap('romaO', demo=True)