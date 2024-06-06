import numpy as np
from spinv import local_chern_marker, make_finite, onsite_disorder
from spinv.example_models import haldane_pythtb
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# GLOBALS

# size of the system
NX = 50
NY = 50
FILENAME = sys.argv[1]
# model masses (use linspace bethween 0 and 3 later)
MASS = float(sys.argv[3])
# disorder strength list (bethween 0 and 10)
W = float(sys.argv[2])
# path for saving figures
OUTPUT_PATH = "./outputs/"
FIG_PATH = "./figures/"

def plot_lcm(matrix, path):

    small_size = 16
    medium_size = 20
    bigger_size = 22

    plt.rc('font', size=medium_size +5)          # controls default text sizes
    plt.rc('axes', titlesize=bigger_size-2+5)     # fontsize of the axes title
    plt.rc('axes', labelsize=medium_size-2+5)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small_size +5)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=small_size +5)    # fontsize of the tick labels
    plt.rc('legend', fontsize=small_size +5)    # legend fontsize
    plt.rc('figure', titlesize=bigger_size +5)  # fontsize of the figure title

    # Define colors
    #colors = [(0, 'green'), (0.33, 'blue'), (0.5, 'white'), (0.67, 'red'), (1, 'yellow')]
    colors = [(0, '#A63919'), (0.33, '#8C4324'), (0.5, 'white'), (0.67, '#292984'), (1, '#010590')]
    # Create colormap
    custom_cmap = LinearSegmentedColormap.from_list('custom_colormap', colors)

    plt.figure()
    plt.gcf().set_facecolor('none')
    plt.title(f"w = {W/2}", color='#51230F')
    #plt.imshow(matrix, cmap='seismic', origin='lower', extent=(0, matrix.shape[1], 0, matrix.shape[0]), vmin=-np.max(np.abs(matrix)), vmax=np.max(np.abs(matrix)))
    plt.imshow(matrix, cmap=custom_cmap, origin='lower', extent=(0, matrix.shape[1], 0, matrix.shape[0]), vmin=-2, vmax=2)
    plt.xlabel('X', color='#51230F')
    plt.ylabel('Y', color='#51230F')
    plt.tick_params(axis='x', colors='#51230F')  # Change x-axis tick color to red
    plt.tick_params(axis='y', colors='#51230F') # Change y-axis tick color to blue

    cbar = plt.colorbar(label='Chern Marker')
    numticks = 4
    cbar.locator = plt.MaxNLocator(numticks)
    cbar.update_ticks()
    # plt.show()
    plt.savefig(path + f'w{(W)}'+ f'M{MASS}' + '.png', bbox_inches='tight', format = 'png', dpi=1200, transparent=True)

    return 0

if __name__ == "__main__":
    
    lcm_data = c = np.load("./inputs/" + FILENAME)

    if os.path.exists(FIG_PATH):
        plot_lcm(lcm_data, FIG_PATH)
    else: 
        print("Could not find the path for figures :(")
