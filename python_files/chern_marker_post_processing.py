import numpy as np
from spinv import local_chern_marker, make_finite, onsite_disorder
from spinv.example_models import haldane_pythtb
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


#CALCULATING INTEGRATED LCM AND (OPTIONALLY) PLOTTING THE RESULTS

# GLOBALS
INTEGRATION_WIDTH = 3
PLOT=False
# size of the system
NX = 50
NY = 50
# model masses (use linspace bethween 0 and 3 later)

# MASS_LIST = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0]
MASS_LIST = np.array([1.2  , 1.275, 1.35 , 1.425, 1.5  , 1.575, 1.65 , 1.725, 1.8  ,
       1.875, 1.95 , 2.025, 2.1  , 2.175, 2.25 , 2.325, 2.4  , 2.475,
       2.55 , 2.625])
# disorder strength list (bethween 0 and 10)
# W_LIST = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
W_LIST = np.linspace(0.0, 9.5, 20)
# path for saving figures
INPUT_PATH = "./inputs/"
OUTPUT_PATH = "./outputs/"
FIG_PATH = "./figures/"

def plot_lcm(matrix, path, w, mass):

    small_size = 16
    medium_size = 20
    bigger_size = 22

    plt.rc('font', size=small_size)          # controls default text sizes
    plt.rc('axes', titlesize=bigger_size-2)     # fontsize of the axes title
    plt.rc('axes', labelsize=medium_size-2)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=small_size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=small_size)    # legend fontsize
    plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title

    # Define colors
    colors = [(0, 'green'), (0.33, 'blue'), (0.5, 'white'), (0.67, 'red'), (1, 'yellow')]
    #colors = [(0, '#A63919'), (0.33, '#8C4324'), (0.5, 'white'), (0.67, '#292984'), (1, '#010590')]
    # Create colormap
    custom_cmap = LinearSegmentedColormap.from_list('custom_colormap', colors)

    plt.figure()
    plt.gcf().set_facecolor('none')
    plt.title(f"w = {w/2}")
    #plt.imshow(matrix, cmap='seismic', origin='lower', extent=(0, matrix.shape[1], 0, matrix.shape[0]), vmin=-np.max(np.abs(matrix)), vmax=np.max(np.abs(matrix)))
    plt.imshow(matrix, cmap=custom_cmap, origin='lower', extent=(0, matrix.shape[1], 0, matrix.shape[0]), vmin=-2, vmax=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    cbar = plt.colorbar(label='Chern Marker')
    numticks = 4
    cbar.locator = plt.MaxNLocator(numticks)
    cbar.update_ticks()
    # plt.show()
    plt.savefig(path + f'w{(w)}'+ f'M{mass}' + '.png', bbox_inches='tight', format = 'png', dpi=1200, transparent=True)

    return 0

if __name__ == "__main__":
    
    topology_class = np.zeros((len(MASS_LIST), len(W_LIST)))
    for mass_index, mass in enumerate(MASS_LIST):
        for disorder_index, w in enumerate(W_LIST):

            filename = f'w{w}' + f'M{mass}' + '.npy'
            lcm_data = np.load(INPUT_PATH + filename)

            l = INTEGRATION_WIDTH
            m = len(lcm_data[0])
            topology_class[mass_index, disorder_index] = sum(sum(lcm_data[0:l])) + sum(sum(lcm_data[m-l:m])) + sum(sum(lcm_data[l:m-l, 0:l])) + sum(sum(lcm_data[l:m-l, m-l:m]))

            # if PLOT:
            #     if os.path.exists(FIG_PATH):
            #         plot_lcm(lcm_data, FIG_PATH, w, mass)
            #     else: 
            #         print("Could not find the path for figures :(")

    # plotting the topology classification
    # heatmap
    small_size = 16
    medium_size = 20
    bigger_size = 22

    plt.rc('font', size=small_size)          # controls default text sizes
    plt.rc('axes', titlesize=bigger_size-2)     # fontsize of the axes title
    plt.rc('axes', labelsize=medium_size-2)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=small_size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=small_size)    # legend fontsize
    plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title

    # Define colors
    #colors = [(0, 'green'), (0.33, 'blue'), (0.5, 'white'), (0.67, 'red'), (1, 'yellow')]
    colors = [(0, '#A63919'), (0.33, '#8C4324'), (0.5, 'white'), (0.67, '#292984'), (1, '#010590')]
    # Create colormap
    custom_cmap = LinearSegmentedColormap.from_list('custom_colormap', colors)

    plt.figure()
    plt.gcf().set_facecolor('none')
    #plt.title("Integrated LCM plot")
    #plt.imshow(matrix, cmap='seismic', origin='lower', extent=(0, matrix.shape[1], 0, matrix.shape[0]), vmin=-np.max(np.abs(matrix)), vmax=np.max(np.abs(matrix)))
    #plt.imshow(topology_class, cmap="cool", origin='lower', extent=(0, topology_class.shape[1], 0, topology_class.shape[0]), vmin=0, vmax=2000)
    plt.imshow(topology_class, cmap="cool", origin='lower', vmin=0, vmax=2000)
    # Remove ticks from both x and y axes
    # Set ticks and labels
    # Set ticks and labels with 5 ticks each
    x_indices = np.linspace(0, len(W_LIST) - 1, 5).astype(int)
    y_indices = np.linspace(0, len(MASS_LIST) - 1, 5).astype(int)

    plt.xticks(ticks=x_indices, labels=np.round(W_LIST[x_indices], 2), rotation=45)
    plt.yticks(ticks=y_indices, labels=np.round(MASS_LIST[y_indices], 3))
    plt.xlabel('disorder strength')
    plt.ylabel('mass')
    cbar = plt.colorbar(label='Integrated LCM')
    numticks = 4
    cbar.locator = plt.MaxNLocator(numticks)
    cbar.update_ticks()
    plt.show()
    #plt.savefig(path + f'w{(w)}'+ f'M{mass}' + '.png', bbox_inches='tight', format = 'png', dpi=1200, transparent=True)

    # separate line for each of the masses
    for i, mass in enumerate(MASS_LIST):
        plt.plot(W_LIST, topology_class[i, :], label=f"Mass: {mass}")
        plt.xlabel("disorder strength")
        plt.ylabel("integrated LCM")
    plt.legend(fontsize=9) 
    plt.show()

    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.mplot3d import Axes3D


    # Create a meshgrid
    x, y = np.meshgrid(W_LIST, MASS_LIST)

    # Create a figure and a 3D Axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.tight_layout()
    plt.gcf().set_facecolor('none')
    # Plot the wireframe
    wire = ax.plot_wireframe(x, y, topology_class, color='#882E15', linewidth=0.5)
    # Remove background and grid
    ax.xaxis.pane.fill = False  # Remove background pane for x-axis
    ax.yaxis.pane.fill = False  # Remove background pane for y-axis
    ax.zaxis.pane.fill = False  # Remove background pane for z-axis
    ax.grid(False)              # Turn off the grid

    ax.view_init(36, 56)
    ax.set_axis_off()

    plt.savefig('wire.png', bbox_inches='tight', format = 'png', dpi=1200, transparent=True)



    # Show plot
    plt.show()




