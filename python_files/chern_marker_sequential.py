import numpy as np
import matplotlib.pyplot as plt
from spinv import local_chern_marker, make_finite, onsite_disorder
from spinv.example_models import haldane_pythtb
from pythtb import tb_model
from tbmodels import Model
from matplotlib.colors import LinearSegmentedColormap
import time
import os

# GLOBALS

# size of the system
NX = 50
NY = 50
# disorder strength list (bethween 0 and 10)
W_LIST = [6.0]
# model masses (use linspace bethween 0 and 3 later)
MASS_LIST = np.array([2.0])
# path for saving figures
PATH = "./figures/"
OUTPUT_PATH = "./outputs/"

INTEGRATION_WIDTH = 3

def lcm_calc(mass, w_list, Nx, Ny):
    """
    Constructs a Haldane model for given [mass: float] parameter
    Performs heavy local_chern_marker() calcluation for each of the disorder strength 
    parameters in the [w_list: float list]
    Returns a dictionary of {w: lcm (Nx*Ny array)}
    Times the procedure
    """

    # Timing of the task
    start_time = time.time()

    print(f'started calculations for mass: {mass}')

    # Create Haldane models through PythTB and TBmodels packages
    hmodel_pbc_pythtb = haldane_pythtb(delta = mass, t = -1, t2 = -1/3, phi = np.pi / 2, L=1)
    # Cut the model to make a sample of finite size (defines))
    hmodel_obc_pythtb = make_finite(model=hmodel_pbc_pythtb, nx_sites=Nx, ny_sites=Ny)
    # Initiallise chern marker matrix dictionary
    chern_matrices = {}

    for w in w_list:
        print(f"computing w = {w/2} for mass {mass}")
        # Add Anderson disorder within [-w/2, w/2]. The argument spinstates specifies the spin of the model
        hmodel_pythtb_disorder = onsite_disorder(model=hmodel_obc_pythtb, w=w, spinstates=1, seed=181)
        print("disorder added")

        # Compute the local Chern markers for TBmodels and PythTB
        chern_matrices[w] = local_chern_marker(model=hmodel_pythtb_disorder, nx_sites=Nx, ny_sites=Ny)
        print("chern marker computed")
        #np.save(OUTPUT_PATH + f'w{w}' + f'M{mass}' + '.npy', chern_matrices[w])
    
    # Timing of the task
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total Elapsed time:", elapsed_time, "seconds", f"for mass {mass}")
    
    # return chern_matrices
    return chern_matrices

def construct_classification(mass_list, w_list, Nx=50, Ny=50, l=3):
    """
    Executes lcm calculation for a mass array
    Integrated lcm along the boundaries to classify the system for each mass and disorder
    l is the width of the integration along the boundaries
    Returns a list of calculated lcm dictionaries
    Returns topology classification for each mass and disorder
    """
    # array to store all the lcm
    chern_dictinoary_array = [None] * len(mass_list)

    #classification array
    topology_class = np.zeros((len(mass_list), len(w_list)))

    # chern_matrx_array = [None] * len(mass_list)
    for mass_index, mass in enumerate(mass_list):

        chern_matrix = lcm_calc(mass, w_list, Nx, Ny)
        # store the lcm for future use
        chern_dictinoary_array[mass_index] = chern_matrix

        for disorder_index, (w, matrix) in enumerate(chern_matrix.items()):

            m = len(matrix[0])
            topology_class[mass_index, disorder_index] = sum(sum(matrix[0:l])) + sum(sum(matrix[m-l:m])) + sum(sum(matrix[l:m-l, 0:l])) + sum(sum(matrix[l:m-l, m-l:m]))
    
    return chern_dictinoary_array, topology_class

def plot_lcm(chern_array, path):

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

    for chern_matrix in chern_array:
        for w, matrix in chern_matrix.items():
            plt.figure()
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
        plt.savefig(path + f'w_{int(w)}_{int((w - int(w))*10)}.png', bbox_inches='tight', format = 'png', dpi=800)
    
    return 0

if __name__ == "__main__":

    lcm_data, lcm_classification = construct_classification(MASS_LIST, W_LIST, NX, NY, l=INTEGRATION_WIDTH)
    print(lcm_classification)

    if os.path.exists(PATH):
        plot_lcm(lcm_data, PATH)
    else: 
        print("Could not find the path for figures :(")

    #b = np.load("./outputs/w4_0M2_0.npy")

