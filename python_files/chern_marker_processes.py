import numpy as np
import matplotlib.pyplot as plt
from spinv import local_chern_marker, make_finite, onsite_disorder
from spinv.example_models import haldane_pythtb
from pythtb import tb_model
from tbmodels import Model
import time
import threading
from matplotlib.colors import LinearSegmentedColormap
from multiprocessing import Pool, cpu_count
from functools import partial
import os


# GLOBALS

# size of the system
NX = 50
NY = 50
# disorder strength list (bethween 0 and 10)
W_LIST = [4]
# model masses (use linspace bethween 0 and 3 later)
MASS_LIST = np.array([0.2, 2])
# path for saving figures
PATH = "./figures/"

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
        print(f"chern marker computed for mass{mass}")
    
    # Timing of the task
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total Elapsed time:", elapsed_time, "seconds", f"for mass {mass}")
    
    # return chern_matrices
    return chern_matrices


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

    # Create colormap
    custom_cmap = LinearSegmentedColormap.from_list('custom_colormap', colors)

    for i, chern_matrix in enumerate(chern_array):
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
        plt.savefig(path + f'mass{i}_w_{int(w)}_{int((w - int(w))*10)}.png', bbox_inches='tight', format = 'png', dpi=800)
    
    return 0

class ReturnableThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, Verbose=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)
    def join(self):
        threading.Thread.join(self)
        return self._return


if __name__ == "__main__":

    print("hi, I'm the processes program, wish me luck :)")

    # execute lcm calculation
    num_processes = cpu_count()
    pool = Pool(processes= num_processes-2)
    # pool = Pool(processes= 2)
    lcm_calc_parametrised=partial(lcm_calc, w_list = W_LIST, Nx = NX, Ny = NY) # put the constant parameters
    chern_dictinoary_array = pool.map(lcm_calc_parametrised, MASS_LIST)
    pool.close()
    pool.join()

    #classification array
    topology_class = np.zeros((len(MASS_LIST), len(W_LIST)))
    # fill in the classification
    for mass_index, mass in enumerate(MASS_LIST):
        for disorder_index, (w, matrix) in enumerate(chern_dictinoary_array[mass_index].items()):
            l = INTEGRATION_WIDTH
            m = len(matrix[0])
            topology_class[mass_index, disorder_index] = sum(sum(matrix[0:l])) + sum(sum(matrix[m-l:m])) + sum(sum(matrix[l:m-l, 0:l])) + sum(sum(matrix[l:m-l, m-l:m]))
        
    # print and plot the results
    print(topology_class)

    if os.path.exists(PATH):
        plot_lcm(chern_dictinoary_array, PATH)
    else: 
        print("Could not find the path for figures :(")