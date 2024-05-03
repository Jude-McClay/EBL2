import numpy as np
import matplotlib.pyplot as plt
from spinv import local_chern_marker, make_finite, onsite_disorder
from spinv.example_models import haldane_pythtb
from pythtb import tb_model
from tbmodels import Model
from multiprocessing import Pool, cpu_count
import time

from matplotlib.colors import LinearSegmentedColormap


# def alt_haldane_pythtb(delta, t, t2, phi, L1,L2,L3,L4):
#     # From http://www.physics.rutgers.edu/pythtb/examples.html#haldane-model
#     lat=[[1.0,0.0],[0.5,np.sqrt(3.0)/2.0]]
#     orb=[[0.0,0.0],[1./3.,1./3.]]
#     model=tb_model(2,2,lat,orb)
#     model.set_onsite([-delta,delta])
#     for lvec in ([ 0, 0], [-1, 0], [ 0,-1]):
#         model.set_hop(t, 0, 1, lvec)
#     for lvec in ([ 1, 0], [-1, 1], [ 0,-1]):
#         model.set_hop(t2*np.exp(1.j*phi), 0, 0, lvec)
#     for lvec in ([-1, 0], [ 1,-1], [ 0, 1]):
#         model.set_hop(t2*np.exp(1.j*phi), 1, 1, lvec)

#     sc_model = model.make_supercell([[L1,L2],[L3,L4]])
#     return sc_model

# define size of the system
Nx = 50
Ny = 50
# List of w values
#w_values = [0, 0.2, 0.4, 0.6, 0.8]  # Add more values as needed
#w_values = [0,2,4,6,8,10]
#w_values = [4]
w_values = [6]


def lcm_calc(mass):
    start_time = time.time()
    print(f'calculations for mass: {mass}')
    # Create Haldane models through PythTB and TBmodels packages
    #hmodel_pbc_pythtb = alt_haldane_pythtb(delta = 0.2, t = -1, t2 = -5, phi = np.pi / 2, L1=1,L2=0,L3=0,L4=1)
    hmodel_pbc_pythtb = haldane_pythtb(delta = mass, t = -1, t2 = -1/3, phi = np.pi / 2, L=1)

    # Cut the model to make a sample of finite size (defines))
    hmodel_obc_pythtb = make_finite(model=hmodel_pbc_pythtb, nx_sites=Nx, ny_sites=Ny)

    # Initiallise chern marker matrix dictionary
    chern_matrices = {}

    for w in w_values:
        print(f"computing w = {w/2} for mass {mass}")
        # Add Anderson disorder within [-w/2, w/2]. The argument spinstates specifies the spin of the model
        hmodel_pythtb_disorder = onsite_disorder(model=hmodel_obc_pythtb, w=w, spinstates=1, seed=181)
        print("disorder added")

        # Compute the local Chern markers for TBmodels and PythTB
        chern_matrices[w] = local_chern_marker(model=hmodel_pythtb_disorder, nx_sites=Nx, ny_sites=Ny)
        print("chern marker computed")
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    # Print the elapsed time
    print("Total Elapsed time:", elapsed_time, "seconds", f"for mass {mass}")
    
    # return chern_matrices
    return chern_matrices


if __name__ == "__main__":
    print("hi")
    num_processes = cpu_count()
    # pool = Pool(processes= num_processes-2)
    pool = Pool(processes= 2)
    
    # model masses for which we want to compute lcm
    # masses = np.linspace(0.2, 2, num_processes-2)
    masses = [0.2, 2]

    chern_matrx_list = pool.map(lcm_calc, masses)
    pool.close()
    pool.join()
 
    #classification array
    topology_class = np.zeros((len(masses), len(w_values)))
    # Loop through the chern_matrices dictionary and plot each matrix
    # define widith of the frame 
    l = 3

    # for plots
    SMALL_SIZE = 16
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE-2)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE-2)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Define colors
    colors = [(0, 'green'), (0.33, 'blue'), (0.5, 'white'), (0.67, 'red'), (1, 'yellow')]

    # Create colormap
    custom_cmap = LinearSegmentedColormap.from_list('custom_colormap', colors)

    for j, chern_matrix in enumerate(chern_matrx_list):
        # init counter
        i = 0
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
            m = len(matrix[0])
            topology_class[j, i] = sum(sum(matrix[0:l])) + sum(sum(matrix[m-l:m])) + sum(sum(matrix[l:m-l, 0:l])) + sum(sum(matrix[l:m-l, m-l:m]))
            print(topology_class[j, i])
            i = i + 1
        plt.show()

# make a plotting routine into a function as well
