import numpy as np
from spinv import local_chern_marker, make_finite, onsite_disorder
from spinv.example_models import haldane_pythtb
import time
import sys
import os

# GLOBALS

# size of the system
NX = 50
NY = 50
# model masses (use linspace bethween 0 and 3 later)
MASS = float(sys.argv[1])
# disorder strength list (bethween 0 and 10)
W = float(sys.argv[2])
# path for saving figures
OUTPUT_PATH = "./outputs/"

def lcm_calc(mass, w, Nx, Ny):
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

    print(f"computing w = {w/2} for mass {mass}")
    # Add Anderson disorder within [-w/2, w/2]. The argument spinstates specifies the spin of the model
    hmodel_pythtb_disorder = onsite_disorder(model=hmodel_obc_pythtb, w=w, spinstates=1, seed=181)
    print("disorder added")

    # Compute the local Chern markers for TBmodels and PythTB
    chern_matrix = local_chern_marker(model=hmodel_pythtb_disorder, nx_sites=Nx, ny_sites=Ny)
    print("chern marker computed")
    
    # Timing of the task
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total Elapsed time:", elapsed_time, "seconds", f"for mass {mass}")
    
    # return chern_matrices
    return chern_matrix

if __name__ == "__main__":

    lcm_matrix = lcm_calc(MASS, W, NX, NY)
    
    # Save the array to a binary file
    if os.path.exists(OUTPUT_PATH):
        #NB: can only safely name up to .0 precision
        #np.save(OUTPUT_PATH + f'w{int(W)}_{int((W - int(W))*10)}' + f'M{int(MASS)}_{int((MASS - int(MASS))*10)}' + '.npy', lcm_matrix)
        np.save(OUTPUT_PATH + f'w{W}' + f'M{MASS}' + '.npy', lcm_matrix)

    else: 
        print("Could not find the ./outputs directory :(")
