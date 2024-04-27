import numpy as np
from spinv import local_chern_marker, make_finite, onsite_disorder
from spinv.example_models import haldane_pythtb

# Create Haldane model through PythTB package
hmodel_pbc_pythtb = haldane_pythtb(delta=0.5, t=-1, t2=0.15, phi=np.pi/2, L=1)

# Cut the model to make a sample of size 10 x 10
Nx = 10
Ny = 10
hmodel_obc_pythtb = make_finite(model=hmodel_pbc_pythtb, nx_sites=Nx, ny_sites=Ny)

# List of w values
w_values = [0, 1, 2, 3, 4]  # Add more values as needed

for w in w_values:
    # Add Anderson disorder within [-w/2, w/2]. The argument spinstates specifies the spin of the model
    hmodel_pythtb_disorder = onsite_disorder(model=hmodel_obc_pythtb, w=w, spinstates=1, seed=184)

    # Compute the local Chern markers for TBmodels and PythTB
    lcm_pythtb = local_chern_marker(model=hmodel_pythtb_disorder, nx_sites=Nx, ny_sites=Ny)

    matrix_array = np.array(lcm_pythtb)

    # Specify the file path where you want to save the matrix
    file_path = f"spinv_data/chern_matrix_data_{Nx:d}x{Ny:d}_w{w:.1f}.csv"  # Format the file name with the current w value

    # Save the matrix to a text file
    np.savetxt(file_path, matrix_array, fmt='%f', delimiter=',')
