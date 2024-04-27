import numpy as np
from spinv import local_chern_marker, make_finite, onsite_disorder
from spinv.example_models import haldane_pythtb, haldane_tbmodels

# Create Haldane model through PythTB package
hmodel_pbc_pythtb = haldane_pythtb(delta = 0.5, t = -1, t2 = 0.15, phi = np.pi / 2, L = 1)

# Cut the model to make a sample of size 10 x 10
hmodel_obc_pythtb = make_finite(model = hmodel_pbc_pythtb, nx_sites = 100, ny_sites = 5)

# Add Anderson disorder within [-w/2, w/2]. The argument spinstates specifies the spin of the model
hmodel_pythtb_disorder = onsite_disorder(model = hmodel_obc_pythtb, w = 0, spinstates = 1, seed = 184)

# Compute the local Chern markers for TBmodels and PythTB
lcm_pythtb = local_chern_marker(model = hmodel_pythtb_disorder, nx_sites = 100, ny_sites = 5)

matrix_array = np.array(lcm_pythtb)

# Specify the file path where you want to save the matrix
file_path = "chern_matrix_data.csv"

# Save the matrix to a text file
np.savetxt(file_path, matrix_array, fmt='%f', delimiter=',')
