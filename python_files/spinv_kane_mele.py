import numpy as np
import matplotlib.pyplot as plt
from spinv import single_point_spin_chern, make_finite, onsite_disorder, local_chern_marker
from spinv.example_models import kane_mele_pythtb, km_anderson_disorder_pythtb

L = 1              # supercell LxL linear size

# Topological phase
r = 1.              # r = rashba/spin_orb
e = 3.              # e = e_onsite/spin_orb
spin_o = 0.3        # spin_orb = spin_orb/t (t=1)

# # Trivial phase
# r = 3. 
# e = 5.5
# spin_o = 0.3

# Create Kane-Mele model in supercell LxL through PythTB package
km_pythtb_model = kane_mele_pythtb(r, e, spin_o, L)

# cut the model to make a sample of size Nx x Ny
Nx = 10
Ny = 10
km_pythtb_model = make_finite(model=km_pythtb_model, nx_sites=Nx, ny_sites=Ny)

# w value list
w_values = [0]

# initialise single point spin chern number (spscn) matrix directory
spscn_matrices = {}

# spscn params
spin_chern = 'down'
which_formula = 'asymmetric'

for w in w_values:
    # Add Anderson disorder within [-w/2, w/2]. The argument spinstates specifies the spin of the model
    km_pythtb_model = onsite_disorder(model=km_pythtb_model, w=w, spinstates=1, seed=184)

    # Compute the local Chern markers for TBmodels and PythTB
    #spscn_matrices[w] = single_point_spin_chern(model=km_pythtb_model, spin=spin_chern, formula=which_formula)
    spscn_matrices = local_chern_marker(model=km_pythtb_model, nx_sites=Nx, ny_sites=Ny)



# Loop through the spscn_matrices dictionary and plot each matrix
for w, matrix in spscn_matrices.items():
    plt.figure()
    plt.title(f"w = {w}")
    plt.imshow(matrix, cmap='seismic', origin='lower', extent=(0, matrix.shape[1], 0, matrix.shape[0]), vmin=-np.max(np.abs(matrix)), vmax=np.max(np.abs(matrix)))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(label='Chern Marker')
plt.show()
