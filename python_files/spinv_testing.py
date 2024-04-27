import numpy as np
from spinv import local_chern_marker, make_finite, onsite_disorder
from spinv.example_models import haldane_pythtb, haldane_tbmodels
from pythtb import tb_model
from tbmodels import Model

def alt_haldane_pythtb(delta, t, t2, phi, L1,L2,L3,L4):
    # From http://www.physics.rutgers.edu/pythtb/examples.html#haldane-model
    lat=[[1.0,0.0],[0.5,np.sqrt(3.0)/2.0]]
    orb=[[0.0,0.0],[1./3.,1./3.]]
    model=tb_model(2,2,lat,orb)
    model.set_onsite([-delta,delta])
    for lvec in ([ 0, 0], [-1, 0], [ 0,-1]):
        model.set_hop(t, 0, 1, lvec)
    for lvec in ([ 1, 0], [-1, 1], [ 0,-1]):
        model.set_hop(t2*np.exp(1.j*phi), 0, 0, lvec)
    for lvec in ([-1, 0], [ 1,-1], [ 0, 1]):
        model.set_hop(t2*np.exp(1.j*phi), 1, 1, lvec)

    sc_model = model.make_supercell([[L1,L2],[L3,L4]])
    return sc_model

# Create Haldane models through PythTB and TBmodels packages
hmodel_pbc_pythtb = alt_haldane_pythtb(delta = 0.5, t = -1, t2 = 0.15, phi = np.pi / 2, L1=0,L2=1,L3=-1,L4=0)

# Cut the models to make a sample of size 10 x 10
hmodel_obc_pythtb = make_finite(model = hmodel_pbc_pythtb, nx_sites = 10, ny_sites = 10)

# Add Anderson disorder within [-w/2, w/2] to the samples. The argument spinstates specifies the spin of the model
hmodel_pythtb_disorder = onsite_disorder(model = hmodel_obc_pythtb, w = 1, spinstates = 1, seed = 184)

h = hmodel_pythtb_disorder._gen_ham()

lcm_pythtb = local_chern_marker(model=hmodel_pythtb_disorder, nx_sites=10, ny_sites=10)

print(lcm_pythtb)

print("Hamiltonian Matrix:")
print(h)

(fig, ax) = hmodel_pythtb_disorder.visualize(dir_first=0, dir_second=1,ph_color="wheel")
fig.savefig("model.pdf")

#matrix_array = np.array(h)

# Specify the file path where you want to save the matrix
#file_path = "chern_matrix_data.csv"

# Save the matrix to a text file
#np.savetxt(file_path, matrix_array, fmt='%f', delimiter=',')
