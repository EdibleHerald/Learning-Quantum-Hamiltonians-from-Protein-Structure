from pyscf import gto, dft
from process_pdb import process_pdb
from qiskit_nature.second_q.drivers import PySCFDriver

coords,atomic_numbers = process_pdb("proteins/training_proteins/1AK9.pdb")

# Map atomic numbers to elements
element_map = {6: 'C', 7: 'N', 8: 'O', 1: 'H', 16: 'S'}
# electron_map = {'C'}

particle_count = 0
full_string_list = []
for coord,a_num in zip(coords,atomic_numbers):
    x,y,z = coord
    # print(element_map[a_num])
    curr_str = f"{element_map[a_num]} {x} {y} {z}"
    full_string_list.append(curr_str)
    full_string_list.append(";")
    particle_count += 1

print(f"Particle Count: {particle_count}")
print(f"Full String List Length: {len(full_string_list)}")
full_string = "".join(full_string_list[0:len(full_string_list)-1])

# print(full_string)
# print(len(full_string_list))
# print("Trying")
# driver = PySCFDriver(atom=full_string,
#                      charge=0,
#                      spin=1)
# print("built")
# problem = driver.run()

# num_alpha = problem.num_alpha
# num_beta = problem.num_beta
# total_electrons = num_alpha + num_beta

# print("done")

# Define active cluster geometry from PDB
try:
    mol = gto.M(
        atom = full_string, # or inline coordinate string
        basis="sto-3g",
        charge=0,
        spin=0 # Singlet: 0, Doublet: 1, Triplet: 2
    )
    print("Spin: 0, Singlet")
except:
    try:
        mol = gto.M(
            atom = full_string, # or inline coordinate string
            basis="sto-3g", # 6-31G
            charge=0,
            spin=1, # Singlet: 0, Doublet: 1, Triplet: 2
            
        )
        print("Spin: 1, Doublet")
    except:
        print("Failed")

# Build and run the Kohn-Sham DFT solver
mf = dft.RKS(mol)
mf.xc = "b3lyp"
e_dft = mf.kernel() # Solves SCF and outputs Ground State Energy in Hartrees
matrix = mf.get_fock()
h_core = mf.get_hcore()

print(f"hcore: {len(h_core)} x {len(h_core[0])}")
print(f"matrix: {len(matrix)} x {len(matrix[0])}")