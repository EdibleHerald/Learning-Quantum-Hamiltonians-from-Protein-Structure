from pyscf import gto, dft
from process_pdb import process_pdb

coords,atomic_numbers = process_pdb("proteins/training_proteins/9ATK.pdb")

# Map atomic numbers to elements
element_map = {6: 'C', 7: 'N', 8: 'O', 1: 'H', 16: 'S'}

full_string = []
for coord,a_num in zip(coords,atomic_numbers):
    x,y,z = coord
    # print(element_map[a_num])
    curr_str = f"{element_map[a_num]} {x} {y} {z};"
    full_string.append(curr_str)

# Define active cluster geometry from PDB
mol = gto.M(
    atom = "".join(full_string), # or inline coordinate string
    basis="sto-3g",
    charge=0,
    spin=0
)

# Build and run the Kohn-Sham DFT solver
mf = dft.RKS(mol)
mf.xc = "b3lyp"
e_dft = mf.kernel() # Solves SCF and outputs Ground State Energy in Hartrees
print(type(mf))
print(type(e_dft))
print(e_dft)
matrix = mf.get_fock()

print(len(matrix))
print(len(matrix[0]))
# print(matrix)
# print(mf.get_fock())
# print(mf.get_hcore())