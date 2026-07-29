import sys
# Use an absolute path or relative path to the directory
sys.path.append("scripts")

import pdb_voxelizier
import cnn_mlp_encoder
import jw_quantum_mapper
import torch
import pyvista as pv
import cirq

tensor = pdb_voxelizier.pdb_to_tensor('proteins/1ENH.pdb')
coefficients = cnn_mlp_encoder.get_hamiltonian(tensor)
qubit_instructions = jw_quantum_mapper.apply_jw(coefficients)

def visualize_tensor(tensor):
    # [batch_size, channels, Depth, Height, Width]
    protein = tensor[0]
    
    protein = protein.to_dense().max(axis=0).values

    array = protein.numpy()
    grid = pv.wrap(array) # Automatically recognizes as a dataset
    grid.plot(volume=True)

