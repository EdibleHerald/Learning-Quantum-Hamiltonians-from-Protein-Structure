import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Bio.PDB import PDBParser
from torch.utils.data import Dataset, DataLoader
import pdb_to_graph
import mldft_surrogate
import verify_twin_pipeline
import pdb_voxelizier
import cnn_mlp_encoder
import jw_quantum_mapper

protein = "proteins/1ACX.pdb"
tensor = pdb_voxelizier.pdb_to_tensor(protein,grid_size=32)
model = cnn_mlp_encoder.ProteinPhysicsEncoder(num_sites=4)

tensor = torch.tensor(tensor,dtype=torch.float32)
# print(tensor.shape)

model.load_state_dict(torch.load("../protein_cnn2.pth"))
model.eval()
with torch.no_grad():
    cnn_coefficients = model(tensor)
cnn_coefficients = (cnn_coefficients.cpu().numpy()[0])
# print(cnn_coefficients)

# 1. Run Track B (ML-DFT)
graph_data = pdb_to_graph.pdb_to_graph(protein, distance_threshold=5.0)
mldft_coefficients = mldft_surrogate.get_mldft_hamiltonian(graph_data, num_qubits=4)

# 2. Run the Verification (Assuming you have 'cnn_coefficients' from Track A)
# Use dummy data here just to test the script if needed:
#cnn_coefficients = mldft_coefficients + np.random.normal(0, 0.01, 10)
verify_twin_pipeline.cross_verify_pipelines(cnn_coefficients, mldft_coefficients, num_sites=4)