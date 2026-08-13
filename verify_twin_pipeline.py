import numpy as np
from multiprocessing import Pool
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Bio.PDB import PDBParser
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append("scripts")
import pdb_to_graph
import mldft_surrogate
import verify_twin_pipeline
import pdb_voxelizier
import cnn_mlp_encoder
import jw_quantum_mapper


def build_matrix(coefficients, num_sites):
    """Reconstructs the 2D Hamiltonian Matrix from 1D coefficients."""
    matrix = np.zeros((num_sites, num_sites))
    
    # 1. Fill Diagonals (Site Energies)
    epsilons = coefficients[:num_sites]
    np.fill_diagonal(matrix, epsilons)
    
    # 2. Fill Off-Diagonals (Hopping Integrals J)
    j_strengths = coefficients[num_sites:]
    idx = 0
    for i in range(num_sites):
        for j in range(i + 1, num_sites):
            matrix[i, j] = j_strengths[idx]
            matrix[j, i] = j_strengths[idx] # Hermitian symmetry
            idx += 1
    return matrix

def cross_verify_pipelines(coeffs_track_A, coeffs_track_B, num_sites=4):
    """
    Performs the Twin Pipeline Checkpoints: 
    1. Mathematical MAE Check 
    2. Physical Ground State Energy Check
    """
    # print("=== TWIN PIPELINE VERIFICATION REPORT ===")
    
    # CHECKPOINT 1: Mathematical Accuracy
    mae = np.mean(np.abs(coeffs_track_A - coeffs_track_B))
    # print(f"\n[Checkpoint 1] Coefficient Mean Absolute Error (MAE): {mae:.6f} eV")
    if mae < 0.05:
        print("-> Status: PASS (High Mathematical Agreement)")
    else:
        print("-> Status: WARNING (Check spatial mapping drift)")
        
    # CHECKPOINT 2: Physical Ground State Energy
    
    # Paralleize matrix building:
    mat_num_list = [(coeffs_track_A,num_sites),(coeffs_track_B,num_sites)]
    results = None
    if __name__ == '__main__':
        with Pool(2) as p:
            print(p.starmap(build_matrix,mat_num_list))
    
    # mat_A = build_matrix(coeffs_track_A, num_sites)
    # mat_B = build_matrix(coeffs_track_B, num_sites)
    # print(mat_A)
    # print(mat_B)
    # print("---")
    print(results)
    mat_A = mat_num_list[0][0]
    mat_B = mat_num_list[1][0]
    
    print(mat_A)
    print(mat_B)
    # Diagonalize both to find E0 (lowest eigenvalue)
    eigenvalues_A = np.linalg.eigvalsh(mat_A)
    eigenvalues_B = np.linalg.eigvalsh(mat_B)
    
    E0_A = eigenvalues_A[0]
    E0_B = eigenvalues_B[0]
    
    delta_E = np.abs(E0_A - E0_B)
    
    # print(f"\n[Checkpoint 2] Physical Ground State Energy (E0)")
    # print(f"Track A (3D CNN) E0 : {E0_A:.6f} eV")
    # print(f"Track B (ML-DFT) E0 : {E0_B:.6f} eV")
    # print(f"Delta E (Error)     : {delta_E:.6f} eV")
    
    # 1 kcal/mol is roughly 0.043 eV
    if delta_E <= 0.043:
        print("-> Status: PASS (Within Chemical Accuracy! Ready for Quantum Simulation.)")
    else:
        print("-> Status: FAIL (Exceeds Chemical Accuracy. Do not send to QPU.)")
    
    # print("=========================================")

# Usage:

protein = "proteins/1ACX.pdb"
tensor = pdb_voxelizier.pdb_to_tensor(protein,grid_size=32)
model1 = cnn_mlp_encoder.ProteinPhysicsEncoder(num_sites=4)
graph = pdb_to_graph.pdb_to_graph(protein)
tensor = torch.tensor(tensor,dtype=torch.float32)
# print(tensor.shape)

model1.load_state_dict(torch.load("protein_cnn3.pth"))
model1.eval()
with torch.no_grad():
    cnn_coefficients = model1(tensor)
cnn_coeffs = (cnn_coefficients.cpu().numpy()[0])


model2 = mldft_surrogate.MLDFT_GNN(4)
model2.load_state_dict(torch.load("protein_gnn3.pth"))
model2.eval()
with torch.no_grad():
    mldft_coefficients = model2(graph)
mldft_coeffs = (mldft_coefficients.cpu().numpy())


cross_verify_pipelines(cnn_coeffs, mldft_coeffs, num_sites=4)
