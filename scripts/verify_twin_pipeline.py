import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio.PDB import PDBParser
from torch.utils.data import Dataset, DataLoader
import pdb_to_graph
import mldft_surrogate
import pdb_voxelizier
import cnn_mlp_encoder
import jw_quantum_mapper
import multiprocessing
multiprocessing.set_start_method("fork") # Forkserver on Linux fails....for some reason. 
from multiprocessing import Pool


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
    
    # Uncomment below to get verbose information
    # print(f"\n[Checkpoint 1] Coefficient Mean Absolute Error (MAE): {mae:.6f} eV")
    # if mae < 0.05:
    #     print("-> Status: PASS (High Mathematical Agreement)")
    # else:
    #     print("-> Status: WARNING (Check spatial mapping drift)")
    
    # CHECKPOINT 2: Physical Ground State Energy
    # Parallelize for small efficiency gain
    if __name__ == "verify_twin_pipeline":
        with Pool(2) as p:
            result = p.starmap(func=build_matrix,iterable=[(coeffs_track_A,num_sites),(coeffs_track_B,num_sites)])
    mat_A = result[0]
    mat_B = result[1]
    
    
    
    # Diagonalize both to find E0 (lowest eigenvalue)
    # Parallelize for small efficiency gain
    if __name__ == "verify_twin_pipeline":
        with Pool(2) as p:
            result = p.map(func=np.linalg.eigvalsh,iterable=[mat_A,mat_B])
    eigenvalues_A = result[0]
    eigenvalues_B = result[1]

    E0_A = eigenvalues_A[0]
    E0_B = eigenvalues_B[0]
    
    delta_E = np.abs(E0_A - E0_B)
    
    # Uncomment below to get verbose information
    # print(f"\n[Checkpoint 2] Physical Ground State Energy (E0)")
    # print(f"Track A (3D CNN) E0 : {E0_A:.6f} eV")
    # print(f"Track B (ML-DFT) E0 : {E0_B:.6f} eV")
    # print(f"Delta E (Error)     : {delta_E:.6f} eV")
    
    # 1 kcal/mol is roughly 0.043 eV
    # if delta_E <= 0.043:
    #     print("-> Status: PASS (Within Chemical Accuracy! Ready for Quantum Simulation.)")
    # else:
    #     print("-> Status: FAIL (Exceeds Chemical Accuracy. Do not send to QPU.)")
    
    # print("=========================================")
    
    # Return tuple of information. If it passed, and each individual value
    return (mae < 0.05 and delta_E <= 0.043,mae,delta_E)

def run_verification(test_protein_dir:str,num_sites:int=4,grid_size:int=32,distance_threshold:float=5.0):
    proteins = [os.path.join(test_protein_dir,x) for x in os.listdir(test_protein_dir)] # Get all protein paths
    
    # print(proteins)
    # Load models for testing:
    # CNN model:
    cnn_model = cnn_mlp_encoder.ProteinPhysicsEncoder(num_sites=num_sites)
    cnn_model.load_state_dict(torch.load("protein_cnn3.pth")) # Load trained model
    cnn_model.eval() # Put into evaluation mode
    
    # GNN model that was used for training:
    gnn_model = mldft_surrogate.MLDFT_GNN(num_sites)
    gnn_model.load_state_dict(torch.load("protein_gnn3.pth"))
    gnn_model.eval()
    
    # Test every protein in test_protein_dir
    for protein in proteins:
        # print(protein)
        tensor = pdb_voxelizier.pdb_to_tensor(protein,grid_size) # Returns nparray
        tensor = torch.tensor(data=tensor,dtype=torch.float32) # Convert nparray into Torch tensor
        
        # 1. Run Track A (CNN) Predictions From Tensor
        with torch.no_grad():
            cnn_coefficients = cnn_model(tensor)
        cnn_coefficients = (cnn_coefficients.cpu().numpy()[0]) # Predicted coefficients as a numpy array

        # 2. Run Track B (GNN) Predictions From Graph (ML-DFT)
        graph_data = pdb_to_graph.pdb_to_graph(protein, distance_threshold) # Convert protein structure into graph representation
        with torch.no_grad():
            mldft_coefficients = gnn_model(graph_data)
        mldft_coefficients = mldft_coefficients.cpu().numpy()


        # 3. Run the Verification (Assuming you have 'cnn_coefficients' from Track A)
        #cnn_coefficients = mldft_coefficients + np.random.normal(0, 0.01, 10)
        
        passed = cross_verify_pipelines(cnn_coefficients, mldft_coefficients, num_sites=num_sites)
        
        if passed[0]:
            print(f"Protein {protein} passed verification")
        else:
            print(f"Protein {protein} did not pass verification")
        
# Usage:
# cross_verify_pipelines(cnn_coeffs, mldft_coeffs, num_sites=4)
# print(run_verification("exampleStructures"))


# cnn_coef = np.array([ 0.5256421 , -0.18717101,  0.11206499,  0.4405638 , -0.07213806, -0.5189328 , -0.50337166, -0.13883707, -0.85241264,  0.37721214])
# mldft_coef = np.array([ 0.3970126 , -0.13355608,  0.077498,  0.319919 , -0.03946331, -0.3948239 , -0.38502923, -0.105671 , -0.63055885,  0.28899044])
# print(cross_verify_pipelines(cnn_coef,mldft_coef,4))

