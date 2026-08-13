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
    # if mae < 0.05:
    #     print("-> Status: PASS (High Mathematical Agreement)")
    # else:
    #     print("-> Status: WARNING (Check spatial mapping drift)")
        
    # CHECKPOINT 2: Physical Ground State Energy
    mat_A = build_matrix(coeffs_track_A, num_sites)
    mat_B = build_matrix(coeffs_track_B, num_sites)
    
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
    # if delta_E <= 0.043:
    #     print("-> Status: PASS (Within Chemical Accuracy! Ready for Quantum Simulation.)")
    # else:
    #     print("-> Status: FAIL (Exceeds Chemical Accuracy. Do not send to QPU.)")
    
    # print("=========================================")
    
    # Return tuple of information. If it passed, and each individual value
    return (mae < 0.05 and delta_E <= 0.043,mae,delta_E)

def run_verification(test_protein_dir:str,num_sites:int=4,grid_size:int=32,distance_threshold:float=5.0):
    proteins = [os.path.join(test_protein_dir,x) for x in os.listdir(test_protein_dir)] # Get all protein paths
    
    print(proteins)
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
run_verification("testing_proteins")