import os
import multiprocessing
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn',force=True)
    multiprocessing.freeze_support()
from multiprocessing import Pool
from psutil import cpu_count
import numpy as np
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
import hash_for_cache
from diskcache import Cache

# Try to only use 75% of the threads the CPU has.
TOTAL_THREAD_COUNT = cpu_count()
# THREAD_COUNT = TOTAL_THREAD_COUNT - (TOTAL_THREAD_COUNT // 4) if TOTAL_THREAD_COUNT > 2 else 1
THREAD_COUNT = TOTAL_THREAD_COUNT // 2

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

def cross_verify_pipelines(coeffs_track_A, coeffs_track_B,num_sites):
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
    # if __name__ == "__main__":
    # if __name__ == "verify_twin_pipeline":
    #     result = pool.starmap(func=build_matrix,iterable=[(coeffs_track_A,num_sites),(coeffs_track_B,num_sites)])
    # mat_A = result[0]
    # mat_B = result[1]
    
    mat_A = build_matrix(coefficients=coeffs_track_A,num_sites=num_sites)
    mat_B = build_matrix(coefficients=coeffs_track_B,num_sites=num_sites)
    
    
    # Diagonalize both to find E0 (lowest eigenvalue)
    # Parallelize for small efficiency gain
    # if __name__ == "__main__":
    # if __name__ == "verify_twin_pipeline":
    #     result = pool.map(func=np.linalg.eigvalsh,iterable=[mat_A,mat_B])
    # eigenvalues_A = result[0]
    # eigenvalues_B = result[1]

    E0_A = np.linalg.eigvalsh(mat_A)[0]
    E0_B = np.linalg.eigvalsh(mat_B)[0]
    
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
    single_bool = mae < 0.05 and delta_E <= 0.043
    return (single_bool,mae,delta_E)

def fetch_protein_data(protein_path:str,grid_size:int,distance_threshold:float):

    tensor = pdb_voxelizier.pdb_to_tensor(protein_path,grid_size) # Returns nparray
    tensor = torch.tensor(data=tensor,dtype=torch.float32) # Convert nparray into Torch tensor

    graph_data = pdb_to_graph.pdb_to_graph(protein_path, distance_threshold) # Convert protein structure into graph representation
    
    return (tensor,graph_data) # (Voxel tensor, graph)

def run_verification(test_protein_dir:str,num_sites:int=4,grid_size:int=32,distance_threshold:float=5.0):
    proteins = [os.path.join(test_protein_dir,x) for x in os.listdir(test_protein_dir)] # Get all protein paths

    # Load models for testing:
    # CNN model:
    cnn_model = cnn_mlp_encoder.ProteinPhysicsEncoder(num_sites=num_sites)
    cnn_model.load_state_dict(torch.load("__temp__/models/protein_cnn.pt")) # Load trained model
    cnn_model.eval() # Put into evaluation mode
    
    # GNN model that was used for training:
    gnn_model = mldft_surrogate.MLDFT_GNN(num_sites)
    gnn_model.load_state_dict(torch.load("__temp__/models/protein_gnn.pt"))
    gnn_model.eval()
    
    failed_protein_dict = dict() # Dictionary of tuples to hold information for proteins that failed verification
    
    # Use only 25% of threads since each thread will itself spawn two threads
    VERIF_THREAD_COUNT = TOTAL_THREAD_COUNT // 4 if TOTAL_THREAD_COUNT >= 4 else 1
    
    arg_list = []
    protein_data = []
    processed_protein_name_list = []
    unprocessed_protein_name_list = []
    # Grab protein voxel if cached, add rest to list to be processed in parallel:
    with Cache("__scriptcache__") as cache:
        
        for protein_name in proteins:
            voxel_hash = hash_for_cache.voxel_json_encoder(protein=protein_name,grid_size=grid_size).hexdigest()
            graph_hash = hash_for_cache.graph_json_encoder(protein=protein_name,distance_threshold=distance_threshold).hexdigest()    

            if voxel_hash in cache and graph_hash in cache:
                tensor = cache[voxel_hash]
                graph_data = cache[graph_hash]
                protein_name_list.append(protein_name)
                protein_data.append((tensor,graph_data))
            else:
                arg_list.append((protein_name,grid_size,distance_threshold))
                unprocessed_protein_name_list.append(protein_name)
    
    # Create a new pool, with 75% thread use
    if __name__ == "verify_twin_pipeline":
        with Pool(THREAD_COUNT) as temp_pool:
            protein_data_temp = temp_pool.starmap(fetch_protein_data,arg_list)
    
    # Get ordered protein name list, to align with protein data tuples:
    processed_protein_name_list += unprocessed_protein_name_list
    protein_data += protein_data_temp
    
    # 1. Run Track A (CNN) Predictions From Tensor
    cnn_coef_list = []
    gnn_coef_list = []
    for tensor,graph_data in protein_data:
        
        with torch.no_grad():
            cnn_coefficients = cnn_model(tensor)
        cnn_coef_list.append(cnn_coefficients.cpu().numpy()[0])

        # 2. Run Track B (GNN) Predictions From Graph (ML-DFT)
        with torch.no_grad():
            mldft_coefficients = gnn_model(graph_data)
        gnn_coef_list.append(mldft_coefficients.cpu().numpy())

    # Create thread pool to parallelize verification function
    if __name__ == "verify_twin_pipeline":
        arg_list = [(x,y,num_sites) for x,y in zip(cnn_coef_list,gnn_coef_list)]
        with Pool(THREAD_COUNT) as pool:
            # 3. Run the Verification (Assuming you have 'cnn_coefficients' from Track A)
            passed = pool.starmap(cross_verify_pipelines,arg_list)

    for result_tuple,protein_name in zip(passed,processed_protein_name_list):
        if not result_tuple[0]:
            failed_protein_dict[protein_name] = (result_tuple[1],result_tuple[2])

    return failed_protein_dict if len(failed_protein_dict) != 0 else None
# Usage:
# cross_verify_pipelines(cnn_coeffs, mldft_coeffs, num_sites=4)
# print(run_verification("exampleStructures"))


# cnn_coef = np.array([ 0.5256421 , -0.18717101,  0.11206499,  0.4405638 , -0.07213806, -0.5189328 , -0.50337166, -0.13883707, -0.85241264,  0.37721214])
# mldft_coef = np.array([ 0.3970126 , -0.13355608,  0.077498,  0.319919 , -0.03946331, -0.3948239 , -0.38502923, -0.105671 , -0.63055885,  0.28899044])
# print(run_verification("testing_proteins",num_sites=4,grid_size=32,distance_threshold=5.0))


