import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import multiprocessing
if __name__ == "verify_twin_pipeline":
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
# import jw_quantum_mapper
import hash_for_cache
from diskcache import Cache
from pretrain_CNN import ProteinDataset,collate_proteins
from process_pdb import process_pdb

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
    
    try:
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
    except Exception as e:
        print(f"Exception: {e}")
        raise Exception("broke")

# Returns untrained GNN predictions and saves model under a name
def get_gnn_predictions(graph_dict_list:list,num_sites:int,pool:Pool):
    gnn_model = mldft_surrogate.MLDFT_GNN(num_sites)
    gnn_model.load_state_dict(torch.load("__temp__/models/protein_gnn.pt"))
    
    # for graph in graph_dict_list:
        # print(f"If leaf: {graph['x'].is_leaf}")
        # print(f"Requires grad: {graph['x'].requires_grad}")
        # graph_dict_list[i]['x'] = torch.Tensor(graph_dict_list[i]['x']).detach()
    
    # Prepare model for predictions by putting into evaluation mode/disabling gradients
    gnn_model.eval()
    for param in gnn_model.parameters():
        param.requires_grad = False
    with torch.no_grad():
            # Parallel process GNN predictions using the calculated graphs
            predicts = pool.map(gnn_model.forward,graph_dict_list)

    return predicts

def get_protein_data(protein_path_list:list(str),batch_size:int,num_sites,grid_size:int=32,distance_threshold:float=5.0,pool:Pool=None):
    
    # Lists for storing protein names, argument tuples, and cached results
    cached_voxel_protein_names = []        
    voxel_args_list,voxel_protein_names = [], []
    graph_args_list,graph_protein_names = [], []
    voxel_list = []
    graph_list = []
    
    # Get voxel/graph caches for each individual protein structure.
    # If no cache exists, add to list for computation to take place.
    with Cache("__scriptcache__") as cache:
        # Get Voxel output hashes:
        for protein_path in protein_path_list:
            # Get protein name so we don't process paths by accident
            protein_name = protein_path.split("/")[-1]
            voxel_hash = hash_for_cache.voxel_json_encoder(protein=protein_name,grid_size=grid_size).hexdigest()
            if voxel_hash in cache:
                # If hash in cache, return cached tensor
                voxel_list.append(cache[voxel_hash])
                cached_voxel_protein_names.append(protein_name)
            else:
                # Else, calculate coordinate/atomic number lists to be processed later
                # (We can discard atomic numbers since our voxelization pipeline doesn't take it into account)
                coordinates,atomic_numbers = process_pdb(pdb_path=protein_path)
                if coordinates is None:
                    continue # Skip protein, has no active sites
                voxel_args_list.append((coordinates,grid_size))
                voxel_protein_names.append(protein_name) # Also append name for later caching

            # Get Graph output hashes:
        
            # Get protein name so we don't process paths by accident
            graph_hash = hash_for_cache.graph_json_encoder(protein=protein_name,distance_threshold=distance_threshold).hexdigest()
            if graph_hash in cache:
                # If hash in cache, return cached tensor
                graph_list.append(cache[graph_hash])
            else:
                # Else, calculate coordinate/atomic number lists to be processed later
                coordinates,atomic_numbers = process_pdb(pdb_path=protein_path)
                if coordinates is None:
                    continue # Skip protein, has no active sites
                graph_args_list.append((coordinates,atomic_numbers,distance_threshold,True))
                graph_protein_names.append(protein_name) # Append name for later caching
    
    # Lists will stay None if not initiated by Pool (i.e. theres no computations to complete)
    temp_voxel_list = None
    temp_graph_list = None

    if __name__ == "verify_twin_pipeline":
        # Get all voxels and graphs we need for training!
        temp_voxel_list = pool.starmap(pdb_voxelizier.protein_to_tensor,voxel_args_list)
        temp_graph_list = pool.starmap(pdb_to_graph.protein_to_graph,graph_args_list)
    
    # Store new lists in cache if not fetched from cache:
    # This basically means every item in the "temp_" lists
    with Cache("__scriptcache__") as cache:
        if temp_voxel_list:
            for result,protein_name in zip(temp_voxel_list,voxel_protein_names):
                voxel_hash = hash_for_cache.voxel_json_encoder(protein=protein_name,grid_size=grid_size).hexdigest()
                cache[voxel_hash] = result
                # Store newly cached protein name into list
                cached_voxel_protein_names.append(protein_name)
        if temp_graph_list:    
            for result,protein_name in zip(temp_graph_list,graph_protein_names):
                graph_hash = hash_for_cache.graph_json_encoder(protein=protein_name,distance_threshold=distance_threshold).hexdigest()
                cache[graph_hash] = result
    
    # Combined cached structures with newly computed ones (if there were any computed)
    voxel_list = (voxel_list + temp_voxel_list) if temp_voxel_list else voxel_list
    graph_list = (graph_list + temp_graph_list) if temp_graph_list else graph_list

    # Needs to be np.array because for some reason ProteinDataset expects it
    voxel_list = np.asarray(voxel_list, dtype = np.float32)
    training_data = get_gnn_predictions(graph_dict_list=graph_list, num_sites=num_sites,pool=pool) # Pass down pool to avoid deadlocking

    
    # We return loader for training, with voxel_list_tuple and training data for later verification
    return voxel_list,training_data,cached_voxel_protein_names

def run_verification(test_protein_dir:str,num_sites:int=4,grid_size:int=32,distance_threshold:float=5.0):
    proteins = [os.path.join(test_protein_dir,x) for x in os.listdir(test_protein_dir)] # Get all protein paths
    
    failed_protein_dict = dict() # Dictionary of tuples to hold information for proteins that failed verification
    
    # Create thread pool to parallelize verification function
    if __name__ == "verify_twin_pipeline":
        with Pool(THREAD_COUNT) as pool:
            # Fetch protein data
            voxel_list,gnn_predictions,voxel_names = get_protein_data(
                protein_path_list=proteins,
                batch_size=32,
                num_sites=num_sites ,
                grid_size=grid_size,
                distance_threshold=distance_threshold,
                pool=pool
            )
            
    processed_protein_name_list = voxel_names # Get all names, are ordered to match the order of cnn/gnn predictions
    
    # Load models for testing:
    # CNN model:
    cnn_model = cnn_mlp_encoder.ProteinPhysicsEncoder(num_sites=num_sites)
    cnn_model.load_state_dict(torch.load("__temp__/models/protein_cnn.pt")) # Load trained model
    cnn_model.eval() # Put into evaluation mode
    
    # 1. Run Track A (CNN) Predictions From Tensor
    cnn_coef_list = []
    gnn_coef_list = [x.cpu().numpy() for x in gnn_predictions]
    
    # Get CNN predictions
    cnn_model.eval()
    for param in cnn_model.parameters():
        param.requires_grad = False
    for voxel in voxel_list:
        voxel_tensor = torch.tensor(voxel, dtype=torch.float32)
        with torch.no_grad():
            cnn_prediction = cnn_model(voxel_tensor)
        cnn_coef_list.append(cnn_prediction.cpu().numpy()[0])
    
    arg_list = [(x,y,num_sites) for x,y in zip(cnn_coef_list,gnn_coef_list)]

    if __name__ == "verify_twin_pipeline":

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


# failed = run_verification(test_protein_dir="proteins/testing_proteins",num_sites=4,grid_size=32,distance_threshold=5.0)
