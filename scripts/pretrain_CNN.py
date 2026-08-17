import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import pdb_to_graph
import mldft_surrogate
import verify_twin_pipeline
import pdb_voxelizier
import cnn_mlp_encoder
import hash_for_cache
import psutil
from multiprocessing import Pool
from diskcache import Cache

# Try to only use 75% of the threads the CPU has.
TOTAL_THREAD_COUNT = psutil.cpu_count()
THREAD_COUNT = TOTAL_THREAD_COUNT - (TOTAL_THREAD_COUNT // 4) if TOTAL_THREAD_COUNT > 2 else 1

class ProteinDataset(Dataset):
    def __init__(self, voxels, coefficients):
        # Ensure that we'll have consistent dimension tensors
        if len(voxels) != len(coefficients):
            raise Exception("ProteinDataset::__init__() - Mismatched list sizes.")
        
        # Voxels
        self.VOXELS = voxels

        # Coefficients
        self.COEFFS = coefficients
    
    # Returns number of proteins  
    def __len__(self):
        return len(self.VOXELS)
    
    # Return voxel and coefficient of given index.
    # (Voxels are numpy arrays so they need conversion while coefficients are already tensors) 
    def __getitem__(self, index):
        voxel = torch.tensor(self.VOXELS[index], dtype=torch.float32)
        coeff = self.COEFFS[index]
        return voxel, coeff

# Returns untrained GNN predictions and saves model under a name
def get_gnn_predictions(graph_dict_list:list,num_sites:int):
    gnn_model = mldft_surrogate.MLDFT_GNN(num_sites)
    
    # Prepare model for predictions by putting into evaluation mode/disabling gradients
    gnn_model.eval()
    for param in gnn_model.parameters():
        param.requires_grad = False
    
    # Prepare list of arguments for parallel predictions
    with Pool(THREAD_COUNT) as p:
        predicts = p.map(gnn_model.forward,graph_dict_list)
    
    # Save model to test against later
    torch.save(gnn_model.state_dict(),"__temp__/models/protein_gnn.pt")
    
    return predicts

# Function used to package all data into DataLoader
def collate_proteins(batch):
        voxels = []
        coefficients = []
    
        for voxel,coeff in batch:
            voxels.append(voxel.squeeze(0))
            coefficients.append(coeff)
        
        curr_batch_size = len(voxel)
        
        # Stack voxel and prediction tensors into one for faster processing! 
        voxels = torch.stack(voxels)
        coefficients = torch.stack(coefficients)

        return voxels, coefficients, curr_batch_size

# Returns DataLoader to be used for training.
def get_protein_data(protein_path_list:list(str),batch_size:int,num_sites:int=4,grid_size:int=32,distance_threshold:float=5.0):
    
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
                # Else, add to args list to be computed later!
                # (Here we use path to protein rather than just name)
                voxel_args_list.append((protein_path,grid_size))
                voxel_protein_names.append(protein_name) # Also append name for later caching

        # Get Graph output hashes:
        for protein_path in protein_path_list:
            # Get protein name so we don't process paths by accident
            protein_name = protein_path.split("/")[-1]
            graph_hash = hash_for_cache.graph_json_encoder(protein=protein_name,distance_threshold=distance_threshold).hexdigest()
            if graph_hash in cache:
                # If hash in cache, return cached tensor
                graph_list.append(cache[graph_hash])
            else:
                # Else, add to args list to be computed later!
                # (Here we use path to protein rather than just name)
                graph_args_list.append((protein_path,distance_threshold))
                graph_protein_names.append(protein_name) # Append name for later caching
    
    with Pool(THREAD_COUNT) as p:
        # Get all voxels and graphs we need for training!
        temp_voxel_list = p.starmap(pdb_voxelizier.pdb_to_tensor,voxel_args_list)
        temp_graph_list = p.starmap(pdb_to_graph.pdb_to_graph,graph_args_list)
    
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
    voxel_list = voxel_list + temp_voxel_list if temp_voxel_list else voxel_list
    graph_list = graph_list + temp_graph_list if temp_graph_list else graph_list

    # Needs to be np.array because for some reason ProteinDataset expects it
    voxel_list = np.asarray(voxel_list, dtype = np.float32)
    training_data = get_gnn_predictions(graph_dict_list=graph_list, num_sites=num_sites)
    
    # Create DataLoader
    dataset = ProteinDataset(voxel_list,training_data)
    
    # Now automatically loads proteins and shuffles order
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=True, collate_fn=collate_proteins)
    
    # Protein names and their accompanying voxel representations.
    voxel_list_tuple = zip(cached_voxel_protein_names,voxel_list)
    
    # We return loader for training, with voxel_list_tuple and training data for later verification
    return loader,voxel_list_tuple,training_data

def return_pretrained_CNN(protein_dir:str, num_sites:int = 4, grid_size:int = 32, distance_threshold:float = 5.0,max_iterations:int=1000,loss_threshold:float=0.043,batch_size:int=32):
    # Create CNN
    cnn_model = cnn_mlp_encoder.ProteinPhysicsEncoder(num_sites)

    # Mean Squared Error (MSE): Helps keep error positive 
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.0001)

    # Get list of all proteins to train against
    proteins = [os.path.join(protein_dir,x) for x in os.listdir(protein_dir)]
    
    # Get protein data:
    loader, voxel_list_tuple, training_data = get_protein_data(protein_path_list=proteins,num_sites=num_sites,grid_size=grid_size,distance_threshold=distance_threshold,batch_size=batch_size)

    # Training loop
    total_loss = float("inf")
    epoch = 0
    MAX_ITER = max_iterations
    
    pref_loss = loss_threshold
    while total_loss > pref_loss and epoch != MAX_ITER:
        # Track loss
        total_loss = 0

        # Unpack batches from loader
        for voxel, target, curr_batch_size in loader:
            # CNN prediction
            prediction = cnn_model(voxel)

            # Compare prediction with target coefficients 
            loss = criterion(prediction,target)
            
            # Remove old gradients 
            optimizer.zero_grad()
            
            # Calculate how values should change
            loss.backward()

            # Update values 
            optimizer.step()
            total_loss += loss.item()
        
        # Handle loss, we take average loss of stacked tensors
        avg_loss = total_loss / curr_batch_size

        # Note: Ideally loss should decrease overtime
        epoch += 1
    
    # Now, since CNN predictions are relatively cheap and we have saved GNN predictions, we verify that the final model can somewhat accurately 
    # predict similar interactions as the GNN. This is worth testing since a failure could indicate a weakness in the training data, loss threshold, or iterations needed. 
    
    # Prepare model for predictions by putting into evaluation mode/disabling gradients
    cnn_model.eval()
    for param in cnn_model.parameters():
        param.requires_grad = False
    
    loss_list = []
    for voxel_tuple,test_data in zip(voxel_list_tuple,training_data):
        voxel_tensor = torch.tensor(voxel_tuple[1])
        prediction = cnn_model(voxel_tensor).squeeze(0) # Sqeeze from tensor size [ batch features ] to [ features ]
        
        loss = criterion(prediction,test_data)
        
        # print(f"Loss is {loss.item()} for protein {voxel_tuple[0]}")
        
        # Test that each protein makes minimum required chemical accuracy of 0.043 eV
        if loss.item() > 0.043:
            loss_list.append((voxel_tuple[0],loss.item()))
    
    if len(loss_list) > 0:
        print(f"Protein(s) have failed verification:")
        for i in loss_list:
            print(f"Protein {i[0]} had loss of {i[1]}")
    else:
        print("Verified model accuracy is satisfactory.")
    
    if total_loss <= pref_loss:
        print(f"Successfully pre-trained CNN model with loss total of {total_loss:.6f} at {epoch} epochs. All predictions meet the 0.043 eV chemical accuracy threshold!")
    else:
        print(f"Training stopped after reaching max iterations of {MAX_ITER}. Loss: {total_loss}")
    
    torch.save(cnn_model.state_dict(),"__temp__/models/protein_cnn.pt")

# Example Use:  
# return_pretrained_CNN("proteins",loss_threshold=0.01)

# THINGS THAT NEED WORKING ON:
# - Consider a better way to improve caching system. Currently, caching voxels/graphs for 30~ proteins is rather expensive at 78Mb
#   e.g. Theres (right now) 28 proteins in protein folder at 12Mb total while the cache sits at 70Mb. 
#        This will only get more expensive as resolution of voxels/graphs grow or as more proteins are added.