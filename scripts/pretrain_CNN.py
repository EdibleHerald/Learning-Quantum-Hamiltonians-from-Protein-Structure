import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Bio.PDB import PDBParser
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

TOTAL_THREAD_COUNT = psutil.cpu_count()
THREAD_COUNT = TOTAL_THREAD_COUNT - (TOTAL_THREAD_COUNT // 4) if TOTAL_THREAD_COUNT > 2 else 1

class ProteinDataset(Dataset):
    def __init__(self, voxels, graphs, coefficients):
        #voxels
        self.voxels = voxels
        #graph values 
        self.graphs = graphs
        #coefficients
        self.coefficients = coefficients
    #number of proteins  
    def __len__(self):
        return len(self.voxels)
    #cinverting numpy array to tensor 
    def __getitem__(self, index):
        voxel = torch.tensor(self.voxels[index], dtype=torch.float32)
        if voxel.ndim == 3:
            voxel = voxel.unsqueeze(0)
        graph = self.graphs[index]
        coefficients = torch.tensor(self.coefficients[index], dtype = torch.float32)
        return voxel, graph, coefficients

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
    torch.save(gnn_model.state_dict(),"protein_gnn3.pth")
    
    return predicts    

# Returns DataLoader to be used for training.
def get_protein_data(protein_path_list:list(str),num_sites:int=4,grid_size:int=32,distance_threshold:float=5.0):
    # proteins = [os.path.join(protein_dir,x) for x in os.listdir(protein_dir)]
    proteins = protein_path_list
    
    # We implement a naive cache system.
    # We turn the list into a frozenset to be hashed, then search for it in our cache. Otherwise we run calculations and then store the result in cache
    # (Smarter would be to cache graph results for each individual protein)
    protein_name_list = [x.split("/")[-1] for x in proteins]
    
    # We want two hashes:
    # One for a list of proteins at a specific grid_size (voxel cache). 
    voxel_hash_exist = False # Bool to check if a cached result exists
    voxel_hash = hash_for_cache.voxel_json_encoder(protein_list=protein_name_list,grid_size=grid_size).hexdigest()
    
    # One for a list of proteins at a specific distance threshold (graph cache)
    graph_hash_exist = False
    graph_hash = hash_for_cache.graph_json_encoder(protein_list=protein_name_list,distance_threshold=distance_threshold).hexdigest()
    
    with Cache("__scriptcache__") as cache:
        if voxel_hash in cache:
            # We just fetch the results!
            voxel_list = cache[voxel_hash]
            voxel_hash_exist = True
            print("Fetching cached voxels!")
        
        if graph_hash in cache:
            # Fetch results
            graph_list = cache[graph_hash]
            graph_hash_exist = True
            print("Fetching cached graphs!")
    
    # Create argument tuples for parallel processing
    voxel_args = [(x,grid_size) for x in proteins] if not voxel_hash_exist else None
    graph_args = [(x,distance_threshold) for x in proteins] if not graph_hash_exist else None
    
    with Pool(THREAD_COUNT) as p:
        # Get all voxels and graphs we need for training!
        voxel_list = p.starmap(pdb_voxelizier.pdb_to_tensor,voxel_args) if not voxel_hash_exist else voxel_list
        graph_list = p.starmap(pdb_to_graph.pdb_to_graph,graph_args) if not graph_hash_exist else graph_list
    
    # Store new lists in cache if not fetched from cache:
    with Cache("__scriptcache__") as cache:
        if not voxel_hash_exist:
            cache[voxel_hash] = voxel_list
        
        if not graph_hash_exist:
            cache[graph_hash] = graph_list

    # print(f"Size of voxel_list = {len(voxel_list)}")
    # print(f"Size of graph_list = {len(graph_list)}")

    # NEEDS WORKING:
    # 1. GENERATE COEFFICIENTS (done!)
    # 2. GET CACHES WORKING (PASS STRING LIST RATHER THAN DIR NAME) (NEEDS WORK)
    # 3. BUILD A JUPYTER NOTEBOOK TO ACTUALLY TEST THIS STUFF (NEEDS WORK)
    
    # Needs to be np.array because for some reason ProteinDataset expects it
    voxel_list = np.asarray(voxel_list, dtype = np.float32)
    training_data = get_gnn_predictions(graph_dict_list=graph_list, num_sites=num_sites)
    
    # print(f"Size of new voxel_list = {len(voxel_list)}")
    # print(f"Size of training_data = {len(training_data)}")
    # print(f"Size of graph_list = {len(graph_list)}")

    # Create DataLoader
    dataset = ProteinDataset(voxel_list,graph_list,training_data)

    # Function used to package all data into DataLoader (misconception: Each data point is called once so we're returning a list with a single tensor each time, odd...)
    # Additionally, we can't shuffle items with batch size of 1.
    def collate_proteins(batch):
        # TO DO: PASS IN DIFFERENT SIZE BATCHES (OR PLAY WITH THAT IDEA)
        # RIGHT NOW:
        # - batch = list of each batch, so batch[0] = first batch of [voxel,graph,coef] and so on
        # print(len(batch[0]))
        voxels = batch[0][0]
        graphs = batch[0][1]
        coefficients = batch[0][2]

        # num = 0
        # for voxel, graph, coeff in batch:
        #     voxels.append(voxel)
        #     graphs.append(graph)
        #     coefficients.append(coeff)
        #     num+=1
        #     # print(f"Iteration {num} !")
        # print(batch)
        # print(type(batch))
        # print(f"Size is: {len(voxels)}")
        # assert len(voxels) == len(graphs) == len(coefficients)
        
        # Stack CNN inputs (aka, combine into one new tensor via a new dimension)
        # voxels = torch.stack(voxels)
        # print(voxels.size())
        # Stack coefficient targets
        # coefficients = torch.stack(coefficients)
        # print(coefficients.size())
        return voxels, graphs, coefficients
    
    # Now automatically loads proteins and shuffles order 
    loader = DataLoader(dataset,batch_size=1,shuffle=False, collate_fn=collate_proteins)
    
    return loader

def return_pretrained_CNN(protein_dir:str, num_sites:int = 4, grid_size:int = 32, distance_threshold:float = 5.0)->None:
    # Create CNN
    cnn_model = cnn_mlp_encoder.ProteinPhysicsEncoder(num_sites)

    #mean squared error : helps keep error positive 
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.0001)

    # Get list of all proteins to train against
    proteins = [os.path.join(protein_dir,x) for x in os.listdir(protein_dir)]
    
    # Get protein data:
    loader = get_protein_data(protein_path_list=proteins,num_sites=num_sites,grid_size=grid_size,distance_threshold=distance_threshold)

    #training loop
    #epoch = iterations 
    loss_history = []
    total_loss = float("inf")
    # for epoch in range(epochs):
    epoch = 0
    MAX_ITER = 1000
    curr_iter = 0
    
    pref_loss = 0.00005
    #test set
    test_set = set()
    while total_loss > pref_loss and curr_iter != MAX_ITER:
        total_loss = 0
        # Track loss
        for voxel, graph, target in loader:
            test_set.add(voxel)
            test_set.add(graph['x'])
            test_set.add(target)
            # CNN prediction
            prediction = cnn_model(voxel).squeeze(0)
            
            # Compare prediction with target coefficients 
            loss = criterion(prediction,target)
            
            # remove old gradients 
            optimizer.zero_grad()
            #calculate how values should change
            loss.backward()
            
            #update values 
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        loss_history.append(total_loss)
        #Note: ideally loss should decrease overtime
        epoch += 1
        curr_iter += 1
        print(f"Epoch {epoch}/{MAX_ITER}, Loss: {total_loss:.6f}")
    
    print(f"Size of set should be {len(proteins)*3}, is {len(test_set)}")

    if total_loss <= pref_loss:
        print(f"Successfully pre-trained CNN model with loss of {total_loss} at {epoch} epochs")
    else:
        print(f"Training stopped after reaching max iterations of {MAX_ITER}. Loss: {total_loss}")
    
    torch.save(cnn_model.state_dict(),"protein_cnn3.pth")

return_pretrained_CNN(protein_dir='exampleStructures')