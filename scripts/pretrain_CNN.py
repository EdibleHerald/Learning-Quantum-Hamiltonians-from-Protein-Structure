import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Bio.PDB import PDBParser
from torch.utils.data import Dataset, DataLoader
import os
import sys
sys.path.append('proteins')
import pdb_to_graph
import mldft_surrogate
import verify_twin_pipeline
import pdb_voxelizier
import cnn_mlp_encoder
import jw_quantum_mapper
import psutil
from multiprocessing import Pool
import functools
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
    
    # for pdb in proteins:
    #     # Convert to voxel grid
    #     voxel = pdb_voxelizier.pdb_to_tensor(pdb, grid_size)
    #     # Store voxel
    #     X.append(voxel[0])
    #     # Try to get gnn input
    #     graph = pdb_to_graph.pdb_to_graph(pdb, distance_threshold)
    #     graphs.append(graph)

    #     #coefficients = mldft_surrogate.get_mldft_hamiltonian(graph, num_qubits = 4)
    #     with torch.no_grad():
    #         coefficients = gnn_model(graph)

    #     coefficients = (coefficients.cpu().numpy())
    #     teach_coefficients.append(coefficients)

    # Create argument tuples for parallel processing
    voxel_args = [(x,grid_size) for x in proteins]
    graph_args = [(x,distance_threshold) for x in proteins]
    with Pool(THREAD_COUNT) as p:
        # Get all voxels and graphs we need for training!
        voxel_list = p.starmap(pdb_voxelizier.pdb_to_tensor,voxel_args)
        graph_list = p.starmap(pdb_to_graph.pdb_to_graph,graph_args)
    
    print(f"Size of voxel_list = {len(voxel_list)}")
    print(f"Size of graph_list = {len(graph_list)}")

    # NEEDS WORKING:
    # 1. GENERATE COEFFICIENTS (done!)
    # 2. GET CACHES WORKING (PASS STRING LIST RATHER THAN DIR NAME) (NEEDS WORK)
    # 3. BUILD A JUPYTER NOTEBOOK TO ACTUALLY TEST THIS STUFF (NEEDS WORK)
    
    # Needs to be np.array because for some reason ProteinDataset expects it
    voxel_list = np.asarray(voxel_list, dtype = np.float32)
    training_data = get_gnn_predictions(graph_dict_list=graph_list, num_sites=num_sites)
    
    print(f"Size of new voxel_list = {len(voxel_list)}")
    print(f"Size of training_data = {len(training_data)}")
    print(f"Size of graph_list = {len(graph_list)}")

    # Create DataLoader
    dataset = ProteinDataset(voxel_list,graph_list,training_data)

    # Function used to package all data into DataLoader
    def collate_proteins(batch):
        voxels = []
        graphs = []
        coefficients = []

        for voxel, graph, coeff in batch:
            voxels.append(voxel)
            graphs.append(graph)
            coefficients.append(coeff)
        
        print(f"Size is: {len(voxels)}")
        assert len(voxels) == len(graphs) == len(coefficients)
        
        # Stack CNN inputs (aka, combine into one new tensor via a new dimension)
        voxels = torch.stack(voxels)
        print(voxels.size())
        # Stack coefficient targets
        coefficients = torch.stack(coefficients)
        print(coefficients.size())
        return voxels, graphs, coefficients
    
    # Now automatically loads proteins and shuffles order 
    loader = DataLoader(dataset,batch_size=1,shuffle=True, collate_fn=collate_proteins)
    
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
    while total_loss > pref_loss and curr_iter != MAX_ITER:
        total_loss = 0
        # Track loss
        for voxel, graph, target in loader:

            # CNN prediction
            prediction = cnn_model(voxel.squeeze(0)).squeeze(0)
            
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
        print(f"Epoch {epoch}/{epochs}, Loss: {total_loss:.6f}")
    
    if total_loss <= pref_loss:
        print(f"Successfully pre-trained CNN model with loss of {total_loss} at {epoch} epochs")

    #return cnn_model.state_dict(), gnn_model.state_dict()
    torch.save(cnn_model.state_dict(),"protein_cnn3.pth")

return_pretrained_CNN(protein_dir='testing_proteins')