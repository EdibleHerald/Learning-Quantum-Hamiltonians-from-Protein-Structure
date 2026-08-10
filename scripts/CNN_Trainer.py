import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
#!pip install biopython
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

def return_pretrained_CNN(protein_dir:str, num_sites:int = 4, grid_size:int = 32, distance_threshold:float = 5.0, epochs:int = 100)->None:

    proteins = [os.path.join("proteins",x) for x in os.listdir('proteins')]
    print(proteins)
    # stores input protein voxel and graphs (x,graph)
    X = []
    graphs = []
    teach_coefficients = []

    #create cnn and gnn
    cnn_model = cnn_mlp_encoder.ProteinPhysicsEncoder(num_sites)
    gnn_model = mldft_surrogate.MLDFT_GNN(num_sites)

    for pdb in proteins:
        #convert to voxel grid
        voxel = pdb_voxelizier.pdb_to_tensor(pdb, grid_size)
        #store voxel
        X.append(voxel[0])
        # try to get gnn input
        graph = pdb_to_graph.pdb_to_graph(pdb, distance_threshold)
        graphs.append(graph)

        #coefficients = mldft_surrogate.get_mldft_hamiltonian(graph, num_qubits = 4)
        with torch.no_grad():
            coefficients = gnn_model(graph)

        coefficients = (coefficients.cpu().numpy())
        #coefficients = np.asarray(coefficients, dtype = np.float32)
        #coefficients = coefficients.reshape(-1)
        teach_coefficients.append(coefficients)
    X = np.asarray(X, dtype = np.float32)
    teach_cofficients = np.asarray(teach_coefficients, dtype = np.float32)

    #print(X.shape)
    #print(teach_cofficients.shape)

    #creating dataloader
    dataset = ProteinDataset(X,graphs, teach_coefficients)
    #now automatcally loads proteins and shuffles order 
    #loader = DataLoader(dataset,batch_size=1,shuffle=True)

    def collate_proteins(batch):
        voxels = []
        graphs = []
        coefficients = []

        for voxel, graph, coeff in batch:
            voxels.append(voxel)
            graphs.append(graph)
            coefficients.append(coeff)
        # Stack CNN inputs
        voxels = torch.stack(voxels)
        # Stack coefficient targets
        coefficients = torch.stack(coefficients)
        return voxels, graphs, coefficients

    loader = DataLoader(dataset,batch_size=1,shuffle=True, collate_fn=collate_proteins)

    
    gnn_model.eval()
    for param in gnn_model.parameters():
        param.requires_grad = False

    #mean squared error : helps keep error positive 
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.0001)

    #training loop
    #epoch = iterations 
    loss_history = []
    for epoch in range(epochs):
        #track loss
        total_loss = 0
        for voxel, graph, target in loader:
            #instead gnn creates target 
            #graph = graph[0]
            print('---')
            print(graph[0])
            print('---')
            with torch.no_grad():
                target = gnn_model(graph[0])
            # cnn prediction
            prediction = cnn_model(voxel)
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
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.6f}")

    torch.save(cnn_model.state_dict(),"protein_cnn3.pth")
    torch.save(gnn_model.state_dict(),"protein_gnn3.pth")

return_pretrained_CNN(protein_dir='proteins')