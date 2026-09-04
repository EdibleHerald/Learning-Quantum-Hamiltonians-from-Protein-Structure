from pretrain_CNN import collate_proteins,ProteinDataset,THREAD_COUNT
import torch
import torch.nn as nn
from multiprocessing import Pool
from process_pdb import process_pdb
from pdb_voxelizier import protein_to_tensor
import numpy as np
from torch.utils.data import Dataset, DataLoader

def get_ground_state_energy(sim_string:str):
    # Define active cluster geometry from PDB
    mol = gto.M(
        atom = "".join(full_string), # Coordinate string
        basis="sto-3g",
        charge=0,
        spin=0
    )

    # Build and run the Kohn-Sham DFT solver
    mf = dft.RKS(mol)
    mf.xc = "b3lyp"
    e_dft = mf.kernel() # Solves SCF and outputs Ground State Energy in Hartrees

    matrix = mf.get_fock()
    return e_dft
    # NOT DONE

def get_dft_values(coord_list,atomic_num_list,num_sites:int,pool:Pool):
    string_list = []
    
    # Map atomic numbers to elements
    element_map = {6: 'C', 7: 'N', 8: 'O', 1: 'H', 16: 'S'}
    
    # Create and store coordinate strings
    for coords,atomic_numbers in zip(coord_list,atomic_num_list):
        full_string = []
        for coord,a_num in zip(coords,atomic_numbers):
            x,y,z = coord
            
            curr_str = f"{element_map[a_num]} {x} {y} {z};"
            full_string.append(curr_str)
        string_list.append(full_string)
    
    # Parallelize ground state calculations
    gse_list = pool.map(get_ground_state_energy,string_list)
    
    return gse_list # Ordered to match proteins in protein_path_list

def return_trained_CNN(protein_dir:str, num_sites, grid_size, distance_threshold,max_iterations:int=100,loss_threshold:float=0.043,batch_size:int=32):
    # Import pretrained model
    cnn_model = cnn_mlp_encoder.ProteinPhysicsEncoder(num_sites)
    cnn_model.load_state_dict(torch.load("__temp__/models/protein_cnn.pt"))
    
    # Mean Squared Error (MSE): Helps keep error positive 
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.0001)

    # Get list of all proteins to test against
    proteins = [os.path.join(protein_dir,x) for x in os.listdir(protein_dir)]
    
    # Get all coordinates/atomic numbers needed
    coord_list = []
    atomic_num_list = []
    protein_path_list = [] # Only store proteins that have catalytic sites that could be fetched
    for path in proteins:
        coords,atomic_nums = process_pdb(pdb_path=path)
        
        if coords is None or atomic_nums is None:
            print(f"Protein catalytic sites for {protein_path} returned no values. Cannot train on this protein, skipped.")
            continue # Skip, no data found
        
        protein_path_list.append(path)
        coord_list.append(coords)
        atomic_num_list.append(atomic_nums)
    
    # Create Pool to pass down to other subprocesses (to avoid deadlocking and repetitive code)
    with Pool(THREAD_COUNT) as p:
        # Compute DFT values
        # (change variable name later)
        result_list = get_dft_values(coord_list=coord_list,atomic_num_list=atomic_num_list,num_sites=num_sites,pool=p)
        # Get Voxelization!
        voxel_arg = [(x,grid_size) for x in coord_list]
        voxel_list = p.starmap(protein_to_tensor)
    
    # Create dataset
    Dataset = ProteinDataset(np.asarray(voxel_list),np.asarray(result_list))
    
    # Create dataloader
    Dataloader = Dataloader(Dataset, batch_size=batch_size,shuffle=True, collate_fn=collate_proteins)
    
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
    
    # Test CNN against trained proteins.   
    # Prepare model for predictions by putting into evaluation mode/disabling gradients
    cnn_model.eval()
    for param in cnn_model.parameters():
        param.requires_grad = False
    
    loss_list = []
    for voxel_tuple,test_data in zip(voxel_list_tuple,training_data):
        voxel_tensor = torch.tensor(voxel_tuple[1])
        prediction = cnn_model(voxel_tensor).squeeze(0) # Sqeeze from tensor size [ batch features ] to [ features ]
        
        loss = criterion(prediction,test_data)
        
        # Test that each protein meets mean average error of 0.05 (5% margin)
        if loss.item() > 0.05:
            loss_list.append((voxel_tuple[0],loss.item()))
    
    if total_loss <= pref_loss:
        print(f"Successfully pre-trained CNN model with loss total of {total_loss:.6f} at {epoch} epochs.")
    else:
        print(f"Training stopped after reaching max iterations of {MAX_ITER}. Loss: {total_loss}")
    
    torch.save(cnn_model.state_dict(),"__temp__/models/protein_cnn.pt")
    return loss_list