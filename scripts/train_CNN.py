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

def verify_model_loss(cnn_model,loader,validation_loader,criterion):
    # Validate CNN performance
    total_loss = 0
    validation_total_loss = 0
    total_samples_train = 0
    total_samples_validation = 0
    cnn_model.eval() # Put model into eval mode
    with torch.no_grad():
        # Validate against training set 
        for voxel, target in loader:
            curr_batch_size = voxel.size(0)
            
            # CNN prediction
            prediction = cnn_model(voxel)
            
            # Compare prediction with target coefficients 
            loss = criterion(prediction,target)

            # Update loss
            total_loss += loss.item() * curr_batch_size
            total_samples_train += curr_batch_size
        
        # Validate against validation set
        for voxel, target in validation_loader:
            curr_batch_size = voxel.size(0)
            
            # CNN prediction
            prediction = cnn_model(voxel)

            # Compare prediction with target coefficients 
            loss = criterion(prediction,target)

            # Update loss
            validation_total_loss += loss.item() * curr_batch_size
            total_samples_validation += curr_batch_size
    
    # Handle loss, we take per-sample loss average
    total_loss = total_loss / total_samples_train
    validation_total_loss = validation_total_loss / total_samples_validation
    
    return total_loss,validation_total_loss

def return_trained_CNN(test_set_dir:str,validation_set_dir:str, num_sites, grid_size, distance_threshold,max_iterations:int=100,loss_threshold:float=0.043,batch_size:int=32,lr:float=0.001,loss_loop_threshold:int=10):
    # Create CNN
    cnn_model = cnn_mlp_encoder.ProteinPhysicsEncoder(num_sites).to(DEVICE)
    
    # Mean Squared Error (MSE): Helps keep error positive 
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=lr)

    # Get list of all proteins to train against
    proteins = [os.path.join(test_set_dir,x) for x in os.listdir(test_set_dir)]
    
    # Get list of all proteins to validate against
    validation_proteins = [os.path.join(validation_set_dir,x) for x in os.listdir(validation_set_dir)]
        
    # Create Pool to pass down to other subprocesses (to avoid deadlocking and repetitive code)
    with Pool(THREAD_COUNT) as p:
        # Get protein data:
        loader = get_protein_data(
                    protein_path_list=proteins,
                    num_sites=num_sites,
                    grid_size=grid_size,
                    distance_threshold=distance_threshold,
                    batch_size=batch_size,
                    pool=p
                    )
        
        # Get validation data:
        validation_loader = get_protein_data(
                                protein_path_list=validation_proteins,
                                num_sites=num_sites,
                                grid_size=grid_size,
                                distance_threshold=distance_threshold,
                                batch_size=batch_size,
                                pool=p
                                )

    # Training loop
    total_loss = float("inf")
    epoch = 0
    MAX_ITER = max_iterations
    
    iterations_under_threshold = 0
    
    total_loss_history = []
    validation_loss_history = []
        
    # Get initial loss before training
    total_loss,validation_total_loss = verify_model_loss(
                                    cnn_model=cnn_model,
                                    loader=loader,
                                    validation_loader=validation_loader,
                                    criterion=criterion
                                    )
    total_loss_history.append(total_loss)
    validation_loss_history.append(validation_total_loss)

    while iterations_under_threshold <= loss_loop_threshold and epoch != MAX_ITER:
        # Put model into training mode
        cnn_model.train()
        
        # Track loss
        total_loss = 0
        validation_total_loss = 0 
        
        # Accumulate batch size to calculate per-sample error
        total_samples_train = 0
        
        # validate CNN
        for voxel, target in loader:
            
            curr_batch_size = voxel.size(0)
            # print(f"curr batch size: {curr_batch_size}")
            
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
        
        total_loss,validation_total_loss = verify_model_loss(
                                            cnn_model=cnn_model,
                                            loader=loader,
                                            validation_loader=validation_loader,
                                            criterion=criterion
                                            )
        
        # Add losses to history list
        total_loss_history.append(total_loss)
        validation_loss_history.append(validation_total_loss)
        
        # Iterate loop counter
        if total_loss < loss_threshold and validation_total_loss < loss_threshold:
            iterations_under_threshold += 1
        
        # Note: Ideally loss should decrease overtime
        epoch += 1
    
    if total_loss <= loss_threshold:
        print(f"Successfully pre-trained CNN model with loss total of {total_loss:.6f} at {epoch} epochs.")
    else:
        print(f"Training stopped after reaching max iterations of {MAX_ITER}. Loss: {total_loss}")
    
    # Save CNN
    torch.save(cnn_model.state_dict(),"__temp__/models/protein_cnn.pt")
    
    return total_loss_history,validation_loss_history