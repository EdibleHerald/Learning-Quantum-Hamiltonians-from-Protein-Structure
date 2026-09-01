import numpy as np
from Bio.PDB import PDBParser
from scipy.spatial import KDTree
from fetch_protein_info import fetch_catalytic_site_values

# Problem:
# The voxelization and graphing functions currently are completely mismatched.
# The voxelization function takes in the entire protein and converts to a voxel.
# The graphing function builds a graph based off a distance threshold that the voxels do not obey.

# The Solution:
# Create a function that takes a protein's PDB file and extract active site coordinates. 
# Then pass those results to both the voxelization function and graphing function in order
# to have some consistency between both models.

# Returns np.ndarray of coordinates for active sites
def active_site_calculation(pdb_filepath:str,distance_threshold:int=5.0):
    """
    Reads a PDB file and converts it into a molecular graph.
    Nodes = Atoms (Coordinates & Elements)
    Edges = Connections based on spatial distance threshold (e.g., < 5.0 Angstroms).
    """
    # Fetch catalytic site values:
    test_site_dictionary = fetch_catalytic_site_values(pdb_filepath.split("/")[-1].split(".")[0])
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_filepath)
    
    coordinates = []
    atomic_numbers = []

    # Standard mapping for simple proteins
    element_map = {'C': 6, 'N': 7, 'O': 8, 'H': 1, 'S': 16}
    
    chain_list = list(test_site_dictionary.keys())
        
    for model in structure:
        # print(model.child_dict)
        # Pick out chains with catalytic residues. Treating chain_list as a stack to cycle through chains.
        for i in range(len(chain_list)):
            print("ran!")
            curr_chain = chain_list.pop(0)
            if model.__contains__(curr_chain): # Entity base class stores children in a dictionary, so this is O(1)!
                chain = model.__getitem__(curr_chain)

                # Turn the residues in the chain into a list, then access based on
                # catalytic site values. 
                all_residues = chain.child_list
                
                # Grab list of wanted residue ids (sorted)
                residue_id_list = sorted(test_site_dictionary[curr_chain])

                # Selected catalytic residues
                residue_list = []
                for residue_id in residue_id_list:
                    residue_list.append(all_residues[residue_id - 1]) # Minus 1 since protein sequence ids are 1-indexed and lists are 0-indexed
                   
                # NEED TO DO: 
                # ADD RESIDUE ATOMS TO LIST AND ALTER VOXEL/GRAPH PIPELINE TO RENDER CORRECT ATOMS
                
                #     # We're looking for particular residues
                #     if residue_id_list and residue.id == residue_id_list[0]:
                #         counter+=1
                #         residue_id_list.pop(0)
                    # for atom in residue:
                    #     coordinates.append(atom.get_coord())
                    #     elem = atom.element.upper()
                    #     atomic_numbers.append(element_map.get(elem, 6)) # Default to Carbon if unknown
                
            else:
                chain_list.append(curr_chain) # If not in model, return to end of stack.
        
    print(counter)
    exit(1)
    
    coords_np = np.array(coordinates)
    atoms_np = np.array(atomic_numbers)
    
    if len(coords_np) == 0:
        raise ValueError("No atoms found in PDB.")
    
    # Build Edges (Adjacency list) based on distance
    edge_set = set() # To avoid duplicates
    edges = []
    
    # Create KDTree to query nearest neighbors.
    kdtree = KDTree(coords_np)

    num_atoms = len(coords_np)
    for i in range(num_atoms):
        # Fetch all points within 'distance_threshold' angstroms, returning sorted indices. 
        atom_list = kdtree.query_ball_point(x=coords_np[i],r=distance_threshold,return_sorted=True)
        
        for j in range(len(atom_list)):
            index = atom_list[j]
            
            # Skip adding edges that references itself (i.e. Not an actual relationship)
            if i == index:
                continue
            
            # Calculate distance between both atoms
            atom_distance = np.linalg.norm(coords_np[i] - coords_np[index])
            if atom_distance < distance_threshold:
                # Create both tuples to check for
                tuple1,list1 = (i,index),[i,index]
                tuple2,list2 = (index,i),[index,i]
                
                # Selectively add edges that don't already exist
                if tuple1 not in edge_set:
                    edges.append(list1)
                    edge_set.add(tuple1)
                
                if tuple2 not in edge_set:
                    edges.append(list2)        
                    edge_set.add(tuple2)
    
    edges_np = np.array(edges).T # Shape [2, num_edges]
    
    return {
        'x': torch.tensor(atoms_np, dtype=torch.float32).view(-1, 1),
        'pos': torch.tensor(coords_np, dtype=torch.float32),
        'edge_index': torch.tensor(edges_np, dtype=torch.long)
    }
    pass

active_site_calculation("proteins/training_proteins/9ATK.pdb")