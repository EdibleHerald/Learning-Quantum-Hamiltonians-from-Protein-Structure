import numpy as np
from Bio.PDB import PDBParser
import torch
from scipy.spatial import KDTree
from active_site_calculation import active_site_calculation

def pdb_to_graph(pdb_filepath, distance_threshold=5.0):
    """
    Reads a PDB file and converts it into a molecular graph.
    Nodes = Atoms (Coordinates & Elements)
    Edges = Connections based on spatial distance threshold (e.g., < 5.0 Angstroms).
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_filepath)
    
    coordinates,atomic_numbers = active_site_calculation(pdb_filepath=pdb_filepath,distance_threshold=distance_threshold)
    
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

# Usage:
# graph_data = pdb_to_graph("proteins/training_proteins/9ATK.pdb")

