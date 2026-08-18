import numpy as np
from Bio.PDB import PDBParser
import torch

def pdb_to_graph(pdb_filepath, distance_threshold=5.0):
    """
    Reads a PDB file and converts it into a molecular graph.
    Nodes = Atoms (Coordinates & Elements)
    Edges = Connections based on spatial distance threshold (e.g., < 5.0 Angstroms).
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_filepath)
    
    coordinates = []
    atomic_numbers = []
    
    # Standard mapping for simple proteins
    element_map = {'C': 6, 'N': 7, 'O': 8, 'H': 1, 'S': 16}
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coordinates.append(atom.get_coord())
                    elem = atom.element.upper()
                    atomic_numbers.append(element_map.get(elem, 6)) # Default to Carbon if unknown
      
    coords_np = np.array(coordinates)
    atoms_np = np.array(atomic_numbers)
    
    if len(coords_np) == 0:
        raise ValueError("No atoms found in PDB.")
    
    # Build Edges (Adjacency list) based on distance
    edges = []
    num_atoms = len(coords_np)
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist = np.linalg.norm(coords_np[i] - coords_np[j])
            if dist < distance_threshold:
                edges.append([i, j])
                edges.append([j, i]) # Undirected graph
    edges_np = np.array(edges).T # Shape [2, num_edges]
    
    return {
        'x': torch.tensor(atoms_np, dtype=torch.float32).view(-1, 1),
        'pos': torch.tensor(coords_np, dtype=torch.float32),
        'edge_index': torch.tensor(edges_np, dtype=torch.long)
    }

# Usage:
# graph_data = pdb_to_graph("testing_proteins/1F3R.pdb")

# 11.5s on average for 1F3R.pdb with no parallelization
# Resulting Edges array:
# [[   0    1    0 ... 3976 3975 3976]
# [   1    0    2 ... 3974 3976 3975]]