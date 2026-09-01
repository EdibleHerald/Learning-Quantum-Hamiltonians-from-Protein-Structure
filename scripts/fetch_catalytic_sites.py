import numpy as np
from Bio.PDB import PDBParser
from scipy.spatial import KDTree
from fetch_protein_info import fetch_catalytic_site_values

import torch

# Problem:
# The voxelization and graphing functions currently are completely mismatched.
# The voxelization function takes in the entire protein and converts to a voxel.
# The graphing function builds a graph based off a distance threshold that the voxels do not obey.

# The Solution:
# Create a function that takes a protein's PDB file and extract active site coordinates. 
# Then pass those results to both the voxelization function and graphing function in order
# to have some consistency between both models.

# Returns np.ndarray of coordinates for active sites
def fetch_catalytic_sites(pdb_filepath:str,distance_threshold:int=5.0):
    """
    Reads a PDB file and converts it into a molecular graph.
    Nodes = Atoms (Coordinates & Elements)
    Edges = Connections based on spatial distance threshold (e.g., < 5.0 Angstroms).
    """
    # Fetch catalytic site values:
    site_dict = fetch_catalytic_site_values(pdb_filepath.split("/")[-1].split(".")[0])
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_filepath)
    
    # coordinates = []
    # atomic_numbers = []

    # Standard mapping for simple proteins
    element_map = {'C': 6, 'N': 7, 'O': 8, 'H': 1, 'S': 16}
    
    chain_list = list(site_dict.keys())
    
    # Residue atom dictionary for export
    residue_dict = dict()
    
    for model in structure:
        # Pick out chains with catalytic residues. Treating chain_list as a stack 
        # to cycle through chains.
        for i in range(len(chain_list)):
            curr_chain = chain_list.pop(0)
            if model.__contains__(curr_chain): # Entity base class stores children in a dictionary, so this is O(1)!
                chain = model.__getitem__(curr_chain)

                # Turn the residues in the chain into a list, then access based on
                # catalytic site values. 
                all_residues = chain.child_list
                
                # Grab list of wanted residue ids (sorted)
                residue_id_list = sorted(site_dict[curr_chain])

                # Selected catalytic residues
                residue_list = []
                for residue_id in residue_id_list:
                    residue_list.append(all_residues[residue_id - 1]) # Minus 1 since protein sequence ids are 1-indexed and lists are 0-indexed

                # Add all atoms within each catalytic site residue
                # to a dictionary for export
                coordinates = []
                atomic_numbers = []
                for residue in residue_list:
                    for atom in residue:
                        coordinates.append(atom.get_coord())
                        elem = atom.element.upper()
                        atomic_numbers.append(element_map.get(elem, 6)) # Default to Carbon if unknown
                
                residue_dict[chain.id] = [np.array(coordinates),np.array(atomic_numbers)]
            else:
                chain_list.append(curr_chain) # If not in model, return to end of stack.
    
    # coords_np = np.array(coordinates)
    # atoms_np = np.array(atomic_numbers)
    
    if len(residue_dict) == 0:
        raise ValueError("No atoms found in PDB.")
    
    # Returns coordinates of extracted atom positions and their
    # accompanying atom element information. 
    return residue_dict