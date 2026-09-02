import sys
sys.path.append("scripts/process_pdb_scripts")
from export_residue_as_pdb import export_residue
from fetch_catalytic_sites import fetch_catalytic_sites
from hydrogen_capping import add_hydrogen_atoms
import os
import tempfile
from Bio.PDB import PDBParser
import numpy as np

def extract_atomic_info(pdb_filepath,parser):
    coordinates = []
    atomic_numbers = []
    
    # Standard mapping for simple proteins
    element_map = {'C': 6, 'N': 7, 'O': 8, 'H': 1, 'S': 16}
    
    # Fetch structure
    structure = parser.get_structure("protein", pdb_filepath)

    # Pick out only residue present
    residue = next(structure[0].get_residues())
    
    for atom in residue:
        coordinates.append(atom.get_coord())
        elem = atom.element.upper()
        atomic_numbers.append(element_map.get(elem, 6)) # Default to Carbon if unknown

    return coordinates, atomic_numbers

def process_pdb(pdb_path:str):
    coordinates = []
    atomic_numbers = []

    # Define Bio.PDB parser
    parser = PDBParser(QUIET=True)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Fetch catalytic residues
        residue_dict = fetch_catalytic_sites(pdb_path)
        print(len(residue_dict))
        if not residue_dict:
            return None,None # No catalytic sites found
        
        # Export each residue into their own pdb files
        export_path_list = export_residue(residue_dict=residue_dict,export_dir=tmpdir)
                
        # Perform hydrogen capping on all residues:
        for path in export_path_list:
            # Overwrite existing pdb
            add_hydrogen_atoms(pdb_file=path)
            
            # Extract coordinate data from this residue
            coords,a_nums = extract_atomic_info(pdb_filepath=path,parser=parser)
            
            # Append coordinates to full coordinate list
            coordinates += coords
            atomic_numbers += a_nums
    
    return np.array(coordinates),np.array(atomic_numbers)

# Example Usage:
# print(process_pdb("proteins/training_proteins/9ATK.pdb"))
