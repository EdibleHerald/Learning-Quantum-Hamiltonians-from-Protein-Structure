from Bio.PDB import PDBIO
from fetch_catalytic_sites import fetch_catalytic_sites


def export_residue(residue_dict:dict,export_dir):
    """
    Exports each BioPython.PDB residue in a dictionary as its own pdb.
    E.g.
    {'residue_name': residue_object }
    Exports as export_path/residue_name.pdb
    Returns nothing.
    """
    
    export_path_list = list()
    
    io = PDBIO()
    for name,residue in residue_dict.items():
        export_path = f"{export_dir}/{name}.pdb"
        io.set_structure(residue)
        io.save(export_path)
        export_path_list.append(export_path)
    
    return export_path_list 