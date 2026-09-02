# import hydride
from pdbfixer import PDBFixer
from openmm.app import PDBFile 

def add_hydrogen_atoms(pdb_file:str):
    """ 
    """

    fixer = PDBFixer(filename=pdb_file)
    fixer.addMissingHydrogens()
    PDBFile.writeFile(fixer.topology,fixer.positions,open(pdb_file,'w'))

# Example Usage:
# add_hydrogen_atoms("__temp__/pdb_exported")