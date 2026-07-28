import numpy as np
import warnings
from Bio.PDB import PDBParser

def pdb_to_tensor(pdb_filepath, grid_size=32):
    """
    Parses a PDB file, extracts atomic coordinates, and maps them to a 3D Voxel Tensor.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_filepath)
    
    # 1. Extract all X,Y,Z coordinates
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coords.append(atom.get_coord())
    
    coords = np.array(coords)
    if len(coords) == 0:
        raise ValueError("No atomic coordinates found in PDB file.")

    # 2. Normalize coordinates to fit within our 3D grid
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    
    # Scale to range [0, grid_size - 1]
    # Add a small epsilon to denominator to prevent division by zero
    scaled_coords = (coords - min_coords) / (max_coords - min_coords + 1e-8)
    grid_coords = np.round(scaled_coords * (grid_size - 1)).astype(int)
    
    # 3. Create the 3D Voxel Grid
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    
    # 4. Populate the grid (Density representation)
    for coord in grid_coords:
        x, y, z = coord
        voxel_grid[x, y, z] += 1.0  # Accumulate mass at this spatial voxel
        
    # Standardize to neural network format: [Batch, Channel, Depth, Height, Width]
    tensor_object = voxel_grid.reshape(1, 1, grid_size, grid_size, grid_size)
    
    return tensor_object

# Usage inside Jupyter:
# tensor = pdb_to_tensor("my_protein.pdb", grid_size=32)
