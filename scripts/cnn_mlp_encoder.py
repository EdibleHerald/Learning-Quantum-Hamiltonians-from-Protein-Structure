import torch
import torch.nn as nn
import torch.nn.functional as F

class ProteinPhysicsEncoder(nn.Module):
    def __init__(self, num_sites):
        super(ProteinPhysicsEncoder, self).__init__()
        
        # 3D CNN for Spatial Feature Extraction
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(2)
        
        # Assuming 32x32x32 input -> pool to 16 -> pool to 8. 
        # Flattened size = 32 * 8 * 8 * 8 = 16384
        self.fc1 = nn.Linear(16384, 128) # The Latent Embedding
        
        # MLP for Physical Translation
        self.num_sites = num_sites
        # We need N Site Energies (Epsilon) and N*(N-1)/2 Coupling Strengths (J)
        self.num_interactions = num_sites + (num_sites * (num_sites - 1)) // 2
        
        self.fc2 = nn.Linear(128, self.num_interactions)
        
    def forward(self, x):
        # Pass through CNN
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        
        x = x.view(x.size(0), -1) # Flatten
        latent_embedding = F.relu(self.fc1(x))
        
        # Pass through MLP to get physical coefficients
        physics_coefficients = self.fc2(latent_embedding)
        return physics_coefficients

def get_hamiltonian(tensor_array, num_qubits=4):
    """
    Helper function to run the tensor through the model.
    num_qubits defines how many sites we are simulating on hardware.
    """
    model = ProteinPhysicsEncoder(num_sites=num_qubits)
    
    # Convert numpy array to PyTorch Tensor
    tensor_pt = torch.from_numpy(tensor_array)
    
    # Forward pass (without calculating gradients for inference)
    with torch.no_grad():
        coeffs = model(tensor_pt)
        
    return coeffs.numpy()[0] # Return the 1D array of physical values

# Usage inside Jupyter:
# coefficients = get_hamiltonian(tensor, num_qubits=4)
