import numpy as np

def build_matrix(coefficients, num_sites):
    """Reconstructs the 2D Hamiltonian Matrix from 1D coefficients."""
    matrix = np.zeros((num_sites, num_sites))
    
    # 1. Fill Diagonals (Site Energies)
    epsilons = coefficients[:num_sites]
    np.fill_diagonal(matrix, epsilons)
    
    # 2. Fill Off-Diagonals (Hopping Integrals J)
    j_strengths = coefficients[num_sites:]
    idx = 0
    for i in range(num_sites):
        for j in range(i + 1, num_sites):
            matrix[i, j] = j_strengths[idx]
            matrix[j, i] = j_strengths[idx] # Hermitian symmetry
            idx += 1
    return matrix

def cross_verify_pipelines(coeffs_track_A, coeffs_track_B, num_sites=4):
    """
    Performs the Twin Pipeline Checkpoints: 
    1. Mathematical MAE Check 
    2. Physical Ground State Energy Check
    """
    print("=== TWIN PIPELINE VERIFICATION REPORT ===")
    
    # CHECKPOINT 1: Mathematical Accuracy
    mae = np.mean(np.abs(coeffs_track_A - coeffs_track_B))
    print(f"\n[Checkpoint 1] Coefficient Mean Absolute Error (MAE): {mae:.6f} eV")
    if mae < 0.05:
        print("-> Status: PASS (High Mathematical Agreement)")
    else:
        print("-> Status: WARNING (Check spatial mapping drift)")
        
    # CHECKPOINT 2: Physical Ground State Energy
    mat_A = build_matrix(coeffs_track_A, num_sites)
    mat_B = build_matrix(coeffs_track_B, num_sites)
    
    # Diagonalize both to find E0 (lowest eigenvalue)
    eigenvalues_A = np.linalg.eigvalsh(mat_A)
    eigenvalues_B = np.linalg.eigvalsh(mat_B)
    
    E0_A = eigenvalues_A[0]
    E0_B = eigenvalues_B[0]
    
    delta_E = np.abs(E0_A - E0_B)
    
    print(f"\n[Checkpoint 2] Physical Ground State Energy (E0)")
    print(f"Track A (3D CNN) E0 : {E0_A:.6f} eV")
    print(f"Track B (ML-DFT) E0 : {E0_B:.6f} eV")
    print(f"Delta E (Error)     : {delta_E:.6f} eV")
    
    # 1 kcal/mol is roughly 0.043 eV
    if delta_E <= 0.043:
        print("-> Status: PASS (Within Chemical Accuracy! Ready for Quantum Simulation.)")
    else:
        print("-> Status: FAIL (Exceeds Chemical Accuracy. Do not send to QPU.)")
    
    print("=========================================")

# Usage:
# cross_verify_pipelines(cnn_coeffs, mldft_coeffs, num_sites=4)
