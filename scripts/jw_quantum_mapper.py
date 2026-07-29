import numpy as np

def apply_jw(coefficients, num_sites=4):
    """
    Applies a symbolic Jordan-Wigner transform to physical coefficients.
    Outputs executable Pauli Strings.
    """
    pauli_strings = []
    
    # 1. Extract Site Potentials (Epsilon)
    # The first 'num_sites' values map to individual site energies
    epsilons = coefficients[:num_sites]
    
    for i in range(num_sites):
        # JW Rule for a_i^dagger a_i: Maps to (I - Z_i)/2. 
        # We drop the Identity term for simulation simplicity, leaving the Z rotation.
        term = f"{-0.5 * epsilons[i]:.4f} * Z{i}"
        pauli_strings.append(term)
        
    # 2. Extract Hopping / Interaction Strengths (J)
    # The remaining values dictate connections between sites
    j_strengths = coefficients[num_sites:]
    
    index = 0
    for i in range(num_sites):
        for j in range(i + 1, num_sites):
            j_val = j_strengths[index]
            
            # JW Rule for a_i^dagger a_j + a_j^dagger a_i:
            # Requires X_i(Z_chain)X_j + Y_i(Z_chain)Y_j
            
            # Build the Z-string tally gates
            z_chain = ""
            for k in range(i + 1, j):
                z_chain += f"Z{k} "
                
            x_term = f"{0.5 * j_val:.4f} * X{i} {z_chain}X{j}"
            y_term = f"{0.5 * j_val:.4f} * Y{i} {z_chain}Y{j}"
            
            pauli_strings.append(x_term)
            pauli_strings.append(y_term)
            index += 1
            
    return pauli_strings

def display_instructions(pauli_strings):
    print("--- QUANTUM HARDWARE INSTRUCTIONS ---")
    for string in pauli_strings:
        print(string)
    print("-------------------------------------")

# Usage inside Jupyter:
# qubit_instructions = apply_jw(coefficients)
# display_instructions(qubit_instructions)
