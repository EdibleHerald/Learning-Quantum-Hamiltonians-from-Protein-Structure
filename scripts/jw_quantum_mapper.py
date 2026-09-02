import numpy as np
from qiskit.quantum_info import SparsePauliOp

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

# To convert our list of strings into a PauliSum, we need to loop through each instruction
def convert_qubit_operators_to_pauli_operators(qubit_operators:list,num_sites:int):
    coef_list = []
    op_list = []
    # mp_coef_list = []
    # multi_qubit_operator_set = list()
    for i in qubit_operators:
        # Split up the operation into coefficient and the gates
        operation_list = i.split("*")
        
        # We grab the coefficient as a float and each operator as a single string   
        coef = float(operation_list[0])
        operators = operation_list[1].strip().split(" ")

        coef_list += [coef]
        
        # Go through each operator in the list, convert to Qiskit 'Pauli' term
        # First, we create a list of identity terms since each Pauli needs to be the same dimension
        start_op_list = list("I" * (num_sites * 2)) # Since spin isn't measured, we need to pad out the qubit operator size for UCCSD to accept it, this padding doesn't change the physics.
        
        # We want to not add even-numbered Y amount of observables due to creating zero gradients
        # (Insert reference here)
        # y_count = None
        for term in operators:
            
            # if term[0] == "Y":
            #     if y_count is None:
            #         y_count = 0
            #     y_count += 1
            
            # First term is letter, next is integer
            start_op_list[int(term[1])] = term[0]
        
        # Additionally, we can also define our multi-qubit operators here! (Note, some single-qubit operators are here too but we deal with that later!)
        fin_str = "".join(start_op_list)
        
        # if y_count is None or y_count % 2 == 1:
        #     multi_qubit_operator_set.append(fin_str)
        #     mp_coef_list.append(coef)
        
        op_list.append(fin_str)
    
    return SparsePauliOp(data=op_list,coeffs=coef_list)

def display_instructions(pauli_strings):
    print("--- QUANTUM HARDWARE INSTRUCTIONS ---")
    for string in pauli_strings:
        print(string)
    print("-------------------------------------")

# Usage inside Jupyter:
# qubit_instructions = apply_jw(coefficients)
# display_instructions(qubit_instructions)
