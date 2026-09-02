from qiskit_nature.second_q.drivers import PySCFDriver

# Use Qiskit Nature to generate operator pools
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import SlaterDeterminant

# H2O molecule, neutral charge (0), singlet state (spin=0, multiplicity=1)
driver = PySCFDriver(atom="O 0 0 0; H 0.75 -0.46 0; H -0.75 -0.46 0",
                     charge=0,
                     spin=0)
problem = driver.run()

num_alpha = problem.num_alpha
num_beta = problem.num_beta
total_electrons = num_alpha + num_beta

print(f"Alpha Electrons: {num_alpha}")
print(f"Beta Electrons: {num_beta}")
# test = SlaterDeterminant(hamiltonian_matrix)

num_particles = (num_alpha,num_beta)
mapper = JordanWignerMapper()
hartree = HartreeFock(num_spatial_orbitals=total_electrons, num_particles=num_particles,qubit_mapper=mapper)
test_ansatz = UCCSD(num_spatial_orbitals=total_electrons,num_particles=num_particles,qubit_mapper=mapper)
# # Extract operators:
# operator_pool = list(test_ansatz.operators) 
# print(len(operator_pool))